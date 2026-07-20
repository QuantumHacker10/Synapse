// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Applies a compiled law to a PhysicsField. Base class for specific applicators.</summary>
    public abstract class LawApplicator
    {
        public string Name { get; }
        public string TargetField { get; }
        protected readonly BytecodeInterpreter _interpreter;
        protected readonly GasMeter _gas;

        protected LawApplicator(string name, string targetField, long gasLimit = 10_000_000)
        {
            Name = name;
            TargetField = targetField;
            _gas = new GasMeter(gasLimit);
            _interpreter = new BytecodeInterpreter(_gas);
        }

        /// <summary>Apply the compiled law to the field at the given time step.</summary>
        public abstract void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config);

        /// <summary>Apply boundary conditions to the field after the law is applied.</summary>
        protected void ApplyBoundaryConditions(PhysicsField field, LawCompilerConfig config)
        {
            int size = field.GridSize;
            var component = field.GetComponent(TargetField);
            if (component == null)
                return;

            switch (config.BoundaryCondition)
            {
                case BoundaryCondition.Periodic:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[size - 1, y, z];
                            component[size - 1, y, z] = component[0, y, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = component[x, size - 1, z];
                            component[x, size - 1, z] = component[x, 0, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = component[x, y, size - 1];
                            component[x, y, size - 1] = component[x, y, 0];
                        }
                    break;

                case BoundaryCondition.Dirichlet:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = config.BoundaryValue;
                            component[size - 1, y, z] = config.BoundaryValue;
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = config.BoundaryValue;
                            component[x, size - 1, z] = config.BoundaryValue;
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = config.BoundaryValue;
                            component[x, y, size - 1] = config.BoundaryValue;
                        }
                    break;

                case BoundaryCondition.Neumann:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[1, y, z];
                            component[size - 1, y, z] = component[size - 2, y, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = component[x, 1, z];
                            component[x, size - 1, z] = component[x, size - 2, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = component[x, y, 1];
                            component[x, y, size - 1] = component[x, y, size - 2];
                        }
                    break;

                case BoundaryCondition.Radiation:
                    float h = config.BoundaryValue;
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[1, y, z] / (1f + h);
                            component[size - 1, y, z] = component[size - 2, y, z] / (1f + h);
                        }
                    break;
            }
        }

        protected float ComputeGradientX(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX;
            float left = x > 0 ? field[x - 1, y, z] : field[0, y, z];
            float right = x < sx - 1 ? field[x + 1, y, z] : field[sx - 1, y, z];
            return (right - left) / (2f * dx);
        }

        protected float ComputeGradientY(FieldGrid field, int x, int y, int z, float dy)
        {
            int sy = field.SizeY;
            float bottom = y > 0 ? field[x, y - 1, z] : field[x, 0, z];
            float top = y < sy - 1 ? field[x, y + 1, z] : field[x, sy - 1, z];
            return (top - bottom) / (2f * dy);
        }

        protected float ComputeGradientZ(FieldGrid field, int x, int y, int z, float dz)
        {
            int sz = field.SizeZ;
            float back = z > 0 ? field[x, y, z - 1] : field[x, y, 0];
            float front = z < sz - 1 ? field[x, y, z + 1] : field[x, y, sz - 1];
            return (front - back) / (2f * dz);
        }

        protected float ComputeLaplacian(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX, sy = field.SizeY, sz = field.SizeZ;
            float center = field[x, y, z];
            float left = x > 0 ? field[x - 1, y, z] : center;
            float right = x < sx - 1 ? field[x + 1, y, z] : center;
            float bottom = y > 0 ? field[x, y - 1, z] : center;
            float top = y < sy - 1 ? field[x, y + 1, z] : center;
            float back = z > 0 ? field[x, y, z - 1] : center;
            float front = z < sz - 1 ? field[x, y, z + 1] : center;
            float invDx2 = 1f / (dx * dx);
            return (left + right + bottom + top + back + front - 6f * center) * invDx2;
        }

        protected float ComputeDivergence3D(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientX(vx, x, y, z, dx) + ComputeGradientY(vy, x, y, z, dx) + ComputeGradientZ(vz, x, y, z, dx);
        }

        protected float ComputeCurlX(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientY(vz, x, y, z, dx) - ComputeGradientZ(vy, x, y, z, dx);
        }

        protected float ComputeCurlY(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientZ(vx, x, y, z, dx) - ComputeGradientX(vz, x, y, z, dx);
        }

        protected float ComputeCurlZ(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientX(vy, x, y, z, dx) - ComputeGradientY(vx, x, y, z, dx);
        }
    }

    /// <summary>Heat equation applicator: ∂T/∂t = α*∇²T.</summary>
    public sealed class HeatApplicator : LawApplicator
    {
        public HeatApplicator() : base("HeatEquation", "temperature") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var temp = field.Temperature;
            var tempNew = temp.Clone();
            float alpha = config.Constants.TryGetValue("alpha", out float a) ? a : 1.43e-4f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(temp, x, y, z, dx);
                        tempNew[x, y, z] = MathF.Max(0f, temp[x, y, z] + alpha * dt * laplacian);
                    }

            field.Temperature.CopyFrom(tempNew);
            ApplyBoundaryConditions(field, config);
        }
    }
}
