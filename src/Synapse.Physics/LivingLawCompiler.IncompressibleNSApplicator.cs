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

    /// <summary>Incompressible Navier-Stokes applicator (simplified).</summary>
    public sealed class IncompressibleNSApplicator : LawApplicator
    {
        public IncompressibleNSApplicator() : base("IncompressibleNS", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float viscosity = config.Constants.TryGetValue("mu", out float mu) ? mu : 1.002e-3f;
            float density = config.Constants.TryGetValue("rho", out float rho) ? rho : 1000f;
            float dx = config.CellSize;
            float nu = viscosity / density;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            var vx = field.VelocityX;
            var vy = field.VelocityY;
            var vz = field.VelocityZ;
            var vxNew = vx.Clone();
            var vyNew = vy.Clone();
            var vzNew = vz.Clone();

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float lapVx = ComputeLaplacian(vx, x, y, z, dx);
                        float lapVy = ComputeLaplacian(vy, x, y, z, dx);
                        float lapVz = ComputeLaplacian(vz, x, y, z, dx);
                        float dVxDx = ComputeGradientX(vx, x, y, z, dx);
                        float dVyDy = ComputeGradientY(vy, x, y, z, dx);
                        float dVzDz = ComputeGradientZ(vz, x, y, z, dx);

                        vxNew[x, y, z] = vx[x, y, z] + dt * (nu * lapVx - vx[x, y, z] * dVxDx);
                        vyNew[x, y, z] = vy[x, y, z] + dt * (nu * lapVy - vy[x, y, z] * dVyDy);
                        vzNew[x, y, z] = vz[x, y, z] + dt * (nu * lapVz - vz[x, y, z] * dVzDz);
                    }

            vx.CopyFrom(vxNew);
            vy.CopyFrom(vyNew);
            vz.CopyFrom(vzNew);
            ApplyBoundaryConditions(field, config);
        }
    }
}
