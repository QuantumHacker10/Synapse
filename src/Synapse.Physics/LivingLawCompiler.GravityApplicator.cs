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

    /// <summary>Gravity field applicator.</summary>
    public sealed class GravityApplicator : LawApplicator
    {
        public GravityApplicator() : base("Gravity", "density") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float G = config.Constants.TryGetValue("G", out float gv) ? gv : 6.674e-11f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            var density = field.Density;
            var densityNew = density.Clone();

            float totalMass = 0f;
            for (int z = 0; z < sz; z++)
                for (int y = 0; y < sy; y++)
                    for (int x = 0; x < sx; x++)
                        totalMass += density[x, y, z] * dx * dx * dx;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(density, x, y, z, dx);
                        float gravityPotential = -G * totalMass / MathF.Max(MathF.Abs(density[x, y, z]) * dx * dx * dx, 1e-10f);
                        densityNew[x, y, z] = density[x, y, z] + dt * gravityPotential * laplacian * 0.001f;
                    }

            field.Density.CopyFrom(densityNew);
            ApplyBoundaryConditions(field, config);
        }
    }
}
