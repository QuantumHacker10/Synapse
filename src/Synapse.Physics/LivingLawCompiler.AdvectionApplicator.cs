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

    /// <summary>Advection applicator using upwind scheme.</summary>
    public sealed class AdvectionApplicator : LawApplicator
    {
        public AdvectionApplicator() : base("Advection", "density") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var density = field.Density;
            var densityNew = density.Clone();
            float vx = config.Constants.TryGetValue("vx", out float vxv) ? vxv : 1f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float dudx = vx > 0
                            ? (density[x, y, z] - density[x - 1, y, z]) / dx
                            : (density[x + 1, y, z] - density[x, y, z]) / dx;
                        densityNew[x, y, z] = density[x, y, z] - vx * dt * dudx;
                    }

            field.Density.CopyFrom(densityNew);
            ApplyBoundaryConditions(field, config);
        }
    }
}
