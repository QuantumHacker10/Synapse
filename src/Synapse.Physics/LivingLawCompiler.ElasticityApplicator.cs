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

    /// <summary>Elasticity applicator: F = -k*Δx.</summary>
    public sealed class ElasticityApplicator : LawApplicator
    {
        public ElasticityApplicator() : base("Elasticity", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float k = config.Constants.TryGetValue("k", out float kv) ? kv : 100f;
            float mass = config.Constants.TryGetValue("m", out float mv) ? mv : 1f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            var vx = field.VelocityX;
            var vy = field.VelocityY;
            var vz = field.VelocityZ;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float div = ComputeDivergence3D(vx, vy, vz, x, y, z, dx);
                        float accel = -k * div / mass;
                        vx[x, y, z] += accel * dt * 0.5f;
                        vy[x, y, z] += accel * dt * 0.5f;
                        vz[x, y, z] += accel * dt * 0.5f;
                    }

            ApplyBoundaryConditions(field, config);
        }
    }
}
