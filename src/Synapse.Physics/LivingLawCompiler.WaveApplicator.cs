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

    /// <summary>Wave equation applicator: ∂²u/∂t² = c²*∇²u.</summary>
    public sealed class WaveApplicator : LawApplicator
    {
        private FieldGrid? _previousState;

        public WaveApplicator() : base("WaveEquation", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var velocity = field.VelocityX;
            if (_previousState == null)
            { _previousState = velocity.Clone(); return; }

            float c = config.Constants.TryGetValue("c", out float cv) ? cv : 340f;
            float dx = config.CellSize;
            float dt2 = dt * dt, c2 = c * c;
            var newState = velocity.Clone();
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(velocity, x, y, z, dx);
                        float uPrev = _previousState[x, y, z];
                        float uCurr = velocity[x, y, z];
                        newState[x, y, z] = 2f * uCurr - uPrev + c2 * dt2 * laplacian;
                    }

            _previousState = velocity.Clone();
            velocity.CopyFrom(newState);
            ApplyBoundaryConditions(field, config);
        }
    }
}
