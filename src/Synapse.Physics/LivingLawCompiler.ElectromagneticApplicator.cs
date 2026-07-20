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

    /// <summary>Electromagnetic wave applicator (Maxwell-like).</summary>
    public sealed class ElectromagneticApplicator : LawApplicator
    {
        public ElectromagneticApplicator() : base("Electromagnetic", "temperature") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float epsilon = config.Constants.TryGetValue("eps", out float eps) ? eps : 8.854e-12f;
            float mu0 = config.Constants.TryGetValue("mu0", out float mu0v) ? mu0v : 1.25663706212e-6f;
            float dx = config.CellSize;
            float c = 1f / MathF.Sqrt(epsilon * mu0);
            float dx2 = dx * dx;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            var temp = field.Temperature;
            var tempNew = temp.Clone();

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(temp, x, y, z, dx);
                        tempNew[x, y, z] = 2f * temp[x, y, z] - temp[x, y, z] + c * c * dt * dt * laplacian;
                    }

            field.Temperature.CopyFrom(tempNew);
            ApplyBoundaryConditions(field, config);
        }
    }
}
