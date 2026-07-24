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

    /// <summary>Comparison result between two law versions.</summary>
    public readonly struct ComparisonResult
    {
        public readonly float MaxDivergence;
        public readonly float MeanDivergence;
        public readonly float RootMeanSquareError;
        public readonly float KolmogorovSmirnovStatistic;
        public readonly float StructuralSimilarity;
        public readonly int ExpressionEditDistance;
        public readonly string[] Differences;
        public readonly bool PhysicallyEquivalent;

        public ComparisonResult(float maxDiv, float meanDiv, float rmse, float ks, float ssim,
            int editDist, string[] diffs, bool physEq)
        {
            MaxDivergence = maxDiv;
            MeanDivergence = meanDiv;
            RootMeanSquareError = rmse;
            KolmogorovSmirnovStatistic = ks;
            StructuralSimilarity = ssim;
            ExpressionEditDistance = editDist;
            Differences = diffs;
            PhysicallyEquivalent = physEq;
        }
    }
}
