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
    // LawComparison — compare simulation results
    // =========================================================================

    /// <summary>Compares simulation results between different law versions.</summary>
    public sealed class LawComparison
    {
        /// <summary>Compare two field grids point-wise.</summary>
        public static ComparisonResult CompareFields(FieldGrid fieldA, FieldGrid fieldB, int expressionEditDistance = 0)
        {
            if (fieldA.TotalCells != fieldB.TotalCells)
                throw new ArgumentException("Field sizes must match");

            int n = fieldA.TotalCells;
            float maxDiv = 0f;
            double sumDiv = 0, sumSqDiff = 0;
            var diffs = new List<string>();

            for (int i = 0; i < n; i++)
            {
                float diff = MathF.Abs(fieldA.Data[i] - fieldB.Data[i]);
                if (diff > maxDiv)
                    maxDiv = diff;
                sumDiv += diff;
                sumSqDiff += (double)diff * diff;
            }

            float meanDiv = (float)(sumDiv / n);
            float rmse = MathF.Sqrt((float)(sumSqDiff / n));

            // Kolmogorov-Smirnov statistic
            var valuesA = new float[n];
            var valuesB = new float[n];
            Array.Copy(fieldA.Data, valuesA, n);
            Array.Copy(fieldB.Data, valuesB, n);
            Array.Sort(valuesA);
            Array.Sort(valuesB);

            float ksMax = 0f;
            for (int i = 0; i < n; i++)
            {
                float cdfA = (float)(i + 1) / n;
                int bIdx = Array.BinarySearch(valuesB, valuesA[i]);
                if (bIdx < 0)
                    bIdx = ~bIdx;
                float cdfB = (float)(bIdx + 1) / n;
                float ksDiff = MathF.Abs(cdfA - cdfB);
                if (ksDiff > ksMax)
                    ksMax = ksDiff;
            }

            // SSIM approximation based on statistics
            float muA = fieldA.Average(), muB = fieldB.Average();
            float sigA = fieldA.StandardDeviation(), sigB = fieldB.StandardDeviation();
            float sigAB = 0f;
            for (int i = 0; i < n; i++)
                sigAB += (fieldA.Data[i] - muA) * (fieldB.Data[i] - muB);
            sigAB /= n;

            float c1 = 0.01f * 0.01f, c2 = 0.03f * 0.03f;
            float ssim = ((2f * muA * muB + c1) * (2f * sigAB + c2)) /
                         ((muA * muA + muB * muB + c1) * (sigA * sigA + sigB * sigB + c2));

            if (maxDiv > 0.01f)
                diffs.Add($"Max divergence: {maxDiv:F6}");
            if (meanDiv > 0.001f)
                diffs.Add($"Mean divergence: {meanDiv:F6}");
            if (rmse > 0.01f)
                diffs.Add($"RMSE: {rmse:F6}");
            if (ksMax > 0.1f)
                diffs.Add($"KS statistic: {ksMax:F4} (distributions differ significantly)");
            if (MathF.Abs(muA - muB) > 0.01f)
                diffs.Add($"Mean difference: {muA:F4} vs {muB:F4}");
            if (MathF.Abs(sigA - sigB) > 0.01f)
                diffs.Add($"Std dev difference: {sigA:F4} vs {sigB:F4}");

            bool physEq = maxDiv < 0.01f && rmse < 0.01f && ksMax < 0.05f;
            return new ComparisonResult(maxDiv, meanDiv, rmse, ksMax, ssim, expressionEditDistance, diffs.ToArray(), physEq);
        }

        /// <summary>Compare two physics fields.</summary>
        public static ComparisonResult ComparePhysicsFields(PhysicsField fieldA, PhysicsField fieldB)
        {
            var tempComp = CompareFields(fieldA.Temperature, fieldB.Temperature);
            var presComp = CompareFields(fieldA.Pressure, fieldB.Pressure);
            var densComp = CompareFields(fieldA.Density, fieldB.Density);

            float maxDiv = MathF.Max(tempComp.MaxDivergence, MathF.Max(presComp.MaxDivergence, densComp.MaxDivergence));
            float meanDiv = (tempComp.MeanDivergence + presComp.MeanDivergence + densComp.MeanDivergence) / 3f;
            float rmse = MathF.Sqrt((tempComp.RootMeanSquareError * tempComp.RootMeanSquareError +
                presComp.RootMeanSquareError * presComp.RootMeanSquareError +
                densComp.RootMeanSquareError * densComp.RootMeanSquareError) / 3f);
            float ks = MathF.Max(tempComp.KolmogorovSmirnovStatistic,
                MathF.Max(presComp.KolmogorovSmirnovStatistic, densComp.KolmogorovSmirnovStatistic));
            float ssim = (tempComp.StructuralSimilarity + presComp.StructuralSimilarity + densComp.StructuralSimilarity) / 3f;

            var diffs = new List<string>();
            diffs.AddRange(tempComp.Differences.Select(d => $"Temperature: {d}"));
            diffs.AddRange(presComp.Differences.Select(d => $"Pressure: {d}"));
            diffs.AddRange(densComp.Differences.Select(d => $"Density: {d}"));
            bool physEq = tempComp.PhysicallyEquivalent && presComp.PhysicallyEquivalent && densComp.PhysicallyEquivalent;
            return new ComparisonResult(maxDiv, meanDiv, rmse, ks, ssim, 0, diffs.ToArray(), physEq);
        }

        /// <summary>Compare two simulation snapshots over time.</summary>
        public static List<ComparisonResult> CompareTimeSeries(
            List<PhysicsField> snapshotsA, List<PhysicsField> snapshotsB)
        {
            int count = Math.Min(snapshotsA.Count, snapshotsB.Count);
            var results = new List<ComparisonResult>(count);
            for (int i = 0; i < count; i++)
                results.Add(ComparePhysicsFields(snapshotsA[i], snapshotsB[i]));
            return results;
        }

        /// <summary>Compute divergence field between two grids.</summary>
        public static FieldGrid ComputeDivergenceField(FieldGrid a, FieldGrid b)
        {
            if (a.SizeX != b.SizeX || a.SizeY != b.SizeY || a.SizeZ != b.SizeZ)
                throw new ArgumentException("Grid sizes must match");
            var div = new FieldGrid(a.SizeX, a.SizeY, a.SizeZ);
            for (int z = 0; z < a.SizeZ; z++)
                for (int y = 0; y < a.SizeY; y++)
                    for (int x = 0; x < a.SizeX; x++)
                        div[x, y, z] = a[x, y, z] - b[x, y, z];
            return div;
        }

        /// <summary>Compute L2 norm of the difference between two fields.</summary>
        public static float ComputeL2Norm(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            double sum = 0;
            for (int i = 0; i < a.TotalCells; i++)
            {
                double diff = a.Data[i] - b.Data[i];
                sum += diff * diff;
            }
            return MathF.Sqrt((float)(sum / a.TotalCells));
        }

        /// <summary>Compute L-infinity norm (max absolute difference).</summary>
        public static float ComputeLInfNorm(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            float maxDiff = 0f;
            for (int i = 0; i < a.TotalCells; i++)
            {
                float diff = MathF.Abs(a.Data[i] - b.Data[i]);
                if (diff > maxDiff)
                    maxDiff = diff;
            }
            return maxDiff;
        }

        /// <summary>Compute correlation coefficient between two fields.</summary>
        public static float ComputeCorrelation(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            int n = a.TotalCells;
            float muA = a.Average(), muB = b.Average();
            float num = 0f, denA = 0f, denB = 0f;
            for (int i = 0; i < n; i++)
            {
                float dA = a.Data[i] - muA;
                float dB = b.Data[i] - muB;
                num += dA * dB;
                denA += dA * dA;
                denB += dB * dB;
            }
            float den = MathF.Sqrt(denA * denB);
            return den > 0f ? num / den : 0f;
        }

        /// <summary>Compute energy norm of the difference.</summary>
        public static float ComputeEnergyNorm(FieldGrid a, FieldGrid b, float dx)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            double sum = 0;
            float invDx2 = 1f / (dx * dx);
            for (int z = 1; z < a.SizeZ - 1; z++)
                for (int y = 1; y < a.SizeY - 1; y++)
                    for (int x = 1; x < a.SizeX - 1; x++)
                    {
                        float valA = a[x, y, z];
                        float valB = b[x, y, z];
                        sum += (valA - valB) * (valA - valB) * dx * dx * dx;
                    }
            return MathF.Sqrt((float)sum);
        }

        /// <summary>Compute spectral analysis difference.</summary>
        public static float ComputeSpectralDifference(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            float muA = a.Average(), muB = b.Average();
            float varA = 0f, varB = 0f, covAB = 0f;
            for (int i = 0; i < a.TotalCells; i++)
            {
                float dA = a.Data[i] - muA;
                float dB = b.Data[i] - muB;
                varA += dA * dA;
                varB += dB * dB;
                covAB += dA * dB;
            }
            float n = a.TotalCells;
            varA /= n;
            varB /= n;
            covAB /= n;
            return MathF.Sqrt(MathF.Max(0f, varA + varB - 2f * covAB));
        }
    }    // =========================================================================
}
