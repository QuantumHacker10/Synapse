using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Encodage multi-résolution par grille de hachage (façon Instant-NGP).
/// Remplace l'encodage positionnel global par des features locales apprises,
/// réparties sur plusieurs niveaux de résolution spatiale.
/// </summary>
public sealed class MultiResolutionHashEncoder
{
    public const int NumLevels = 8;
    public const int FeaturesPerLevel = 2;
    public const int HashTableSizeLog2 = 14;
    public const float BaseResolution = 16f;
    public const float FinestResolution = 512f;

    private readonly float[][] _hashTables;
    private readonly float _growthFactor;

    public int OutputDimension => NumLevels * FeaturesPerLevel;
    public int TableSize => 1 << HashTableSizeLog2;

    /// <summary>
    /// Accès en lecture aux tables de hachage (pour sérialisation / entraînement).
    /// </summary>
    public ReadOnlySpan<float> GetLevelTable(int level) => _hashTables[level];

    /// <summary>
    /// Accès mutable pour l'entraînement (SGD sur les features).
    /// </summary>
    public Span<float> GetMutableLevelTable(int level) => _hashTables[level];

    /// <summary>
    /// Contribution d'un coin de voxel lors de l'interpolation trilinéaire.
    /// </summary>
    public readonly struct CornerContribution
    {
        public readonly int Level;
        public readonly int HashIndex;
        public readonly int Feature;
        public readonly float Weight;

        public CornerContribution(int level, int hashIndex, int feature, float weight)
        {
            Level = level;
            HashIndex = hashIndex;
            Feature = feature;
            Weight = weight;
        }
    }

    public MultiResolutionHashEncoder(Random random)
    {
        _growthFactor = MathF.Exp(
            (MathF.Log(FinestResolution) - MathF.Log(BaseResolution)) / (NumLevels - 1));

        int tableSize = TableSize;
        _hashTables = new float[NumLevels][];
        for (int l = 0; l < NumLevels; l++)
        {
            _hashTables[l] = new float[tableSize * FeaturesPerLevel];
            for (int i = 0; i < _hashTables[l].Length; i++)
                _hashTables[l][i] = (float)(random.NextDouble() * 2e-4 - 1e-4);
        }
    }

    /// <summary>
    /// Reconstruit l'encodeur à partir de tables sérialisées.
    /// </summary>
    public MultiResolutionHashEncoder(ReadOnlySpan<float> serializedTables)
    {
        _growthFactor = MathF.Exp(
            (MathF.Log(FinestResolution) - MathF.Log(BaseResolution)) / (NumLevels - 1));

        int expected = GetTotalParameterCount();
        if (serializedTables.Length < expected)
            throw new ArgumentException($"Expected at least {expected} hash table entries, got {serializedTables.Length}.");

        int tableSize = TableSize;
        _hashTables = new float[NumLevels][];
        int offset = 0;
        for (int l = 0; l < NumLevels; l++)
        {
            _hashTables[l] = new float[tableSize * FeaturesPerLevel];
            serializedTables.Slice(offset, _hashTables[l].Length).CopyTo(_hashTables[l]);
            offset += _hashTables[l].Length;
        }
    }

    public static int GetTotalParameterCount() =>
        NumLevels * (1 << HashTableSizeLog2) * FeaturesPerLevel;

    public void Encode(Vector3 position, Span<float> output) =>
        Encode(position, output, contributions: null);

    /// <summary>
    /// Encode une position et optionnellement enregistre les contributions
    /// pour la rétropropagation vers les tables de hachage.
    /// </summary>
    public void Encode(Vector3 position, Span<float> output, List<CornerContribution>? contributions)
    {
        if (output.Length < OutputDimension)
            throw new ArgumentException($"Output span must have at least {OutputDimension} elements.");

        contributions?.Clear();

        for (int l = 0; l < NumLevels; l++)
        {
            float resolution = BaseResolution * MathF.Pow(_growthFactor, l);
            Vector3 gridPos = position * resolution;

            Vector3 floor = new(MathF.Floor(gridPos.X), MathF.Floor(gridPos.Y), MathF.Floor(gridPos.Z));
            Vector3 frac = gridPos - floor;

            Span<float> accum = stackalloc float[FeaturesPerLevel];
            accum.Clear();

            int tableEntries = _hashTables[l].Length / FeaturesPerLevel;

            for (int corner = 0; corner < 8; corner++)
            {
                int cx = corner & 1;
                int cy = (corner >> 1) & 1;
                int cz = (corner >> 2) & 1;

                float weight =
                    (cx == 1 ? frac.X : 1 - frac.X) *
                    (cy == 1 ? frac.Y : 1 - frac.Y) *
                    (cz == 1 ? frac.Z : 1 - frac.Z);

                int hash = HashCoord(
                    (int)floor.X + cx, (int)floor.Y + cy, (int)floor.Z + cz,
                    tableEntries);

                for (int f = 0; f < FeaturesPerLevel; f++)
                {
                    accum[f] += weight * _hashTables[l][hash * FeaturesPerLevel + f];
                    contributions?.Add(new CornerContribution(l, hash, f, weight));
                }
            }

            for (int f = 0; f < FeaturesPerLevel; f++)
                output[l * FeaturesPerLevel + f] = accum[f];
        }
    }

    /// <summary>
    /// Applique un gradient sur une feature encodée vers les entrées de table touchées.
    /// </summary>
    public void AccumulateFeatureGradient(
        ReadOnlySpan<CornerContribution> contributions,
        ReadOnlySpan<float> encodedGradients,
        float learningRate)
    {
        for (int i = 0; i < contributions.Length; i++)
        {
            var c = contributions[i];
            int encodedIdx = c.Level * FeaturesPerLevel + c.Feature;
            float grad = encodedGradients[encodedIdx] * c.Weight;
            _hashTables[c.Level][c.HashIndex * FeaturesPerLevel + c.Feature] -= learningRate * grad;
        }
    }

    public float[] Serialize()
    {
        float[] result = new float[GetTotalParameterCount()];
        int offset = 0;
        for (int l = 0; l < NumLevels; l++)
        {
            _hashTables[l].CopyTo(result, offset);
            offset += _hashTables[l].Length;
        }
        return result;
    }

    public MultiResolutionHashEncoder Clone()
    {
        var tables = Serialize();
        return new MultiResolutionHashEncoder(tables);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HashCoord(int x, int y, int z, int tableSize)
    {
        const uint p1 = 1u, p2 = 2654435761u, p3 = 805459861u;
        uint h = (uint)x * p1 ^ (uint)y * p2 ^ (uint)z * p3;
        return (int)(h % (uint)tableSize);
    }
}
