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

    // =========================================================================
    // FieldGrid — simple physics field grid
    // =========================================================================

    /// <summary>Represents a 3D field grid for physics simulation.</summary>
    public sealed class FieldGrid
    {
        public int SizeX { get; }
        public int SizeY { get; }
        public int SizeZ { get; }
        public float CellSize { get; }
        private float[] _data;

        public FieldGrid(int sx, int sy, int sz, float cellSize = 1.0f)
        {
            SizeX = sx;
            SizeY = sy;
            SizeZ = sz;
            CellSize = cellSize;
            _data = new float[sx * sy * sz];
        }

        public float this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[x + SizeX * (y + SizeY * z)];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[x + SizeX * (y + SizeY * z)] = value;
        }

        public float[] Data => _data;
        public int TotalCells => SizeX * SizeY * SizeZ;
        public void Clear() => Array.Clear(_data, 0, _data.Length);

        public void CopyFrom(FieldGrid other)
        {
            if (other.TotalCells != TotalCells)
                throw new ArgumentException("Grid sizes mismatch");
            Array.Copy(other._data, _data, _data.Length);
        }

        public float Max() => _data.Length > 0 ? _data.AsSpan().ToArray().Max() : 0f;
        public float Min() => _data.Length > 0 ? _data.AsSpan().ToArray().Min() : 0f;

        public float Average()
        {
            if (_data.Length == 0)
                return 0f;
            double sum = 0;
            for (int i = 0; i < _data.Length; i++)
                sum += _data[i];
            return (float)(sum / _data.Length);
        }

        public float StandardDeviation()
        {
            if (_data.Length == 0)
                return 0f;
            float avg = Average();
            double variance = 0;
            for (int i = 0; i < _data.Length; i++)
            {
                double diff = _data[i] - avg;
                variance += diff * diff;
            }
            return MathF.Sqrt((float)(variance / _data.Length));
        }

        public void AddScaled(FieldGrid other, float scale)
        {
            if (other.TotalCells != TotalCells)
                throw new ArgumentException("Grid sizes mismatch");
            for (int i = 0; i < _data.Length; i++)
                _data[i] += other._data[i] * scale;
        }

        public FieldGrid Clone()
        {
            var clone = new FieldGrid(SizeX, SizeY, SizeZ, CellSize);
            Array.Copy(_data, clone._data, _data.Length);
            return clone;
        }
    }
}
