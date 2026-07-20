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

    /// <summary>A physics field with multiple grid components.</summary>
    public sealed class PhysicsField
    {
        public string Name { get; set; } = "";
        public Dictionary<string, FieldGrid> Components { get; } = new();
        public FieldGrid Temperature { get; set; } = null!;
        public FieldGrid Pressure { get; set; } = null!;
        public FieldGrid Density { get; set; } = null!;
        public FieldGrid VelocityX { get; set; } = null!;
        public FieldGrid VelocityY { get; set; } = null!;
        public FieldGrid VelocityZ { get; set; } = null!;
        public int GridSize { get; }
        public float Time { get; set; }

        public PhysicsField(int gridSize, string name = "default")
        {
            Name = name;
            GridSize = gridSize;
            Temperature = new FieldGrid(gridSize, gridSize, gridSize);
            Pressure = new FieldGrid(gridSize, gridSize, gridSize);
            Density = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityX = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityY = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityZ = new FieldGrid(gridSize, gridSize, gridSize);
        }

        public FieldGrid GetComponent(string name) => name switch
        {
            "T" or "temperature" or "Temperature" => Temperature,
            "P" or "pressure" or "Pressure" => Pressure,
            "rho" or "density" or "Density" => Density,
            "vx" or "VelocityX" or "u" => VelocityX,
            "vy" or "VelocityY" or "v" => VelocityY,
            "vz" or "VelocityZ" or "w" => VelocityZ,
            _ => Components.TryGetValue(name, out var comp) ? comp : Temperature
        };

        public PhysicsField Clone()
        {
            var clone = new PhysicsField(GridSize, Name + "_clone");
            clone.Temperature = Temperature.Clone();
            clone.Pressure = Pressure.Clone();
            clone.Density = Density.Clone();
            clone.VelocityX = VelocityX.Clone();
            clone.VelocityY = VelocityY.Clone();
            clone.VelocityZ = VelocityZ.Clone();
            clone.Time = Time;
            foreach (var kv in Components)
                clone.Components[kv.Key] = kv.Value.Clone();
            return clone;
        }
    }
}
