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
    // LawSerializer — serialization/deserialization of laws and bytecode
    // =========================================================================

    /// <summary>Serializes and deserializes law entries, bytecodes, and version trees.</summary>
    public sealed class LawSerializer
    {
        private static readonly JsonSerializerOptions _options = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>Serialize a law entry to JSON.</summary>
        public static string SerializeLawEntry(LawEntry entry) => JsonSerializer.Serialize(entry, _options);

        /// <summary>Deserialize a law entry from JSON.</summary>
        public static LawEntry? DeserializeLawEntry(string json) => JsonSerializer.Deserialize<LawEntry>(json, _options);

        /// <summary>Serialize multiple law entries to JSON.</summary>
        public static string SerializeLawEntries(IReadOnlyList<LawEntry> entries) =>
            JsonSerializer.Serialize(entries, _options);

        /// <summary>Deserialize law entries from JSON.</summary>
        public static LawEntry[] DeserializeLawEntries(string json) =>
            JsonSerializer.Deserialize<LawEntry[]>(json, _options) ?? Array.Empty<LawEntry>();

        /// <summary>Save a law library to a JSON file.</summary>
        public static void SaveLibrary(LawLibrary library, string filePath)
        {
            var json = SerializeLawEntries(library.AllEntries);
            File.WriteAllText(filePath, json);
        }

        /// <summary>Load a law library from a JSON file.</summary>
        public static LawLibrary LoadLibrary(string filePath)
        {
            var library = new LawLibrary();
            var entries = DeserializeLawEntries(File.ReadAllText(filePath));
            foreach (var entry in entries)
                library.Register(entry);
            return library;
        }

        /// <summary>Serialize version tree history to JSON.</summary>
        public static string SerializeVersionHistory(LawVersionTree tree)
        {
            var history = tree.ExportHistory();
            var records = history.Select(h => new
            {
                h.Id,
                h.Expression,
                h.Timestamp,
                h.Description
            });
            return JsonSerializer.Serialize(records, _options);
        }

        /// <summary>Serialize compilation results.</summary>
        public static string SerializeCompilationResult(CompilationResult result) =>
            JsonSerializer.Serialize(new
            {
                result.Success,
                result.Message,
                result.Errors,
                result.Warnings,
                result.InstructionCount,
                result.CompilationTimeMs,
                ResultDimension = result.Bytecode?.ResultDimension.ToString() ?? "N/A"
            }, _options);

        /// <summary>Serialize comparison results.</summary>
        public static string SerializeComparisonResult(ComparisonResult result) =>
            JsonSerializer.Serialize(new
            {
                result.MaxDivergence,
                result.MeanDivergence,
                result.RootMeanSquareError,
                result.KolmogorovSmirnovStatistic,
                result.StructuralSimilarity,
                result.ExpressionEditDistance,
                result.Differences,
                result.PhysicallyEquivalent
            }, _options);

        /// <summary>Serialize simulation results.</summary>
        public static string SerializeSimulationResult(SimulationResult result) =>
            JsonSerializer.Serialize(new
            {
                result.TotalTime,
                result.Converged,
                result.Iterations,
                result.ErrorMessage,
                SnapshotCount = result.Snapshots.Count,
                result.TimeSteps,
                result.EnergyHistory,
                result.ErrorHistory
            }, _options);

        /// <summary>Export a law to a compact string format.</summary>
        public static string ExportCompact(LawEntry entry)
        {
            return $"ID:{entry.Id}|CAT:{entry.Category}|NAME:{entry.Name}|EXPR:{entry.Expression}";
        }

        /// <summary>Import a law from a compact string format.</summary>
        public static LawEntry? ImportCompact(string compact)
        {
            var parts = compact.Split('|');
            if (parts.Length < 4)
                return null;
            var entry = new LawEntry();
            foreach (var part in parts)
            {
                var kv = part.Split(':', 2);
                if (kv.Length != 2)
                    continue;
                switch (kv[0])
                {
                    case "ID":
                        entry.Id = kv[1];
                        break;
                    case "CAT":
                        entry.Category = kv[1];
                        break;
                    case "NAME":
                        entry.Name = kv[1];
                        break;
                    case "EXPR":
                        entry.Expression = kv[1];
                        break;
                    case "DESC":
                        entry.Description = kv[1];
                        break;
                }
            }
            return entry;
        }

        /// <summary>Serialize a PhysicsField to JSON (for checkpointing).</summary>
        public static string SerializeField(PhysicsField field)
        {
            return JsonSerializer.Serialize(new
            {
                field.Name,
                field.GridSize,
                field.Time,
                Temperature = field.Temperature.Data,
                Pressure = field.Pressure.Data,
                Density = field.Density.Data,
                VelocityX = field.VelocityX.Data,
                VelocityY = field.VelocityY.Data,
                VelocityZ = field.VelocityZ.Data
            }, _options);
        }

        /// <summary>Deserialize a PhysicsField from JSON.</summary>
        public static PhysicsField? DeserializeField(string json)
        {
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            int gridSize = root.GetProperty("gridSize").GetInt32();
            string name = root.GetProperty("name").GetString() ?? "deserialized";
            float time = root.GetProperty("time").GetSingle();
            var field = new PhysicsField(gridSize, name);
            field.Time = time;

            if (root.TryGetProperty("temperature", out var tempEl))
                CopyJsonArrayToGrid(tempEl, field.Temperature);
            if (root.TryGetProperty("pressure", out var presEl))
                CopyJsonArrayToGrid(presEl, field.Pressure);
            if (root.TryGetProperty("density", out var densEl))
                CopyJsonArrayToGrid(densEl, field.Density);
            if (root.TryGetProperty("velocityX", out var vxEl))
                CopyJsonArrayToGrid(vxEl, field.VelocityX);
            if (root.TryGetProperty("velocityY", out var vyEl))
                CopyJsonArrayToGrid(vyEl, field.VelocityY);
            if (root.TryGetProperty("velocityZ", out var vzEl))
                CopyJsonArrayToGrid(vzEl, field.VelocityZ);

            return field;
        }

        private static void CopyJsonArrayToGrid(JsonElement array, FieldGrid grid)
        {
            int idx = 0;
            int total = Math.Min(array.GetArrayLength(), grid.TotalCells);
            foreach (var item in array.EnumerateArray())
            {
                if (idx >= total)
                    break;
                int z = idx / (grid.SizeX * grid.SizeY);
                int rem = idx % (grid.SizeX * grid.SizeY);
                int y = rem / grid.SizeX;
                int x = rem % grid.SizeX;
                grid[x, y, z] = item.GetSingle();
                idx++;
            }
        }

        /// <summary>Save a checkpoint of the entire compiler state.</summary>
        public static void SaveCheckpoint(LivingLawCompiler compiler, string directory)
        {
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "compiler_state.json"), compiler.ExportState());
            File.WriteAllText(Path.Combine(directory, "library.json"),
                SerializeLawEntries(compiler.GetAllLaws()));
            foreach (var law in compiler.GetAllLaws())
            {
                string lawFile = Path.Combine(directory, $"law_{law.Id}.json");
                File.WriteAllText(lawFile, SerializeLawEntry(law));
            }
        }

        /// <summary>Load a checkpoint of the compiler state.</summary>
        public static (LawLibrary Library, string? StateJson) LoadCheckpoint(string directory)
        {
            string stateJson = File.Exists(Path.Combine(directory, "compiler_state.json"))
                ? File.ReadAllText(Path.Combine(directory, "compiler_state.json")) : "";
            var library = File.Exists(Path.Combine(directory, "library.json"))
                ? LoadLibrary(Path.Combine(directory, "library.json")) : LawLibrary.LoadBuiltIn();
            return (library, stateJson);
        }
    }
}
