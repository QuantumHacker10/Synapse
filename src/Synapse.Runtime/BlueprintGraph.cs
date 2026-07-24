using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Sentience;

namespace Synapse.Runtime
{
    public enum BlueprintNodeKind
    {
        Entry, Exit, Sequence, Selector, Condition, Action,
        LawApply, EvolveStep, LlmQuery, SpawnAgent, Wait
    }

    public sealed class BlueprintPin
    {
        public string Name { get; set; } = "Out";
        public bool IsInput { get; set; }
    }

    public sealed class BlueprintNode
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public BlueprintNodeKind Kind { get; set; } = BlueprintNodeKind.Action;
        public string Title { get; set; } = "Node";
        public float X { get; set; }
        public float Y { get; set; }
        public string? Payload { get; set; }
        public List<BlueprintPin> Inputs { get; set; } = new();
        public List<BlueprintPin> Outputs { get; set; } = new();
    }

    public sealed class BlueprintEdge
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public Guid FromNodeId { get; set; }
        public int FromPin { get; set; }
        public Guid ToNodeId { get; set; }
        public int ToPin { get; set; }
    }

    public sealed class BlueprintDocument
    {
        public string Name { get; set; } = "Untitled Blueprint";
        public List<BlueprintNode> Nodes { get; set; } = new();
        public List<BlueprintEdge> Edges { get; set; } = new();

        public static BlueprintDocument CreateDefault()
        {
            var entry = new BlueprintNode
            {
                Kind = BlueprintNodeKind.Entry,
                Title = "Entry",
                X = 40,
                Y = 80,
                Outputs = { new BlueprintPin { Name = "Exec", IsInput = false } }
            };
            var action = new BlueprintNode
            {
                Kind = BlueprintNodeKind.Action,
                Title = "Patrol",
                Payload = "patrol",
                X = 260,
                Y = 80,
                Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } },
                Outputs = { new BlueprintPin { Name = "Then", IsInput = false } }
            };
            var exit = new BlueprintNode
            {
                Kind = BlueprintNodeKind.Exit,
                Title = "Exit",
                X = 480,
                Y = 80,
                Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } }
            };
            return new BlueprintDocument
            {
                Name = "Agent Patrol",
                Nodes = { entry, action, exit },
                Edges =
                {
                    new BlueprintEdge { FromNodeId = entry.Id, FromPin = 0, ToNodeId = action.Id, ToPin = 0 },
                    new BlueprintEdge { FromNodeId = action.Id, FromPin = 0, ToNodeId = exit.Id, ToPin = 0 }
                }
            };
        }

        private const long MaxBlueprintBytes = 5L * 1024 * 1024;

        public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
        {
            var full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);
            await using var stream = File.Create(full);
            await JsonSerializer.SerializeAsync(stream, this, BlueprintJsonContext.Default.BlueprintDocument, cancellationToken);
        }

        public static async Task<BlueprintDocument> LoadAsync(string path, CancellationToken cancellationToken = default)
        {
            var full = Path.GetFullPath(path);
            var info = new FileInfo(full);
            if (!info.Exists)
                throw new FileNotFoundException("Blueprint file not found.", full);
            if (info.Length > MaxBlueprintBytes)
                throw new InvalidDataException($"Blueprint file exceeds {MaxBlueprintBytes} byte limit.");
            await using var stream = File.OpenRead(full);
            return await JsonSerializer.DeserializeAsync(stream, BlueprintJsonContext.Default.BlueprintDocument, cancellationToken)
                   ?? throw new InvalidDataException($"Blueprint file '{full}' is empty or invalid JSON.");
        }

        public (bool Ok, string Message) Validate()
        {
            const int maxNodes = 512;
            const int maxEdges = 2_048;
            const int maxTitle = 128;
            const int maxPayload = 4_096;

            if (Nodes.Count == 0)
                return (false, "Blueprint has no nodes");
            if (Nodes.Count > maxNodes)
                return (false, $"Blueprint exceeds {maxNodes} nodes");
            if (Edges.Count > maxEdges)
                return (false, $"Blueprint exceeds {maxEdges} edges");
            if (Nodes.Count(n => n.Kind == BlueprintNodeKind.Entry) != 1)
                return (false, "Blueprint must have exactly one Entry node");
            if (!Nodes.Any(n => n.Kind == BlueprintNodeKind.Exit))
                return (false, "Missing Exit node");

            var ids = new HashSet<Guid>();
            foreach (var node in Nodes)
            {
                if (node.Id == Guid.Empty)
                    return (false, "Node Id cannot be empty");
                if (!ids.Add(node.Id))
                    return (false, $"Duplicate node Id '{node.Id}'");
                if (string.IsNullOrWhiteSpace(node.Title) || node.Title.Length > maxTitle)
                    return (false, $"Node '{node.Id}' has invalid Title");
                if (node.Payload is { Length: > maxPayload })
                    return (false, $"Node '{node.Title}' Payload exceeds {maxPayload} characters");
                if (!float.IsFinite(node.X) || !float.IsFinite(node.Y))
                    return (false, $"Node '{node.Title}' has non-finite coordinates");
                if (node.Inputs.Count > 32 || node.Outputs.Count > 32)
                    return (false, $"Node '{node.Title}' has too many pins");
            }

            foreach (var edge in Edges)
            {
                var from = Nodes.FirstOrDefault(n => n.Id == edge.FromNodeId);
                var to = Nodes.FirstOrDefault(n => n.Id == edge.ToNodeId);
                if (from == null || to == null)
                    return (false, $"Dangling edge {edge.Id}");
                if (edge.FromPin < 0 || edge.FromPin >= Math.Max(1, from.Outputs.Count))
                    return (false, $"Edge {edge.Id} FromPin out of range");
                if (edge.ToPin < 0 || edge.ToPin >= Math.Max(1, to.Inputs.Count))
                    return (false, $"Edge {edge.Id} ToPin out of range");
                if (edge.FromNodeId == edge.ToNodeId)
                    return (false, $"Self-loop edge {edge.Id} is not allowed");
            }

            return (true, $"OK — {Nodes.Count} nodes, {Edges.Count} edges");
        }

        public string CompileToBehaviorTreeName()
        {
            var tree = CompileToBehaviorTreeBlueprint();
            var action = tree.Children.FirstOrDefault(c => c.NodeType == BehaviorNodeType.Action)
                         ?? FindFirstAction(tree);
            return action?.ActionType ?? "patrol";
        }

        public BehaviorTreeBlueprint CompileToBehaviorTreeBlueprint() => BlueprintCompiler.Compile(this);

        private static BehaviorTreeBlueprint? FindFirstAction(BehaviorTreeBlueprint node)
        {
            if (node.NodeType == BehaviorNodeType.Action)
                return node;
            foreach (var child in node.Children)
            {
                var found = FindFirstAction(child);
                if (found != null)
                    return found;
            }
            return null;
        }
    }

    public sealed class SculptStroke
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        public float Radius { get; set; } = 0.5f;
        public float Strength { get; set; } = 0.1f;
        public bool Invert { get; set; }
    }

    public sealed class SculptSession
    {
        private readonly List<SculptStroke> _strokes = new();
        public IReadOnlyList<SculptStroke> Strokes => _strokes;
        public float BrushRadius { get; set; } = 0.5f;
        public float BrushStrength { get; set; } = 0.15f;
        public bool Invert { get; set; }

        public void ApplyStroke(float x, float y, float z)
        {
            _strokes.Add(new SculptStroke
            {
                X = x,
                Y = y,
                Z = z,
                Radius = BrushRadius,
                Strength = BrushStrength,
                Invert = Invert
            });
        }

        public void Clear() => _strokes.Clear();

        /// <summary>Sample cumulative displacement at a world point (soft brush falloff).</summary>
        public float SampleDisplacement(float x, float y, float z)
        {
            float sum = 0;
            foreach (var s in _strokes)
            {
                float dx = x - s.X, dy = y - s.Y, dz = z - s.Z;
                float d2 = dx * dx + dy * dy + dz * dz;
                float r2 = s.Radius * s.Radius;
                if (d2 >= r2)
                    continue;
                float w = 1f - d2 / r2;
                w = w * w;
                sum += (s.Invert ? -s.Strength : s.Strength) * w;
            }
            return sum;
        }
    }

    [JsonSerializable(typeof(BlueprintDocument))]
    [JsonSerializable(typeof(BlueprintNode))]
    [JsonSerializable(typeof(BlueprintEdge))]
    [JsonSerializable(typeof(BlueprintPin))]
    [JsonSerializable(typeof(List<BlueprintNode>))]
    [JsonSerializable(typeof(List<BlueprintEdge>))]
    [JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    internal sealed partial class BlueprintJsonContext : JsonSerializerContext
    {
    }
}
