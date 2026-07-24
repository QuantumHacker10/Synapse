using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Synapse.Runtime;

/// <summary>
/// Compact scene collaboration patches exchanged over WAN and usable by the WASM Studio.
/// Encodes entity transforms (+ optional active law) without shipping the full document every frame.
/// </summary>
public static class ScenePatchCodec
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false
    };

    public sealed class ScenePatch
    {
        public int Version { get; set; } = 1;
        public string? SceneName { get; set; }
        public string? ActiveLawId { get; set; }
        public long Sequence { get; set; }
        public List<EntityTransformPatch> Entities { get; set; } = new();
    }

    public sealed class EntityTransformPatch
    {
        public string Id { get; set; } = "";
        public string? Name { get; set; }
        public string? Type { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        public float Sx { get; set; } = 1;
        public float Sy { get; set; } = 1;
        public float Sz { get; set; } = 1;
        public bool Removed { get; set; }
    }

    public static byte[] Encode(SceneDocument scene, long sequence)
    {
        ArgumentNullException.ThrowIfNull(scene);
        var patch = FromScene(scene, sequence);
        return Encoding.UTF8.GetBytes(JsonSerializer.Serialize(patch, JsonOptions));
    }

    public static ScenePatch FromScene(SceneDocument scene, long sequence)
    {
        ArgumentNullException.ThrowIfNull(scene);
        var patch = new ScenePatch
        {
            SceneName = scene.Name,
            ActiveLawId = scene.ActiveLawId,
            Sequence = sequence
        };
        foreach (var e in scene.Entities)
        {
            patch.Entities.Add(new EntityTransformPatch
            {
                Id = e.Id.ToString("N"),
                Name = e.Name,
                Type = e.Type,
                X = e.Position.X,
                Y = e.Position.Y,
                Z = e.Position.Z,
                Sx = e.Scale.X,
                Sy = e.Scale.Y,
                Sz = e.Scale.Z
            });
        }

        return patch;
    }

    public static bool TryDecode(ReadOnlySpan<byte> payload, out ScenePatch? patch)
    {
        patch = null;
        try
        {
            patch = JsonSerializer.Deserialize<ScenePatch>(payload, JsonOptions);
            return patch != null && patch.Version >= 1;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// Applies a remote patch onto <paramref name="scene"/>. Returns the number of entities touched.
    /// </summary>
    public static int Apply(SceneDocument scene, ScenePatch patch, bool replaceMissing = true)
    {
        ArgumentNullException.ThrowIfNull(scene);
        ArgumentNullException.ThrowIfNull(patch);

        if (!string.IsNullOrWhiteSpace(patch.SceneName))
            scene.Name = patch.SceneName!;
        if (!string.IsNullOrWhiteSpace(patch.ActiveLawId))
            scene.ActiveLawId = patch.ActiveLawId;

        int touched = 0;
        var seen = new HashSet<Guid>();
        foreach (var ep in patch.Entities)
        {
            if (!Guid.TryParse(ep.Id, out var id))
                continue;

            if (ep.Removed)
            {
                touched += scene.Entities.RemoveAll(e => e.Id == id);
                continue;
            }

            seen.Add(id);
            var existing = scene.Entities.Find(e => e.Id == id);
            if (existing == null)
            {
                if (!replaceMissing)
                    continue;
                existing = new SceneEntityData
                {
                    Id = id,
                    Name = ep.Name ?? $"Peer_{id.ToString("N")[..8]}",
                    Type = string.IsNullOrWhiteSpace(ep.Type) ? "Mesh" : ep.Type!
                };
                scene.Entities.Add(existing);
            }

            if (!string.IsNullOrWhiteSpace(ep.Name))
                existing.Name = ep.Name!;
            if (!string.IsNullOrWhiteSpace(ep.Type))
                existing.Type = ep.Type!;
            existing.Position = new Vec3(ep.X, ep.Y, ep.Z);
            existing.Scale = new Vec3(ep.Sx, ep.Sy, ep.Sz);
            touched++;
        }

        return touched;
    }
}
