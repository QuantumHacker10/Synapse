using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Synapse.Web.Studio;

/// <summary>Lightweight scene document for the WASM Studio (mirrors Runtime SceneDocument shape).</summary>
public sealed class WasmSceneDocument
{
    public string Name { get; set; } = "Untitled";
    public string Version { get; set; } = "2.2";
    public string? ActiveLawId { get; set; } = "heat_equation";
    public List<WasmEntity> Entities { get; set; } = new();
    public WasmCamera Camera { get; set; } = new();

    public static WasmSceneDocument CreateDemo() => new()
    {
        Name = "Demo WASM Scene",
        ActiveLawId = "heat_equation",
        Entities =
        [
            new WasmEntity { Id = "ground", Name = "Ground", Type = "Mesh", Position = new WasmVec3(0, 0, 0), Scale = new WasmVec3(10, 0.2f, 10) },
            new WasmEntity { Id = "alpha", Name = "Agent_Alpha", Type = "Agent", Position = new WasmVec3(2, 0.5f, 0) },
            new WasmEntity { Id = "beta", Name = "Agent_Beta", Type = "Agent", Position = new WasmVec3(-2, 0.5f, 1) },
            new WasmEntity { Id = "cube", Name = "Cube", Type = "Mesh", Position = new WasmVec3(0, 1, -2), Scale = new WasmVec3(1, 1, 1) },
        ]
    };

    public static WasmSceneDocument? TryParse(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Prefer dedicated WASM shape; also accept Runtime SceneDocument JSON.
            if (root.TryGetProperty("entities", out var entitiesEl) ||
                root.TryGetProperty("Entities", out entitiesEl))
            {
                var scene = new WasmSceneDocument
                {
                    Name = ReadString(root, "name", "Name") ?? "Untitled",
                    Version = ReadString(root, "version", "Version") ?? "2.2",
                    ActiveLawId = ReadString(root, "activeLawId", "ActiveLawId")
                };

                foreach (var e in entitiesEl.EnumerateArray())
                {
                    var id = ReadString(e, "id", "Id") ?? Guid.NewGuid().ToString("N");
                    var pos = ReadVec3(e, "position", "Position");
                    var scale = ReadVec3(e, "scale", "Scale", new WasmVec3(1, 1, 1));
                    scene.Entities.Add(new WasmEntity
                    {
                        Id = id,
                        Name = ReadString(e, "name", "Name") ?? "Entity",
                        Type = ReadString(e, "type", "Type") ?? "Mesh",
                        Position = pos,
                        Scale = scale
                    });
                }

                return scene;
            }

            return JsonSerializer.Deserialize(json, WasmSceneJsonContext.Default.WasmSceneDocument);
        }
        catch
        {
            return null;
        }
    }

    private static string? ReadString(JsonElement el, string camel, string pascal)
    {
        if (el.TryGetProperty(camel, out var v) || el.TryGetProperty(pascal, out v))
            return v.ValueKind == JsonValueKind.String ? v.GetString() : v.ToString();
        return null;
    }

    private static WasmVec3 ReadVec3(JsonElement parent, string camel, string pascal, WasmVec3? fallback = null)
    {
        if (!parent.TryGetProperty(camel, out var v) && !parent.TryGetProperty(pascal, out v))
            return fallback ?? new WasmVec3();

        if (v.ValueKind == JsonValueKind.Array && v.GetArrayLength() >= 3)
            return new WasmVec3(v[0].GetSingle(), v[1].GetSingle(), v[2].GetSingle());

        float x = 0, y = 0, z = 0;
        if (v.TryGetProperty("x", out var xv) || v.TryGetProperty("X", out xv))
            x = xv.GetSingle();
        if (v.TryGetProperty("y", out var yv) || v.TryGetProperty("Y", out yv))
            y = yv.GetSingle();
        if (v.TryGetProperty("z", out var zv) || v.TryGetProperty("Z", out zv))
            z = zv.GetSingle();
        return new WasmVec3(x, y, z);
    }

    public string ToJson() => JsonSerializer.Serialize(this, WasmSceneJsonContext.Default.WasmSceneDocument);

    public string ToGltfJson()
    {
        // Minimal glTF 2.0 with one node per entity (translation/scale only).
        var nodes = new List<object>();
        var meshes = new List<object>();
        var sceneNodes = new List<int>();
        for (int i = 0; i < Entities.Count; i++)
        {
            var e = Entities[i];
            sceneNodes.Add(i);
            nodes.Add(new
            {
                name = e.Name,
                translation = new[] { e.Position.X, e.Position.Y, e.Position.Z },
                scale = new[] { e.Scale.X, e.Scale.Y, e.Scale.Z },
                mesh = 0,
                extras = new { synapseType = e.Type, synapseId = e.Id }
            });
        }

        // Unit cube positions + indices (inline base64 buffer).
        float[] positions =
        [
            -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
            -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f, -0.5f, 0.5f,-0.5f
        ];
        ushort[] indices =
        [
            0,1,2, 0,2,3, 1,5,6, 1,6,2, 5,4,7, 5,7,6,
            4,0,3, 4,3,7, 3,2,6, 3,6,7, 4,5,1, 4,1,0
        ];
        var posBytes = new byte[positions.Length * 4];
        Buffer.BlockCopy(positions, 0, posBytes, 0, posBytes.Length);
        var idxBytes = new byte[indices.Length * 2];
        Buffer.BlockCopy(indices, 0, idxBytes, 0, idxBytes.Length);
        var bin = new byte[posBytes.Length + idxBytes.Length];
        Buffer.BlockCopy(posBytes, 0, bin, 0, posBytes.Length);
        Buffer.BlockCopy(idxBytes, 0, bin, posBytes.Length, idxBytes.Length);
        string uri = "data:application/octet-stream;base64," + Convert.ToBase64String(bin);

        meshes.Add(new
        {
            primitives = new[]
            {
                new
                {
                    attributes = new { POSITION = 0 },
                    indices = 1,
                    mode = 4
                }
            }
        });

        var gltf = new
        {
            asset = new { version = "2.0", generator = "Synapse.Web.Studio" },
            scene = 0,
            scenes = new[] { new { name = Name, nodes = sceneNodes } },
            nodes,
            meshes,
            accessors = new object[]
            {
                new { bufferView = 0, componentType = 5126, count = 8, type = "VEC3", max = new[] { 0.5f, 0.5f, 0.5f }, min = new[] { -0.5f, -0.5f, -0.5f } },
                new { bufferView = 1, componentType = 5123, count = indices.Length, type = "SCALAR" }
            },
            bufferViews = new object[]
            {
                new { buffer = 0, byteOffset = 0, byteLength = posBytes.Length, target = 34962 },
                new { buffer = 0, byteOffset = posBytes.Length, byteLength = idxBytes.Length, target = 34963 }
            },
            buffers = new[] { new { byteLength = bin.Length, uri } },
            extras = new { activeLawId = ActiveLawId, synapseVersion = Version }
        };

        return JsonSerializer.Serialize(gltf, new JsonSerializerOptions { WriteIndented = true });
    }
}

public sealed class WasmEntity
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");
    public string Name { get; set; } = "Entity";
    public string Type { get; set; } = "Mesh";
    public WasmVec3 Position { get; set; } = new();
    public WasmVec3 Scale { get; set; } = new(1, 1, 1);
}

public sealed class WasmCamera
{
    public WasmVec3 Position { get; set; } = new(0, 2, 8);
    public float Yaw { get; set; }
    public float Pitch { get; set; } = -0.2f;
}

public sealed class WasmVec3
{
    public WasmVec3() { }
    public WasmVec3(float x, float y, float z) { X = x; Y = y; Z = z; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
}

[JsonSerializable(typeof(WasmSceneDocument))]
[JsonSerializable(typeof(WasmEntity))]
[JsonSerializable(typeof(List<WasmEntity>))]
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal partial class WasmSceneJsonContext : JsonSerializerContext;
