using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Runtime
{
    public sealed class SceneDocument
    {
        public const int MaxEntities = 10_000;
        public const int MaxJoints = 10_000;
        public const int MaxNameLength = 256;
        public const int MaxStringLength = 1_024;
        public const float MaxCoordinate = 1_000_000f;
        private const long MaxSceneBytes = 10L * 1024 * 1024;

        public string Name { get; set; } = "Untitled";
        public string Version { get; set; } = "1.0";
        public string? ActiveLawId { get; set; } = "heat_equation";
        public string? ActiveLawExpression { get; set; }
        public List<SceneEntityData> Entities { get; set; } = new();
        /// <summary>Persisted physics joints (hinge, ball-socket, distance, …).</summary>
        public List<SceneJointData> Joints { get; set; } = new();
        public CameraData Camera { get; set; } = new();
        public Dictionary<string, string> Assets { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        public static SceneDocument CreateDemo()
        {
            return new SceneDocument
            {
                Name = "Demo Scene",
                ActiveLawId = "heat_equation",
                Entities =
                {
                    new SceneEntityData
                    {
                        Id = Guid.Parse("11111111-1111-1111-1111-111111111111"),
                        Name = "Ground",
                        Type = "Mesh",
                        Position = new Vec3(0, 0, 0),
                        Scale = new Vec3(10, 0.2f, 10)
                    },
                    new SceneEntityData
                    {
                        Id = Guid.Parse("22222222-2222-2222-2222-222222222222"),
                        Name = "Agent_Alpha",
                        Type = SceneEntityKinds.Agent,
                        Position = new Vec3(2, 0, 0),
                        BehaviorProfile = "patrol"
                    },
                    new SceneEntityData
                    {
                        Id = Guid.Parse("33333333-3333-3333-3333-333333333333"),
                        Name = "Agent_Beta",
                        Type = SceneEntityKinds.Agent,
                        Position = new Vec3(-2, 0, 1),
                        BehaviorProfile = "guard"
                    },
                    new SceneEntityData
                    {
                        Id = Guid.Parse("44444444-4444-4444-4444-444444444444"),
                        Name = "NeuralForm",
                        Type = "Genome",
                        Position = new Vec3(0, 1.5f, -2),
                        GenomeId = "demo-sdf"
                    }
                }
            };
        }

        public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
        {
            Validate();
            var full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);
            await using var stream = File.Create(full);
            await JsonSerializer.SerializeAsync(stream, this, SceneDocumentJsonContext.Default.SceneDocument, cancellationToken);
        }

        public static async Task<SceneDocument> LoadAsync(string path, CancellationToken cancellationToken = default)
        {
            var full = Path.GetFullPath(path);
            var info = new FileInfo(full);
            if (!info.Exists)
                throw new FileNotFoundException("Scene file not found.", full);
            if (info.Length > MaxSceneBytes)
                throw new InvalidDataException($"Scene file exceeds {MaxSceneBytes} byte limit.");
            await using var stream = File.OpenRead(full);
            var doc = await JsonSerializer.DeserializeAsync(stream, SceneDocumentJsonContext.Default.SceneDocument, cancellationToken)
                      ?? throw new InvalidDataException($"Scene file '{full}' is empty or invalid JSON.");
            doc.NormalizeLegacyTypes();
            doc.Validate(Path.GetDirectoryName(full));
            return doc;
        }

        /// <summary>Validates scene integrity. Optionally jails MeshPath under <paramref name="projectRoot"/>.</summary>
        public void Validate(string? projectRoot = null)
        {
            if (string.IsNullOrWhiteSpace(Name) || Name.Length > MaxNameLength)
                throw new InvalidDataException("Scene Name is missing or too long.");
            if (Entities.Count > MaxEntities)
                throw new InvalidDataException($"Scene exceeds {MaxEntities} entities.");
            if (Joints.Count > MaxJoints)
                throw new InvalidDataException($"Scene exceeds {MaxJoints} joints.");
            if (Assets.Count > 4_096)
                throw new InvalidDataException("Scene Assets dictionary exceeds 4096 entries.");

            RequireFiniteOptionalString(ActiveLawId, nameof(ActiveLawId));
            if (ActiveLawExpression is { Length: > 8_192 })
                throw new InvalidDataException("ActiveLawExpression exceeds 8192 characters.");

            var ids = new HashSet<Guid>();
            foreach (var e in Entities)
            {
                if (e.Id == Guid.Empty)
                    throw new InvalidDataException("Entity Id cannot be empty.");
                if (!ids.Add(e.Id))
                    throw new InvalidDataException($"Duplicate entity Id '{e.Id}'.");
                if (string.IsNullOrWhiteSpace(e.Name) || e.Name.Length > MaxNameLength)
                    throw new InvalidDataException($"Entity '{e.Id}' has invalid Name.");
                if (string.IsNullOrWhiteSpace(e.Type) || e.Type.Length > 64)
                    throw new InvalidDataException($"Entity '{e.Name}' has invalid Type.");
                RequireFiniteVec3(e.Position, $"Entity '{e.Name}' Position");
                RequireFiniteVec3(e.Rotation, $"Entity '{e.Name}' Rotation");
                RequireFiniteVec3(e.Scale, $"Entity '{e.Name}' Scale");
                if (e.Scale.X <= 0 || e.Scale.Y <= 0 || e.Scale.Z <= 0)
                    throw new InvalidDataException($"Entity '{e.Name}' Scale must be positive.");
                RequireFiniteOptionalString(e.BehaviorProfile, "BehaviorProfile");
                RequireFiniteOptionalString(e.GenomeId, "GenomeId");
                RequireFiniteOptionalString(e.LawId, "LawId");
                ValidateMeshPath(e.MeshPath, projectRoot, e.Name);
            }

            var entityIds = ids;
            foreach (var j in Joints)
            {
                if (j.Id == Guid.Empty)
                    j.Id = Guid.NewGuid();
                if (string.IsNullOrWhiteSpace(j.Name) || j.Name.Length > MaxNameLength)
                    throw new InvalidDataException($"Joint '{j.Id}' has invalid Name.");
                if (string.IsNullOrWhiteSpace(j.Type) || j.Type.Length > 64)
                    throw new InvalidDataException($"Joint '{j.Name}' has invalid Type.");
                if (j.BodyA == Guid.Empty || !entityIds.Contains(j.BodyA))
                    throw new InvalidDataException($"Joint '{j.Name}' BodyA is missing from entities.");
                if (j.BodyB != Guid.Empty && !entityIds.Contains(j.BodyB))
                    throw new InvalidDataException($"Joint '{j.Name}' BodyB is missing from entities.");
                RequireFiniteVec3(j.LocalAnchorA, $"Joint '{j.Name}' LocalAnchorA");
                RequireFiniteVec3(j.LocalAnchorB, $"Joint '{j.Name}' LocalAnchorB");
                RequireFiniteVec3(j.LocalAxisA, $"Joint '{j.Name}' LocalAxisA");
                RequireFiniteVec3(j.LocalAxisB, $"Joint '{j.Name}' LocalAxisB");
                if (!float.IsFinite(j.RestLength) || !float.IsFinite(j.Stiffness) || !float.IsFinite(j.Damping))
                    throw new InvalidDataException($"Joint '{j.Name}' has non-finite scalar parameters.");
            }

            RequireFiniteVec3(Camera.Position, "Camera.Position");
            if (!float.IsFinite(Camera.Yaw) || !float.IsFinite(Camera.Pitch) || !float.IsFinite(Camera.Fov))
                throw new InvalidDataException("Camera orientation/FOV must be finite.");
            if (Camera.Fov is <= 1f or >= 179f)
                throw new InvalidDataException("Camera Fov must be in (1, 179).");
        }

        private void NormalizeLegacyTypes()
        {
            foreach (var e in Entities)
                e.Type = SceneEntityKinds.Normalize(e.Type);
        }

        private static void ValidateMeshPath(string? meshPath, string? projectRoot, string entityName)
        {
            if (string.IsNullOrWhiteSpace(meshPath))
                return;
            if (meshPath.Length > MaxStringLength)
                throw new InvalidDataException($"Entity '{entityName}' MeshPath is too long.");
            if (meshPath.Contains("..", StringComparison.Ordinal))
                throw new InvalidDataException($"Entity '{entityName}' MeshPath must not contain '..'.");

            var ext = Path.GetExtension(meshPath).ToLowerInvariant();
            if (ext is not (".gltf" or ".glb" or ".obj" or ".fbx" or ".usda" or ".usd" or ""))
                throw new InvalidDataException($"Entity '{entityName}' MeshPath has unsupported extension '{ext}'.");

            if (string.IsNullOrWhiteSpace(projectRoot))
                return;

            var root = Path.GetFullPath(projectRoot);
            var assetsRoot = Directory.Exists(Path.Combine(root, "assets"))
                ? Path.Combine(root, "assets")
                : root;

            string candidate = Path.IsPathRooted(meshPath)
                ? Path.GetFullPath(meshPath)
                : Path.GetFullPath(Path.Combine(assetsRoot, meshPath));

            Synapse.Core.Security.PathSecurity.EnsureUnderRoot(assetsRoot, candidate);
        }

        private static void RequireFiniteVec3(Vec3 v, string label)
        {
            if (!float.IsFinite(v.X) || !float.IsFinite(v.Y) || !float.IsFinite(v.Z))
                throw new InvalidDataException($"{label} must be finite.");
            if (MathF.Abs(v.X) > MaxCoordinate || MathF.Abs(v.Y) > MaxCoordinate || MathF.Abs(v.Z) > MaxCoordinate)
                throw new InvalidDataException($"{label} exceeds coordinate limits.");
        }

        private static void RequireFiniteOptionalString(string? value, string label)
        {
            if (value == null)
                return;
            if (value.Length > MaxStringLength)
                throw new InvalidDataException($"{label} exceeds {MaxStringLength} characters.");
        }
    }

    public sealed class SceneEntityData
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string Name { get; set; } = "Entity";
        public string Type { get; set; } = "Empty";
        public Vec3 Position { get; set; } = new();
        public Vec3 Rotation { get; set; } = new();
        public Vec3 Scale { get; set; } = new(1, 1, 1);
        public bool Visible { get; set; } = true;
        public string? BehaviorProfile { get; set; }
        public string? GenomeId { get; set; }
        public string? LawId { get; set; }
        /// <summary>Optional mesh asset path (glTF/OBJ) for SynapseMeshProvider.</summary>
        public string? MeshPath { get; set; }
        /// <summary>When true with <see cref="MeshPath"/>, bake a G-DNN SDF after load.</summary>
        public bool BakeNeuralSdf { get; set; }
        /// <summary>When true, treat the entity as a raycast vehicle chassis.</summary>
        public bool IsVehicle { get; set; }
    }

    /// <summary>Scene-serialized bilateral joint for Synapse Omnia assemblies.</summary>
    public sealed class SceneJointData
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string Name { get; set; } = "Joint";
        /// <summary>Hinge | BallSocket | Slider | Fixed | Distance</summary>
        public string Type { get; set; } = "Hinge";
        public Guid BodyA { get; set; }
        /// <summary>Guid.Empty = world anchor.</summary>
        public Guid BodyB { get; set; }
        public Vec3 LocalAnchorA { get; set; } = new();
        public Vec3 LocalAnchorB { get; set; } = new();
        public Vec3 LocalAxisA { get; set; } = new(0, 1, 0);
        public Vec3 LocalAxisB { get; set; } = new(0, 1, 0);
        public float RestLength { get; set; } = 1f;
        public float Stiffness { get; set; } = 1f;
        public float Damping { get; set; } = 0.1f;
        public float Compliance { get; set; }
        public float MinLimit { get; set; } = float.NegativeInfinity;
        public float MaxLimit { get; set; } = float.PositiveInfinity;
    }

    public sealed class CameraData
    {
        public Vec3 Position { get; set; } = new(0, 2, 5);
        public float Yaw { get; set; } = -90f;
        public float Pitch { get; set; }
        public float Fov { get; set; } = 60f;
    }

    public sealed class Vec3
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        public Vec3() { }
        public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; }
        public Vector3 ToVector3() => new(X, Y, Z);
        public static Vec3 From(Vector3 v) => new(v.X, v.Y, v.Z);
    }

    /// <summary>
    /// Canonical scene entity type labels for the simulation atelier.
    /// Legacy "Character" (game-engine wording) is accepted on load and rewritten to <see cref="Agent"/>.
    /// </summary>
    public static class SceneEntityKinds
    {
        public const string Agent = "Agent";

        public static bool IsSentientAgent(string? type) =>
            !string.IsNullOrWhiteSpace(type) &&
            (type.Equals(Agent, StringComparison.OrdinalIgnoreCase) ||
             type.Equals("Character", StringComparison.OrdinalIgnoreCase) ||
             type.Equals("Sentient", StringComparison.OrdinalIgnoreCase));

        public static string Normalize(string? type)
        {
            if (IsSentientAgent(type))
                return Agent;
            return string.IsNullOrWhiteSpace(type) ? "Empty" : type;
        }
    }

    [JsonSerializable(typeof(SceneDocument))]
    [JsonSerializable(typeof(SceneEntityData))]
    [JsonSerializable(typeof(SceneJointData))]
    [JsonSerializable(typeof(CameraData))]
    [JsonSerializable(typeof(Vec3))]
    [JsonSerializable(typeof(List<SceneEntityData>))]
    [JsonSerializable(typeof(List<SceneJointData>))]
    [JsonSerializable(typeof(Dictionary<string, string>))]
    [JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    internal sealed partial class SceneDocumentJsonContext : JsonSerializerContext
    {
    }
}
