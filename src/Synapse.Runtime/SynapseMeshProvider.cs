// ============================================================================
// Synapse Omnia — Runtime/SynapseMeshProvider.cs
// MeshProvider adapted to Synapse: MeshLoader → collision cook → optional G-DNN SDF bake.
// ============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;
using GDNN.Rendering.MeshIO;
using GDNN.Streaming;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;

namespace Synapse.Runtime
{
    /// <summary>
    /// Synapse Omnia mesh provider: loads artist meshes, cooks physics colliders,
    /// and can bake G-DNN neural SDFs into the scene asset stream.
    /// </summary>
    public sealed class SynapseMeshProvider : IMeshProvider
    {
        private readonly MeshLoader _loader = new();
        private readonly Dictionary<string, MeshCollisionSource> _sources = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, MeshAsset> _assets = new(StringComparer.OrdinalIgnoreCase);
        private readonly ISynapseLogger? _logger;

        public SynapseMeshProvider(ISynapseLogger? logger = null)
        {
            _logger = logger;
        }

        public IEnumerable<string> MeshIds => _sources.Keys;

        public bool TryGetMesh(string meshId, out MeshCollisionSource source) =>
            _sources.TryGetValue(meshId, out source);

        public MeshAsset? GetAsset(string meshId) =>
            _assets.TryGetValue(meshId, out var a) ? a : null;

        /// <summary>Registers an already-loaded mesh asset under <paramref name="meshId"/>.</summary>
        public MeshCollisionSource RegisterAsset(string meshId, MeshAsset asset)
        {
            ArgumentNullException.ThrowIfNull(asset);
            var source = ToCollisionSource(meshId, asset);
            _sources[meshId] = source;
            _assets[meshId] = asset;
            return source;
        }

        /// <summary>Loads a mesh file (glTF/OBJ/…) and registers it for physics + rendering.</summary>
        public async Task<MeshCollisionSource?> LoadAsync(
            string meshId,
            string filePath,
            CancellationToken cancellationToken = default)
        {
            var result = await _loader.LoadAsync(filePath, config: null, ct: cancellationToken).ConfigureAwait(false);
            if (!result.Success || result.Asset == null)
            {
                _logger?.Warn("MeshProvider", $"Failed to load '{filePath}': {result.ErrorMessage}");
                return null;
            }

            return RegisterAsset(meshId, result.Asset);
        }

        /// <summary>
        /// Cooks a collider for a Synapse body. Dynamic → convex hull; static → triangle mesh.
        /// </summary>
        public Collider? CookCollider(string meshId, BodyType bodyType, int maxHullVertices = 48)
        {
            if (!_sources.TryGetValue(meshId, out var source))
                return null;
            return MeshCollisionCooker.CookForBodyType(source, bodyType, maxHullVertices);
        }

        /// <summary>
        /// Optional G-DNN bake: train a neural SDF from the mesh and write a <c>.gnn</c>
        /// under the scene <see cref="AssetStreamer.AssetRootDirectory"/> for Omnia streaming.
        /// </summary>
        public async Task<string?> BakeNeuralSdfAsync(
            string meshId,
            MeshToSdfOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            if (!_assets.TryGetValue(meshId, out var asset))
                return null;

            try
            {
                string root = AssetStreamer.AssetRootDirectory
                    ?? Path.Combine(AppContext.BaseDirectory, "assets");
                Directory.CreateDirectory(root);
                string outPath = Path.Combine(root, $"{Sanitize(meshId)}.gnn");

                var bakeOptions = new MeshToSdfOptions
                {
                    SampleCount = options?.SampleCount ?? 1024,
                    Epochs = options?.Epochs ?? 8,
                    RandomSeed = options?.RandomSeed ?? 42,
                    LearningRate = options?.LearningRate ?? 5e-3f,
                    HashLearningRate = options?.HashLearningRate ?? 5e-2f,
                    LoadConfig = options?.LoadConfig,
                    OutputAssetPath = outPath,
                    TargetMeshId = options?.TargetMeshId
                };

                var bake = await Task.Run(
                    () => MeshToSdfPipeline.TrainFromAsset(asset, bakeOptions, cancellationToken),
                    cancellationToken).ConfigureAwait(false);

                if (!bake.Success || bake.Network == null)
                {
                    _logger?.Warn("MeshProvider", $"SDF bake failed for '{meshId}': {bake.ErrorMessage}");
                    return null;
                }

                _logger?.Info("MeshProvider", $"Baked G-DNN SDF for '{meshId}' → {outPath}");
                return outPath;
            }
            catch (Exception ex)
            {
                _logger?.Warn("MeshProvider", $"SDF bake failed for '{meshId}': {ex.Message}");
                return null;
            }
        }

        /// <summary>Creates a rigid body from a registered mesh id.</summary>
        public RigidBody? CreateBodyFromMesh(
            string meshId,
            BodyType bodyType,
            Vector3 position,
            float mass = 1f,
            string? name = null)
        {
            var collider = CookCollider(meshId, bodyType);
            if (collider == null)
                return null;

            var body = new RigidBody
            {
                Name = name ?? meshId,
                Type = bodyType,
                Collider = collider,
                Position = position,
                Material = PhysicsMaterial.Default
            };
            if (bodyType == BodyType.Dynamic)
                body.SetMass(mass);
            else
                body.SetMass(0f);
            return body;
        }

        public static MeshCollisionSource ToCollisionSource(string meshId, MeshAsset asset)
        {
            var verts = new List<Vector3>();
            var indices = new List<int>();
            int baseVertex = 0;
            foreach (var prim in asset.Primitives)
            {
                foreach (var v in prim.Vertices)
                    verts.Add(v.Position);
                foreach (var idx in prim.Indices)
                    indices.Add(baseVertex + (int)idx);
                baseVertex += prim.Vertices.Count;
            }
            return new MeshCollisionSource(meshId, verts.ToArray(), indices.ToArray());
        }

        private static string Sanitize(string id)
        {
            var chars = id.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_').ToArray();
            return new string(chars);
        }
    }
}
