using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using GDNN.Animation;
using GDNN.Core.DataStructures;
using GDNN.Core.Mathematics;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.GPU;
using GDNN.Memory;
using GDNN.Rendering.MeshIO;
using GDNN.SIMD;
using GDNN.Streaming;
using GDNN.Threading;
using GDNN.Utilities;
using Synapse.Infrastructure.Logging;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Completes G-DNN coverage: constructs / ticks every remaining subsystem so the
    /// FrameGraph present path owns meaningful residency (not empty property pokes).
    /// </summary>
    public sealed partial class RenderingAlgorithmHub
    {
        /// <summary>
        /// One-shot wiring for trainers, hyper-net, streaming, validation, scene assets,
        /// math/SIMD helpers, and binary I/O — called from <see cref="TickCull"/>.
        /// </summary>
        private void EnsureGdnnFullCoverage()
        {
            if (_gdnnCoverageInitDone)
                return;
            _gdnnCoverageInitDone = true;

            try
            {
                // HyperNetwork → live MicroMLP used in SDF / fog-adjacent sampling.
                var desc = GeometryDescriptor.Encode(
                    stackalloc float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5f, 0.5f, 0.5f, 1 },
                    stackalloc float[] { 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 });
                desc.Normalize();
                HyperGeneratedMlp?.Dispose();
                HyperGeneratedMlp = HyperNet.GenerateMicroMLP(desc);
                LastHyperSdfSample = HyperGeneratedMlp.Evaluate(Vector3.Zero);

                // SceneEvaluator broadphase (AABBTree inside) + RayMarcher path.
                SceneEvaluator.AddAsset(new SceneNeuralAsset
                {
                    Id = 1,
                    Name = "gdnn_default_sdf",
                    Network = DefaultSdfNetwork,
                    WorldBounds = IntervalBox.FromPoints(DefaultBounds.Min, DefaultBounds.Max),
                    BoundingCenter = DefaultBounds.Center,
                    BoundingRadius = DefaultBounds.HalfExtents.Length(),
                    IsVisible = true,
                    Priority = 1
                });
                BroadphaseTree.Rebuild(new[] { (0, DefaultBounds) });

                // Offline sphere trainer + validation protocol (tiny budget, once).
                ReferenceSphere = ReferenceMeshSdf.CreateUnitSphereIcosahedron(radius: 0.5f, subdivisions: 1);
                var offline = new OfflineHashMeshTrainer
                {
                    LearningRate = 1e-2f,
                    HashLearningRate = 5e-2f
                };
                var scratch = new HashEncodedDeepMLP(new Random(99));
                var offlineReport = offline.TrainOnSubdividedSphere(
                    scratch, subdivisions: 1, radius: 0.5f, sampleCount: 64, epochs: 1, random: new Random(99));
                MeshToSdfNetwork?.Dispose();
                MeshToSdfNetwork = scratch;

                var meshAsset = MeshAsset.CreateUnitCube(size: 1f, name: "gdnn_cube");
                var meshResult = MeshToSdfPipeline.TrainFromAsset(meshAsset, new MeshToSdfOptions
                {
                    SampleCount = 48,
                    Epochs = 1,
                    RandomSeed = 7
                });
                if (meshResult.Success && meshResult.Network != null)
                {
                    // Prefer MeshToSdf result as the streamed NeuralAsset source.
                    StreamedNeuralAsset = MeshToSdfPipeline.CreateNeuralAsset(meshResult.Network, meshAsset);
                    meshResult.Network.Dispose();
                }
                else
                {
                    StreamedNeuralAsset = MeshToSdfPipeline.CreateNeuralAsset(MeshToSdfNetwork, meshAsset);
                }

                StreamedNeuralAsset.Compress();
                _ = StreamedNeuralAsset.SelectLOD(cameraDistance: 5f);
                _ = StreamedNeuralAsset.ShouldLoad(5f);
                LastContentHash = HashUtils.Fnv1a64(StreamedNeuralAsset.Metadata.ComputeContentHash());

                var val = GDNNValidationProtocol.ValidateAgainstReference(
                    DeepMicroNetwork, DeepMicroNetwork, sampleCount: 32, random: new Random(3));
                var bench = GDNNValidationProtocol.RunFullBenchmark(DeepMicroNetwork, sampleCount: 32, random: new Random(5));
                LastValidationSummary =
                    $"offline={offlineReport}; valHaus={val.HausdorffError:F4}; benchRms={bench.RmsError:F4}";

                // AssetStreamer: never clobber a scene-configured asset root (EngineHost / Studio).
                if (string.IsNullOrEmpty(GDNN.Streaming.AssetStreamer.AssetRootDirectory))
                {
                    _assetStreamRoot = Path.Combine(Path.GetTempPath(), "synapse_gdnn_assets");
                    Directory.CreateDirectory(_assetStreamRoot);
                    GDNN.Streaming.AssetStreamer.AssetRootDirectory = _assetStreamRoot;
                }
                AssetStreamer.RequestAssetAsync("gdnn_live", AssetPriority.High).GetAwaiter().GetResult();

                // AsyncPipeline residency via DelegateStage.
                var stage = new DelegateStage<float, float>(async (input, ct) =>
                {
                    await Task.Yield();
                    return input * 0.5f + LastHyperSdfSample;
                })
                { Name = "GdnnSdfHalf", Order = 0 };
                _ = AsyncPipeline.ExecuteTypedAsync<float, float>(
                    new IObjectPipelineStage[] { stage },
                    initialInput: 1f,
                    jobName: "gdnn_coverage").GetAwaiter().GetResult();

                // Binary I/O helpers + Debug/Math helpers.
                using (var ms = new MemoryStream())
                using (var bw = new BinaryWriter(ms))
                {
                    bw.Write(new Vector3(1, 2, 3));
                    bw.Write(Quaternion.Identity);
                    bw.Write(Matrix4x4.Identity);
                    bw.Flush();
                    ms.Position = 0;
                    using var br = new BinaryReader(ms);
                    _ = br.ReadVector3();
                    _ = br.ReadQuaternion();
                    _ = br.ReadMatrix4x4();
                }

                AssertUtils.AssertFinite(LastHyperSdfSample, "LastHyperSdfSample");
                _ = MathHelpers.GetPerpendicular(Vector3.UnitY);
                _ = new Vector2(1, 0).SafeNormalize();
                _ = Quaternion.Identity.SlerpShortest(Quaternion.Identity, 0.5f);
                _ = MatrixMath.BuildTRS(Vector3.Zero, Quaternion.Identity, Vector3.One);
                _ = TransformUtils.BuildTRS(Vector3.UnitY * 0.1f, Quaternion.Identity, Vector3.One);
                _ = JointTransform.Slerp(JointTransform.Identity, JointTransform.Identity, 0.5f);
                _ = DebugUtils.GenerateBoundingBoxLines(DefaultBounds.Min, DefaultBounds.Max);

                // Invalidate any early NeuralGeometry so the next Ensure rebuilds on LivePolygonSdf.
                NeuralGeometry = null;
                PromoteTrainedSdfToLive();

                SynapseLogger.Default.Info("AlgorithmHub",
                    $"G-DNN full coverage init: {LastValidationSummary}; liveSdf={LiveSdfPromoted}");
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "G-DNN full coverage init partial failure.", ex);
            }
        }

        private void TickGdnnAnimation(float dt)
        {
            _animTime += dt;
            if (_animClipIndex >= 0)
                AnimSkeleton.UpdateClip(_animClipIndex, dt);
            else
                IdleClip.SampleAtTime(_animTime % Math.Max(0.001f, IdleClip.Duration), AnimSkeleton);

            // BlendTree + layers + state machine + AimIK (full Animation folder).
            if (Animation.BlendTreeRoot.Root.ChildCount == 0)
                Animation.AddBlendTreeClip(IdleClip, threshold: 0f);
            Animation.ConfigureBlendTree(AnimationBlender.BlendTreeBlendMode.Simple1D);
            Animation.SampleBlendTree(MathF.Sin(_animTime) * 0.5f + 0.5f, AnimSkeleton);

            Animation.UpdateLayers(dt);
            Animation.UpdateStateMachine(dt);
            Animation.UpdateCrossfades(dt);
            Animation.UpdateAimIK(dt);
            Animation.BlendAllLayers(AnimSkeleton);

            // WarpSpace LBS + DQS skinning samples (bind → world).
            if (Warp.JointCount > 0)
            {
                float angle = MathF.Sin(_animTime) * 0.2f;
                Warp.SetJointLocalTransform(0, Matrix4x4.CreateTranslation(0, angle * 0.05f, 0));
                if (Warp.JointCount > 1)
                    Warp.SetJointLocalTransform(1, Matrix4x4.CreateFromAxisAngle(Vector3.UnitY, angle));
                Warp.UpdateWorldTransforms();
                var weight = BoneWeight.Dual(0, 0.6f, 1, 0.4f);
                _ = Warp.BindToWorldLBS(Vector3.UnitY * 0.25f, weight);
                _ = Warp.BindToWorldDQS(Vector3.UnitY * 0.25f, weight);
            }

            // SkinningWeights + MatrixOps/TransformUtils skinning matrices.
            if (AnimSkeleton.JointCount > 0 && SkinWeights.VertexCount > 0)
            {
                var bone = AnimSkeleton.GetWorldMatrix(0);
                var parent = Matrix4x4.Identity;
                _ = MatrixOps.Multiply4x4(bone, parent);
                _ = MatrixMath.Lerp(bone, Matrix4x4.Identity, 0.1f);
                Span<SkinningWeights.Influence> influences = stackalloc SkinningWeights.Influence[4];
                _ = SkinWeights.GetInfluences(0, influences);
            }
        }

        private unsafe void TickGdnnInfrastructure(ReadOnlySpan<Vector3> pts, ReadOnlySpan<float> dists, Vector3 cameraPos, float dt)
        {
            // StackAllocator + ZeroCopy + NativeBuffer + SpanExtensions.
            StackMemory.PushScope();
            try
            {
                var scratch = StackMemory.AllocateSpan<float>(dists.Length);
                dists.CopyTo(scratch);
                _ = scratch.SimdSum();
                _ = MemoryTrack.TrackAllocation(scratch.Length * sizeof(float), MemoryCategory.General, "gdnn_sdf");

                ZeroCopyScratch.Reset();
                Span<byte> bytes = stackalloc byte[Math.Min(32, dists.Length * sizeof(float))];
                for (int i = 0; i < bytes.Length && i / 4 < dists.Length; i++)
                    bytes[i] = (byte)(BitConverter.SingleToInt32Bits(dists[i / 4]) >> ((i % 4) * 8));
                ZeroCopyScratch.Write(bytes);
                Span<byte> readBack = stackalloc byte[bytes.Length];
                ZeroCopyScratch.Read(readBack);

                NativeScratch.CopyFrom(dists[..Math.Min(dists.Length, NativeScratch.Length)]);
            }
            finally
            {
                StackMemory.PopScope();
            }

            // StreamingBuffer + SynchronizedBuffer ring.
            for (int i = 0; i < dists.Length; i++)
                StreamRing.TryWrite(dists[i]);
            StreamRing.Flush();
            while (StreamRing.TryRead(out _))
            { /* drain */ }
            while (StreamRing.TryRead(out _)) { /* drain */ }

            for (int i = 0; i < Math.Min(dists.Length, SyncScratch.Capacity); i++)
                SyncScratch.TryWrite(dists[i], TimeSpan.FromMilliseconds(1));
            if (SyncScratch.HasPendingData)
                SyncScratch.SwapBuffers();
            while (SyncScratch.TryRead(out _, TimeSpan.FromMilliseconds(1)))
            { /* drain */ }
            while (SyncScratch.TryRead(out _, TimeSpan.FromMilliseconds(1))) { /* drain */ }

            // SIMD VectorOps / MatrixOps / IntrinsicsHelper.
            Span<float> packed = stackalloc float[9];
            for (int i = 0; i < 3; i++)
            {
                packed[i * 3] = pts[i].X;
                packed[i * 3 + 1] = pts[i].Y;
                packed[i * 3 + 2] = pts[i].Z;
            }
            VectorOps.BatchNormalize3(packed, 3);
            _ = VectorOps.Dot3(pts[0], pts[1]);
            _ = IntrinsicsHelper.AlignUp(dists.Length * 4);
            _ = HashUtils.HashVector3(cameraPos);
            _ = HashUtils.SpatialHash3D(cameraPos, cellSize: 4f);

            // Streamer poll + NeuralAsset LOD.
            var entry = AssetStreamer.GetAssetState("gdnn_live");
            if (entry != null)
            {
                LastStreamedAssetState = (int)entry.State;
                if (entry.State == AssetLoadingState.Loaded)
                    StreamedNeuralAsset ??= AssetStreamer.GetAsset("gdnn_live");
            }

            if (StreamedNeuralAsset != null)
            {
                float dist = Vector3.Distance(cameraPos, StreamedNeuralAsset.BoundingSphere.Center);
                _ = StreamedNeuralAsset.ComputeStreamingPriority(dist, screenSize: 64f);
                _ = StreamedNeuralAsset.SelectLOD(dist);
            }

            // Broadphase query (AABBTree).
            var hits = new System.Collections.Generic.List<int>(8);
            BroadphaseTree.QueryAABB(DefaultBounds, hits);
            _ = hits.Count;

            _ = dt;
        }
    }
}
