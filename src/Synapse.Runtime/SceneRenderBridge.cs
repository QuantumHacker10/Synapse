using System;
using System.Numerics;
using GDNN.Rendering.Engine;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime
{
    /// <summary>
    /// Syncs runtime scene documents with the Vulkan deferred renderer and L-DNN bridge.
    /// </summary>
    public static class SceneRenderBridge
    {
        public static void SyncDocument(RenderEngine? renderEngine, SceneDocument scene, ViewportEditorState editor, ISynapseLogger logger)
        {
            var renderer = renderEngine?.SceneRenderer;
            if (renderer == null || !renderer.IsInitialized) return;

            foreach (var entity in scene.Entities)
            {
                if (!entity.Visible) continue;
                renderer.SyncEntityProxy(
                    entity.Id,
                    entity.Type,
                    entity.Position.ToVector3(),
                    entity.Scale.ToVector3(),
                    entity.Rotation.ToVector3());
            }

            var selected = scene.Entities.Find(e => e.Id == editor.SelectedEntityId);
            renderer.SyncEditorGizmos(
                editor.ShowGrid,
                editor.ShowGizmos,
                editor.SelectedEntityId,
                selected?.Position.ToVector3() ?? Vector3.Zero,
                selected?.Scale.ToVector3() ?? Vector3.One);

            renderer.PushLightsToGlobalIllumination();
            logger.Debug("SceneRenderBridge", $"Synced {scene.Entities.Count} entities to renderer");
        }

        public static void SyncDocument(RenderEngine? renderEngine, SceneDocument scene, ISynapseLogger logger)
            => SyncDocument(renderEngine, scene, new ViewportEditorState(), logger);

        public static void ApplyEvolutionVisual(
            RenderEngine? renderEngine,
            SceneDocument scene,
            int generation,
            double fitness,
            ISynapseLogger logger)
        {
            var genomeEntity = scene.Entities.Find(e =>
                e.Type.Equals("Genome", StringComparison.OrdinalIgnoreCase));

            float scale = Math.Clamp(0.4f + (float)fitness * 0.6f, 0.3f, 3f);
            if (genomeEntity == null)
            {
                genomeEntity = new SceneEntityData
                {
                    Id = Guid.NewGuid(),
                    Name = $"EvolvedForm g{generation}",
                    Type = "Genome",
                    Position = new Vec3(0, 1.5f, -2),
                    GenomeId = $"neat-g{generation}"
                };
                scene.Entities.Add(genomeEntity);
            }

            genomeEntity.Name = $"EvolvedForm g{generation} (fit={fitness:F2})";
            genomeEntity.GenomeId = $"neat-g{generation}";
            genomeEntity.Scale = new Vec3(scale, scale, scale);

            SyncDocument(renderEngine, scene, logger);
            logger.Info("Evolution", $"Applied best genome visual scale={scale:F2} gen={generation}");
        }
    }
}
