using System;
using System.Numerics;
using FluentAssertions;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.Polygonization;
using Xunit;

namespace Synapse.Tests.GDNN;

/// <summary>
/// Entraîne un HashEncodedDeepMLP sur une sphère avec un échantillonnage stratifié
/// (bande de surface / intérieur / volume) — indispensable pour que le réseau
/// apprenne un vrai changement de signe et pas un champ positif partout.
/// </summary>
internal static class SphereNetworkFactory
{
    public static HashEncodedDeepMLP CreateTrained(int seed = 7)
    {
        var network = new HashEncodedDeepMLP(new Random(seed));
        var trainer = new HashEncodedDeepMLPTrainer(network)
        {
            LearningRate = 1e-2f,
            HashLearningRate = 1e-1f
        };

        const int samples = 2048;
        var random = new Random(seed);
        var points = new Vector3[samples];
        var targets = new float[samples];
        for (int i = 0; i < samples; i++)
        {
            Vector3 dir;
            do
            {
                dir = new Vector3(
                    random.NextSingle() * 2 - 1,
                    random.NextSingle() * 2 - 1,
                    random.NextSingle() * 2 - 1);
            } while (dir.LengthSquared() < 1e-4f);
            dir = Vector3.Normalize(dir);

            // 60 % bande de surface, 20 % intérieur profond, 20 % volume.
            points[i] = (i % 5) switch
            {
                0 or 1 or 2 => dir * (0.5f + (random.NextSingle() * 2 - 1) * 0.12f),
                3 => dir * (random.NextSingle() * 0.42f),
                _ => new Vector3(
                    random.NextSingle() * 2 - 1,
                    random.NextSingle() * 2 - 1,
                    random.NextSingle() * 2 - 1)
            };
            targets[i] = ReferenceMeshSdf.AnalyticSphereSdf(points[i], 0.5f);
        }

        trainer.Fit(points, targets, epochs: 24, new Random(seed));
        return network;
    }
}

public class OnlineSdfTrainerTests
{
    private static readonly AABB Domain = new(Vector3.Zero, new Vector3(1f));

    [Fact]
    public void TrainSlice_WithoutData_DoesNothing()
    {
        using var network = new HashEncodedDeepMLP(new Random(7));
        var trainer = new OnlineSdfTrainer(network, Domain);

        var report = trainer.TrainSlice(budgetMs: 5.0);

        report.StepsExecuted.Should().Be(0);
        report.GeometryVersion.Should().Be(0);
        trainer.TryConsumeDirtyBounds(out _).Should().BeFalse();
    }

    [Fact]
    public void TrainSlice_RespectsTimeBudget()
    {
        using var network = new HashEncodedDeepMLP(new Random(7));
        var trainer = new OnlineSdfTrainer(network, Domain);
        EnqueueSphereSamples(trainer, 8192);

        var report = trainer.TrainSlice(budgetMs: 3.0);

        report.StepsExecuted.Should().BeGreaterThan(0);
        // Marge large : la granularité est un pas de SGD, pas une interruption.
        report.ElapsedMs.Should().BeLessThan(100.0);
        report.PendingRemaining.Should().BeGreaterThan(0, "le budget doit borner le travail");
    }

    [Fact]
    public void TrainSlice_ConsumingEdits_IncrementsVersionAndTracksDirtyBounds()
    {
        using var network = new HashEncodedDeepMLP(new Random(7));
        var trainer = new OnlineSdfTrainer(network, Domain);
        EnqueueSphereSamples(trainer, 256);

        trainer.TrainSlice(budgetMs: 50.0);

        trainer.GeometryVersion.Should().Be(1);
        trainer.TryConsumeDirtyBounds(out var dirty).Should().BeTrue();
        dirty.HalfExtents.X.Should().BeGreaterThan(0f);
        trainer.TryConsumeDirtyBounds(out _).Should().BeFalse("les bornes sont consommées");
    }

    [Fact]
    public void TrainSlices_ReduceErrorAgainstAnalyticTarget()
    {
        using var network = new HashEncodedDeepMLP(new Random(7));
        var trainer = new OnlineSdfTrainer(network, Domain);

        float rmsBefore = MeasureRms(network);
        EnqueueSphereSamples(trainer, 2048);
        for (int i = 0; i < 200 && trainer.PendingCount > 0; i++)
            trainer.TrainSlice(budgetMs: 25.0);

        MeasureRms(network).Should().BeLessThan(rmsBefore);
    }

    [Fact]
    public void ApplySphereEdit_Subtract_RemovesMaterial_WhileAnchorsPreserveTheRest()
    {
        using var network = SphereNetworkFactory.CreateTrained();

        var brushCenter = new Vector3(0.5f, 0f, 0f);
        var oppositePoint = new Vector3(-0.5f, 0f, 0f);
        float atBrushBefore = network.Evaluate(brushCenter);
        float atOppositeBefore = network.Evaluate(oppositePoint);

        var trainer = new OnlineSdfTrainer(network, Domain);
        trainer.ApplySphereEdit(brushCenter, radius: 0.25f, SdfEditOperation.Subtract);
        for (int i = 0; i < 200 && trainer.PendingCount > 0; i++)
            trainer.TrainSlice(budgetMs: 25.0);
        // Quelques tranches de replay pour consolider.
        for (int i = 0; i < 5; i++)
            trainer.TrainSlice(budgetMs: 10.0);

        // La matière a été retirée : le SDF au centre de la brosse devient positif.
        network.Evaluate(brushCenter).Should().BeGreaterThan(atBrushBefore + 0.05f);

        // Les ancres d'auto-distillation préservent le côté opposé.
        MathF.Abs(network.Evaluate(oppositePoint) - atOppositeBefore).Should().BeLessThan(0.15f);
    }

    private static void EnqueueSphereSamples(OnlineSdfTrainer trainer, int count)
    {
        var random = new Random(11);
        var points = new Vector3[count];
        var targets = new float[count];
        for (int i = 0; i < count; i++)
        {
            points[i] = new Vector3(
                random.NextSingle() * 2f - 1f,
                random.NextSingle() * 2f - 1f,
                random.NextSingle() * 2f - 1f);
            targets[i] = ReferenceMeshSdf.AnalyticSphereSdf(points[i], 0.5f);
        }
        trainer.EnqueueSamples(points, targets, Domain);
    }

    private static float MeasureRms(HashEncodedDeepMLP network)
    {
        var random = new Random(23);
        float sumSq = 0f;
        const int count = 512;
        for (int i = 0; i < count; i++)
        {
            var p = new Vector3(
                random.NextSingle() * 2f - 1f,
                random.NextSingle() * 2f - 1f,
                random.NextSingle() * 2f - 1f);
            float diff = network.Evaluate(p) - ReferenceMeshSdf.AnalyticSphereSdf(p, 0.5f);
            sumSq += diff * diff;
        }
        return MathF.Sqrt(sumSq / count);
    }
}

public class NeuralClusterRendererTests
{
    private static NeuralPolygonLodChain BuildSphereChain()
    {
        var sdf = new AnalyticSphereSdf(0.5f);
        var bounds = new AABB(Vector3.Zero, new Vector3(0.8f));
        return NeuralPolygonLodChain.Build(sdf, bounds, baseResolution: 32, levelCount: 3);
    }

    private static CameraView FrontCamera(float distance = 2f) =>
        CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, distance), Vector3.Zero, MathF.PI / 3f,
            aspect: 16f / 9f, nearPlane: 0.01f, farPlane: 100f, screenHeightPixels: 1080);

    [Fact]
    public void SelectVisible_FacingObject_ReturnsClusters_AndCullsBackfaces()
    {
        var renderer = new NeuralClusterRenderer(BuildSphereChain());

        var visible = renderer.SelectVisible(FrontCamera(), out var stats);

        visible.Should().NotBeEmpty();
        stats.VisibleClusters.Should().Be(visible.Count);
        stats.VisibleTriangles.Should().BeGreaterThan(0);

        // Sur une sphère, une part substantielle des clusters est dos à la caméra.
        stats.BackfaceCulled.Should().BeGreaterThan(stats.ClustersTotal / 5);
    }

    [Fact]
    public void SelectVisible_LookingAway_CullsEverything()
    {
        var renderer = new NeuralClusterRenderer(BuildSphereChain());
        var camera = CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), new Vector3(0, 0, 4f), MathF.PI / 3f,
            aspect: 16f / 9f, nearPlane: 0.01f, farPlane: 100f, screenHeightPixels: 1080);

        var visible = renderer.SelectVisible(camera, out var stats);

        visible.Should().BeEmpty();
        stats.FrustumCulled.Should().Be(stats.ClustersTotal);
    }

    [Fact]
    public void SelectVisible_FartherCamera_PicksCoarserLevel()
    {
        var renderer = new NeuralClusterRenderer(BuildSphereChain());

        renderer.SelectVisible(FrontCamera(1.5f), out var near);
        renderer.SelectVisible(FrontCamera(300f), out var far);

        far.SelectedLevel.Should().BeGreaterThan(near.SelectedLevel);
    }

    [Fact]
    public void IsBackfacing_ConeTest_IsConservative()
    {
        var meshlet = new NeuralMeshlet
        {
            VertexIndices = [0],
            LocalIndices = [0, 0, 0],
            Bounds = new AABB(Vector3.Zero, new Vector3(0.05f)),
            ConeAxis = Vector3.UnitZ,
            ConeCutoff = 0.9f
        };

        // Caméra du côté des normales : jamais cullé.
        NeuralClusterRenderer.IsBackfacing(meshlet, new Vector3(0, 0, 5f)).Should().BeFalse();

        // Caméra à l'opposé : entièrement dos, cullé.
        NeuralClusterRenderer.IsBackfacing(meshlet, new Vector3(0, 0, -5f)).Should().BeTrue();

        // Cône dégénéré : jamais cullé (silhouette possible).
        var degenerate = new NeuralMeshlet
        {
            VertexIndices = [0],
            LocalIndices = [0, 0, 0],
            Bounds = meshlet.Bounds,
            ConeAxis = Vector3.UnitZ,
            ConeCutoff = -1f
        };
        NeuralClusterRenderer.IsBackfacing(degenerate, new Vector3(0, 0, -5f)).Should().BeFalse();
    }
}

public class NeuralGeometryPipelineTests
{
    private static readonly AABB Bounds = new(Vector3.Zero, new Vector3(0.8f));

    private static HashEncodedDeepMLP CreateTrainedSphereNetwork()
        => SphereNetworkFactory.CreateTrained();

    private static CameraView FrontCamera() =>
        CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), Vector3.Zero, MathF.PI / 3f,
            aspect: 16f / 9f, nearPlane: 0.01f, farPlane: 100f, screenHeightPixels: 1080);

    [Fact]
    public void Tick_WithoutEdits_RendersWithoutRebuilding()
    {
        using var network = CreateTrainedSphereNetwork();
        var pipeline = new NeuralGeometryPipeline(network, Bounds);

        var report = pipeline.Tick(FrontCamera(), trainBudgetMs: 1.0);

        report.Rebuilt.Should().BeFalse();
        report.VisibleClusters.Should().NotBeEmpty();
        report.Culling.VisibleTriangles.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Tick_AfterSculptEdit_RetrainsAndRebuildsGeometry()
    {
        using var network = CreateTrainedSphereNetwork();
        var pipeline = new NeuralGeometryPipeline(network, Bounds, new NeuralGeometryPipelineOptions
        {
            BaseResolution = 24,
            LevelCount = 2
        });
        var meshBefore = pipeline.Chain.Levels[0].Mesh;

        var brushCenter = new Vector3(0.5f, 0f, 0f);
        pipeline.Trainer.ApplySphereEdit(brushCenter, radius: 0.25f, SdfEditOperation.Subtract);

        bool rebuilt = false;
        int totalSteps = 0;
        for (int tick = 0; tick < 150 && !(rebuilt && pipeline.Trainer.PendingCount == 0); tick++)
        {
            var report = pipeline.Tick(FrontCamera(), trainBudgetMs: 5.0);
            rebuilt |= report.Rebuilt;
            totalSteps += report.Training.StepsExecuted;
        }

        rebuilt.Should().BeTrue("l'édit doit déclencher une ré-extraction");
        totalSteps.Should().BeGreaterThan(0);
        pipeline.Trainer.PendingCount.Should().Be(0);

        // La matière retirée doit se voir dans le SDF appris.
        network.Evaluate(brushCenter).Should().BeGreaterThan(0.02f);

        // Et la chaîne a bien été régénérée depuis le réseau modifié : le contenu
        // géométrique doit différer (le nombre de triangles peut coïncider).
        var meshAfter = pipeline.Chain.Levels[0].Mesh;
        bool identical = meshAfter.VertexCount == meshBefore.VertexCount;
        if (identical)
        {
            for (int i = 0; i < meshAfter.VertexCount && identical; i++)
                identical = (meshAfter.Positions[i] - meshBefore.Positions[i]).LengthSquared() < 1e-12f;
        }
        identical.Should().BeFalse("la géométrie extraite doit refléter le réseau ré-entraîné");
    }

    [Fact]
    public void RenderFrame_RasterizesVisibleClustersIntoVisibilityBuffer()
    {
        using var network = CreateTrainedSphereNetwork();
        var pipeline = new NeuralGeometryPipeline(network, Bounds);

        var target = new RasterTarget(128, 128);
        var stats = pipeline.RenderFrame(FrontCamera(), target, out var visible);

        visible.Should().NotBeEmpty();
        stats.TrianglesRasterized.Should().BeGreaterThan(0);
        target.CountCoveredPixels().Should().BeGreaterThan(200);
    }

    [Fact]
    public void Tick_ReportsSparseExtractionMetrics()
    {
        using var network = CreateTrainedSphereNetwork();
        var pipeline = new NeuralGeometryPipeline(network, Bounds);

        // L'extraction initiale (constructeur) a utilisé le mode épars.
        var report = pipeline.Polygonizer.LastReport;
        report.SdfEvaluations.Should().BeGreaterThan(0);
        report.SdfEvaluations.Should().BeLessThan(report.TheoreticalDenseEvaluations);
    }
}
