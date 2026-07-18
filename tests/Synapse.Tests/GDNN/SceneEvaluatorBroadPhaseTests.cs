using System;
using System.Numerics;
using FluentAssertions;
using GDNN.Evaluation;
using Xunit;

namespace Synapse.Tests.GDNN;

/// <summary>
/// Broad-phase BVH du SceneEvaluator : mêmes résultats que le chemin linéaire,
/// avec des assets écartés (comptés dans les stats de frame).
/// </summary>
public class SceneEvaluatorBroadPhaseTests
{
    private const float Radius = 0.5f;

    private static SceneNeuralAsset CreateSphereAsset(Vector3 center)
    {
        var asset = new SceneNeuralAsset
        {
            Network = new AnalyticSphereSdf(Radius),
            Transform = Matrix4x4.CreateTranslation(center),
            WorldBounds = IntervalBox.FromCenterHalfExtents(center, new Vector3(Radius + 0.1f)),
            BoundingCenter = center,
            BoundingRadius = Radius + 0.1f
        };
        asset.UpdateInverseTransform();
        return asset;
    }

    private static SceneEvaluator CreateTwoSphereScene(bool broadPhase)
    {
        var evaluator = new SceneEvaluator(new SceneEvaluatorConfig
        {
            EnableBroadPhase = broadPhase,
            EnableFrustumCulling = false
        });
        evaluator.AddAsset(CreateSphereAsset(Vector3.Zero));
        evaluator.AddAsset(CreateSphereAsset(new Vector3(10f, 0f, 0f)));
        return evaluator;
    }

    [Fact]
    public void TraceRay_WithBroadPhase_MatchesLinearPath()
    {
        using var withBvh = CreateTwoSphereScene(broadPhase: true);
        using var linear = CreateTwoSphereScene(broadPhase: false);

        var ray = new TracingRay(new Vector3(0f, 0f, -3f), Vector3.UnitZ, 100f);
        var hitBvh = withBvh.TraceRay(ray, ray.Origin);
        var hitLinear = linear.TraceRay(ray, ray.Origin);

        hitBvh.DidHit.Should().BeTrue();
        hitLinear.DidHit.Should().BeTrue();
        Vector3.Distance(hitBvh.HitPoint, hitLinear.HitPoint).Should().BeLessThan(1e-3f);
        hitBvh.AssetId.Should().Be(hitLinear.AssetId);

        // Le rayon vise la sphère à l'origine : le point d'impact est sur sa surface.
        MathF.Abs(hitBvh.HitPoint.Length() - Radius).Should().BeLessThan(0.02f);
    }

    [Fact]
    public void TraceRay_TowardSecondSphere_HitsIt()
    {
        using var evaluator = CreateTwoSphereScene(broadPhase: true);

        var ray = new TracingRay(new Vector3(10f, 0f, -3f), Vector3.UnitZ, 100f);
        var hit = evaluator.TraceRay(ray, ray.Origin);

        hit.DidHit.Should().BeTrue();
        Vector3.Distance(hit.HitPoint, new Vector3(10f, 0f, -Radius)).Should().BeLessThan(0.02f);
    }

    [Fact]
    public void TraceRay_MissingEverything_ReturnsNoHit()
    {
        using var evaluator = CreateTwoSphereScene(broadPhase: true);

        var ray = new TracingRay(new Vector3(0f, 5f, -3f), Vector3.UnitZ, 100f);
        evaluator.TraceRay(ray, ray.Origin).DidHit.Should().BeFalse();
    }

    [Fact]
    public void BroadPhase_CullsFarAssets_AndReportsCountInFrameStats()
    {
        using var evaluator = CreateTwoSphereScene(broadPhase: true);
        evaluator.BeginFrame(Matrix4x4.Identity, new Vector3(0f, 0f, -3f), Vector3.UnitZ);

        // Rayon vers la sphère à l'origine : celle à x=10 doit être écartée
        // sans évaluation de son réseau.
        var ray = new TracingRay(new Vector3(0f, 0f, -3f), Vector3.UnitZ, 100f);
        evaluator.TraceRay(ray, ray.Origin);

        var stats = evaluator.EndFrame(totalEvalTimeMs: 1f);
        stats.BroadPhaseCulled.Should().BeGreaterThan(0);
    }

    [Fact]
    public void EvaluatePointGrid_WithBroadPhase_MatchesLinearNearSurfaces()
    {
        using var withBvh = CreateTwoSphereScene(broadPhase: true);
        using var linear = CreateTwoSphereScene(broadPhase: false);

        // Points proches des surfaces : le broad-phase ne doit rien changer.
        Vector3[] points =
        [
            new(0f, 0f, 0.4f),
            new(0f, 0f, 0.6f),
            new(10f, 0.45f, 0f),
            new(9.4f, 0f, 0f)
        ];

        var resultsBvh = new float[points.Length];
        var resultsLinear = new float[points.Length];
        withBvh.EvaluatePointGrid(points, resultsBvh, Vector3.Zero);
        linear.EvaluatePointGrid(points, resultsLinear, Vector3.Zero);

        for (int i = 0; i < points.Length; i++)
            MathF.Abs(resultsBvh[i] - resultsLinear[i]).Should().BeLessThan(1e-4f,
                $"le point {i} est proche d'une surface");
    }

    [Fact]
    public void EvaluatePoint_NearSecondSphere_PicksIt()
    {
        using var evaluator = CreateTwoSphereScene(broadPhase: true);

        var hit = evaluator.EvaluatePoint(new Vector3(10.4f, 0f, 0f));

        hit.DidHit.Should().BeTrue();
        // SDF proche de la surface de la sphère en (10,0,0) : |10.4-10| - 0.5 = -0.1.
        hit.SdfValue.Should().BeApproximately(-0.1f, 0.02f);
    }
}
