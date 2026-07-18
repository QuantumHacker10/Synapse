using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;
using GDNN.Polygonization;
using Xunit;

namespace Synapse.Tests.GDNN;

/// <summary>
/// SDF analytique (sphère) derrière ISdfNetwork : vérité terrain exacte
/// pour valider la polygonisation indépendamment de l'entraînement.
/// </summary>
internal sealed class AnalyticSphereSdf : ISdfNetwork
{
    private readonly float _radius;

    public AnalyticSphereSdf(float radius) => _radius = radius;

    public float Evaluate(Vector3 point) => point.Length() - _radius;

    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        for (int i = 0; i < points.Length; i++)
            distances[i] = Evaluate(points[i]);
    }

    public Vector3 ComputeGradient(Vector3 point)
        => point.LengthSquared() > 1e-12f ? Vector3.Normalize(point) : Vector3.UnitY;

    public float EvaluateWithGradient(Vector3 point, out Vector3 gradient)
    {
        gradient = ComputeGradient(point);
        return Evaluate(point);
    }
}

/// <summary>SDF analytique d'une boîte : arêtes vives pour valider le QEF.</summary>
internal sealed class AnalyticBoxSdf : ISdfNetwork
{
    private readonly Vector3 _halfExtents;

    public AnalyticBoxSdf(Vector3 halfExtents) => _halfExtents = halfExtents;

    public float Evaluate(Vector3 point)
    {
        Vector3 q = Vector3.Abs(point) - _halfExtents;
        Vector3 outside = Vector3.Max(q, Vector3.Zero);
        float inside = MathF.Min(MathF.Max(q.X, MathF.Max(q.Y, q.Z)), 0f);
        return outside.Length() + inside;
    }

    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        for (int i = 0; i < points.Length; i++)
            distances[i] = Evaluate(points[i]);
    }

    public Vector3 ComputeGradient(Vector3 point)
    {
        const float eps = 1e-4f;
        var g = new Vector3(
            Evaluate(point + new Vector3(eps, 0, 0)) - Evaluate(point - new Vector3(eps, 0, 0)),
            Evaluate(point + new Vector3(0, eps, 0)) - Evaluate(point - new Vector3(0, eps, 0)),
            Evaluate(point + new Vector3(0, 0, eps)) - Evaluate(point - new Vector3(0, 0, eps)));
        return g.LengthSquared() > 1e-12f ? Vector3.Normalize(g) : Vector3.UnitY;
    }

    public float EvaluateWithGradient(Vector3 point, out Vector3 gradient)
    {
        gradient = ComputeGradient(point);
        return Evaluate(point);
    }
}

public class NeuralPolygonizerTests
{
    private const float Radius = 0.5f;
    private static readonly AABB Bounds = new(Vector3.Zero, new Vector3(0.8f));

    [Fact]
    public void Extract_Sphere_VerticesLieOnSurface()
    {
        var sdf = new AnalyticSphereSdf(Radius);
        var mesh = new NeuralPolygonizer().Extract(sdf, Bounds, resolution: 32);

        mesh.VertexCount.Should().BeGreaterThan(100);
        mesh.TriangleCount.Should().BeGreaterThan(100);

        foreach (var p in mesh.Positions)
            MathF.Abs(p.Length() - Radius).Should().BeLessThan(0.01f,
                "après projection SDF, les sommets doivent être sur l'iso-surface");
    }

    [Fact]
    public void Extract_Sphere_NormalsPointOutward()
    {
        var sdf = new AnalyticSphereSdf(Radius);
        var mesh = new NeuralPolygonizer().Extract(sdf, Bounds, resolution: 24);

        for (int i = 0; i < mesh.VertexCount; i++)
        {
            Vector3 outward = Vector3.Normalize(mesh.Positions[i]);
            Vector3.Dot(mesh.Normals[i], outward).Should().BeGreaterThan(0.99f);
        }
    }

    [Fact]
    public void Extract_Sphere_TriangleWindingMatchesNormals()
    {
        var sdf = new AnalyticSphereSdf(Radius);
        var mesh = new NeuralPolygonizer().Extract(sdf, Bounds, resolution: 24);

        int outwardFacing = 0;
        for (int t = 0; t < mesh.TriangleCount; t++)
        {
            Vector3 a = mesh.Positions[mesh.Indices[t * 3]];
            Vector3 b = mesh.Positions[mesh.Indices[t * 3 + 1]];
            Vector3 c = mesh.Positions[mesh.Indices[t * 3 + 2]];
            Vector3 faceNormal = Vector3.Cross(b - a, c - a);
            Vector3 centroid = (a + b + c) / 3f;
            if (Vector3.Dot(faceNormal, centroid) > 0f)
                outwardFacing++;
        }

        // Tolérance : quelques triangles quasi dégénérés peuvent osciller.
        (outwardFacing / (float)mesh.TriangleCount).Should().BeGreaterThan(0.98f);
    }

    [Fact]
    public void Extract_Sphere_IsWatertight()
    {
        var sdf = new AnalyticSphereSdf(Radius);
        var mesh = new NeuralPolygonizer().Extract(sdf, Bounds, resolution: 20);

        // Surface fermée : chaque arête doit être partagée par exactement 2 triangles.
        var edgeCounts = new Dictionary<(int, int), int>();
        for (int t = 0; t < mesh.TriangleCount; t++)
        {
            for (int e = 0; e < 3; e++)
            {
                int v0 = mesh.Indices[t * 3 + e];
                int v1 = mesh.Indices[t * 3 + (e + 1) % 3];
                var key = v0 < v1 ? (v0, v1) : (v1, v0);
                edgeCounts[key] = edgeCounts.GetValueOrDefault(key) + 1;
            }
        }

        foreach (var count in edgeCounts.Values)
            count.Should().Be(2);
    }

    [Fact]
    public void Extract_Report_ContainsMeasuredMetrics()
    {
        var sdf = new AnalyticSphereSdf(Radius);
        var polygonizer = new NeuralPolygonizer { UseSparseTraversal = false };
        polygonizer.Extract(sdf, Bounds, resolution: 16);

        var report = polygonizer.LastReport;
        report.SdfEvaluations.Should().Be(17 * 17 * 17);
        report.TheoreticalDenseEvaluations.Should().Be(17 * 17 * 17);
        report.VertexCount.Should().BeGreaterThan(0);
        report.MaxSurfaceDeviation.Should().BeLessThan(0.01f);
        report.GeometricError.Should().BeApproximately(
            (Bounds.Size / 16f).Length(), 1e-5f);
    }

    [Fact]
    public void SparseTraversal_ProducesIdenticalTopology_WithFarFewerEvaluations()
    {
        var sdf = new AnalyticSphereSdf(Radius);

        var dense = new NeuralPolygonizer { UseSparseTraversal = false };
        var sparse = new NeuralPolygonizer { UseSparseTraversal = true };
        var denseMesh = dense.Extract(sdf, Bounds, resolution: 32);
        var sparseMesh = sparse.Extract(sdf, Bounds, resolution: 32);

        sparseMesh.VertexCount.Should().Be(denseMesh.VertexCount);
        sparseMesh.TriangleCount.Should().Be(denseMesh.TriangleCount);

        // Le gain épars doit être mesurable, pas déclaré : ici < 40 % du coût dense.
        sparse.LastReport.SdfEvaluations.Should().BeLessThan(
            (int)(dense.LastReport.SdfEvaluations * 0.4f));
        AssertWatertight(sparseMesh);
    }

    [Fact]
    public void DualContouringQef_ReconstructsSharpEdges_BetterThanSurfaceNets()
    {
        var box = new AnalyticBoxSdf(new Vector3(0.3f, 0.25f, 0.35f));
        var bounds = new AABB(Vector3.Zero, new Vector3(0.6f));

        var surfaceNets = new NeuralPolygonizer
        {
            VertexPlacement = VertexPlacementMode.SurfaceNets,
            SurfaceProjectionIterations = 0
        };
        var qef = new NeuralPolygonizer
        {
            VertexPlacement = VertexPlacementMode.DualContouringQef
        };

        surfaceNets.Extract(box, bounds, resolution: 20);
        var qefMesh = qef.Extract(box, bounds, resolution: 20);

        // Sur des arêtes vives, le QEF place les sommets sur l'intersection des
        // plans : la déviation doit être strictement meilleure que le barycentre.
        qef.LastReport.MaxSurfaceDeviation.Should().BeLessThan(
            surfaceNets.LastReport.MaxSurfaceDeviation);
        AssertWatertight(qefMesh);
    }

    internal static void AssertWatertight(NeuralPolygonMesh mesh)
    {
        var edgeCounts = new Dictionary<(int, int), int>();
        for (int t = 0; t < mesh.TriangleCount; t++)
        {
            for (int e = 0; e < 3; e++)
            {
                int v0 = mesh.Indices[t * 3 + e];
                int v1 = mesh.Indices[t * 3 + (e + 1) % 3];
                var key = v0 < v1 ? (v0, v1) : (v1, v0);
                edgeCounts[key] = edgeCounts.GetValueOrDefault(key) + 1;
            }
        }
        foreach (var count in edgeCounts.Values)
            count.Should().Be(2);
    }

    [Fact]
    public void Extract_HigherResolution_ReducesSurfaceDeviation()
    {
        var sdf = new AnalyticSphereSdf(Radius);

        var coarse = new NeuralPolygonizer { SurfaceProjectionIterations = 0 };
        var fine = new NeuralPolygonizer { SurfaceProjectionIterations = 0 };
        coarse.Extract(sdf, Bounds, resolution: 8);
        fine.Extract(sdf, Bounds, resolution: 48);

        fine.LastReport.MaxSurfaceDeviation.Should().BeLessThan(
            coarse.LastReport.MaxSurfaceDeviation);
    }
}

public class NeuralMeshletBuilderTests
{
    private static NeuralPolygonMesh ExtractSphereMesh(int resolution = 24)
    {
        var sdf = new AnalyticSphereSdf(0.5f);
        var bounds = new AABB(Vector3.Zero, new Vector3(0.8f));
        return new NeuralPolygonizer().Extract(sdf, bounds, resolution);
    }

    [Fact]
    public void Build_RespectsMeshShaderLimits()
    {
        var meshlets = new NeuralMeshletBuilder().Build(ExtractSphereMesh());

        meshlets.Should().NotBeEmpty();
        foreach (var m in meshlets)
        {
            m.VertexCount.Should().BeInRange(1, NeuralMeshletBuilder.MaxVertices);
            m.TriangleCount.Should().BeInRange(1, NeuralMeshletBuilder.MaxTriangles);
            m.LocalIndices.Length.Should().Be(m.TriangleCount * 3);
        }
    }

    [Fact]
    public void Build_CoversEveryTriangleExactlyOnce()
    {
        var mesh = ExtractSphereMesh();
        var meshlets = new NeuralMeshletBuilder().Build(mesh);

        int total = 0;
        foreach (var m in meshlets)
            total += m.TriangleCount;
        total.Should().Be(mesh.TriangleCount);
    }

    [Fact]
    public void Build_LocalIndicesResolveToValidGlobalVertices()
    {
        var mesh = ExtractSphereMesh();
        var meshlets = new NeuralMeshletBuilder().Build(mesh);

        foreach (var m in meshlets)
        {
            foreach (byte local in m.LocalIndices)
            {
                local.Should().BeLessThan((byte)m.VertexCount);
                m.VertexIndices[local].Should().BeInRange(0, mesh.VertexCount - 1);
            }
        }
    }

    [Fact]
    public void Build_ConeAndBoundsAreConsistent()
    {
        var mesh = ExtractSphereMesh();
        var meshlets = new NeuralMeshletBuilder().Build(mesh);

        foreach (var m in meshlets)
        {
            m.ConeAxis.Length().Should().BeApproximately(1f, 1e-3f);
            m.ConeCutoff.Should().BeInRange(-1f, 1f);

            foreach (int v in m.VertexIndices)
            {
                Vector3 p = mesh.Positions[v];
                (p.X >= m.Bounds.Min.X - 1e-5f && p.X <= m.Bounds.Max.X + 1e-5f).Should().BeTrue();
                (p.Y >= m.Bounds.Min.Y - 1e-5f && p.Y <= m.Bounds.Max.Y + 1e-5f).Should().BeTrue();
                (p.Z >= m.Bounds.Min.Z - 1e-5f && p.Z <= m.Bounds.Max.Z + 1e-5f).Should().BeTrue();
            }
        }
    }
}

public class NeuralPolygonLodChainTests
{
    private static NeuralPolygonLodChain BuildChain(int baseResolution = 32, int levels = 3)
    {
        var sdf = new AnalyticSphereSdf(0.5f);
        var bounds = new AABB(Vector3.Zero, new Vector3(0.8f));
        return NeuralPolygonLodChain.Build(sdf, bounds, baseResolution, levels);
    }

    [Fact]
    public void Build_ProducesDecreasingResolutionAndIncreasingError()
    {
        var chain = BuildChain();

        chain.Levels.Should().HaveCount(3);
        for (int i = 1; i < chain.Levels.Count; i++)
        {
            chain.Levels[i].GridResolution.Should().Be(chain.Levels[i - 1].GridResolution / 2);
            chain.Levels[i].GeometricError.Should().BeGreaterThan(chain.Levels[i - 1].GeometricError);
            chain.Levels[i].Mesh.TriangleCount.Should().BeLessThan(chain.Levels[i - 1].Mesh.TriangleCount);
        }
    }

    [Fact]
    public void SelectLod_NearCamera_ReturnsFinestLevel()
    {
        var chain = BuildChain();
        var lod = chain.SelectLod(
            distance: 0.1f, verticalFovRadians: MathF.PI / 3f,
            screenHeightPixels: 1080, pixelErrorThreshold: 1.0f);

        lod.Level.Should().Be(0);
    }

    [Fact]
    public void SelectLod_FarCamera_ReturnsCoarserLevel()
    {
        var chain = BuildChain();
        var near = chain.SelectLod(1f, MathF.PI / 3f, 1080);
        var far = chain.SelectLod(500f, MathF.PI / 3f, 1080);

        far.Level.Should().BeGreaterThan(near.Level);
    }

    [Fact]
    public void SelectLod_SelectedLevelSatisfiesPixelBudget_WhenPossible()
    {
        var chain = BuildChain();
        const float fov = MathF.PI / 3f;
        const int height = 1080;
        const float budget = 1.5f;

        foreach (float distance in new[] { 5f, 20f, 100f, 400f })
        {
            var lod = chain.SelectLod(distance, fov, height, budget);
            float error = NeuralPolygonLodChain.ProjectedScreenError(
                lod.GeometricError, distance, fov, height);

            float finestError = NeuralPolygonLodChain.ProjectedScreenError(
                chain.Levels[0].GeometricError, distance, fov, height);

            if (finestError <= budget)
            {
                // Le budget est atteignable : le niveau choisi doit le respecter.
                error.Should().BeLessThanOrEqualTo(budget);
            }
            else
            {
                // Budget intenable : le plus fin disponible est rendu.
                lod.Level.Should().Be(0);
            }
        }
    }

    [Fact]
    public void Build_EveryLevelHasMeshlets()
    {
        var chain = BuildChain();
        foreach (var level in chain.Levels)
            level.Meshlets.Should().NotBeEmpty();
    }
}
