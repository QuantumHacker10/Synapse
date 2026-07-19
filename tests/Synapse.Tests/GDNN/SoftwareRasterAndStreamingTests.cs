using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Core.DataStructures;
using GDNN.GPU;
using GDNN.Polygonization;
using GDNN.Streaming;
using Xunit;

namespace Synapse.Tests.GDNN;

// ---------------------------------------------------------------------------
// Rasterisation software
// ---------------------------------------------------------------------------

public class SoftwareRasterizerTests
{
    private const int Size = 64;

    /// <summary>Caméra identité : les positions du maillage sont déjà en NDC.</summary>
    private static CameraView NdcCamera() => new()
    {
        Position = new Vector3(0, 0, 1f),
        ViewProjection = Matrix4x4.Identity,
        VerticalFovRadians = MathF.PI / 3f,
        ScreenHeightPixels = Size
    };

    private static (NeuralPolygonMesh Mesh, NeuralMeshlet Meshlet) SingleTriangle(
        Vector3 a, Vector3 b, Vector3 c)
    {
        var mesh = new NeuralPolygonMesh
        {
            Positions = [a, b, c],
            Normals = [Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ],
            Indices = [0, 1, 2]
        };
        var meshlet = new NeuralMeshlet
        {
            VertexIndices = [0, 1, 2],
            LocalIndices = [0, 1, 2],
            Bounds = mesh.ComputeBounds(),
            ConeAxis = Vector3.UnitZ,
            ConeCutoff = -1f
        };
        return (mesh, meshlet);
    }

    [Fact]
    public void Rasterize_FrontFacingTriangle_CoversInteriorNotExterior()
    {
        // CCW en NDC (y vers le haut) = face avant.
        var (mesh, meshlet) = SingleTriangle(
            new Vector3(-0.5f, -0.5f, 0.5f),
            new Vector3(0.5f, -0.5f, 0.5f),
            new Vector3(-0.5f, 0.5f, 0.5f));

        var target = new RasterTarget(Size, Size);
        var stats = new SoftwareRasterizer().Rasterize(target, mesh, [meshlet], NdcCamera());

        stats.TrianglesRasterized.Should().Be(1);
        stats.TrianglesBackfaceCulled.Should().Be(0);

        // Barycentre du triangle en écran : NDC (-0.17, -0.17) → pixel ~(26, 37).
        target.IsCovered(26, 37).Should().BeTrue();
        target.TryDecode(26, 37, out int meshletIdx, out int triIdx).Should().BeTrue();
        meshletIdx.Should().Be(0);
        triIdx.Should().Be(0);

        // Coin opposé de l'écran : vide.
        target.IsCovered(60, 4).Should().BeFalse();
        target.CountCoveredPixels().Should().BeGreaterThan(50);
    }

    [Fact]
    public void Rasterize_BackFacingTriangle_IsCulled()
    {
        // Winding inversé (CW en NDC) = dos.
        var (mesh, meshlet) = SingleTriangle(
            new Vector3(-0.5f, -0.5f, 0.5f),
            new Vector3(-0.5f, 0.5f, 0.5f),
            new Vector3(0.5f, -0.5f, 0.5f));

        var target = new RasterTarget(Size, Size);
        var stats = new SoftwareRasterizer().Rasterize(target, mesh, [meshlet], NdcCamera());

        stats.TrianglesBackfaceCulled.Should().Be(1);
        stats.TrianglesRasterized.Should().Be(0);
        target.CountCoveredPixels().Should().Be(0);
    }

    [Fact]
    public void Rasterize_DepthTest_NearerTriangleWins()
    {
        // Deux triangles superposés : z NDC 0.8 (loin) puis 0.3 (proche).
        var (meshFar, meshletFar) = SingleTriangle(
            new Vector3(-0.6f, -0.6f, 0.8f),
            new Vector3(0.6f, -0.6f, 0.8f),
            new Vector3(-0.6f, 0.6f, 0.8f));
        var (meshNear, _) = SingleTriangle(
            new Vector3(-0.6f, -0.6f, 0.3f),
            new Vector3(0.6f, -0.6f, 0.3f),
            new Vector3(-0.6f, 0.6f, 0.3f));

        // Un seul maillage combiné, deux meshlets.
        var mesh = new NeuralPolygonMesh
        {
            Positions = [.. meshFar.Positions, .. meshNear.Positions],
            Normals = [.. meshFar.Normals, .. meshNear.Normals],
            Indices = [0, 1, 2, 3, 4, 5]
        };
        NeuralMeshlet MakeMeshlet(int offset) => new()
        {
            VertexIndices = [offset, offset + 1, offset + 2],
            LocalIndices = [0, 1, 2],
            Bounds = mesh.ComputeBounds(),
            ConeAxis = Vector3.UnitZ,
            ConeCutoff = -1f
        };

        var target = new RasterTarget(Size, Size);
        new SoftwareRasterizer().Rasterize(
            target, mesh, [MakeMeshlet(0), MakeMeshlet(3)], NdcCamera());

        // Le centre du recouvrement doit appartenir au meshlet 1 (le plus proche).
        target.TryDecode(26, 37, out int meshletIdx, out _).Should().BeTrue();
        meshletIdx.Should().Be(1);
        target.ClosenessAt(26, 37).Should().BeApproximately(0.7f, 0.01f);
    }

    [Fact]
    public void Rasterize_SharedEdge_NoGapNoDoubleCoverage()
    {
        // Quad découpé en deux triangles le long de la diagonale : la règle
        // top-left en virgule fixe doit donner une partition exacte.
        Vector3 p00 = new(-0.5f, -0.5f, 0.5f), p10 = new(0.5f, -0.5f, 0.5f);
        Vector3 p01 = new(-0.5f, 0.5f, 0.5f), p11 = new(0.5f, 0.5f, 0.5f);

        var (meshA, meshletA) = SingleTriangle(p00, p10, p11); // CCW
        var (meshB, meshletB) = SingleTriangle(p00, p11, p01); // CCW

        var targetA = new RasterTarget(Size, Size);
        var targetB = new RasterTarget(Size, Size);
        var rasterizer = new SoftwareRasterizer();
        rasterizer.Rasterize(targetA, meshA, [meshletA], NdcCamera());
        rasterizer.Rasterize(targetB, meshB, [meshletB], NdcCamera());

        int both = 0, union = 0;
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                bool a = targetA.IsCovered(x, y);
                bool b = targetB.IsCovered(x, y);
                if (a && b)
                    both++;
                if (a || b)
                    union++;
            }

        both.Should().Be(0, "un pixel d'arête partagée appartient à exactement un triangle");

        // Tous les pixels strictement intérieurs au quad sont couverts (pas de trou).
        // Quad en écran : x ∈ [16,48], y ∈ [16,48].
        for (int y = 17; y < 47; y++)
            for (int x = 17; x < 47; x++)
                (targetA.IsCovered(x, y) || targetB.IsCovered(x, y)).Should()
                    .BeTrue($"pixel ({x},{y}) intérieur au quad");

        union.Should().BeGreaterThan(30 * 30);
    }

    [Fact]
    public void RenderFrame_SphereChain_FillsVisibilityBuffer()
    {
        var sdf = new AnalyticSphereSdf(0.5f);
        var bounds = new AABB(Vector3.Zero, new Vector3(0.8f));
        var chain = NeuralPolygonLodChain.Build(sdf, bounds, baseResolution: 32, levelCount: 2);
        var renderer = new NeuralClusterRenderer(chain);
        var camera = CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), Vector3.Zero, MathF.PI / 3f,
            aspect: 1f, nearPlane: 0.01f, farPlane: 100f, screenHeightPixels: 256);

        var visible = renderer.SelectVisible(camera, out var cullStats);
        var target = new RasterTarget(256, 256);
        var stats = new SoftwareRasterizer().Rasterize(
            target, chain.Levels[cullStats.SelectedLevel].Mesh, visible, camera);

        stats.TrianglesRasterized.Should().BeGreaterThan(100);
        int covered = target.CountCoveredPixels();
        covered.Should().BeGreaterThan(2000, "la sphère doit remplir une part visible de l'écran");

        // Chaque pixel couvert doit décoder vers un cluster/triangle valides.
        for (int y = 0; y < 256; y += 8)
            for (int x = 0; x < 256; x += 8)
            {
                if (!target.TryDecode(x, y, out int m, out int t))
                    continue;
                m.Should().BeInRange(0, visible.Count - 1);
                t.Should().BeInRange(0, visible[m].TriangleCount - 1);
            }
    }
}

public class MeshletRasterizerShaderTests
{
    [Fact]
    public void GeneratedGlsl_ContainsNaniteStyleConstructs()
    {
        string glsl = MeshletRasterizerShaderGenerator.GenerateGlsl();

        glsl.Should().Contain("#version 450");
        glsl.Should().Contain("GL_EXT_shader_atomic_int64");
        glsl.Should().Contain("local_size_x = 128");
        glsl.Should().Contain("shared vec4 s_clip[64]");
        glsl.Should().Contain("atomicMax(visibility[");
        glsl.Should().Contain("isTopLeftEdge");
        glsl.Should().Contain("barrier();");
    }

    [Fact]
    public void GeneratedGlsl_CompilesToSpirv_WhenToolchainAvailable()
    {
        if (!SpirvToolchain.IsGlslangAvailable)
            return; // pas de toolchain sur cet agent CI : la structure est testée ci-dessus

        string glsl = MeshletRasterizerShaderGenerator.GenerateGlsl();
        bool ok = SpirvToolchain.TryCompileGlsl(glsl, out byte[] spirv, out string log);

        ok.Should().BeTrue(log);
        spirv.Length.Should().BeGreaterThan(0);
    }
}

// ---------------------------------------------------------------------------
// Streaming hors-cœur
// ---------------------------------------------------------------------------

public class MeshletStreamingTests : IDisposable
{
    private readonly string _tempDir;

    public MeshletStreamingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "gdnn-streaming-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        { Directory.Delete(_tempDir, recursive: true); }
        catch (IOException) { }
    }

    private static NeuralPolygonLodChain BuildSphereChain(int baseResolution = 32, int levels = 3)
    {
        var sdf = new AnalyticSphereSdf(0.5f);
        var bounds = new AABB(Vector3.Zero, new Vector3(0.8f));
        return NeuralPolygonLodChain.Build(sdf, bounds, baseResolution, levels);
    }

    [Fact]
    public void Codec_RoundTrip_IsWithinQuantizationBounds()
    {
        var chain = BuildSphereChain(24, 1);
        var mesh = chain.Levels[0].Mesh;
        var meshlet = chain.Levels[0].Meshlets[0];

        byte[] encoded = MeshletCodec.Encode(mesh, meshlet);
        var decoded = MeshletCodec.Decode(encoded);

        decoded.VertexCount.Should().Be(meshlet.VertexCount);
        decoded.TriangleCount.Should().Be(meshlet.TriangleCount);
        decoded.LocalIndices.Should().Equal(meshlet.LocalIndices);
        decoded.ConeCutoff.Should().BeApproximately(meshlet.ConeCutoff, 1e-6f);

        // Borne d'erreur garantie : taille du meshlet / 65535 par axe.
        Vector3 maxError = meshlet.Bounds.Size / 65535f + new Vector3(1e-6f);
        for (int v = 0; v < meshlet.VertexCount; v++)
        {
            Vector3 diff = Vector3.Abs(
                decoded.Positions[v] - mesh.Positions[meshlet.VertexIndices[v]]);
            diff.X.Should().BeLessThanOrEqualTo(maxError.X);
            diff.Y.Should().BeLessThanOrEqualTo(maxError.Y);
            diff.Z.Should().BeLessThanOrEqualTo(maxError.Z);

            Vector3.Dot(decoded.Normals[v], mesh.Normals[meshlet.VertexIndices[v]])
                .Should().BeGreaterThan(0.98f);
        }
    }

    [Fact]
    public void Codec_CompressesBelowRawSize()
    {
        var chain = BuildSphereChain(24, 1);
        var mesh = chain.Levels[0].Mesh;

        long encodedTotal = 0, rawTotal = 0;
        foreach (var meshlet in chain.Levels[0].Meshlets)
        {
            encodedTotal += MeshletCodec.Encode(mesh, meshlet).Length;
            rawTotal += meshlet.VertexCount * 24 + meshlet.LocalIndices.Length; // fp32 pos+normales
        }

        encodedTotal.Should().BeLessThan(rawTotal / 2,
            "quantification 16 bits + oct-encoding + Deflate doivent au moins diviser la taille par 2");
    }

    [Fact]
    public void PageFile_BuildAndOpen_ExposesFullDirectory()
    {
        var chain = BuildSphereChain();
        string path = Path.Combine(_tempDir, "sphere.gdnnpage");

        MeshletPageFile.Build(chain, path);
        File.Exists(path).Should().BeTrue();

        using var file = MeshletPageFile.Open(path);
        file.LevelCount.Should().Be(chain.Levels.Count);
        int expected = chain.Levels.Sum(l => l.Meshlets.Count);
        file.Directory.Should().HaveCount(expected);

        // Le répertoire porte les bornes : nécessaires au culling sans chargement.
        foreach (var entry in file.Directory)
            entry.Bounds.HalfExtents.Length().Should().BeGreaterThan(0f);
    }

    [Fact]
    public void Streamer_GetOrLoad_MatchesSourceGeometry()
    {
        var chain = BuildSphereChain();
        string path = Path.Combine(_tempDir, "sphere.gdnnpage");
        MeshletPageFile.Build(chain, path);

        using var file = MeshletPageFile.Open(path);
        using var streamer = new MeshletStreamer(file, memoryBudgetBytes: 64 * 1024 * 1024);

        var key = new ClusterKey(0, 5);
        var cluster = streamer.GetOrLoad(key);
        var source = chain.Levels[0].Meshlets[5];

        cluster.VertexCount.Should().Be(source.VertexCount);
        cluster.TriangleCount.Should().Be(source.TriangleCount);
        for (int v = 0; v < cluster.VertexCount; v++)
        {
            Vector3 expected = chain.Levels[0].Mesh.Positions[source.VertexIndices[v]];
            (cluster.Positions[v] - expected).Length().Should().BeLessThan(1e-3f);
        }

        streamer.Stats.CacheMisses.Should().Be(1);
        streamer.GetOrLoad(key);
        streamer.Stats.CacheHits.Should().Be(1);
    }

    [Fact]
    public void Streamer_EnforcesMemoryBudget_WithLruEviction()
    {
        var chain = BuildSphereChain();
        string path = Path.Combine(_tempDir, "sphere.gdnnpage");
        MeshletPageFile.Build(chain, path);

        using var file = MeshletPageFile.Open(path);
        // Budget minuscule : quelques clusters seulement.
        using var streamer = new MeshletStreamer(file, memoryBudgetBytes: 16 * 1024);

        foreach (var entry in file.Directory.Where(e => e.Key.Level == 0))
            streamer.GetOrLoad(entry.Key);

        streamer.Stats.ResidentBytes.Should().BeLessThanOrEqualTo(16 * 1024);
        streamer.Stats.Evictions.Should().BeGreaterThan(0);
        streamer.Stats.ResidentClusters.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Streamer_QueryVisible_UsesDirectoryOnly_AndSortsByDistance()
    {
        var chain = BuildSphereChain();
        string path = Path.Combine(_tempDir, "sphere.gdnnpage");
        MeshletPageFile.Build(chain, path);

        using var file = MeshletPageFile.Open(path);
        using var streamer = new MeshletStreamer(file, memoryBudgetBytes: 64 * 1024 * 1024);

        var facing = CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), Vector3.Zero, MathF.PI / 3f, 1f, 0.01f, 100f, 1080);
        var visible = streamer.QueryVisible(facing, level: 0);

        visible.Should().NotBeEmpty();
        streamer.Stats.CacheMisses.Should().Be(0, "la requête ne doit rien charger");

        // Tri par distance : le premier cluster est plus proche que le dernier.
        AABB BoundsOf(ClusterKey k) => file.Directory.First(e => e.Key == k).Bounds;
        float first = (BoundsOf(visible[0]).Center - facing.Position).Length();
        float last = (BoundsOf(visible[^1]).Center - facing.Position).Length();
        first.Should().BeLessThanOrEqualTo(last);

        var away = CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), new Vector3(0, 0, 4f), MathF.PI / 3f, 1f, 0.01f, 100f, 1080);
        streamer.QueryVisible(away, level: 0).Should().BeEmpty();
    }

    [Fact]
    public async Task Streamer_PrefetchAsync_MakesVisibleClustersResident()
    {
        var chain = BuildSphereChain();
        string path = Path.Combine(_tempDir, "sphere.gdnnpage");
        MeshletPageFile.Build(chain, path);

        using var file = MeshletPageFile.Open(path);
        using var streamer = new MeshletStreamer(file, memoryBudgetBytes: 64 * 1024 * 1024);

        var camera = CameraView.CreatePerspectiveLookAt(
            new Vector3(0, 0, 2f), Vector3.Zero, MathF.PI / 3f, 1f, 0.01f, 100f, 1080);
        var visible = streamer.QueryVisible(camera, level: 0);

        await streamer.PrefetchAsync(visible);

        foreach (var key in visible)
            streamer.TryGetResident(key, out _).Should().BeTrue($"cluster {key} doit être résident");
    }
}
