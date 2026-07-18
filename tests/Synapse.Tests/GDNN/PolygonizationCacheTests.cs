using System;
using System.IO;
using System.Numerics;
using FluentAssertions;
using GDNN.Core.DataStructures;
using GDNN.Polygonization;
using Xunit;

namespace Synapse.Tests.GDNN;

/// <summary>
/// Cache disque des chaînes de LOD polygonales : round-trip fidèle,
/// invalidation par les poids du réseau, et branchement dans le pipeline.
/// </summary>
public class PolygonizationCacheTests : IDisposable
{
    private static readonly AABB Bounds = new(Vector3.Zero, new Vector3(0.8f));
    private readonly string _cacheDir;

    public PolygonizationCacheTests()
    {
        _cacheDir = Path.Combine(Path.GetTempPath(), "gdnn-polycache-" + Guid.NewGuid().ToString("N"));
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_cacheDir))
                Directory.Delete(_cacheDir, recursive: true);
        }
        catch (IOException)
        {
            // Nettoyage best-effort du répertoire temporaire.
        }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void StoreThenLoad_RoundTripsChainExactly()
    {
        using var network = SphereNetworkFactory.CreateTrained();
        var chain = NeuralPolygonLodChain.Build(network, Bounds, baseResolution: 24, levelCount: 2);

        var cache = new PolygonizationCache(_cacheDir);
        string key = PolygonizationCache.ComputeKey(network, Bounds, 24, 2);

        cache.Store(key, chain);
        cache.TryLoad(key, out var loaded).Should().BeTrue();

        loaded!.Levels.Count.Should().Be(chain.Levels.Count);
        loaded.Bounds.Should().Be(chain.Bounds);

        for (int level = 0; level < chain.Levels.Count; level++)
        {
            var original = chain.Levels[level];
            var restored = loaded.Levels[level];

            restored.Level.Should().Be(original.Level);
            restored.GridResolution.Should().Be(original.GridResolution);
            restored.GeometricError.Should().Be(original.GeometricError);
            restored.Mesh.VertexCount.Should().Be(original.Mesh.VertexCount);
            restored.Mesh.Indices.Should().Equal(original.Mesh.Indices);
            restored.Meshlets.Count.Should().Be(original.Meshlets.Count);

            for (int v = 0; v < original.Mesh.VertexCount; v++)
            {
                restored.Mesh.Positions[v].Should().Be(original.Mesh.Positions[v]);
                restored.Mesh.Normals[v].Should().Be(original.Mesh.Normals[v]);
            }
        }
    }

    [Fact]
    public void TryLoad_UnknownKey_ReturnsFalse()
    {
        var cache = new PolygonizationCache(_cacheDir);
        cache.TryLoad("DEADBEEF", out var chain).Should().BeFalse();
        chain.Should().BeNull();
    }

    [Fact]
    public void ComputeKey_ChangesWhenWeightsChange()
    {
        using var network = SphereNetworkFactory.CreateTrained();

        string keyBefore = PolygonizationCache.ComputeKey(network, Bounds, 24, 2);
        network.OutputBias += 0.5f;
        string keyAfter = PolygonizationCache.ComputeKey(network, Bounds, 24, 2);

        keyAfter.Should().NotBe(keyBefore);
    }

    [Fact]
    public void ComputeKey_ChangesWithExtractionParameters()
    {
        using var network = SphereNetworkFactory.CreateTrained();

        string key24 = PolygonizationCache.ComputeKey(network, Bounds, 24, 2);
        string key32 = PolygonizationCache.ComputeKey(network, Bounds, 32, 2);

        key32.Should().NotBe(key24);
    }

    [Fact]
    public void Pipeline_SecondConstruction_LoadsChainFromCache()
    {
        using var network = SphereNetworkFactory.CreateTrained();
        var options = new NeuralGeometryPipelineOptions
        {
            BaseResolution = 24,
            LevelCount = 2,
            CacheDirectory = _cacheDir
        };

        var first = new NeuralGeometryPipeline(network, Bounds, options);
        first.InitialChainFromCache.Should().BeFalse("premier run : rien en cache");

        var second = new NeuralGeometryPipeline(network, Bounds, options);
        second.InitialChainFromCache.Should().BeTrue("le réseau n'a pas changé");

        second.Chain.Levels.Count.Should().Be(first.Chain.Levels.Count);
        for (int level = 0; level < first.Chain.Levels.Count; level++)
        {
            second.Chain.Levels[level].Mesh.VertexCount
                .Should().Be(first.Chain.Levels[level].Mesh.VertexCount);
        }
    }

    [Fact]
    public void Pipeline_WithoutCacheDirectory_NeverUsesCache()
    {
        using var network = SphereNetworkFactory.CreateTrained();
        var options = new NeuralGeometryPipelineOptions { BaseResolution = 24, LevelCount = 2 };

        var pipeline = new NeuralGeometryPipeline(network, Bounds, options);

        pipeline.InitialChainFromCache.Should().BeFalse();
        Directory.Exists(_cacheDir).Should().BeFalse("aucun fichier ne doit être écrit");
    }
}
