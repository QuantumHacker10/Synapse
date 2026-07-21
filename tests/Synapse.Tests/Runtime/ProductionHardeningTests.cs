using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Memory;
using GDNN.Streaming;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class ProductionHardeningTests
{
    [Fact]
    public void ZeroCopyBuffer_CreateFromFile_MapsAndRoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-zcb-{Guid.NewGuid():N}.bin");
        try
        {
            using (var buffer = ZeroCopyBuffer.CreateFromFile(path, capacity: 64))
            {
                buffer.Capacity.Should().Be(64);
                buffer.GetSpan<byte>(0, 64)[0] = 0xAB;
                buffer.GetSpan<byte>(0, 64)[63] = 0xCD;
            }

            using var reopen = ZeroCopyBuffer.CreateFromFile(path, capacity: 64);
            reopen.GetSpan<byte>(0, 64)[0].Should().Be(0xAB);
            reopen.GetSpan<byte>(0, 64)[63].Should().Be(0xCD);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void MappedBuffer_KeepsPointerAliveAcrossOpen()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-mapped-{Guid.NewGuid():N}.bin");
        try
        {
            using var mapped = new MappedBuffer(path, 32);
            mapped.Open();
            mapped.IsOpen.Should().BeTrue();
            mapped.GetSpan<byte>(0, 32)[0] = 42;
            mapped.GetSpan<byte>(0, 32)[0].Should().Be(42);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task AssetStreamer_FailsClosedWhenAssetMissing()
    {
        var previousRoot = AssetStreamer.AssetRootDirectory;
        var previousSynth = AssetStreamer.AllowSyntheticPlaceholders;
        var root = Path.Combine(Path.GetTempPath(), $"synapse-assets-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        AssetStreamer? streamer = null;
        try
        {
            AssetStreamer.AssetRootDirectory = root;
            AssetStreamer.AllowSyntheticPlaceholders = false;
            streamer = new AssetStreamer(new StreamerConfig
            {
                MaxConcurrentDownloads = 1,
                MemoryBudgetBytes = 16 * 1024 * 1024
            });

            await streamer.RequestAssetAsync("missing-asset-id", AssetPriority.Immediate);

            AssetEntry? entry = null;
            for (int i = 0; i < 50; i++)
            {
                entry = streamer.GetAssetState("missing-asset-id");
                if (entry?.State is AssetLoadingState.Failed or AssetLoadingState.Loaded)
                    break;
                await Task.Delay(50);
            }

            entry.Should().NotBeNull();
            entry!.State.Should().Be(AssetLoadingState.Failed);
            entry.LastError.Should().NotBeNullOrWhiteSpace();
        }
        finally
        {
            streamer?.Dispose();
            AssetStreamer.AssetRootDirectory = previousRoot;
            AssetStreamer.AllowSyntheticPlaceholders = previousSynth;
            if (Directory.Exists(root))
                Directory.Delete(root, true);
        }
    }

    [Fact]
    public void LawMarketplace_RejectsOversizedExpression()
    {
        var package = new SynapseLawPackage
        {
            Id = "bad",
            Name = "Bad",
            Expression = new string('x', LawMarketplace.MaxExpressionLength + 1)
        };
        var act = () => LawMarketplace.ValidatePackage(package);
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void PluginHost_RequireManifest_RefusesBareDll()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var hostPlugins = new PluginHost(logger, PluginTrustMode.RequireManifest);
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-plugin-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var dll = Path.Combine(dir, "fake.dll");
        File.WriteAllBytes(dll, Encoding.UTF8.GetBytes("not-a-real-assembly"));
        try
        {
            var engine = new EngineHost(new SynapseConfig(), logger);
            hostPlugins.LoadPluginAssembly(dll, engine).Should().BeFalse();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public async Task GpuUploadStage_RejectsEmptyAsset()
    {
        var stage = StreamingPipelineStages.CreateGpuUploadStage();
        var asset = new NeuralAsset();
        var act = async () => await stage.ProcessAsync(asset, default);
        await act.Should().ThrowAsync<InvalidDataException>();
    }

    [Fact]
    public async Task GpuUploadStage_PreparesValidWeights()
    {
        var stage = StreamingPipelineStages.CreateGpuUploadStage();
        var mlp = new MicroMLP();
        var asset = new NeuralAsset
        {
            CompressedWeights = mlp.CompressWeights()
        };

        var prepared = await stage.ProcessAsync(asset, default);
        prepared.IsGpuUploadPrepared.Should().BeTrue();
        prepared.Metadata.ContentHash.Should().NotBeNullOrWhiteSpace();
    }
}
