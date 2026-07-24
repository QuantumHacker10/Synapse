using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Platform;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class V23FeatureTests
{
    [Fact]
    public void CpuCapabilityProbe_DoesNotRequireAvx512()
    {
        CpuCapabilityProbe.InvalidateCache();
        Environment.SetEnvironmentVariable("SYNAPSE_SIMD_MAX", "avx2");
        Environment.SetEnvironmentVariable("SYNAPSE_ALLOW_AVX512", null);
        try
        {
            CpuCapabilityProbe.InvalidateCache();
            var caps = CpuCapabilityProbe.Probe();
            caps.MeetsMinimumCpu.Should().BeTrue();
            ((byte)caps.EffectiveCeiling).Should().BeLessThanOrEqualTo((byte)SimdCeiling.Avx2);
            caps.BaselineLabel.Should().NotBeNullOrWhiteSpace();
            caps.Summary.Should().Contain("/");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_SIMD_MAX", null);
            CpuCapabilityProbe.InvalidateCache();
        }
    }

    [Fact]
    public void NativePlatform_SummaryIncludesSimdBaseline()
    {
        NativePlatform.InvalidateProbeCache();
        CpuCapabilityProbe.InvalidateCache();
        var caps = NativePlatform.Probe();
        caps.Cpu.Should().NotBeNull();
        caps.Summary.Should().Contain("SIMD=");
        caps.Summary.Should().Contain("GLFW");
    }

    [Fact]
    public async Task Usdc_RoundTripSynapseMeshPack()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tetra-{Guid.NewGuid():N}.usdc");
        try
        {
            var points = new List<Vector3>
            {
                new(0, 0, 0), new(1, 0, 0), new(0, 1, 0), new(0, 0, 1)
            };
            var indices = new List<uint> { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 };
            UsdBinaryLoader.WriteSynapseMeshPack(path, points, indices);

            File.Exists(path).Should().BeTrue();
            var header = new byte[8];
            await using (var fs = File.OpenRead(path))
                (await fs.ReadAsync(header)).Should().Be(8);
            UsdBinaryLoader.IsUsdc(header).Should().BeTrue();

            var loader = new MeshLoader();
            loader.CanLoad(path).Should().BeTrue();
            loader.SupportedFormats.Should().Contain(".usdc");
            var result = await loader.LoadAsync(path);
            result.Success.Should().BeTrue(result.ErrorMessage ?? "usdc");
            result.Asset!.Primitives.Should().NotBeEmpty();
            result.Asset.Primitives[0].Vertices.Count.Should().Be(4);
            result.Asset.Primitives[0].Indices.Count.Should().Be(12);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task Usdc_SampleFile_LoadsWhenPresent()
    {
        var sample = ResolveSamplePath("samples/meshes/tetra.usdc");
        if (!File.Exists(sample))
        {
            // Generate beside other samples if missing (CI checkout).
            sample = Path.Combine(Path.GetTempPath(), "tetra-sample.usdc");
            UsdBinaryLoader.WriteSynapseMeshPack(
                sample,
                new[] { new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1) },
                new uint[] { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 });
        }

        var result = await new UsdBinaryLoader().LoadAsync(sample);
        result.Success.Should().BeTrue(result.ErrorMessage ?? "sample");
    }

    [Fact]
    public void Blueprint_HotReload_UpdatesRegisteredTree()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 9 }, logger);
        host.InitializeModules();
        host.BlueprintLiveEdit = true;

        var doc = BlueprintDocument.CreateDefault();
        var agent = host.CompileAndSpawnBlueprint(doc, Vector3.Zero);
        agent.Should().NotBeNull();

        doc.Nodes.Add(new BlueprintNode
        {
            Kind = BlueprintNodeKind.Action,
            Title = "LiveAction",
            Payload = "wave",
            X = 300,
            Y = 160,
            Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } },
            Outputs = { new BlueprintPin { Name = "Then", IsInput = false } }
        });

        host.HotReloadBlueprint(doc).Should().BeTrue();
        host.Sentience!.GetBehaviorTree(doc.Name).Should().NotBeNull();
    }

    [Fact]
    public async Task Blueprint_SaveLoad_RoundTrip()
    {
        var path = Path.Combine(Path.GetTempPath(), $"bp-{Guid.NewGuid():N}.blueprint.json");
        try
        {
            var doc = BlueprintDocument.CreateDefault();
            doc.Name = "RoundTrip";
            await doc.SaveAsync(path);
            var loaded = await BlueprintDocument.LoadAsync(path);
            loaded.Name.Should().Be("RoundTrip");
            loaded.Nodes.Count.Should().Be(doc.Nodes.Count);
            loaded.Validate().Ok.Should().BeTrue();
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void BlueprintCompiler_MapsCommonNodeKinds()
    {
        var entry = new BlueprintNode { Kind = BlueprintNodeKind.Entry, Title = "E", Outputs = { new BlueprintPin() } };
        var seq = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Sequence,
            Title = "S",
            Inputs = { new BlueprintPin { IsInput = true } },
            Outputs = { new BlueprintPin() }
        };
        var action = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Action,
            Title = "A",
            Payload = "patrol",
            Inputs = { new BlueprintPin { IsInput = true } },
            Outputs = { new BlueprintPin() }
        };
        var exit = new BlueprintNode { Kind = BlueprintNodeKind.Exit, Title = "X", Inputs = { new BlueprintPin { IsInput = true } } };
        var doc = new BlueprintDocument
        {
            Name = "Kinds",
            Nodes = { entry, seq, action, exit },
            Edges =
            {
                new BlueprintEdge { FromNodeId = entry.Id, ToNodeId = seq.Id },
                new BlueprintEdge { FromNodeId = seq.Id, ToNodeId = action.Id },
                new BlueprintEdge { FromNodeId = action.Id, ToNodeId = exit.Id }
            }
        };
        var tree = BlueprintCompiler.Compile(doc);
        tree.Should().NotBeNull();
        tree.Name.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void PluginHost_MissingDirectory_ReturnsZero()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var plugins = new PluginHost(logger);
        var host = new EngineHost(new SynapseConfig(), logger);
        plugins.LoadFromDirectory(Path.Combine(Path.GetTempPath(), $"no-plugins-{Guid.NewGuid():N}"), host)
            .Should().Be(0);
        plugins.LoadedPlugins.Should().BeEmpty();
    }

    [Fact]
    public void RhiDeviceCreationInfo_DefaultsToVulkan12()
    {
        var info = new global::GDNN.RHI.Vulkan.RhiDeviceCreationInfo();
        info.ApiVersion.Should().Be(global::GDNN.RHI.Vulkan.RhiDeviceCreationInfo.VK_API_VERSION_1_2);
    }

    [Fact]
    public void LivingLawLibrary_SearchAndCategories()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        var laws = host.ListLaws();
        laws.Should().NotBeEmpty();
        laws.Should().Contain(l => l.Id.Contains("heat", StringComparison.OrdinalIgnoreCase));
        host.ApplyLaw("heat_equation").Success.Should().BeTrue();
    }

    private static string ResolveSamplePath(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.GetFullPath(Path.Combine(dir.FullName, relative));
            if (File.Exists(candidate) || Directory.Exists(Path.GetDirectoryName(candidate)!))
                return candidate;
            dir = dir.Parent;
        }
        return Path.GetFullPath(relative);
    }
}
