using System;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.Runtime;
using Synapse.VR;
using Synapse.Web;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class V2FeatureTests
{
    [Fact]
    public void SimulationReproducibility_SameSeedSameSequence()
    {
        SimulationReproducibility.SetSeed(1234);
        var a = SimulationReproducibility.Random.Next();
        SimulationReproducibility.SetSeed(1234);
        var b = SimulationReproducibility.Random.Next();
        a.Should().Be(b);
    }

    [Fact]
    public void SeedFromString_IsStable()
    {
        SimulationReproducibility.SeedFromString("demo-scene")
            .Should().Be(SimulationReproducibility.SeedFromString("demo-scene"));
    }

    [Fact]
    public async Task BenchmarkRunner_ProducesReport()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var runner = new BenchmarkRunner(logger);
        var suite = new BenchmarkSuiteConfig
        {
            Name = "unit",
            WarmupFrames = 2,
            MeasureFrames = 5,
            SimulationSeed = 7
        };

        var report = await runner.RunAsync(suite);
        report.PhysicsMsAvg.Should().BeGreaterThanOrEqualTo(0);
        report.EntityCount.Should().BeGreaterThan(0);
        report.SynapseVersion.Should().StartWith("2.");
    }

    [Fact]
    public void ApplyLaw_ActivatesLibraryEntry()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 1 }, logger);
        host.InitializeModules();
        host.ApplyLaw("heat_equation").Success.Should().BeTrue();
    }

    [Fact]
    public async Task SceneGlTFExporter_WritesGltf()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-export-{Guid.NewGuid():N}.gltf");
        try
        {
            var result = await SceneGlTFExporter.ExportAsync(SceneDocument.CreateDemo(), path);
            result.Success.Should().BeTrue();
            File.Exists(path).Should().BeTrue();
            File.ReadAllText(path).Should().Contain("SYNAPSE OMNIA");
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task LawMarketplace_ImportsPackage()
    {
        var market = new LawMarketplace();
        var path = ResolveSamplePath("samples/laws/custom_heat_wave.synapse-law");
        var package = await market.ImportAsync(path);
        package.Id.Should().NotBeNullOrWhiteSpace();
        market.FindById(package.Id).Should().NotBeNull();
    }

    [Fact]
    public async Task BlueprintRuntimeExecutor_RunsUntilExit()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        var executor = new BlueprintRuntimeExecutor(host, logger);
        executor.Load(BlueprintDocument.CreateDefault());

        for (int i = 0; i < 10 && executor.IsRunning; i++)
            await executor.TickAsync(0.016f);

        executor.IsRunning.Should().BeFalse();
    }

    [Fact]
    public async Task FbxAndUsdLoaders_ParseSampleMeshes()
    {
        var fbxPath = ResolveSamplePath("samples/meshes/tetra.fbx");
        var usdPath = ResolveSamplePath("samples/meshes/tetra.usda");

        var loader = new MeshLoader();
        (await loader.LoadAsync(fbxPath)).Success.Should().BeTrue();
        var usd = await loader.LoadAsync(usdPath);
        usd.Success.Should().BeTrue(usd.ErrorMessage ?? "unknown");
    }

    [Fact]
    public async Task NetworkVrWeb_FoundationTypesWork()
    {
        await using var session = SimulationPeerHub.CreateLocalSession();
        session.IsHost.Should().BeTrue();
        await session.BroadcastScenePatchAsync(new byte[] { 1, 2, 3 });

        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var vr = VrSessionFactory.Create(logger);
        vr.RuntimeName.Should().NotBeNullOrEmpty();
        (await vr.TryInitializeAsync()).Should().BeTrue();
        vr.IsSimulated.Should().BeTrue();
        vr.Swapchain.Should().NotBeNull();

        var preview = WebEditorBuilder.FromScene("Demo", "scene.glb", "heat_equation", 4);
        WebEditorBuilder.ToHtml(preview).Should().Contain("Synapse Web Editor");
    }

    [Fact]
    public async Task MultiPeerHub_AcceptsLocalConnection()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = SimulationPeerHub.CreateMultiPeerHub(logger);
        await host.StartHostAsync(port: 0);
        host.ListenPort.Should().BeGreaterThan(0);

        await using var client = SimulationPeerHub.CreateMultiPeerHub(logger);
        await client.ConnectAsync("127.0.0.1", host.ListenPort);
        await Task.Delay(50);
        host.PeerCount.Should().BeGreaterOrEqualTo(2);
    }

    [Fact]
    public async Task WebEditorBuilder_WritesSiteBundle()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-web-{Guid.NewGuid():N}");
        var bundle = WebEditorBuilder.FromScene("Demo", "scene.glb", "heat_equation", 4);
        await WebEditorBuilder.WriteSiteAsync(dir, bundle);
        File.Exists(Path.Combine(dir, "index.html")).Should().BeTrue();
        File.Exists(Path.Combine(dir, "editor.js")).Should().BeTrue();
        Directory.Delete(dir, true);
    }

    [Fact]
    public void MonolithSplits_ProduceExpectedFiles()
    {
        var simDir = ResolveRepoPath("src/Synapse.Simulation");
        Directory.GetFiles(simDir, "EntityBehaviorSystem.*.cs").Length.Should().BeGreaterThan(10);
        var physDir = ResolveRepoPath("src/Synapse.Physics");
        Directory.GetFiles(physDir, "LivingLawLibrary.*.cs").Length.Should().BeGreaterThan(5);
    }

    private static string ResolveRepoPath(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (Directory.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }
        return Path.GetFullPath(relative);
    }

    [Fact]
    public void LivingLawCompilerSplit_StillCompilesLaws()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        host.ListLaws().Should().NotBeEmpty();
        host.CompileLaw("heat_equation", "T").Success.Should().BeTrue();
    }

    private static string ResolveSamplePath(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        return Path.GetFullPath(relative);
    }
}
