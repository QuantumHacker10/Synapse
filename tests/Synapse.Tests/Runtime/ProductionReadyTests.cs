using System;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class ProductionReadyTests
{
    [Fact]
    public void HealthReport_CoreReady_AfterInitialize()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        var report = host.GetProductionHealth();
        report.IsCoreReady.Should().BeTrue();
        report.ProductVersion.Should().StartWith("2.10");
        report.UsdRuntimeReady.Should().BeTrue(report.UsdRuntimeDetail);
        report.ExperimentalNotes.Should().NotBeEmpty();
        report.ExperimentalNotes.Should().OnlyContain(n =>
            n.Contains("OpenXR", StringComparison.OrdinalIgnoreCase) ||
            n.Contains("WAN", StringComparison.OrdinalIgnoreCase) ||
            n.Contains("OpenUSD", StringComparison.OrdinalIgnoreCase) ||
            n.Contains("USD", StringComparison.OrdinalIgnoreCase));
        report.ToString().Should().Contain("Synapse");
        report.ToString().Should().Contain("usd=ok");
    }

    [Fact]
    public async Task DisposeAsync_IsIdempotent()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        await host.DisposeAsync();
        await host.DisposeAsync(); // must not throw
    }

    [Fact]
    public void PluginMarketplace_VerifiesCatalogHash()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var dir = Path.Combine(Path.GetTempPath(), $"mkt-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var dll = Path.Combine(dir, "dummy.dll");
            File.WriteAllBytes(dll, new byte[] { 1, 2, 3, 4, 5 });
            var hash = PluginMarketplace.ComputeFileSha256(dll);
            PluginMarketplace.WriteCatalogTemplate(dir, new[]
            {
                new PluginCatalogEntry
                {
                    Id = "dummy",
                    Name = "Dummy",
                    FileName = "dummy.dll",
                    Sha256 = hash
                }
            });

            var market = PluginMarketplace.FromDirectory(dir, logger);
            market.Entries.Should().ContainSingle(e => e.Id == "dummy");
            market.VerifyInstalledOrWarn(); // no throw
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void UsdAscii_AppliesTranslateXform()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"usd-xf-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var path = Path.Combine(dir, "moved.usda");
            File.WriteAllText(path, """
                #usda 1.0
                def Mesh "M"
                {
                    double3 xformOp:translate = (10, 0, 0)
                    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                    int[] faceVertexIndices = [0, 1, 2, -1]
                }
                """);
            var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
            result.Success.Should().BeTrue(result.ErrorMessage);
            result.Asset!.Primitives[0].Vertices[0].Position.X.Should().BeApproximately(10f, 0.001f);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ParseTranslate_ReadsTuple()
    {
        UsdAsciiLoader.ParseTranslate("double3 xformOp:translate = (1.5, 2, -3)")
            .Should().Be(new Vector3(1.5f, 2f, -3f));
    }
}
