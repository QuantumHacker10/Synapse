using System;
using System.IO;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Xunit;

namespace Synapse.Tests.Infrastructure;

public sealed class SynapseConfigCliTests
{
    [Theory]
    [InlineData("--vr")]
    [InlineData("--enable-vr")]
    public void ApplyCli_VrFlag_SetsEnableVr(string flag)
    {
        var config = SynapseConfig.Load(path: NonexistentPath(), args: [flag]);
        config.EnableVr.Should().BeTrue();
    }

    [Fact]
    public void ApplyCli_WanRdvAndExportWeb_AreParsed()
    {
        var export = Path.Combine(Path.GetTempPath(), $"synapse-web-{Guid.NewGuid():N}");
        var config = SynapseConfig.Load(path: NonexistentPath(), args:
        [
            "--wan-session", "room-alpha",
            "--wan-port", "8123",
            "--wan-host",
            "--wan-rdv", "9123",
            "--export-web", export
        ]);

        config.WanSessionCode.Should().Be("room-alpha");
        config.WanPort.Should().Be(8123);
        config.WanHost.Should().BeTrue();
        config.WanRendezvousPort.Should().Be(9123);
        config.ExportWebPath.Should().Be(export);
    }

    [Fact]
    public void ApplyCli_WanSessionAlias_IsAccepted()
    {
        var config = SynapseConfig.Load(path: NonexistentPath(), args: ["--wan-code", "legacy-code"]);
        config.WanSessionCode.Should().Be("legacy-code");
    }

    [Fact]
    public void ApplyEnvironment_SynapseVrTrue_EnablesVr()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR");
        var previousWan = Environment.GetEnvironmentVariable("SYNAPSE_WAN_SESSION");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR", "1");
            Environment.SetEnvironmentVariable("SYNAPSE_WAN_SESSION", "env-room");
            var config = SynapseConfig.Load(path: NonexistentPath(), args: Array.Empty<string>());
            config.EnableVr.Should().BeTrue();
            config.WanSessionCode.Should().Be("env-room");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR", previous);
            Environment.SetEnvironmentVariable("SYNAPSE_WAN_SESSION", previousWan);
        }
    }

    [Fact]
    public void Load_FromTempJson_ThenCliOverrides()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-cfg-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, """
            {
              "enableVr": false,
              "wanSessionCode": "from-json",
              "wanPort": 7000,
              "qualityPreset": "Medium"
            }
            """);
        try
        {
            var config = SynapseConfig.Load(path: path, args: ["--vr", "--wan-port", "7779", "--quality", "Ultra"]);
            config.EnableVr.Should().BeTrue();
            config.WanSessionCode.Should().Be("from-json");
            config.WanPort.Should().Be(7779);
            config.QualityPreset.Should().Be("Ultra");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ApplyCli_HeadlessAndSeed()
    {
        var config = SynapseConfig.Load(path: NonexistentPath(), args: ["--headless", "--seed", "99"]);
        config.Headless.Should().BeTrue();
        config.SimulationSeed.Should().Be(99);
    }

    private static string NonexistentPath() =>
        Path.Combine(Path.GetTempPath(), $"missing-synapse-config-{Guid.NewGuid():N}.json");
}
