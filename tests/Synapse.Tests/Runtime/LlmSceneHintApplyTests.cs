using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

/// <summary>
/// Studio/runtime wiring: LLM replies with lighting/SDF JSON update the scene document.
/// </summary>
public class LlmSceneHintApplyTests
{
    [Fact]
    public void ApplyLlmSceneHints_LightingJson_CreatesLightEntity()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);

        const string reply = """
            {
              "directionalDirection": [0.2, -1.0, 0.1],
              "color": "#FFCC88",
              "intensity": 2.5,
              "fogDensity": 0.03,
              "enableClouds": true
            }
            """;

        string status = host.ApplyLlmSceneHints(reply);

        status.Should().StartWith("Applied:");
        status.Should().Contain("lighting");
        host.Scene.Entities.Should().Contain(e => e.Type == "Light" && e.Name.Contains("LLM"));
    }

    [Fact]
    public void ApplyLlmSceneHints_SdfJson_CreatesVolumeEntityAtCenter()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);

        const string reply = """
            {
              "primitive": "sphere",
              "center": [1.5, 0.5, -2.0],
              "radius": 0.75
            }
            """;

        string status = host.ApplyLlmSceneHints(reply);

        status.Should().Contain("sdf:sphere");
        var entity = host.Scene.Entities.Should().ContainSingle(e => e.Name.Contains("LLM_SDF")).Subject;
        entity.Type.Should().Be("Volume");
        entity.Position.X.Should().BeApproximately(1.5f, 1e-4f);
        entity.Position.Y.Should().BeApproximately(0.5f, 1e-4f);
        entity.Position.Z.Should().BeApproximately(-2.0f, 1e-4f);
        entity.Scale.X.Should().BeApproximately(0.75f, 1e-4f);
    }

    [Fact]
    public void ApplyLlmSceneHints_UnrelatedText_ReportsNothingFound()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);

        string status = host.ApplyLlmSceneHints("Hello, how are you?");

        status.Should().Contain("No lighting/SDF");
    }
}
