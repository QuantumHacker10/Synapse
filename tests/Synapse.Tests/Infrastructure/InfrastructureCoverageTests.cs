using System;
using FluentAssertions;
using GDNN.Diagnostics;
using GDNN.Rendering.Quality;
using Synapse.Infrastructure;
using Synapse.Infrastructure.Logging;
using Xunit;

namespace Synapse.Tests.Infrastructure;

public sealed class InfrastructureCoverageTests
{
    [Fact]
    public void SynapseProduct_ExposesNameAndVersion()
    {
        SynapseProduct.Name.Should().NotBeNullOrWhiteSpace();
        SynapseProduct.Version.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void RuntimeQualityManager_StartsAtConfiguredPreset()
    {
        using var quality = new RuntimeQualityManager(QualityPreset.High, AdaptationMode.Dynamic);
        quality.CurrentLevel.Preset.Should().Be(QualityPreset.High);
    }

    [Fact]
    public void SynapseLogger_LogsWithoutThrowing()
    {
        ISynapseLogger logger = new SynapseLogger(null, LogLevel.Debug, consoleEnabled: false);
        logger.Debug("Test", "debug line");
        logger.Info("Test", "info line");
        logger.Warn("Test", "warn line");
        logger.Error("Test", "error line");
        (logger as IDisposable)?.Dispose();
    }

    [Fact]
    public void FrameTimer_MeasuresElapsed()
    {
        var timer = new FrameTimer();
        timer.BeginFrame();
        timer.EndFrame();
        timer.CpuFrameTimeMs.Should().BeGreaterThanOrEqualTo(0);
        timer.FrameCount.Should().Be(1);
    }
}
