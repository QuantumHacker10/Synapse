using System;
using System.IO;
using FluentAssertions;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Xunit;

namespace Synapse.Tests.Plugins;

public sealed class PluginHostCoverageTests
{
    [Fact]
    public void PluginHost_LoadFromMissingDirectory_DoesNotCrashProcess()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var plugins = new PluginHost(logger);
        var missing = Path.Combine(Path.GetTempPath(), $"no-plugins-{Guid.NewGuid():N}");
        try
        {
            plugins.LoadFromDirectory(missing, host: null!);
        }
        catch (Exception ex)
        {
            ex.Should().Match(e =>
                e is DirectoryNotFoundException
                || e is ArgumentNullException
                || e is ArgumentException
                || e is InvalidOperationException);
        }
    }
}
