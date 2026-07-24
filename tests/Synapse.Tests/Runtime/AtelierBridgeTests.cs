using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

/// <summary>
/// Covers the first-class atelier bridges surfaced on <see cref="EngineHost"/> and
/// wired into Synapse Studio: law marketplace, glTF export, digital twins,
/// behavior-tree inspection, and the viewport scale tool.
/// </summary>
public sealed class AtelierBridgeTests
{
    [Fact]
    public async Task ImportExportLawPackage_RoundTripsThroughEngineHost()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 1 }, logger);
        host.InitializeModules();

        var path = Path.Combine(Path.GetTempPath(), $"synapse-atelier-{Guid.NewGuid():N}.synapse-law");
        try
        {
            await host.ExportActiveLawPackageAsync(path);
            File.Exists(path).Should().BeTrue();

            var imported = await host.ImportLawPackageAsync(path, compileAndApply: false);
            imported.Id.Should().NotBeNullOrWhiteSpace();
            imported.Expression.Should().NotBeNullOrWhiteSpace();
            host.ListMarketplaceLaws().Should().Contain(p => p.Id == imported.Id);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task ExportSceneGlTFAsync_CreatesFile()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 1 }, logger);
        host.InitializeModules();

        var path = Path.Combine(Path.GetTempPath(), $"synapse-atelier-{Guid.NewGuid():N}.gltf");
        try
        {
            var result = await host.ExportSceneGlTFAsync(path);
            result.Success.Should().BeTrue(result.ErrorMessage ?? "unknown");
            File.Exists(path).Should().BeTrue();
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task RegisterTwin_ListsAndExportsSnapshot()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 1 }, logger);
        host.InitializeModules();

        var twin = host.RegisterTwin("sensor-alpha", new System.Collections.Generic.Dictionary<string, object>
        {
            ["Name"] = "Alpha",
            ["Temperature"] = 21.5
        });

        host.ListTwins().Should().Contain(t => t.Id == twin.Id);
        host.TwinStatusText.Should().Contain("enregistré");

        var path = Path.Combine(Path.GetTempPath(), $"synapse-twin-{Guid.NewGuid():N}.json");
        try
        {
            await host.ExportTwinSnapshotAsync(twin.Id, path);
            File.Exists(path).Should().BeTrue();
            File.ReadAllText(path).Should().Contain("sensor-alpha");
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void GetAgentBehaviorTreeText_ReturnsTreeAfterSpawnAgent()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 1 }, logger);
        host.InitializeModules();

        var agent = host.SpawnAgent("patrol", new Vector3(1, 0, 1));
        var text = host.GetAgentBehaviorTreeText(agent.EntityId);

        text.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void ApplyScaleDrag_ChangesEntityScale()
    {
        var editor = new ViewportEditorState
        {
            ToolMode = ViewportToolMode.Scale,
            ActiveGizmoAxis = GizmoAxis.None,
            DragStartScale = Vector3.One,
            DragStartMouseX = 100f,
            DragStartMouseY = 100f
        };
        var entity = new SceneEntityData { Scale = Vec3.From(Vector3.One) };

        ViewportInteraction.ApplyScaleDrag(editor, entity, mouseX: 200f, mouseY: 100f);

        entity.Scale.ToVector3().X.Should().BeGreaterThan(1f);
    }
}
