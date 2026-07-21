using System;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class SceneAndBlueprintHardeningTests
{
    [Fact]
    public async Task SceneDocument_RejectsCorruptJsonFallback()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-bad-scene-{Guid.NewGuid():N}.synapse");
        File.WriteAllText(path, "null");
        try
        {
            var act = async () => await SceneDocument.LoadAsync(path);
            await act.Should().ThrowAsync<InvalidDataException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SceneDocument_RejectsDuplicateEntityIds()
    {
        var doc = SceneDocument.CreateDemo();
        doc.Entities.Add(new SceneEntityData
        {
            Id = doc.Entities[0].Id,
            Name = "Dup",
            Type = "Empty",
            Scale = new Vec3(1, 1, 1)
        });
        var act = () => doc.Validate();
        act.Should().Throw<InvalidDataException>().WithMessage("*Duplicate*");
    }

    [Fact]
    public void SceneDocument_RejectsPathTraversalMesh()
    {
        var doc = SceneDocument.CreateDemo();
        doc.Entities[0].MeshPath = "../etc/passwd.glb";
        var root = Path.Combine(Path.GetTempPath(), $"synapse-scene-root-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var act = () => doc.Validate(root);
            act.Should().Throw<InvalidDataException>();
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public void Blueprint_RejectsSelfLoop()
    {
        var bp = BlueprintDocument.CreateDefault();
        var node = bp.Nodes[1];
        bp.Edges.Add(new BlueprintEdge
        {
            FromNodeId = node.Id,
            FromPin = 0,
            ToNodeId = node.Id,
            ToPin = 0
        });
        var (ok, msg) = bp.Validate();
        ok.Should().BeFalse();
        msg.Should().Contain("Self-loop");
    }

    [Fact]
    public async Task BlueprintExecutor_StopsWithoutOutgoingEdge()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new Synapse.Infrastructure.Configuration.SynapseConfig(), logger);
        host.InitializeModules();

        var entry = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Entry,
            Title = "Entry",
            Outputs = { new BlueprintPin { Name = "Exec", IsInput = false } }
        };
        var action = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Action,
            Title = "DeadEnd",
            Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } }
        };
        var exit = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Exit,
            Title = "Exit",
            Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } }
        };
        var doc = new BlueprintDocument
        {
            Name = "DeadEnd",
            Nodes = { entry, action, exit },
            Edges =
            {
                new BlueprintEdge { FromNodeId = entry.Id, FromPin = 0, ToNodeId = action.Id, ToPin = 0 }
                // no edge from action -> exit
            }
        };

        var executor = new BlueprintRuntimeExecutor(host, logger);
        executor.Load(doc);
        await executor.TickAsync(0.016f);
        executor.IsRunning.Should().BeFalse();
        executor.LastError.Should().Contain("no outgoing edge");
    }

    [Fact]
    public void UrlSecurity_BlocksPrivateDnsLiteral()
    {
        var act = () => Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(
            "https://192.168.0.5/secret", allowLoopbackHttp: false, resolveDns: false);
        act.Should().Throw<ArgumentException>();
    }
}
