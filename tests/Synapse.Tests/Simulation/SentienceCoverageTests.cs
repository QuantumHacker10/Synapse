using System.Numerics;
using FluentAssertions;
using GDNN.Sentience;
using Xunit;

namespace Synapse.Tests.Simulation;

public sealed class SentienceCoverageTests
{
    [Fact]
    public void SentienceManager_CreateAndRemoveEntity()
    {
        var manager = new SentienceManager();
        var entity = manager.CreateEntity(EntityType.Sentient, Vector3.UnitX, "patrol");
        manager.EntityCount.Should().BeGreaterThanOrEqualTo(1);
        manager.RemoveEntity(entity.EntityId);
        manager.EntityCount.Should().Be(0);
    }

    [Fact]
    public void BehaviorCompiler_CompileFromLlmPrompt_ProducesTree()
    {
        var compiler = new BehaviorCompiler();
        var tree = compiler.CompileFromLLMPrompt("coverage-tree", "always idle");
        tree.Should().NotBeNull();
        tree.Name.Should().NotBeNullOrWhiteSpace();
    }
}
