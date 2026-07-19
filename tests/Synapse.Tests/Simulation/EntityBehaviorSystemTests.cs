using System;
using System.Numerics;
using FluentAssertions;
using GDNN.Sentience;
using Xunit;

namespace Synapse.Tests.Simulation;

public class EntityBehaviorSystemTests
{
    [Fact]
    public void Entity_Create_ShouldHaveDefaultValues()
    {
        var entity = new SentientEntity(Guid.NewGuid(), EntityType.NPC);

        entity.EntityId.Should().NotBe(Guid.Empty);
        entity.Position.Should().Be(Vector3.Zero);
        entity.IsAlive.Should().BeTrue();
    }

    [Fact]
    public void PerceptionSystem_AddPerception_ShouldIncreaseCount()
    {
        var entity = new SentientEntity(Guid.NewGuid(), EntityType.NPC);
        var perception = new PerceptionEvent(
            PerceptionType.Visual,
            Guid.NewGuid(),
            0,
            0.8f,
            Vector3.Zero,
            Vector3.UnitZ,
            string.Empty,
            1.0f);

        entity.AddPerception(perception);

        entity.GetPendingPerceptions().Should().HaveCount(1);
    }

    [Fact]
    public void EmotionalModel_GetEmotion_ShouldReturnDefaultValue()
    {
        var model = new EmotionalModel();
        var entity = new SentientEntity(Guid.NewGuid(), EntityType.NPC);

        model.GetCurrentState(entity).Should().Be(EmotionalState.Neutral);
    }

    [Fact]
    public void EmotionalModel_SetEmotion_ShouldClampValue()
    {
        var model = new EmotionalModel();
        var entity = new SentientEntity(Guid.NewGuid(), EntityType.NPC);
        var perception = new PerceptionEvent(
            PerceptionType.Tactile,
            Guid.NewGuid(),
            0,
            2.0f,
            Vector3.Zero,
            Vector3.Zero,
            string.Empty,
            1.0f);

        var response = model.ReactToEvent(entity, perception);

        response.Intensity.Should().BeLessOrEqualTo(1.0f);
    }

    [Fact]
    public void MemorySystem_Store_ShouldIncreaseMemoryCount()
    {
        var memory = new MemorySystem();
        var entity = new SentientEntity(Guid.NewGuid(), EntityType.NPC);
        var entry = new MemoryEntry
        {
            Content = "Test memory",
            Importance = 0.5f
        };

        memory.Store(entity, entry);

        memory.Retrieve(entity, new MemoryQuery()).Should().HaveCount(1);
    }
}
