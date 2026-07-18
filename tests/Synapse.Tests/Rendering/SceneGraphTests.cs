using FluentAssertions;
using GDNN.Scene;
using Xunit;

namespace Synapse.Tests.Rendering;

public class SceneGraphTests
{
    [Fact]
    public void SceneNode_Create_ShouldHaveDefaultTransform()
    {
        var node = new SceneNode();

        node.Transform.Should().NotBeNull();
        node.Children.Should().BeEmpty();
    }

    [Fact]
    public void SceneNode_AddChild_ShouldIncreaseChildCount()
    {
        var parent = new SceneNode();
        var child = new SceneNode();

        parent.AddChild(child);

        parent.Children.Should().HaveCount(1);
    }

    [Fact]
    public void SceneNode_RemoveChild_ShouldDecreaseChildCount()
    {
        var parent = new SceneNode();
        var child = new SceneNode();
        parent.AddChild(child);

        parent.RemoveChild(child);

        parent.Children.Should().BeEmpty();
    }
}
