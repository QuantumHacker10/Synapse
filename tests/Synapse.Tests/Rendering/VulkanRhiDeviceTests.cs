using FluentAssertions;
using GDNN.RHI.Vulkan;
using Xunit;

namespace Synapse.Tests.Rendering;

public sealed class VulkanRhiDeviceTests
{
    [Fact]
    public void VulkanResult_Success_IsZero()
    {
        ((int)VulkanResult.Success).Should().Be(0);
    }

    [Fact]
    public void CreateIdentityMatrix4x4_ReturnsIdentity()
    {
        var m = VulkanExtensions.CreateIdentityMatrix4x4();

        m.Should().HaveCount(16);
        m[0].Should().Be(1f);
        m[5].Should().Be(1f);
        m[10].Should().Be(1f);
        m[15].Should().Be(1f);
        m[1].Should().Be(0f);
        m[12].Should().Be(0f);
    }

    [Fact]
    public void CreatePerspectiveMatrix_ProducesFiniteValues()
    {
        var m = VulkanExtensions.CreatePerspectiveMatrix(
            fovY: 1.0f,
            aspectRatio: 16f / 9f,
            nearPlane: 0.1f,
            farPlane: 100f);

        m.Should().HaveCount(16);
        m.Should().OnlyContain(v => float.IsFinite(v));
        m[11].Should().Be(-1f);
    }

    [Fact]
    public void CreateLookAtMatrix_ProducesFiniteValues()
    {
        var m = VulkanExtensions.CreateLookAtMatrix(
            eyeX: 0, eyeY: 0, eyeZ: 5,
            centerX: 0, centerY: 0, centerZ: 0,
            upX: 0, upY: 1, upZ: 0);

        m.Should().HaveCount(16);
        m.Should().OnlyContain(v => float.IsFinite(v));
    }

    [Fact]
    public void BufferDescription_DefaultsAreValid()
    {
        var desc = new BufferDescription
        {
            Size = 256,
            Usage = BufferUsageFlag.VertexBuffer,
            MemoryProperties = MemoryPropertyFlag.DeviceLocal
        };

        desc.Size.Should().Be(256);
        desc.Usage.Should().Be(BufferUsageFlag.VertexBuffer);
    }

    [Fact]
    public void PipelineStageFlag_ToDisplayString_IsReadable()
    {
        PipelineStageFlag.TopOfPipe.ToDisplayString().Should().Be("Top of Pipe");
        PipelineStageFlag.AllCommands.ToDisplayString().Should().Be("All Commands");
    }
}
