using FluentAssertions;
using Synapse.Core;
using Xunit;

namespace Synapse.Tests.Core;

public class PhysicsStateTests
{
    [Fact]
    public void Vector3D_Add_ShouldReturnCorrectSum()
    {
        var a = new Vector3D(1.0, 2.0, 3.0);
        var b = new Vector3D(4.0, 5.0, 6.0);

        var result = a + b;

        result.X.Should().Be(5.0);
        result.Y.Should().Be(7.0);
        result.Z.Should().Be(9.0);
    }

    [Fact]
    public void Vector3D_Magnitude_ShouldReturnCorrectLength()
    {
        var v = new Vector3D(3.0, 4.0, 0.0);

        var magnitude = v.Length();

        magnitude.Should().Be(5.0);
    }

    [Fact]
    public void QuaternionD_Multiply_ShouldComposeRotations()
    {
        var q1 = QuaternionD.Identity;
        var q2 = QuaternionD.Identity;

        var result = q1 * q2;

        result.Should().Be(QuaternionD.Identity);
    }
}
