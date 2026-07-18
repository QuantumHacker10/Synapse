using FluentAssertions;
using Synapse.Core;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime
{
    public sealed class BlueprintAndAdTests
    {
        [Fact]
        public void Blueprint_ValidatesAndCompiles()
        {
            var bp = BlueprintDocument.CreateDefault();
            var (ok, msg) = bp.Validate();
            ok.Should().BeTrue(msg);
            bp.CompileToBehaviorTreeName().Should().Be("patrol");
        }

        [Fact]
        public void SculptSession_AppliesDisplacement()
        {
            var sculpt = new SculptSession { BrushRadius = 1f, BrushStrength = 0.5f };
            sculpt.ApplyStroke(0, 0, 0);
            sculpt.SampleDisplacement(0, 0, 0).Should().BeApproximately(0.5f, 0.01f);
            sculpt.SampleDisplacement(2, 0, 0).Should().Be(0);
        }

        [Fact]
        public void DiffScalar2_TracksSecondDerivative()
        {
            var x = DiffScalar.Variable(2.0).WithSecondDerivative(0);
            var product = x * x;
            product.Value.Should().BeApproximately(4, 1e-9);
            product.Derivative.Should().BeApproximately(4, 1e-9);
            product.SecondDerivative.Should().BeApproximately(2, 1e-9);
        }
    }
}
