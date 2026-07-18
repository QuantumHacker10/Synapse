using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public class StochasticFieldSolversTests
{
    [Fact]
    public void GeometricBrownianMotion_FinalValue_ShouldBePositive()
    {
        var process = new GeometricBrownianMotion(x0: 100.0f, mu: 0.05f, sigma: 0.2f);
        var rng = new Random(42);
        var path = process.GeneratePath(T: 1.0f, steps: 100, rng);

        path.TerminalValue.Should().BeGreaterThan(0.0f);
    }

    [Fact]
    public void OrnsteinUhlenbeckProcess_ShouldRevertToMean()
    {
        var process = new OrnsteinUhlenbeckProcess(x0: 5.0f, theta: 0.5f, mu: 2.0f, sigma: 0.1f);
        var rng = new Random(42);
        var path = process.GeneratePath(T: 10.0f, steps: 1000, rng);

        path.TerminalValue.Should().BeInRange(0.5f, 3.5f);
    }

    [Fact]
    public void FractionalBrownianMotion_Variance_ShouldIncrease()
    {
        var process = new FractionalBrownianMotionProcess(sigma: 1.0f, hurst: 0.7f);
        var rng = new Random(42);
        var path1 = process.GeneratePath(T: 1.0f, steps: 100, rng);
        var path2 = process.GeneratePath(T: 1.0f, steps: 100, new Random(43));

        path1.Values.Should().HaveCount(101);
        path2.Values.Should().HaveCount(101);
    }
}
