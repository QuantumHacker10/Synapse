using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public class SphSolverTests
{
    private static float WendlandC2(float r, float h)
    {
        float q = r / h;
        if (q >= 1f)
        {
            return 0f;
        }

        float t = 1f - 0.5f * q;
        float h9 = h;
        for (int i = 0; i < 8; i++)
        {
            h9 *= h;
        }

        float kernelNorm = 315.0f / (64.0f * MathF.PI * h9);
        return kernelNorm * t * t * t * (2f * q + 1f);
    }

    [Fact]
    public void WendlandKernel_AtZeroDistance_ShouldReturnMaximum()
    {
        var result = WendlandC2(0.0f, 1.0f);

        result.Should().BeGreaterThan(0.0f);
    }

    [Fact]
    public void WendlandKernel_AtLargeDistance_ShouldApproachZero()
    {
        var result = WendlandC2(10.0f, 1.0f);

        result.Should().BeApproximately(0.0f, 1e-10f);
    }

    [Fact]
    public void SphSolver_ParticleCount_ShouldMatchInput()
    {
        const int particleCount = 100;
        var solver = new SphSolver(new SphConfig { NumParticles = particleCount });
        solver.InitializeCube(0f, 0f, 0f, spacing: 0.5f);

        solver.Particles.Count.Should().Be(particleCount);
    }
}
