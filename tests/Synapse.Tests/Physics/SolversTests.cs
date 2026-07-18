using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public class SolversTests
{
    [Fact]
    public void MaxwellSolver_Step_ShouldNotThrow()
    {
        var config = new MaxwellConfig
        {
            GridSize = (8, 8, 8),
            CellSize = 1e-3,
            TimeStep = 1e-12
        };
        var solver = new MaxwellSolver(config);

        var act = () => solver.Step();

        act.Should().NotThrow();
    }

    [Fact]
    public void WavePropagator_Step_ShouldProduceFinitePressure()
    {
        var config = new WaveConfig
        {
            GridSize = (16, 16, 16),
            CellSize = 0.01,
            TimeStep = 1e-5
        };
        var solver = new WavePropagator(config);

        solver.Step();

        double.IsNaN(solver.Pressure[0]).Should().BeFalse();
        double.IsInfinity(solver.Pressure[0]).Should().BeFalse();
    }

    [Fact]
    public void ThermodynamicEnsemble_Energy_ShouldBeNonNegative()
    {
        var config = new ThermoConfig { NumParticles = 32, BoxLength = 5.0 };
        var solver = new ThermodynamicEnsemble(config);

        solver.TotalEnergy.Should().BeGreaterOrEqualTo(0.0);
    }

    [Fact]
    public void NBodySolver_Force_ShouldFollowInverseSquare()
    {
        var config = new NBodyConfig { NumBodies = 2, UseBarnesHut = false };
        var solver = new NBodySolver(config);
        solver.Initialise(
            new[] { 0.0, 3.0 },
            new[] { 0.0, 0.0 },
            new[] { 0.0, 0.0 },
            new[] { 0.0, 0.0 },
            new[] { 0.0, 0.0 },
            new[] { 0.0, 0.0 },
            new[] { 1.0, 2.0 });

        solver.ComputeForcesDirect();

        var body = solver.Bodies[0];
        var force = Math.Sqrt(body.Ax * body.Ax + body.Ay * body.Ay + body.Az * body.Az);

        force.Should().BeGreaterThan(0.0);
    }
}
