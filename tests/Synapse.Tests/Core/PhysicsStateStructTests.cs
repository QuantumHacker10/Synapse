using System.Runtime.InteropServices;
using FluentAssertions;
using Synapse.Core;
using Xunit;

namespace Synapse.Tests.Core;

/// <summary>
/// Tests for the PhysicsState struct (256-byte thermodynamic state).
/// Complements PhysicsStateTests.cs which covers Vector3D/QuaternionD.
/// </summary>
public class PhysicsStateStructTests
{
    private const double Tolerance = 1e-10;
    private const double R = 8.314462618; // J/(mol·K)

    [Fact]
    public void PhysicsState_ShouldHave256ByteSize()
    {
        Marshal.SizeOf<PhysicsState>().Should().Be(256);
    }

    [Fact]
    public void PhysicsState_DefaultFields_ShouldBeZero()
    {
        var state = default(PhysicsState);

        state.Density.Should().Be(0);
        state.Pressure.Should().Be(0);
        state.Temperature.Should().Be(0);
        state.Velocity.Should().Be(Vector3D.Zero);
        state.Position.Should().Be(Vector3D.Zero);
    }

    [Fact]
    public void PIdealGas_ShouldMatchIdealGasLaw()
    {
        var state = default(PhysicsState);
        double n = 1.0;   // mol
        double T = 300.0; // K
        double V = 0.024; // m³

        double P = state.PIdealGas(n, T, V, R);

        P.Should().BeApproximately(n * R * T / V, Tolerance);
    }

    [Fact]
    public void CpFromCv_ShouldAddGasConstant()
    {
        var state = default(PhysicsState);
        double cv = 20.8; // J/(mol·K) for diatomic

        state.CpFromCv(cv, R).Should().BeApproximately(cv + R, Tolerance);
    }

    [Fact]
    public void GammaFromCv_ShouldComputeHeatCapacityRatio()
    {
        var state = default(PhysicsState);
        double cv = 20.8;

        state.GammaFromCv(cv, R).Should().BeApproximately((cv + R) / cv, Tolerance);
    }

    [Fact]
    public void ComputeStrainTensor_ShouldSymmetrizeVelocityGradient()
    {
        var state = default(PhysicsState);
        var L = new Tensor3D(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9);

        var strain = state.ComputeStrainTensor(L);

        strain.M12.Should().BeApproximately(strain.M21, Tolerance);
        strain.M13.Should().BeApproximately(strain.M31, Tolerance);
        strain.M23.Should().BeApproximately(strain.M32, Tolerance);
    }

    [Fact]
    public void ComputeRotationTensor_ShouldAntisymmetrize()
    {
        var state = default(PhysicsState);
        var L = new Tensor3D(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9);

        var rotation = state.ComputeRotationTensor(L);

        rotation.M12.Should().BeApproximately(-rotation.M21, Tolerance);
        rotation.M13.Should().BeApproximately(-rotation.M31, Tolerance);
        rotation.M23.Should().BeApproximately(-rotation.M32, Tolerance);
    }

    [Fact]
    public void ComputeVorticity_ShouldReturnCurlOfVelocityGradient()
    {
        var state = default(PhysicsState);
        var L = new Tensor3D(
            0, -3, 2,
            3, 0, -1,
            -2, 1, 0);

        var w = state.ComputeVorticity(L);

        w.X.Should().BeApproximately(2.0, Tolerance);
        w.Y.Should().BeApproximately(4.0, Tolerance);
        w.Z.Should().BeApproximately(6.0, Tolerance);
    }

    [Fact]
    public void RungeKutta4_ShouldIntegrateSimpleODE()
    {
        var state = default(PhysicsState);
        // dy/dt = y → solution y(t) = y0 * e^t
        Vector3D F(double t, Vector3D y) => y;
        var y0 = new Vector3D(1.0, 0.0, 0.0);
        double h = 0.01;
        double t = 0.0;

        var y1 = state.RungeKutta4(F, t, y0, h);

        y1.X.Should().BeApproximately(Math.Exp(h), 1e-6);
    }

    [Fact]
    public void FicksLaw_ShouldReturnNegativeDiffusionFlux()
    {
        var state = default(PhysicsState);
        double D = 1e-9;
        var gradC = new Vector3D(1.0, 0.0, 0.0);

        var flux = state.FicksLaw(D, gradC);

        flux.X.Should().BeApproximately(-D, Tolerance);
        flux.Y.Should().Be(0);
        flux.Z.Should().Be(0);
    }

    [Fact]
    public void FouriersLaw_ShouldReturnNegativeHeatFlux()
    {
        var state = default(PhysicsState);
        double lambda = 0.5;
        var gradT = new Vector3D(0.0, 100.0, 0.0);

        var flux = state.FouriersLaw(lambda, gradT);

        flux.Y.Should().BeApproximately(-50.0, Tolerance);
    }

    [Fact]
    public void CompressibilityFactor_IdealGas_ShouldBeUnity()
    {
        var state = default(PhysicsState);
        double n = 1.0, T = 300.0, V = 0.024465;
        double P = state.PIdealGas(n, T, V, R);

        state.CompressibilityFactor(P, V, n, T, R).Should().BeApproximately(1.0, 1e-6);
    }

    [Fact]
    public void InternalEnergyIdealGas_ShouldScaleWithTemperature()
    {
        var state = default(PhysicsState);
        double cv = 20.8;
        double T = 300.0;

        state.InternalEnergyIdealGas(T, cv).Should().BeApproximately(cv * T, Tolerance);
    }
}
