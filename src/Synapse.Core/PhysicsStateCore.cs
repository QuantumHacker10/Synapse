// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

// SECTION 12: PHYSICSSTATE EXTENSIONS
[StructLayout(LayoutKind.Explicit, Size = 256)]
public partial struct PhysicsState
{
    // ── CORE FIELDS (referenced by FieldSampleResult and PhysicsConstraint) ──
    [FieldOffset(0)] public double Density;
    [FieldOffset(8)] public double Pressure;
    [FieldOffset(16)] public double Temperature;
    [FieldOffset(24)] public double Entropy;
    [FieldOffset(32)] public double InternalEnergy;
    [FieldOffset(40)] public double _kineticEnergy;
    [FieldOffset(48)] public double Norm;
    [FieldOffset(56)] public Vector3D Velocity;
    [FieldOffset(80)] public Vector3D HeatFlux;
    [FieldOffset(104)] public Vector3D Position;
    [FieldOffset(128)] public Tensor3D VelocityGradient;

    public readonly double KineticEnergy => _kineticEnergy;

    // ── 12.1 TENSOR OPERATIONS ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeStrainTensor(Tensor3D L) => (L + L.Transpose()) * 0.5;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeRotationTensor(Tensor3D L) => (L - L.Transpose()) * 0.5;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeGreenLagrangeStrain(Tensor3D F) => (F.Transpose() * F - Tensor3D.Identity) * 0.5;
    public Tensor3D ComputeAlmansiStrain(Tensor3D F) => (Tensor3D.Identity - (F * F.Transpose()).Inverse()) * 0.5;
    public Tensor3D ComputeCauchyStress(double p, double mu, double lam, Tensor3D e)
        => -Tensor3D.Identity * p + e * (2.0 * mu) + Tensor3D.Identity * (lam * e.Trace);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeFirstPiolaKirchhoff(Tensor3D F, Tensor3D S) => F * S;
    public Tensor3D ComputeSecondPiolaKirchhoff(Tensor3D F, Tensor3D sigma)
        => F.Inverse() * sigma * F.Inverse().Transpose() * F.Determinant;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D ComputeVorticity(Tensor3D L) => new Vector3D(L.M32 - L.M23, L.M13 - L.M31, L.M21 - L.M12);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeEnstrophy(Vector3D w) => 0.5 * w.LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeStrainInvariant2(Tensor3D e) { double t = e.Trace; return 0.5 * (e.FrobeniusNormSquared - t * t); }
    public double ComputeOkuboWeiss(Tensor3D L) => ComputeStrainTensor(L).FrobeniusNormSquared - ComputeRotationTensor(L).FrobeniusNormSquared;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeLeftCauchyGreen(Tensor3D F) => F * F.Transpose();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeRightCauchyGreen(Tensor3D F) => F.Transpose() * F;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeVolumeRatio(Tensor3D F) => F.Determinant;
    public Tensor3D ComputeMandelStress(Tensor3D F, Tensor3D S) => ComputeRightCauchyGreen(F).Inverse() * S;

    // ── 12.2 THERMODYNAMIC STATES ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InternalEnergyIdealGas(double T, double cv) => cv * T;
    public double HelmholtzFreeEnergy(double T, double V, double n, double cv, double R, double T0, double V0)
        => cv * T - T * n * (cv * Math.Log(T / T0) + R * Math.Log(V / V0));
    public double GibbsFreeEnergy(double T, double P, double n, double cp, double R, double S0, double P0)
        => cp * T - T * (S0 + n * R * Math.Log(P / P0));
    public double SackurTetrodeEntropy(double n, double V, double T, double m, double kB, double h)
        => n * 8.314462618 * (Math.Log(V * Math.Pow(2.0 * Math.PI * m * kB * T / (h * h), 1.5)) + 2.5);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CpFromCv(double cv, double R) => cv + R;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GammaFromCv(double cv, double R) => (cv + R) / cv;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EnthalpyIdealGas(double T, double n, double cv, double R) => n * (cv + R) * T;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InternalEnergyVdW(double T, double n, double V, double cv, double a) => cv * n * T - a * n * n / V;
    public double EntropyVdW(double n, double T, double V, double cv, double R, double b) => n * cv * Math.Log(T) + n * R * Math.Log(V - n * b);
    public double EntropyOfMixing(double[] x, double R) { double s = 0; for (int i = 0; i < x.Length; i++) if (x[i] > 1e-15) s -= x[i] * Math.Log(x[i]); return R * s; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ChemicalPotential(double T, double Pi, double mu0, double R, double P0) => mu0 + R * T * Math.Log(Pi / P0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EquilibriumConstant(double dG0, double T, double R) => Math.Exp(-dG0 / (R * T));
    public double VanHoffEquation(double K1, double T1, double T2, double dH0, double R) => K1 * Math.Exp(-dH0 / R * (1.0 / T2 - 1.0 / T1));

    // ── 12.3 EQUATIONS OF STATE ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PIdealGas(double n, double T, double V, double R) => n * R * T / V;
    public double PVdW(double n, double T, double V, double R, double a, double b) => n * R * T / (V - b * n) - a * n * n / (V * V);
    public double PVdW_Iterative(double n, double T, double P, double R, double a, double b)
    {
        double v = n * R * T / P;
        for (int i = 0; i < 50; i++)
        {
            double f = P - n * R * T / (v - b * n) + a * n * n / (v * v);
            double df = -n * R * T / ((v - b * n) * (v - b * n)) + 2.0 * a * n * n / (v * v * v);
            double dv = f / df;
            v -= dv;
            if (Math.Abs(dv) < 1e-14 * Math.Abs(v))
                break;
            if (v < b * n * 1.01)
                v = b * n * 1.01;
        }
        return v;
    }
    public double PRK(double n, double T, double V, double R, double a, double b) => n * R * T / (V - b * n) - a * n * n / (Math.Sqrt(T) * V * (V + b * n));
    public double PPengRobinson(double n, double T, double V, double R, double a, double b, double omega)
    {
        double kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
        double Tc = a / (0.45724 * R * R);
        double Tr = T / Tc;
        double alpha = (1.0 + kappa * (1.0 - Math.Sqrt(Tr)));
        alpha *= alpha;
        return n * R * T / (V - b * n) - a * alpha * n * n / (V * V + 2.0 * b * V - b * b);
    }
    public double PBWR(double n, double T, double V, double R, double A0, double B0, double C0, double a, double b, double c, double alpha, double gamma)
    {
        double nRT = n * R * T, n2 = n * n, n3 = n2 * n, n6 = n3 * n3;
        double V2 = V * V, V6 = V2 * V2 * V2, T2 = T * T;
        return nRT / V + (B0 * nRT - A0 - C0 / T2) * n2 / V2 + (b * nRT - a) * n3 / V6 + a * alpha * n6 / V6 + c * n3 * Math.Exp(-gamma * n2 / V2) / (V * V2);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CompressibilityFactor(double P, double V, double n, double T, double R) => P * V / (n * R * T);
    public double FugacityCoeffVdW(double Z, double A, double B) => Z <= B ? 1e-10 : Math.Exp(Z - 1.0 - Math.Log(Z - B) - A / Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Fugacity(double phi, double P) => phi * P;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SpeedOfSound(double gamma, double R, double T, double M) => Math.Sqrt(gamma * R * T / M);
    public double JouleThomson(double T, double V, double dVdT, double Cp) => (T * dVdT - V) / Cp;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InversionTempVdW(double a, double b, double R) => 2.0 * a / (R * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SecondVirial(double T, double a, double b, double R) => b - a / (R * T);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoyleTemp(double a, double b, double R) => a / (R * b);

    // ── 12.4 TRANSPORT PROPERTIES ──
    public double ViscosityChapmanEnskog(double T, double m, double sigma, double Omega)
    {
        double kB = 1.380649e-23;
        return (5.0 / 16.0) * Math.Sqrt(Math.PI * m * kB * T) / (Math.PI * sigma * sigma * Omega);
    }
    public double ViscositySutherland(double T, double eta0, double T0, double S)
        => eta0 * Math.Pow(T / T0, 1.5) * (T0 + S) / (T + S);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ThermalConductivityEucken(double eta, double cp, double R, double M) => eta * (cp + 1.25 * R) / M;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PrandtlNumber(double cp, double eta, double lambda) => cp * eta / lambda;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SchmidtNumber(double eta, double rho, double D) => eta / (rho * D);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double LewisNumber(double alpha, double D) => alpha / D;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double KnudsenNumber(double mfp, double L_char) => mfp / L_char;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double MeanFreePath(double T, double P, double sigma) => 1.380649e-23 * T / (Math.Sqrt(2.0) * Math.PI * sigma * sigma * P);
    public double DiffusionCoefficient(double T, double P, double sigma, double m)
    {
        double kB = 1.380649e-23;
        double n = P / (kB * T);
        return 0.375 * Math.Sqrt(kB * T / (Math.PI * m)) / (sigma * sigma * n);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D FicksLaw(double D, Vector3D gradC) => gradC * (-D);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D FouriersLaw(double lambda, Vector3D gradT) => gradT * (-lambda);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double NewtonViscosity(double eta, double dudy) => eta * dudy;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PowerLawViscosity(double K, double shearRate, double n) => Math.Abs(shearRate) < 1e-15 ? double.MaxValue : K * Math.Pow(Math.Abs(shearRate), n - 1.0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ArrheniusRate(double A, double Ea, double T, double R) => A * Math.Exp(-Ea / (R * T));
    public double WLFViscosity(double eta0, double T, double Tg, double c1, double c2) => eta0 * Math.Pow(10.0, -c1 * (T - Tg) / (c2 + T - Tg));

    // ── 12.5 EVOLUTION / TIME INTEGRATION ──
    public Vector3D RungeKutta4(Func<double, Vector3D, Vector3D> f, double t, Vector3D y, double h)
    {
        var k1 = f(t, y);
        var k2 = f(t + h * 0.5, y + k1 * (h * 0.5));
        var k3 = f(t + h * 0.5, y + k2 * (h * 0.5));
        var k4 = f(t + h, y + k3 * h);
        return y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
    }
    public void VelocityVerlet(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        var a = acc(pos);
        pos = pos + vel * h + a * (0.5 * h * h);
        var an = acc(pos);
        vel = vel + (a + an) * (0.5 * h);
    }
    public void Leapfrog(ref Vector3D pos, ref Vector3D vHalf, Func<Vector3D, Vector3D> acc, double h)
    {
        pos = pos + vHalf * h;
        vHalf = vHalf + acc(pos) * h;
    }
    public void SymplecticEuler(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        vel = vel + acc(pos) * h;
        pos = pos + vel * h;
    }
    public Vector3D StormerVerlet(Vector3D pos, Vector3D posPrev, Func<Vector3D, Vector3D> acc, double h)
        => pos * 2.0 - posPrev + acc(pos) * (h * h);
    public void YoshidaSuzuki4(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        double w1 = 1.0 / (2.0 - Math.Pow(2.0, 1.0 / 3.0));
        double w0 = 1.0 - 2.0 * w1;
        VelocityVerlet(ref pos, ref vel, acc, h * w1);
        VelocityVerlet(ref pos, ref vel, acc, h * w0);
        VelocityVerlet(ref pos, ref vel, acc, h * w1);
    }
    public (Vector3D pos, Vector3D vel, double H) HMCStep(Vector3D pos, Vector3D mom, Func<Vector3D, double> PE, Func<Vector3D, Vector3D> force, double h, int steps)
    {
        double H0 = 0.5 * mom.LengthSquared() + PE(pos);
        for (int i = 0; i < steps; i++)
        { mom = mom + force(pos) * (h * 0.5); pos = pos + mom * h; mom = mom + force(pos) * (h * 0.5); }
        return (pos, mom, 0.5 * mom.LengthSquared() + PE(pos));
    }
    public (Vector3D pos, Vector3D vel) LangevinBAOAB(Vector3D pos, Vector3D vel, Func<Vector3D, Vector3D> force, double m, double gamma, double T, double kB, double h, Random rng)
    {
        double hh = h * 0.5;
        vel = vel + force(pos) * (hh / m);
        double c1 = Math.Exp(-gamma * hh);
        vel = vel * c1;
        double noise = Math.Sqrt((1.0 - c1 * c1) * kB * T / m);
        vel = vel + new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng)) * noise;
        vel = vel * c1;
        pos = pos + vel * hh;
        vel = vel + force(pos) * (hh / m);
        return (pos, vel);
    }
    public Vector3D BrownianStep(Vector3D pos, Func<Vector3D, Vector3D> force, double gamma, double T, double kB, double h, Random rng)
        => pos + force(pos) * (h / gamma) + new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng)) * Math.Sqrt(2.0 * kB * T * h / gamma);
    private static double NormalSample(Random rng) { double u1 = rng.NextDouble(), u2 = rng.NextDouble(); return Math.Sqrt(-2.0 * Math.Log(u1 + 1e-300)) * Math.Cos(2.0 * Math.PI * u2); }
    public (Vector3D state, double newH) AdaptiveRKF45(Func<double, Vector3D, Vector3D> f, double t, Vector3D y, double h, double tol, double hMin, double hMax)
    {
        var k1 = f(t, y);
        var k2 = f(t + h * 0.25, y + k1 * (h * 0.25));
        var k3 = f(t + h * 0.375, y + k1 * (h * 3.0 / 32) + k2 * (h * 9.0 / 32));
        var k4 = f(t + h * 12.0 / 13, y + k1 * (h * 1932.0 / 2197) - k2 * (h * 7200.0 / 2197) + k3 * (h * 7296.0 / 2197));
        var k5 = f(t + h, y + k1 * (h * 439.0 / 216) - k2 * (h * 8.0) + k3 * (h * 3680.0 / 513) - k4 * (h * 845.0 / 4104));
        var k6 = f(t + h * 0.5, y - k1 * (h * 8.0 / 27) + k2 * (h * 2.0) - k3 * (h * 3544.0 / 2565) + k4 * (h * 1859.0 / 4104) - k5 * (h * 11.0 / 40));
        var y4 = y + k1 * (h * 25.0 / 216) + k3 * (h * 1408.0 / 2565) + k4 * (h * 2197.0 / 4104) + k5 * (-h / 5.0);
        var y5 = y + k1 * (h * 16.0 / 135) + k3 * (h * 6656.0 / 12825) + k4 * (h * 28561.0 / 56430) + k5 * (-h * 9.0 / 50) + k6 * (h * 2.0 / 55);
        double err = Math.Max((y5 - y4).Length(), 1e-15);
        double newH = Math.Max(hMin, Math.Min(hMax, h * Math.Pow(tol / err, 0.2)));
        return err <= tol ? (y5, newH) : AdaptiveRKF45(f, t, y, newH, tol, hMin, hMax);
    }
    public Vector3D EulerMaruyama(Func<Vector3D, Vector3D> drift, Func<Vector3D, Vector3D> diff, Vector3D y, double h, Random rng)
    {
        var Z = new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng));
        return y + drift(y) * h + new Vector3D(diff(y).X * Z.X, diff(y).Y * Z.Y, diff(y).Z * Z.Z) * Math.Sqrt(h);
    }
    public double TotalEnergy(double m, Vector3D v, Func<Vector3D, double> PE, Vector3D r) => 0.5 * m * v.LengthSquared() + PE(r);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Momentum(double m, Vector3D v) => v * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double KineticEnergyCalc(double m, Vector3D v) => 0.5 * m * v.LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D AngularMomentum(double m, Vector3D r, Vector3D v) => Vector3D.Cross(r, v) * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Torque(Vector3D r, Vector3D F) => Vector3D.Cross(r, F);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PowerCalc(Vector3D F, Vector3D v) => Vector3D.Dot(F, v);
    public double WorkAlongPath(Func<Vector3D, Vector3D> field, Vector3D[] path)
    {
        double w = 0;
        for (int i = 0; i < path.Length - 1; i++)
        { var mid = (path[i] + path[i + 1]) * 0.5; w += Vector3D.Dot(field(mid), path[i + 1] - path[i]); }
        return w;
    }

    // ── 12.6 PHASE & EQUILIBRIUM ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ClausiusClapeyron(double L, double T, double dV) => L / (T * dV);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double AntoineEquation(double T, double A, double B, double C) => Math.Pow(10.0, A - B / (C + T));
    public double VaporPressureCC(double P1, double T1, double T2, double dHvap, double R) => P1 * Math.Exp(-dHvap / R * (1.0 / T2 - 1.0 / T1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TroutonsRule(double Tb, double Kt) => Kt * Tb;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RaoultLaw(double x, double P0) => x * P0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double HenrysLaw(double KH, double C) => KH * C;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoilingPointElevation(double i, double Kb, double m) => i * Kb * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double OsmoticPressure(double i, double M, double R, double T) => i * M * R * T;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GibbsPhaseRule(int C, int P) => C - P + 2;
    public double RegularSolutionGibbs(double x1, double T, double R, double omega)
    { double x2 = 1.0 - x1; return R * T * (x1 * Math.Log(x1 + 1e-30) + x2 * Math.Log(x2 + 1e-30)) + omega * x1 * x2; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsInsideSpinodal(double x1, double T, double R, double omega) => R * T / (x1 * (1.0 - x1) + 1e-30) < 2.0 * omega;
    public double FloryHuggins(double phi, int N, double chi) => (phi / N) * Math.Log(phi + 1e-30) + (1.0 - phi) * Math.Log(1.0 - phi + 1e-30) + chi * phi * (1.0 - phi);

    // ── 12.7 FIELD OPERATORS ──
    public double Divergence(Vector3D[,,] F, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = F.GetLength(0), ny = F.GetLength(1), nz = F.GetLength(2);
        double dFx = (ix + 1 < nx ? F[ix + 1, iy, iz].X : F[ix, iy, iz].X) - (ix > 0 ? F[ix - 1, iy, iz].X : F[ix, iy, iz].X);
        double dFy = (iy + 1 < ny ? F[ix, iy + 1, iz].Y : F[ix, iy, iz].Y) - (iy > 0 ? F[ix, iy - 1, iz].Y : F[ix, iy, iz].Y);
        double dFz = (iz + 1 < nz ? F[ix, iy, iz + 1].Z : F[ix, iy, iz].Z) - (iz > 0 ? F[ix, iy, iz - 1].Z : F[ix, iy, iz].Z);
        return dFx / (2.0 * dx) + dFy / (2.0 * dy) + dFz / (2.0 * dz);
    }
    public Vector3D Curl(Vector3D[,,] F, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = F.GetLength(0), ny = F.GetLength(1), nz = F.GetLength(2);
        double dFzdy = ((iy + 1 < ny ? F[ix, iy + 1, iz].Z : F[ix, iy, iz].Z) - (iy > 0 ? F[ix, iy - 1, iz].Z : F[ix, iy, iz].Z)) / (2.0 * dy);
        double dFydz = ((iz + 1 < nz ? F[ix, iy, iz + 1].Y : F[ix, iy, iz].Y) - (iz > 0 ? F[ix, iy, iz - 1].Y : F[ix, iy, iz].Y)) / (2.0 * dz);
        double dFxdz = ((iz + 1 < nz ? F[ix, iy, iz + 1].X : F[ix, iy, iz].X) - (iz > 0 ? F[ix, iy, iz - 1].X : F[ix, iy, iz].X)) / (2.0 * dz);
        double dFzdx = ((ix + 1 < nx ? F[ix + 1, iy, iz].Z : F[ix, iy, iz].Z) - (ix > 0 ? F[ix - 1, iy, iz].Z : F[ix, iy, iz].Z)) / (2.0 * dx);
        double dFydx = ((ix + 1 < nx ? F[ix + 1, iy, iz].Y : F[ix, iy, iz].Y) - (ix > 0 ? F[ix - 1, iy, iz].Y : F[ix, iy, iz].Y)) / (2.0 * dx);
        double dFxdy = ((iy + 1 < ny ? F[ix, iy + 1, iz].X : F[ix, iy, iz].X) - (iy > 0 ? F[ix, iy - 1, iz].X : F[ix, iy, iz].X)) / (2.0 * dy);
        return new Vector3D(dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy);
    }
    public Vector3D Gradient(double[,,] f, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double dFx = ((ix + 1 < nx ? f[ix + 1, iy, iz] : f[ix, iy, iz]) - (ix > 0 ? f[ix - 1, iy, iz] : f[ix, iy, iz])) / (2.0 * dx);
        double dFy = ((iy + 1 < ny ? f[ix, iy + 1, iz] : f[ix, iy, iz]) - (iy > 0 ? f[ix, iy - 1, iz] : f[ix, iy, iz])) / (2.0 * dy);
        double dFz = ((iz + 1 < nz ? f[ix, iy, iz + 1] : f[ix, iy, iz]) - (iz > 0 ? f[ix, iy, iz - 1] : f[ix, iy, iz])) / (2.0 * dz);
        return new Vector3D(dFx, dFy, dFz);
    }
    public double Laplacian(double[,,] f, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double c = f[ix, iy, iz];
        return ((ix + 1 < nx ? f[ix + 1, iy, iz] : c) - 2.0 * c + (ix > 0 ? f[ix - 1, iy, iz] : c)) / (dx * dx)
            + ((iy + 1 < ny ? f[ix, iy + 1, iz] : c) - 2.0 * c + (iy > 0 ? f[ix, iy - 1, iz] : c)) / (dy * dy)
            + ((iz + 1 < nz ? f[ix, iy, iz + 1] : c) - 2.0 * c + (iz > 0 ? f[ix, iy, iz - 1] : c)) / (dz * dz);
    }
    public double AdvectionUpwind(double[,,] f, Vector3D vel, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double c = f[ix, iy, iz];
        double dphidx = vel.X > 0 ? (ix > 0 ? c - f[ix - 1, iy, iz] : 0) / dx : (ix + 1 < nx ? f[ix + 1, iy, iz] - c : 0) / dx;
        double dphidy = vel.Y > 0 ? (iy > 0 ? c - f[ix, iy - 1, iz] : 0) / dy : (iy + 1 < ny ? f[ix, iy + 1, iz] - c : 0) / dy;
        double dphidz = vel.Z > 0 ? (iz > 0 ? c - f[ix, iy, iz - 1] : 0) / dz : (iz + 1 < nz ? f[ix, iy, iz + 1] - c : 0) / dz;
        return -(vel.X * dphidx + vel.Y * dphidy + vel.Z * dphidz);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DiffusionExplicit(double[,,] f, double alpha, int ix, int iy, int iz, double dx, double dy, double dz) => alpha * Laplacian(f, ix, iy, iz, dx, dy, dz);
    public double WaveEquation(double[,,] fCur, double[,,] fPrev, double c, int ix, int iy, int iz, double dx, double dy, double dz, double dt)
        => 2.0 * fCur[ix, iy, iz] - fPrev[ix, iy, iz] + c * c * dt * dt * Laplacian(fCur, ix, iy, iz, dx, dy, dz);

    // ── 12.8 PARTICLE & RELATIVISTIC PHYSICS ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double LorentzFactor(double v, double c) => 1.0 / Math.Sqrt(1.0 - (v / c) * (v / c));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RelativisticEnergy(double gamma, double m, double c) => gamma * m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D RelativisticMomentum(double gamma, double m, Vector3D v) => v * (gamma * m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RelativisticKE(double gamma, double m, double c) => (gamma - 1.0) * m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EnergyMomentum(double p, double m, double c) { double pc = p * c, mc2 = m * c * c; return Math.Sqrt(pc * pc + mc2 * mc2); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DopplerShift(double f0, double vr, double c) { double b = vr / c; return f0 * Math.Sqrt((1.0 + b) / (1.0 - b)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComptonWavelength(double m, double h, double c) => h / (m * c);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DeBroglieWavelength(double p, double h) => h / p;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SchwarzschildRadius(double M, double G, double c) => 2.0 * G * M / (c * c);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GravitationalRedshift(double M, double r, double G, double c) => G * M / (r * c * c);
    public double HawkingTemperature(double M, double G, double c, double hbar, double kB) => hbar * c * c * c / (8.0 * Math.PI * G * M * kB);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double StefanBoltzmann(double A, double T, double sigma) => sigma * A * Math.Pow(T, 4);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double WienDisplacement(double T, double b) => b / T;
    public double PlanckLaw(double lambda, double T, double h, double c, double kB)
    {
        double hc = h * c;
        return 2.0 * h * c * c / Math.Pow(lambda, 5) / (Math.Exp(hc / (lambda * kB * T)) - 1.0);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EinsteinME(double m, double c) => m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoltzmannEntropy(long Omega, double kB) => kB * Math.Log(Omega);
    public double GibbsEntropy(double[] p, double kB) { double s = 0; for (int i = 0; i < p.Length; i++) if (p[i] > 1e-30) s -= p[i] * Math.Log(p[i]); return kB * s; }

    // ── 12.9 GEOMETRY & MATH UTILITIES ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D CrossProduct(Vector3D a, Vector3D b) => Vector3D.Cross(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DotProduct(Vector3D a, Vector3D b) => Vector3D.Dot(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TripleScalar(Vector3D a, Vector3D b, Vector3D c) => Vector3D.Dot(a, Vector3D.Cross(b, c));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D VectorTriple(Vector3D a, Vector3D b, Vector3D c) => b * Vector3D.Dot(a, c) - c * Vector3D.Dot(a, b);
    public Tensor3D OuterProduct(Vector3D a, Vector3D b) => new Tensor3D(a.X * b.X, a.X * b.Y, a.X * b.Z, a.Y * b.X, a.Y * b.Y, a.Y * b.Z, a.Z * b.X, a.Z * b.Y, a.Z * b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Project(Vector3D a, Vector3D b) { double d = Vector3D.Dot(b, b); return d < 1e-30 ? Vector3D.Zero : b * (Vector3D.Dot(a, b) / d); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Reject(Vector3D a, Vector3D b) => a - Project(a, b);
    public double AngleBetween(Vector3D a, Vector3D b) { double c = Math.Max(-1, Math.Min(1, Vector3D.Dot(a.Normalized(), b.Normalized()))); return Math.Acos(c); }
    public double SignedAngleAxis(Vector3D a, Vector3D b, Vector3D axis)
    {
        var ax = Project(a, axis.Normalized());
        var bx = Project(b, axis.Normalized());
        var ap = a - ax;
        var bp = b - bx;
        return Math.Atan2(Vector3D.Dot(axis, Vector3D.Cross(ap, bp)), Vector3D.Dot(ap, bp));
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SmoothMin(double a, double b, double k) { double h = Math.Max(k - Math.Abs(a - b), 0) / k; return Math.Min(a, b) - h * h * h * k / 6.0; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SmoothMax(double a, double b, double k) => -SmoothMin(-a, -b, k);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Remap(double v, double inMin, double inMax, double outMin, double outMax) => outMin + (v - inMin) * (outMax - outMin) / (inMax - inMin);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Lerp(double a, double b, double t) => a + Math.Max(0, Math.Min(1, t)) * (b - a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InverseLerp(double a, double b, double v) => Math.Abs(b - a) < 1e-30 ? 0 : (v - a) / (b - a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Smoothstep(double e0, double e1, double x) { double t = Math.Max(0, Math.Min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Smootherstep(double e0, double e1, double x) { double t = Math.Max(0, Math.Min(1, (x - e0) / (e1 - e0))); return t * t * t * (t * (t * 15 - 10) * t); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Clamp(double v, double lo, double hi) => Math.Max(lo, Math.Min(hi, v));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Wrap(double v, double lo, double hi) { double r = hi - lo; return lo + ((v - lo) % r + r) % r; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PingPong(double v, double len) { v = Math.Abs(v); double m = v % (2 * len); return len - Math.Abs(m - len); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Distance3D(Vector3D a, Vector3D b) => (b - a).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DistanceSq(Vector3D a, Vector3D b) => (b - a).LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ManhattanDist(Vector3D a, Vector3D b) => Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y) + Math.Abs(a.Z - b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ChebyshevDist(Vector3D a, Vector3D b) => Math.Max(Math.Max(Math.Abs(a.X - b.X), Math.Abs(a.Y - b.Y)), Math.Abs(a.Z - b.Z));
    public double MinkowskiDist(Vector3D a, Vector3D b, double p) => Math.Pow(Math.Pow(Math.Abs(a.X - b.X), p) + Math.Pow(Math.Abs(a.Y - b.Y), p) + Math.Pow(Math.Abs(a.Z - b.Z), p), 1.0 / p);
    public Vector3D HermiteInterp(Vector3D p0, Vector3D m0, Vector3D p1, Vector3D m1, double t)
    { double t2 = t * t, t3 = t2 * t; return p0 * (2 * t3 - 3 * t2 + 1) + m0 * (t3 - 2 * t2 + t) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2); }
    public Vector3D CatmullRom(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t, double tau)
    { double t2 = t * t, t3 = t2 * t; return (p1 * 2 + (p2 - p0) * tau * t + (p0 * 2 - p1 * 5 + p2 * 4 - p3) * tau * t2 + (-p1 + p2 * 3 - p3 + p0 * -1) * tau * t3) * 0.5; }
    public Vector3D Bezier(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    { double u = 1 - t, u2 = u * u, u3 = u2 * u, t2 = t * t, t3 = t2 * t; return p0 * u3 + p1 * (3 * u2 * t) + p2 * (3 * u * t2) + p3 * t3; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Fade(double t) => t * t * t * (t * (t * 6 - 15) + 10);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float FastInvSqrt(float x) { int i = BitConverter.SingleToInt32Bits(x); i = 0x5f3759df - (i >> 1); float y = BitConverter.Int32BitsToSingle(i); return y * (1.5f - 0.5f * x * y * y); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Cross2D(Vector3D a, Vector3D b) => a.X * b.Y - a.Y * b.X;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TriangleArea(Vector3D a, Vector3D b, Vector3D c) => 0.5 * Vector3D.Cross(b - a, c - a).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D TriangleCentroid(Vector3D a, Vector3D b, Vector3D c) => (a + b + c) * (1.0 / 3.0);
    public (double u, double v, double w) Barycentric(Vector3D p, Vector3D a, Vector3D b, Vector3D c)
    {
        var v0 = b - a;
        var v1 = c - a;
        var v2 = p - a;
        double d00 = Vector3D.Dot(v0, v0), d01 = Vector3D.Dot(v0, v1), d11 = Vector3D.Dot(v1, v1);
        double d20 = Vector3D.Dot(v2, v0), d21 = Vector3D.Dot(v2, v1);
        double den = d00 * d11 - d01 * d01;
        if (Math.Abs(den) < 1e-30)
            return (1.0 / 3, 1.0 / 3, 1.0 / 3);
        double v = (d11 * d20 - d01 * d21) / den;
        double w = (d00 * d21 - d01 * d20) / den;
        return (1 - v - w, v, w);
    }
    public bool PointInTriangle(Vector3D p, Vector3D a, Vector3D b, Vector3D c) { var (u, v, w) = Barycentric(p, a, b, c); return u >= -1e-6 && v >= -1e-6 && w >= -1e-6; }
    public Vector3D ClosestPointSegment(Vector3D p, Vector3D a, Vector3D b)
    { var ab = b - a; double t = Math.Max(0, Math.Min(1, Vector3D.Dot(p - a, ab) / Vector3D.Dot(ab, ab))); return a + ab * t; }
    public double DistPointSegment(Vector3D p, Vector3D a, Vector3D b) => (p - ClosestPointSegment(p, a, b)).Length();
    public Vector3D ClosestPointTriangle(Vector3D p, Vector3D a, Vector3D b, Vector3D c)
    {
        var (u, v, w) = Barycentric(p, a, b, c);
        if (u >= 0 && v >= 0 && w >= 0)
            return p;
        var best = ClosestPointSegment(p, a, b);
        double bestD = (p - best).LengthSquared();
        var cand = ClosestPointSegment(p, b, c);
        double d = (p - cand).LengthSquared();
        if (d < bestD)
        { best = cand; bestD = d; }
        cand = ClosestPointSegment(p, c, a);
        d = (p - cand).LengthSquared();
        if (d < bestD)
            best = cand;
        return best;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SignedVolumeTetra(Vector3D a, Vector3D b, Vector3D c, Vector3D d) => Vector3D.Dot(a - d, Vector3D.Cross(b - d, c - d)) / 6.0;
    public double HullVolume(Vector3D[] verts, (int i, int j, int k)[] tris)
    { double v = 0; var o = verts[0]; foreach (var t in tris) v += SignedVolumeTetra(verts[t.i], verts[t.j], verts[t.k], o); return Math.Abs(v); }
    public double MeshArea(Vector3D[] verts, (int i, int j, int k)[] tris) { double a = 0; foreach (var t in tris) a += TriangleArea(verts[t.i], verts[t.j], verts[t.k]); return a; }
    public Symmetric3x3 InertiaTensor((double m, Vector3D r)[] particles)
    {
        double ixx = 0, iyy = 0, izz = 0, ixy = 0, ixz = 0, iyz = 0;
        foreach (var (m, r) in particles)
        { ixx += m * (r.Y * r.Y + r.Z * r.Z); iyy += m * (r.X * r.X + r.Z * r.Z); izz += m * (r.X * r.X + r.Y * r.Y); ixy -= m * r.X * r.Y; ixz -= m * r.X * r.Z; iyz -= m * r.Y * r.Z; }
        return new Symmetric3x3(ixx, ixy, ixz, iyy, iyz, izz);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D PrincipalMoments(Symmetric3x3 I) { I.MaxEigenvalues(out double v1, out double v2, out double v3); return new Vector3D(v1, v2, v3); }
    public Vector3D CenterOfMass((double m, Vector3D r)[] p) { double M = 0; Vector3D s = Vector3D.Zero; foreach (var (m, r) in p) { M += m; s = s + r * m; } return M > 1e-30 ? s * (1 / M) : Vector3D.Zero; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ReducedMass(double m1, double m2) => m1 * m2 / (m1 + m2);
    public Vector3D GravForce(double m1, double m2, Vector3D r1, Vector3D r2, double G)
    { var d = r2 - r1; double dsq = d.LengthSquared(); return dsq < 1e-30 ? Vector3D.Zero : d.Normalized() * (G * m1 * m2 / dsq); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GravPE(double m1, double m2, double r, double G) => -G * m1 * m2 / r;
    public Vector3D CoulombF(double q1, double q2, Vector3D r1, Vector3D r2, double ke)
    { var d = r2 - r1; double dsq = d.LengthSquared(); return dsq < 1e-30 ? Vector3D.Zero : d.Normalized() * (ke * q1 * q2 / dsq); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CoulombV(double q, double r, double ke) => ke * q / r;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D HookesLaw(double k, Vector3D x) => x * (-k);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SpringPE(double k, Vector3D x) => 0.5 * k * x.LengthSquared();
    public Vector3D DampedOscillator(double m, double c, double k, Vector3D x, Vector3D v) => (x * (-k) + v * (-c)) * (1.0 / m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double NatFreq(double k, double m) => Math.Sqrt(k / m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DampRatio(double c, double k, double m) => c / (2.0 * Math.Sqrt(k * m));
    public double DampedPeriod(double k, double m, double c) { double w0 = Math.Sqrt(k / m); double z = c / (2 * Math.Sqrt(k * m)); return 2 * Math.PI / (w0 * Math.Sqrt(1 - z * z)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ResonanceAmp(double F0, double c, double w0) => F0 / (c * w0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double QualityFactor(double zeta) => 1.0 / (2.0 * zeta);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RotKE(double I, double w) => 0.5 * I * w * w;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PrecessionFreq(double m, double g, double r, double I, double w) => m * g * r / (I * w);

    // ── 12.10 RANDOM & NOISE ──
    public Vector3D RandomUnitVector(Random rng)
    {
        double theta = 2.0 * Math.PI * rng.NextDouble();
        double phi = Math.Acos(2.0 * rng.NextDouble() - 1.0);
        return new Vector3D(Math.Sin(phi) * Math.Cos(theta), Math.Sin(phi) * Math.Sin(theta), Math.Cos(phi));
    }
    public Vector3D RandomPointInSphere(Random rng)
    {
        double u = rng.NextDouble();
        double r = Math.Pow(u, 1.0 / 3.0);
        return RandomUnitVector(rng) * r;
    }
    public Vector3D RandomPointOnSphere(Random rng) => RandomUnitVector(rng);
    public double PerlinNoise3D(double x, double y, double z)
    {
        int xi = (int)Math.Floor(x), yi = (int)Math.Floor(y), zi = (int)Math.Floor(z);
        double xf = x - xi, yf = y - yi, zf = z - zi;
        double u = Fade(xf), v = Fade(yf), w2 = Fade(zf);
        int h000 = PerlinHash(xi, yi, zi), h100 = PerlinHash(xi + 1, yi, zi), h010 = PerlinHash(xi, yi + 1, zi), h110 = PerlinHash(xi + 1, yi + 1, zi);
        int h001 = PerlinHash(xi, yi, zi + 1), h101 = PerlinHash(xi + 1, yi, zi + 1), h011 = PerlinHash(xi, yi + 1, zi + 1), h111 = PerlinHash(xi + 1, yi + 1, zi + 1);
        double x1 = LerpUnclamped(Grad(h000, xf, yf, zf), Grad(h100, xf - 1, yf, zf), u);
        double x2 = LerpUnclamped(Grad(h010, xf, yf - 1, zf), Grad(h110, xf - 1, yf - 1, zf), u);
        double y1 = LerpUnclamped(x1, x2, v);
        x1 = LerpUnclamped(Grad(h001, xf, yf, zf - 1), Grad(h101, xf - 1, yf, zf - 1), u);
        x2 = LerpUnclamped(Grad(h011, xf, yf - 1, zf - 1), Grad(h111, xf - 1, yf - 1, zf - 1), u);
        double y2 = LerpUnclamped(x1, x2, v);
        return LerpUnclamped(y1, y2, w2);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int PerlinHash(int x, int y, int z) { int h = x * 374761393 + y * 668265263 + z * 1274126177; h = (h ^ (h >> 13)) * 1274126177; return h ^ (h >> 16); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Grad(int hash, double x, double y, double z)
    {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double LerpUnclamped(double a, double b, double t) => a + t * (b - a);
}
