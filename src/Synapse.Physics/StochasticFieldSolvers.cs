#nullable enable

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 1 — ENUMS and SHARED TYPES
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>Type of stochastic process governing a field cell.</summary>
public enum StochasticProcessType : byte
{
    /// <summary>Geometric Brownian Motion.</summary>
    GeometricBrownian = 0,
    /// <summary>Poisson arrival process.</summary>
    Poisson = 1,
    /// <summary>Markov chain.</summary>
    MarkovChain = 2,
    /// <summary>Ornstein-Uhlenbeck mean-reverting.</summary>
    OrnsteinUhlenbeck = 3,
    /// <summary>Merton jump-diffusion.</summary>
    JumpDiffusion = 4,
    /// <summary>Fractional Brownian Motion.</summary>
    FractionalBrownian = 5
}

/// <summary>Numerical scheme for SDE time-stepping.</summary>
public enum SDEScheme : byte
{
    /// <summary>Euler-Maruyama: first-order convergence.</summary>
    EulerMaruyama = 0,
    /// <summary>Milstein: higher strong convergence for scalar SDEs.</summary>
    Milstein = 1,
    /// <summary>Runge-Kutta-Maruyama: multi-stage stochastic scheme.</summary>
    RungeKuttaMaruyama = 2,
    /// <summary>Stratonovich interpretation.</summary>
    Stratonovich = 3
}

/// <summary>Covariance kernel type for spatial correlation.</summary>
public enum CovarianceKernelType : byte
{
    /// <summary>Matern covariance kernel.</summary>
    Matern = 0,
    /// <summary>Exponential covariance kernel.</summary>
    Exponential = 1,
    /// <summary>Gaussian (squared-exponential) kernel.</summary>
    Gaussian = 2,
    /// <summary>Rational quadratic kernel.</summary>
    RationalQuadratic = 3,
    /// <summary>Power-law covariance kernel.</summary>
    PowerLaw = 4
}

/// <summary>Financial derivative type.</summary>
public enum OptionType : byte
{
    /// <summary>European call.</summary>
    Call = 0,
    /// <summary>European put.</summary>
    Put = 1
}

/// <summary>Epidemiological compartment.</summary>
public enum EpiCompartment : byte
{
    /// <summary>Susceptible.</summary>
    Susceptible = 0,
    /// <summary>Exposed (SEIR).</summary>
    Exposed = 1,
    /// <summary>Infectious.</summary>
    Infectious = 2,
    /// <summary>Recovered.</summary>
    Recovered = 3,
    /// <summary>Deceased.</summary>
    Deceased = 4,
    /// <summary>Vaccinated.</summary>
    Vaccinated = 5,
    /// <summary>Quarantined.</summary>
    Quarantined = 6
}

/// <summary>Variance reduction technique for Monte Carlo.</summary>
public enum VarianceReductionTechnique : byte
{
    /// <summary>No variance reduction.</summary>
    None = 0,
    /// <summary>Antithetic variates.</summary>
    Antithetic = 1,
    /// <summary>Control variates.</summary>
    ControlVariates = 2,
    /// <summary>Importance sampling.</summary>
    ImportanceSampling = 3,
    /// <summary>Stratified sampling.</summary>
    Stratified = 4
}

/// <summary>Configuration for the stochastic field system.</summary>
public sealed class StochasticFieldConfig
{
    /// <summary>Grid resolution in X.</summary>
    public int ResolutionX { get; init; } = 64;
    /// <summary>Grid resolution in Y.</summary>
    public int ResolutionY { get; init; } = 64;
    /// <summary>Grid resolution in Z.</summary>
    public int ResolutionZ { get; init; } = 64;
    /// <summary>Cell spacing in X.</summary>
    public float SpacingX { get; init; } = 1.0f;
    /// <summary>Cell spacing in Y.</summary>
    public float SpacingY { get; init; } = 1.0f;
    /// <summary>Cell spacing in Z.</summary>
    public float SpacingZ { get; init; } = 1.0f;
    /// <summary>Origin offset X.</summary>
    public float OriginX { get; init; } = 0.0f;
    /// <summary>Origin offset Y.</summary>
    public float OriginY { get; init; } = 0.0f;
    /// <summary>Origin offset Z.</summary>
    public float OriginZ { get; init; } = 0.0f;
    /// <summary>Default process type.</summary>
    public StochasticProcessType DefaultProcess { get; init; } = StochasticProcessType.GeometricBrownian;
    /// <summary>Time-step size.</summary>
    public float TimeStep { get; init; } = 0.01f;
    /// <summary>Spatial correlation length.</summary>
    public float CorrelationLength { get; init; } = 3.0f;
    /// <summary>Coupling strength.</summary>
    public float CouplingStrength { get; init; } = 0.1f;
    /// <summary>Random seed.</summary>
    public int Seed { get; init; } = 42;
    /// <summary>Monte Carlo paths.</summary>
    public int MonteCarloPaths { get; init; } = 1000;
    /// <summary>Default drift.</summary>
    public float DefaultDrift { get; init; } = 0.05f;
    /// <summary>Default diffusion.</summary>
    public float DefaultDiffusion { get; init; } = 0.2f;
    /// <summary>Hurst parameter.</summary>
    public float HurstParameter { get; init; } = 0.5f;
    /// <summary>SDE scheme.</summary>
    public SDEScheme Scheme { get; init; } = SDEScheme.EulerMaruyama;
    /// <summary>Covariance kernel.</summary>
    public CovarianceKernelType KernelType { get; init; } = CovarianceKernelType.Matern;
    /// <summary>Matern smoothness parameter.</summary>
    public float MaternNu { get; init; } = 1.5f;
    /// <summary>Enable parallel execution.</summary>
    public bool ParallelExecution { get; init; } = true;
    /// <summary>Max degree of parallelism.</summary>
    public int MaxDegreeOfParallelism { get; init; } = -1;
}

/// <summary>Statistical summary of the stochastic field.</summary>
public readonly struct StochasticFieldStats
{
    /// <summary>Dominant process type.</summary>
    public StochasticProcessType ProcessType { get; init; }
    /// <summary>Total cells.</summary>
    public int TotalCells { get; init; }
    /// <summary>Minimum value.</summary>
    public float MinValue { get; init; }
    /// <summary>Maximum value.</summary>
    public float MaxValue { get; init; }
    /// <summary>Mean value.</summary>
    public float Mean { get; init; }
    /// <summary>Variance.</summary>
    public float Variance { get; init; }
    /// <summary>Standard deviation.</summary>
    public float StdDev { get; init; }
    /// <summary>Skewness.</summary>
    public float Skewness { get; init; }
    /// <summary>Excess kurtosis.</summary>
    public float Kurtosis { get; init; }
    /// <summary>Entropy.</summary>
    public float Entropy { get; init; }
    /// <summary>Current simulation time.</summary>
    public float TimeStep { get; init; }
    /// <summary>Sum of all values.</summary>
    public float Sum { get; init; }
    /// <summary>Sum of squares.</summary>
    public float SumOfSquares { get; init; }
    /// <summary>Median value.</summary>
    public float Median { get; init; }
    /// <summary>Interquartile range.</summary>
    public float IQR { get; init; }

    public override string ToString() =>
        $"""StochasticField | {ProcessType} | {TotalCells:N0} cells | Range=[{MinValue:F4},{MaxValue:F4}] | μ={Mean:F4} σ={StdDev:F4}""";
}

/// <summary>A single stochastic path realization.</summary>
public sealed class StochasticPath
{
    /// <summary>Time values.</summary>
    public float[] Times { get; }
    /// <summary>Process values.</summary>
    public float[] Values { get; }
    /// <summary>Path length.</summary>
    public int Length => Values.Length;
    /// <summary>Path identifier.</summary>
    public long PathId { get; init; }
    /// <summary>Path weight for importance sampling.</summary>
    public float Weight { get; set; } = 1.0f;

    /// <summary>Creates a path with given length.</summary>
    public StochasticPath(int length) { Times = new float[length]; Values = new float[length]; }
    /// <summary>Creates a path from existing arrays.</summary>
    public StochasticPath(float[] times, float[] values) { Times = times; Values = values; }
    /// <summary>Terminal value.</summary>
    public float TerminalValue => Values.Length > 0 ? Values[^1] : 0.0f;

    /// <summary>Maximum drawdown.</summary>
    public float MaxDrawdown()
    {
        if (Values.Length < 2)
            return 0.0f;
        float peak = Values[0], maxDD = 0.0f;
        for (int i = 1; i < Values.Length; i++)
        {
            if (Values[i] > peak)
                peak = Values[i];
            float dd = (peak - Values[i]) / (MathF.Abs(peak) + 1e-10f);
            if (dd > maxDD)
                maxDD = dd;
        }
        return maxDD;
    }

    /// <summary>Realized volatility.</summary>
    public float RealizedVolatility()
    {
        if (Values.Length < 2)
            return 0.0f;
        float sumSq = 0.0f;
        for (int i = 1; i < Values.Length; i++)
        {
            float r = MathF.Log(Values[i] / (MathF.Abs(Values[i - 1]) + 1e-10f) + 1e-10f);
            sumSq += r * r;
        }
        return MathF.Sqrt(sumSq / (Values.Length - 1));
    }

    /// <summary>Sharpe ratio.</summary>
    public float SharpeRatio(float riskFreeRate = 0.02f)
    {
        if (Values.Length < 2)
            return 0.0f;
        float sumR = 0.0f, sumSq = 0.0f;
        int n = Values.Length - 1;
        for (int i = 1; i < Values.Length; i++)
        {
            float r = MathF.Log(Values[i] / (MathF.Abs(Values[i - 1]) + 1e-10f) + 1e-10f);
            sumR += r;
            sumSq += r * r;
        }
        float mean = sumR / n;
        float std = MathF.Sqrt(MathF.Max(sumSq / n - mean * mean, 1e-12f));
        return (mean - riskFreeRate) / std;
    }

    /// <summary>Interpolated value at time t.</summary>
    public float InterpolateAt(float t)
    {
        if (Values.Length == 0)
            return 0.0f;
        if (Values.Length == 1)
            return Values[0];
        if (t <= Times[0])
            return Values[0];
        if (t >= Times[^1])
            return Values[^1];
        for (int i = 0; i < Values.Length - 1; i++)
        {
            if (t >= Times[i] && t <= Times[i + 1])
            {
                float frac = (t - Times[i]) / (Times[i + 1] - Times[i] + 1e-10f);
                return Values[i] + frac * (Values[i + 1] - Values[i]);
            }
        }
        return Values[^1];
    }
}

/// <summary>Monte Carlo pricing result.</summary>
public readonly struct MonteCarloResult
{
    /// <summary>Estimated value.</summary>
    public float Estimate { get; init; }
    /// <summary>Standard error.</summary>
    public float StandardError { get; init; }
    /// <summary>95% CI lower.</summary>
    public float CI95Lower { get; init; }
    /// <summary>95% CI upper.</summary>
    public float CI95Upper { get; init; }
    /// <summary>Number of paths.</summary>
    public int NumPaths { get; init; }
    /// <summary>Sample variance.</summary>
    public float Variance { get; init; }
    /// <summary>Elapsed milliseconds.</summary>
    public double ElapsedMs { get; init; }
    /// <summary>Relative standard error.</summary>
    public float RelativeSE => MathF.Abs(Estimate) > 1e-10f ? StandardError / MathF.Abs(Estimate) * 100f : 0.0f;
    public override string ToString() =>
        $"""MC({NumPaths:N0}) = {Estimate:F6} ± {StandardError:F6} [CI95: {CI95Lower:F6}, {CI95Upper:F6}] ({ElapsedMs:F1}ms)""";
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — STOCHASTIC STATE
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Stochastic state of a single point in the abstract field.
/// Each cell carries a full stochastic process state.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct StochasticFieldState
{
    /// <summary>Current realized value.</summary>
    public float CurrentValue;
    /// <summary>Expected value (mean).</summary>
    public float Mean;
    /// <summary>Variance (uncertainty).</summary>
    public float Variance;
    /// <summary>Previous value (for temporal derivatives).</summary>
    public float PreviousValue;
    /// <summary>Drift coefficient (μ).</summary>
    public float Drift;
    /// <summary>Diffusion coefficient (σ).</summary>
    public float Diffusion;
    /// <summary>Spatial correlation [-1,1].</summary>
    public float Correlation;
    /// <summary>Local entropy.</summary>
    public float Entropy;
    /// <summary>Skewness of local distribution.</summary>
    public float Skewness;
    /// <summary>Kurtosis of local distribution.</summary>
    public float Kurtosis;
    /// <summary>Third moment.</summary>
    public float ThirdMoment;
    /// <summary>Fractional Brownian memory.</summary>
    public float FractionalMemory;
    /// <summary>First-order sensitivity (delta).</summary>
    public float Delta;
    /// <summary>Jump intensity (jump-diffusion).</summary>
    public float JumpIntensity;
    /// <summary>Mean reversion speed (OU).</summary>
    public float MeanReversionSpeed;
    /// <summary>Equilibrium mean (OU).</summary>
    public float EquilibriumMean;
    /// <summary>Hurst exponent (fBm).</summary>
    public float HurstExponent;
    /// <summary>Markov chain state index.</summary>
    public int MarkovState;
    /// <summary>Process type.</summary>
    public StochasticProcessType ProcessType;
    /// <summary>Padding.</summary>
    private float _pad0;

    /// <summary>Default zero state.</summary>
    public static StochasticFieldState Zero => default;

    /// <summary>Instantaneous return rate.</summary>
    public readonly float InstantaneousReturn(float dt = 0.01f)
    {
        float prev = MathF.Abs(PreviousValue) < 1e-12f ? CurrentValue : PreviousValue;
        return (CurrentValue - prev) / (prev * dt + 1e-12f);
    }

    /// <summary>Information ratio.</summary>
    public readonly float InformationRatio(float benchmark = 0.0f)
    {
        float excess = Mean - benchmark;
        float te = MathF.Sqrt(MathF.Max(Variance, 1e-12f));
        return excess / te;
    }

    /// <summary>Signal-to-noise ratio.</summary>
    public readonly float SignalToNoiseRatio()
    {
        float signal = MathF.Abs(Mean);
        float noise = MathF.Sqrt(MathF.Max(Variance, 1e-12f));
        return signal / noise;
    }

    /// <summary>Returns true if cell is active.</summary>
    public readonly bool IsActive(float threshold = 1e-6f) =>
        MathF.Abs(CurrentValue) > threshold || MathF.Abs(Drift) > threshold || MathF.Abs(Diffusion) > threshold;

    /// <summary>Fano factor (variance/mean).</summary>
    public readonly float FanoFactor()
    {
        float absMean = MathF.Abs(Mean);
        return absMean > 1e-12f ? Variance / absMean : 0.0f;
    }
}

/// <summary>SEIR epidemiological state for spatial field cells.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct SEIRState
{
    /// <summary>Susceptible fraction.</summary>
    public float S;
    /// <summary>Exposed fraction.</summary>
    public float E;
    /// <summary>Infectious fraction.</summary>
    public float I;
    /// <summary>Recovered fraction.</summary>
    public float R;
    /// <summary>Vaccinated fraction.</summary>
    public float V;
    /// <summary>Quarantined fraction.</summary>
    public float Q;
    /// <summary>Deceased fraction.</summary>
    public float D;
    /// <summary>Effective R0.</summary>
    public float R0Effective;
    /// <summary>Contact rate β.</summary>
    public float Beta;
    /// <summary>Incubation rate σ.</summary>
    public float Sigma;
    /// <summary>Recovery rate γ.</summary>
    public float Gamma;
    /// <summary>Vaccination rate.</summary>
    public float VaccinationRate;
    /// <summary>Quarantine compliance.</summary>
    public float QuarantineRate;
    /// <summary>Population density.</summary>
    public float PopulationDensity;
    /// <summary>Spatial diffusion coefficient.</summary>
    public float DiffusionCoeff;
    private float _pad0;

    /// <summary>Computes R0 = β/γ.</summary>
    public readonly float ComputeR0() => Gamma > 1e-12f ? Beta / Gamma : 0.0f;
    /// <summary>Effective R including immunity.</summary>
    public readonly float ComputeReffective()
    {
        float total = S + E + I + R + V + Q + D;
        return total < 1e-12f ? 0.0f : ComputeR0() * S / total;
    }
    /// <summary>Herd immunity threshold: 1 - 1/R0.</summary>
    public readonly float HerdImmunityThreshold()
    {
        float r0 = ComputeR0();
        return r0 > 1.0f ? 1.0f - 1.0f / r0 : 0.0f;
    }
    /// <summary>Normalizes compartments to sum to 1.</summary>
    public void Normalize()
    {
        float total = S + E + I + R + V + Q + D;
        if (total < 1e-12f)
            return;
        float inv = 1.0f / total;
        S *= inv;
        E *= inv;
        I *= inv;
        R *= inv;
        V *= inv;
        Q *= inv;
        D *= inv;
    }
}

/// <summary>Financial field state for option pricing.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct FinancialFieldState
{
    /// <summary>Spot price.</summary>
    public float SpotPrice;
    /// <summary>Implied volatility.</summary>
    public float ImpliedVolatility;
    /// <summary>Risk-free rate.</summary>
    public float RiskFreeRate;
    /// <summary>Dividend yield.</summary>
    public float DividendYield;
    /// <summary>Time to maturity (years).</summary>
    public float TimeToMaturity;
    /// <summary>Strike price.</summary>
    public float StrikePrice;
    /// <summary>Option price.</summary>
    public float OptionPrice;
    /// <summary>Delta.</summary>
    public float Delta;
    /// <summary>Gamma.</summary>
    public float Gamma;
    /// <summary>Vega.</summary>
    public float Vega;
    /// <summary>Theta.</summary>
    public float Theta;
    /// <summary>Rho.</summary>
    public float Rho;
    /// <summary>Vanna.</summary>
    public float Vanna;
    /// <summary>Volga.</summary>
    public float Volga;
    /// <summary>Option type.</summary>
    public OptionType Type;
    private float _pad0;

    /// <summary>Computes Black-Scholes price and all Greeks.</summary>
    public void ComputeBlackScholes()
    {
        float S = SpotPrice, K = StrikePrice, r = RiskFreeRate, q = DividendYield;
        float sigma = ImpliedVolatility, T = TimeToMaturity;
        if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0)
        { OptionPrice = Delta = Gamma = Vega = Theta = Rho = Vanna = Volga = 0; return; }
        float sqrtT = MathF.Sqrt(T), sigmaSqrtT = sigma * sqrtT;
        float d1 = (MathF.Log(S / K) + (r - q + 0.5f * sigma * sigma) * T) / sigmaSqrtT;
        float d2 = d1 - sigmaSqrtT;
        float Nd1 = NormalCDF(d1), Nd2 = NormalCDF(d2), npd1 = NormalPDF(d1);
        if (Type == OptionType.Call)
        {
            OptionPrice = S * MathF.Exp(-q * T) * Nd1 - K * MathF.Exp(-r * T) * Nd2;
            Delta = MathF.Exp(-q * T) * Nd1;
        }
        else
        {
            float Nmd1 = NormalCDF(-d1), Nmd2 = NormalCDF(-d2);
            OptionPrice = K * MathF.Exp(-r * T) * Nmd2 - S * MathF.Exp(-q * T) * Nmd1;
            Delta = -MathF.Exp(-q * T) * Nmd1;
        }
        Gamma = MathF.Exp(-q * T) * npd1 / (S * sigmaSqrtT);
        Vega = S * MathF.Exp(-q * T) * npd1 * sqrtT / 100.0f;
        Theta = (Type == OptionType.Call
            ? -S * MathF.Exp(-q * T) * npd1 * sigma / (2 * sqrtT) - r * K * MathF.Exp(-r * T) * Nd2 + q * S * MathF.Exp(-q * T) * Nd1
            : -S * MathF.Exp(-q * T) * npd1 * sigma / (2 * sqrtT) + r * K * MathF.Exp(-r * T) * NormalCDF(-d2) - q * S * MathF.Exp(-q * T) * NormalCDF(-d1)) / 365.0f;
        Rho = (Type == OptionType.Call ? K * T * MathF.Exp(-r * T) * Nd2 : -K * T * MathF.Exp(-r * T) * NormalCDF(-d2)) / 100.0f;
        Vanna = -npd1 * (d2 / sigma) * MathF.Exp(-q * T) / 100.0f;
        Volga = Vega * d1 * d2 / sigma;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalCDF(float x)
    {
        float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0 ? -1.0f : 1.0f;
        x = MathF.Abs(x) / MathF.Sqrt(2.0f);
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return 0.5f * (1.0f + sign * y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalPDF(float x) => MathF.Exp(-0.5f * x * x) / MathF.Sqrt(2.0f * MathF.PI);
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — PERLIN NOISE GENERATOR
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Perlin noise generator for initializing stochastic fields.
/// Produces smooth 3D noise for parameter fields and initial conditions.
/// </summary>
public sealed unsafe class PerlinNoiseGenerator : IDisposable
{
    private readonly int* _permutation;
    private readonly int _tableSize;

    /// <summary>Creates a new Perlin noise generator.</summary>
    public PerlinNoiseGenerator(int seed = 42)
    {
        _tableSize = 256;
        _permutation = (int*)NativeMemory.AlignedAlloc((nuint)(_tableSize * 2 * sizeof(int)), 64);
        var rng = new Random(seed);
        for (int i = 0; i < _tableSize; i++)
            _permutation[i] = i;
        for (int i = _tableSize - 1; i > 0; i--)
        { int j = rng.Next(i + 1); (_permutation[i], _permutation[j]) = (_permutation[j], _permutation[i]); }
        for (int i = 0; i < _tableSize; i++)
            _permutation[_tableSize + i] = _permutation[i];
    }

    /// <summary>Evaluates 3D Perlin noise at (x,y,z). Returns [-1,1].</summary>
    public float Noise3D(float x, float y, float z)
    {
        int xi = (int)MathF.Floor(x) & (_tableSize - 1);
        int yi = (int)MathF.Floor(y) & (_tableSize - 1);
        int zi = (int)MathF.Floor(z) & (_tableSize - 1);
        float xf = x - MathF.Floor(x), yf = y - MathF.Floor(y), zf = z - MathF.Floor(z);
        float u = Fade(xf), v = Fade(yf), w = Fade(zf);
        int aaa = _permutation[_permutation[_permutation[xi] + yi] + zi];
        int aba = _permutation[_permutation[_permutation[xi] + yi + 1] + zi];
        int aab = _permutation[_permutation[_permutation[xi] + yi] + zi + 1];
        int abb = _permutation[_permutation[_permutation[xi] + yi + 1] + zi + 1];
        int baa = _permutation[_permutation[_permutation[xi + 1] + yi] + zi];
        int bba = _permutation[_permutation[_permutation[xi + 1] + yi + 1] + zi];
        int bab = _permutation[_permutation[_permutation[xi + 1] + yi] + zi + 1];
        int bbb = _permutation[_permutation[_permutation[xi + 1] + yi + 1] + zi + 1];
        float x1 = Lerp(Grad(aaa, xf, yf, zf), Grad(baa, xf - 1, yf, zf), u);
        float x2 = Lerp(Grad(aba, xf, yf - 1, zf), Grad(bba, xf - 1, yf - 1, zf), u);
        float y1 = Lerp(x1, x2, v);
        x1 = Lerp(Grad(aab, xf, yf, zf - 1), Grad(bab, xf - 1, yf, zf - 1), u);
        x2 = Lerp(Grad(abb, xf, yf - 1, zf - 1), Grad(bbb, xf - 1, yf - 1, zf - 1), u);
        float y2 = Lerp(x1, x2, v);
        return Lerp(y1, y2, w);
    }

    /// <summary>Fractal Brownian motion.</summary>
    public float FractalBrownianMotion(float x, float y, float z, int octaves = 4,
        float persistence = 0.5f, float lacunarity = 2.0f)
    {
        float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, maxAmplitude = 0.0f;
        for (int i = 0; i < octaves; i++)
        { value += amplitude * Noise3D(x * frequency, y * frequency, z * frequency); maxAmplitude += amplitude; amplitude *= persistence; frequency *= lacunarity; }
        return value / maxAmplitude;
    }

    /// <summary>Ridged multifractal noise.</summary>
    public float RidgedMultifractal(float x, float y, float z, int octaves = 4,
        float persistence = 0.5f, float lacunarity = 2.0f, float offset = 0.8f)
    {
        float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, weight = 1.0f;
        for (int i = 0; i < octaves; i++)
        { float signal = MathF.Abs(Noise3D(x * frequency, y * frequency, z * frequency)); signal = offset - signal; signal *= signal; signal *= weight; weight = MathF.Min(MathF.Max(signal * persistence, 0.0f), 1.0f); value += amplitude * signal; amplitude *= persistence; frequency *= lacunarity; }
        return value;
    }

    /// <summary>Turbulence noise.</summary>
    public float Turbulence(float x, float y, float z, int octaves = 4)
    {
        float value = 0.0f, amplitude = 1.0f, frequency = 1.0f;
        for (int i = 0; i < octaves; i++)
        { value += amplitude * MathF.Abs(Noise3D(x * frequency, y * frequency, z * frequency)); amplitude *= 0.5f; frequency *= 2.0f; }
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float Fade(float t) => t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float Lerp(float a, float b, float t) => a + t * (b - a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Grad(int hash, float x, float y, float z)
    {
        int h = hash & 15;
        float u = h < 8 ? x : y, v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    /// <summary>Frees unmanaged memory.</summary>
    public void Dispose() { if (_permutation != null) NativeMemory.AlignedFree(_permutation); }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 4 — STOCHASTIC FIELD (3D GRID CORE)
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// 3D differentiable stochastic field. Stores a regular grid of StochasticFieldState cells,
/// each governed by an SDE. Supports Euler-Maruyama, Milstein, and RK-Maruyama
/// time-stepping, spatial correlation propagation, and Perlin noise initialization.
/// </summary>
public sealed unsafe class StochasticField : IDisposable
{
    private StochasticFieldState* _data;
    private float* _scratchBuffer;
    private float* _correlationBuffer;
    private int _resX, _resY, _resZ;
    private float _spacingX, _spacingY, _spacingZ;
    private float _originX, _originY, _originZ;
    private StochasticProcessType _processType;
    private float _timeStep;
    private float _currentTime;
    private long _stepCount;
    private bool _disposed;
    private readonly Random _rng;

    /// <summary>Resolution X.</summary>
    public int ResX => _resX;
    /// <summary>Resolution Y.</summary>
    public int ResY => _resY;
    /// <summary>Resolution Z.</summary>
    public int ResZ => _resZ;
    /// <summary>Total cells.</summary>
    public int TotalCells => _resX * _resY * _resZ;
    /// <summary>Process type.</summary>
    public StochasticProcessType ProcessType => _processType;
    /// <summary>Time-step.</summary>
    public float TimeStep => _timeStep;
    /// <summary>Current simulation time.</summary>
    public float CurrentTime => _currentTime;
    /// <summary>Step count.</summary>
    public long StepCount => _stepCount;
    /// <summary>Memory usage in bytes.</summary>
    public long MemoryBytes => (long)TotalCells * sizeof(StochasticFieldState) + (long)TotalCells * 2 * sizeof(float);

    /// <summary>Creates a 3D stochastic field.</summary>
    public StochasticField(int resX, int resY, int resZ, StochasticProcessType processType,
        float spacingX, float spacingY, float spacingZ, float timeStep = 0.01f,
        float originX = 0.0f, float originY = 0.0f, float originZ = 0.0f, int seed = 42)
    {
        _resX = resX;
        _resY = resY;
        _resZ = resZ;
        _spacingX = spacingX;
        _spacingY = spacingY;
        _spacingZ = spacingZ;
        _originX = originX;
        _originY = originY;
        _originZ = originZ;
        _processType = processType;
        _timeStep = timeStep;
        _rng = new Random(seed);
        int total = resX * resY * resZ;
        int byteSize = total * sizeof(StochasticFieldState);
        _data = (StochasticFieldState*)NativeMemory.AlignedAlloc((nuint)byteSize, 64);
        NativeMemory.Clear(_data, (nuint)byteSize);
        _scratchBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        _correlationBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        NativeMemory.Clear(_scratchBuffer, (nuint)(total * sizeof(float)));
        NativeMemory.Clear(_correlationBuffer, (nuint)(total * sizeof(float)));
    }

    /// <summary>Convenience constructor with uniform resolution.</summary>
    public StochasticField(int resolution, StochasticProcessType processType = StochasticProcessType.GeometricBrownian,
        float spacing = 1.0f, float timeStep = 0.01f, int seed = 42)
        : this(resolution, resolution, resolution, processType, spacing, spacing, spacing, timeStep, 0, 0, 0, seed) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Index(int x, int y, int z) => (z * _resY + y) * _resX + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool InBounds(int x, int y, int z) => (uint)x < (uint)_resX && (uint)y < (uint)_resY && (uint)z < (uint)_resZ;

    /// <summary>Reference access at (x,y,z), clamped to bounds.</summary>
    public ref StochasticFieldState At(int x, int y, int z)
    { x = Math.Clamp(x, 0, _resX - 1); y = Math.Clamp(y, 0, _resY - 1); z = Math.Clamp(z, 0, _resZ - 1); return ref _data[Index(x, y, z)]; }

    /// <summary>Unchecked access — caller must ensure valid indices.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref StochasticFieldState UncheckedAt(int x, int y, int z) => ref _data[Index(x, y, z)];

    /// <summary>Raw data pointer.</summary>
    public StochasticFieldState* DataPointer => _data;

    /// <summary>Converts world position to grid indices.</summary>
    public void WorldToGrid(float worldX, float worldY, float worldZ, out int gx, out int gy, out int gz)
    {
        gx = Math.Clamp((int)MathF.Floor((worldX - _originX) / _spacingX), 0, _resX - 1);
        gy = Math.Clamp((int)MathF.Floor((worldY - _originY) / _spacingY), 0, _resY - 1);
        gz = Math.Clamp((int)MathF.Floor((worldZ - _originZ) / _spacingZ), 0, _resZ - 1);
    }

    /// <summary>Converts grid indices to world position.</summary>
    public void GridToWorld(int gx, int gy, int gz, out float worldX, out float worldY, out float worldZ)
    {
        worldX = _originX + (gx + 0.5f) * _spacingX;
        worldY = _originY + (gy + 0.5f) * _spacingY;
        worldZ = _originZ + (gz + 0.5f) * _spacingZ;
    }

    // ── Sampling ──────────────────────────────────────────────────────────

    /// <summary>Samples at continuous world position via trilinear interpolation.</summary>
    public StochasticFieldState Sample(float worldX, float worldY, float worldZ)
    {
        float fx = (worldX - _originX) / _spacingX, fy = (worldY - _originY) / _spacingY, fz = (worldZ - _originZ) / _spacingZ;
        int x0 = Math.Clamp((int)MathF.Floor(fx), 0, _resX - 2), y0 = Math.Clamp((int)MathF.Floor(fy), 0, _resY - 2), z0 = Math.Clamp((int)MathF.Floor(fz), 0, _resZ - 2);
        float tx = Math.Clamp(fx - x0, 0f, 1f), ty = Math.Clamp(fy - y0, 0f, 1f), tz = Math.Clamp(fz - z0, 0f, 1f);
        ref StochasticFieldState s000 = ref _data[Index(x0, y0, z0)], s100 = ref _data[Index(x0 + 1, y0, z0)];
        ref StochasticFieldState s010 = ref _data[Index(x0, y0 + 1, z0)], s110 = ref _data[Index(x0 + 1, y0 + 1, z0)];
        ref StochasticFieldState s001 = ref _data[Index(x0, y0, z0 + 1)], s101 = ref _data[Index(x0 + 1, y0, z0 + 1)];
        ref StochasticFieldState s011 = ref _data[Index(x0, y0 + 1, z0 + 1)], s111 = ref _data[Index(x0 + 1, y0 + 1, z0 + 1)];
        return LerpTrilinear(s000, s100, s010, s110, s001, s101, s011, s111, tx, ty, tz);
    }

    /// <summary>Samples scalar value at continuous position.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float SampleValue(float worldX, float worldY, float worldZ) => Sample(worldX, worldY, worldZ).CurrentValue;

    /// <summary>Bilinear sampling on XY at fixed Z layer.</summary>
    public float SampleValue2D(float worldX, float worldY, int zLayer)
    {
        float fx = (worldX - _originX) / _spacingX, fy = (worldY - _originY) / _spacingY;
        int x0 = Math.Clamp((int)MathF.Floor(fx), 0, _resX - 2), y0 = Math.Clamp((int)MathF.Floor(fy), 0, _resY - 2);
        float tx = Math.Clamp(fx - x0, 0f, 1f), ty = Math.Clamp(fy - y0, 0f, 1f);
        float v00 = _data[Index(x0, y0, zLayer)].CurrentValue, v10 = _data[Index(x0 + 1, y0, zLayer)].CurrentValue;
        float v01 = _data[Index(x0, y0 + 1, zLayer)].CurrentValue, v11 = _data[Index(x0 + 1, y0 + 1, zLayer)].CurrentValue;
        return (v00 + (v10 - v00) * tx) + ((v01 + (v11 - v01) * tx) - (v00 + (v10 - v00) * tx)) * ty;
    }

    // ── Writing ───────────────────────────────────────────────────────────

    /// <summary>Sets state at grid cell.</summary>
    public void Set(int x, int y, int z, StochasticFieldState state) { if (InBounds(x, y, z)) _data[Index(x, y, z)] = state; }
    /// <summary>Fills all cells with same state.</summary>
    public void FillUniform(StochasticFieldState state) { int total = TotalCells; for (int i = 0; i < total; i++) _data[i] = state; }
    /// <summary>Sets values from flat array.</summary>
    public void SetValuesFromFlat(float* values, int count) { int total = Math.Min(count, TotalCells); for (int i = 0; i < total; i++) _data[i].CurrentValue = values[i]; }
    /// <summary>Copies values to flat array.</summary>
    public void CopyValuesToFlat(float* values, int count) { int total = Math.Min(count, TotalCells); for (int i = 0; i < total; i++) values[i] = _data[i].CurrentValue; }

    // ── Perlin Initialization ─────────────────────────────────────────────

    /// <summary>Initializes field with 3D Perlin noise.</summary>
    public void InitializePerlin(float amplitude = 1.0f, float frequency = 1.0f, int octaves = 4, int seed = 42)
    {
        using var perlin = new PerlinNoiseGenerator(seed);
        for (int z = 0; z < _resZ; z++)
            for (int y = 0; y < _resY; y++)
                for (int x = 0; x < _resX; x++)
                {
                    float nx = x * frequency / _resX, ny = y * frequency / _resY, nz = z * frequency / _resZ;
                    float noise = perlin.FractalBrownianMotion(nx, ny, nz, octaves) * amplitude;
                    float gx = perlin.FractalBrownianMotion(nx + 0.01f, ny, nz, octaves) * amplitude;
                    float gy = perlin.FractalBrownianMotion(nx, ny + 0.01f, nz, octaves) * amplitude;
                    float gz = perlin.FractalBrownianMotion(nx, ny, nz + 0.01f, octaves) * amplitude;
                    float gradientMag = MathF.Sqrt((gx - noise) * (gx - noise) + (gy - noise) * (gy - noise) + (gz - noise) * (gz - noise));
                    int idx = Index(x, y, z);
                    _data[idx] = new StochasticFieldState { CurrentValue = noise, Mean = noise, Variance = amplitude * amplitude * 0.01f * (1.0f + gradientMag), PreviousValue = noise, Drift = 0.05f * (1.0f + gradientMag), Diffusion = 0.2f * (1.0f + 0.5f * MathF.Abs(noise)), ProcessType = _processType, HurstExponent = 0.5f };
                }
    }

    /// <summary>Initializes with spatially varying drift and diffusion.</summary>
    public void InitializeSpatialHeterogeneity(float valueAmplitude = 1.0f, float driftScale = 0.1f, float diffusionScale = 0.3f, float frequency = 1.0f, int octaves = 3, int seed = 42)
    {
        using var perlinValue = new PerlinNoiseGenerator(seed);
        using var perlinDrift = new PerlinNoiseGenerator(seed + 1);
        using var perlinDiff = new PerlinNoiseGenerator(seed + 2);
        for (int z = 0; z < _resZ; z++)
            for (int y = 0; y < _resY; y++)
                for (int x = 0; x < _resX; x++)
                {
                    float nx = x * frequency / _resX, ny = y * frequency / _resY, nz = z * frequency / _resZ;
                    float value = perlinValue.FractalBrownianMotion(nx, ny, nz, octaves) * valueAmplitude;
                    float drift = perlinDrift.FractalBrownianMotion(nx, ny, nz, octaves) * driftScale;
                    float diffusion = MathF.Abs(perlinDiff.FractalBrownianMotion(nx, ny, nz, octaves)) * diffusionScale + 0.05f;
                    int idx = Index(x, y, z);
                    _data[idx].CurrentValue = value;
                    _data[idx].Mean = value;
                    _data[idx].PreviousValue = value;
                    _data[idx].Drift = drift;
                    _data[idx].Diffusion = diffusion;
                    _data[idx].ProcessType = _processType;
                    _data[idx].Variance = diffusion * diffusion * 0.1f;
                    _data[idx].Entropy = MathF.Log(1.0f + _data[idx].Variance);
                }
    }

    // ── Euler-Maruyama ────────────────────────────────────────────────────

    /// <summary>Advances field by one step using Euler-Maruyama: dX = a dt + b dW.</summary>
    public void StepEulerMaruyama(float dt, float? globalDrift = null, float? globalDiffusion = null)
    {
        _timeStep = dt;
        int total = TotalCells;
        float sqrtDt = MathF.Sqrt(dt);
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _data[i];
            s.PreviousValue = s.CurrentValue;
            float mu = globalDrift ?? s.Drift, sigma = globalDiffusion ?? s.Diffusion;
            float dW = NextGaussian() * sqrtDt;
            switch (s.ProcessType)
            {
                case StochasticProcessType.GeometricBrownian:
                    s.CurrentValue = GBMEuler(s.CurrentValue, mu, sigma, dt, dW);
                    break;
                case StochasticProcessType.OrnsteinUhlenbeck:
                    s.CurrentValue = OUEuler(s.CurrentValue, s.Mean, s.MeanReversionSpeed > 0 ? s.MeanReversionSpeed : mu, sigma, dt, dW);
                    break;
                case StochasticProcessType.Poisson:
                    float lambda = MathF.Max(0.0001f, mu * s.CurrentValue * dt);
                    s.CurrentValue += NextPoisson(lambda);
                    break;
                case StochasticProcessType.JumpDiffusion:
                    s.CurrentValue = JumpDiffEuler(s.CurrentValue, mu, sigma, s.JumpIntensity > 0 ? s.JumpIntensity : 0.1f, dt, dW);
                    break;
                case StochasticProcessType.FractionalBrownian:
                    s.CurrentValue = FBMEuler(s.CurrentValue, mu, sigma, s.HurstExponent > 0 ? s.HurstExponent : 0.5f, dt, dW, ref s.FractionalMemory);
                    break;
            }
            s.Variance = 0.99f * s.Variance + 0.01f * sigma * sigma * dt;
            s.Entropy = MathF.Log(1.0f + s.Variance);
        }
        _currentTime += dt;
        _stepCount++;
    }

    // ── Milstein ──────────────────────────────────────────────────────────

    /// <summary>Advances field by one Milstein step (higher strong order).</summary>
    public void StepMilstein(float dt, float? globalDrift = null, float? globalDiffusion = null)
    {
        _timeStep = dt;
        int total = TotalCells;
        float sqrtDt = MathF.Sqrt(dt);
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _data[i];
            s.PreviousValue = s.CurrentValue;
            float mu = globalDrift ?? s.Drift, sigma = globalDiffusion ?? s.Diffusion;
            float dW = NextGaussian() * sqrtDt, dW2 = dW * dW;
            switch (s.ProcessType)
            {
                case StochasticProcessType.GeometricBrownian:
                    s.CurrentValue = GBMMilstein(s.CurrentValue, mu, sigma, dt, dW, dW2);
                    break;
                case StochasticProcessType.OrnsteinUhlenbeck:
                    float theta = s.MeanReversionSpeed > 0 ? s.MeanReversionSpeed : mu;
                    s.CurrentValue = OUEuler(s.CurrentValue, s.Mean, theta, sigma, dt, dW);
                    break;
                case StochasticProcessType.Poisson:
                    float lambda = MathF.Max(0.0001f, mu * s.CurrentValue * dt);
                    s.CurrentValue += NextPoisson(lambda);
                    break;
                case StochasticProcessType.JumpDiffusion:
                    s.CurrentValue = JumpDiffMilstein(s.CurrentValue, mu, sigma, s.JumpIntensity > 0 ? s.JumpIntensity : 0.1f, dt, dW, dW2);
                    break;
                case StochasticProcessType.FractionalBrownian:
                    s.CurrentValue = FBMEuler(s.CurrentValue, mu, sigma, s.HurstExponent > 0 ? s.HurstExponent : 0.5f, dt, dW, ref s.FractionalMemory);
                    break;
            }
            s.Variance = 0.99f * s.Variance + 0.01f * sigma * sigma * dt;
            s.Entropy = MathF.Log(1.0f + s.Variance);
        }
        _currentTime += dt;
        _stepCount++;
    }

    // ── Runge-Kutta-Maruyama ──────────────────────────────────────────────

    /// <summary>Advances field by 2-stage Runge-Kutta-Maruyama step.</summary>
    public void StepRungeKuttaMaruyama(float dt, float? globalDrift = null, float? globalDiffusion = null)
    {
        _timeStep = dt;
        int total = TotalCells;
        float sqrtDt = MathF.Sqrt(dt);
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _data[i];
            s.PreviousValue = s.CurrentValue;
            float mu = globalDrift ?? s.Drift, sigma = globalDiffusion ?? s.Diffusion;
            float dW1 = NextGaussian() * sqrtDt, dW2 = NextGaussian() * sqrtDt;
            float avgDW = 0.5f * (dW1 + dW2), x = s.CurrentValue;
            switch (s.ProcessType)
            {
                case StochasticProcessType.GeometricBrownian:
                    float drift1 = mu * x, diff1 = sigma * x, xMid = x + drift1 * dt + diff1 * dW1;
                    float drift2 = mu * xMid, diff2 = sigma * xMid;
                    s.CurrentValue = MathF.Max(x + 0.5f * (drift1 + drift2) * dt + 0.5f * (diff1 * dW1 + diff2 * dW2), 1e-10f);
                    break;
                case StochasticProcessType.OrnsteinUhlenbeck:
                    float theta = s.MeanReversionSpeed > 0 ? s.MeanReversionSpeed : mu;
                    float d1 = theta * (s.Mean - x), xM = x + d1 * dt + sigma * dW1, d2 = theta * (s.Mean - xM);
                    s.CurrentValue = x + 0.5f * (d1 + d2) * dt + sigma * avgDW;
                    break;
                case StochasticProcessType.Poisson:
                    s.CurrentValue += NextPoisson(MathF.Max(0.0001f, mu * x * dt));
                    break;
                case StochasticProcessType.JumpDiffusion:
                    float jd1 = mu * x, js1 = sigma * x, xJ = x + jd1 * dt + js1 * dW1;
                    float jd2 = mu * xJ, js2 = sigma * xJ;
                    s.CurrentValue = MathF.Max(x + 0.5f * (jd1 + jd2) * dt + 0.5f * (js1 * dW1 + js2 * dW2), 1e-10f);
                    if (NextPoisson((s.JumpIntensity > 0 ? s.JumpIntensity : 0.1f) * dt) > 0)
                        s.CurrentValue += NextGaussian() * sigma * MathF.Abs(s.CurrentValue);
                    break;
                default:
                    s.CurrentValue = x + mu * dt + sigma * avgDW;
                    break;
            }
            s.Variance = 0.99f * s.Variance + 0.01f * sigma * sigma * dt;
            s.Entropy = MathF.Log(1.0f + s.Variance);
        }
        _currentTime += dt;
        _stepCount++;
    }

    // ── Stratonovich ──────────────────────────────────────────────────────

    /// <summary>Advances field using Stratonovich interpretation (corrected drift).</summary>
    public void StepStratonovich(float dt, float? globalDrift = null, float? globalDiffusion = null)
    {
        _timeStep = dt;
        int total = TotalCells;
        float sqrtDt = MathF.Sqrt(dt);
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _data[i];
            s.PreviousValue = s.CurrentValue;
            float mu = globalDrift ?? s.Drift, sigma = globalDiffusion ?? s.Diffusion;
            float dW = NextGaussian() * sqrtDt;
            switch (s.ProcessType)
            {
                case StochasticProcessType.GeometricBrownian:
                    float itoDrift = mu + 0.5f * sigma * sigma;
                    s.CurrentValue = MathF.Max(s.CurrentValue * MathF.Exp((itoDrift - 0.5f * sigma * sigma) * dt + sigma * dW), 1e-10f);
                    break;
                case StochasticProcessType.OrnsteinUhlenbeck:
                    s.CurrentValue = OUEuler(s.CurrentValue, s.Mean, mu, sigma, dt, dW);
                    break;
                case StochasticProcessType.Poisson:
                    s.CurrentValue += NextPoisson(MathF.Max(0.0001f, mu * s.CurrentValue * dt));
                    break;
                case StochasticProcessType.JumpDiffusion:
                    float itoD = mu + 0.5f * sigma * sigma;
                    s.CurrentValue = MathF.Max(s.CurrentValue * MathF.Exp((itoD - 0.5f * sigma * sigma) * dt + sigma * dW), 1e-10f);
                    if (NextPoisson(0.1f * dt) > 0)
                        s.CurrentValue += NextGaussian() * sigma * MathF.Abs(s.CurrentValue);
                    break;
                default:
                    s.CurrentValue += mu * dt + sigma * dW;
                    break;
            }
            s.Variance = 0.99f * s.Variance + 0.01f * sigma * sigma * dt;
            s.Entropy = MathF.Log(1.0f + s.Variance);
        }
        _currentTime += dt;
        _stepCount++;
    }

    // ── Unified Step ──────────────────────────────────────────────────────

    /// <summary>Advances field by one step using specified scheme.</summary>
    public void Step(float dt, SDEScheme scheme = SDEScheme.EulerMaruyama, float? globalDrift = null, float? globalDiffusion = null)
    {
        switch (scheme)
        {
            case SDEScheme.EulerMaruyama:
                StepEulerMaruyama(dt, globalDrift, globalDiffusion);
                break;
            case SDEScheme.Milstein:
                StepMilstein(dt, globalDrift, globalDiffusion);
                break;
            case SDEScheme.RungeKuttaMaruyama:
                StepRungeKuttaMaruyama(dt, globalDrift, globalDiffusion);
                break;
            case SDEScheme.Stratonovich:
                StepStratonovich(dt, globalDrift, globalDiffusion);
                break;
            default:
                StepEulerMaruyama(dt, globalDrift, globalDiffusion);
                break;
        }
    }

    // ── Correlation Propagation ───────────────────────────────────────────

    /// <summary>Propagates spatial correlation via 6-point nearest-neighbor coupling.</summary>
    public void PropagateCorrelation(float couplingStrength)
    {
        int total = TotalCells;
        for (int i = 0; i < total; i++)
            _scratchBuffer[i] = _data[i].CurrentValue;
        for (int z = 1; z < _resZ - 1; z++)
            for (int y = 1; y < _resY - 1; y++)
                for (int x = 1; x < _resX - 1; x++)
                {
                    int idx = (z * _resY + y) * _resX + x;
                    float c = _scratchBuffer[idx];
                    float avg = (_scratchBuffer[idx - 1] + _scratchBuffer[idx + 1] + _scratchBuffer[idx - _resX] + _scratchBuffer[idx + _resX] + _scratchBuffer[idx - _resX * _resY] + _scratchBuffer[idx + _resX * _resY]) / 6.0f;
                    _data[idx].CurrentValue = c + couplingStrength * (avg - c);
                    _data[idx].Correlation = 1.0f - MathF.Abs(c - avg) / (MathF.Abs(c) + 1e-6f);
                }
    }

    /// <summary>Anisotropic Laplacian coupling with different rates per axis.</summary>
    public void PropagateAnisotropicCorrelation(float couplingX, float couplingY, float couplingZ)
    {
        int total = TotalCells;
        for (int i = 0; i < total; i++)
            _scratchBuffer[i] = _data[i].CurrentValue;
        for (int z = 1; z < _resZ - 1; z++)
            for (int y = 1; y < _resY - 1; y++)
                for (int x = 1; x < _resX - 1; x++)
                {
                    int idx = (z * _resY + y) * _resX + x;
                    float c = _scratchBuffer[idx];
                    float lap = couplingX * (_scratchBuffer[idx - 1] + _scratchBuffer[idx + 1] - 2 * c) + couplingY * (_scratchBuffer[idx - _resX] + _scratchBuffer[idx + _resX] - 2 * c) + couplingZ * (_scratchBuffer[idx - _resX * _resY] + _scratchBuffer[idx + _resX * _resY] - 2 * c);
                    _data[idx].CurrentValue = c + 0.5f * lap;
                }
    }

    /// <summary>Gaussian-weighted spatial correlation.</summary>
    public void PropagateGaussianCorrelation(float correlationLength)
    {
        int total = TotalCells, radius = (int)MathF.Ceiling(correlationLength * 2.0f);
        float sigma2 = 2.0f * correlationLength * correlationLength;
        for (int i = 0; i < total; i++)
            _scratchBuffer[i] = _data[i].CurrentValue;
        for (int z = radius; z < _resZ - radius; z++)
            for (int y = radius; y < _resY - radius; y++)
                for (int x = radius; x < _resX - radius; x++)
                {
                    int idx = (z * _resY + y) * _resX + x;
                    float centerVal = _scratchBuffer[idx], weightedSum = 0, weightSum = 0;
                    for (int dz = -radius; dz <= radius; dz++)
                        for (int dy = -radius; dy <= radius; dy++)
                            for (int dx = -radius; dx <= radius; dx++)
                            {
                                float dist2 = (float)(dx * dx + dy * dy + dz * dz);
                                float w = MathF.Exp(-dist2 / sigma2);
                                weightedSum += w * _scratchBuffer[((z + dz) * _resY + (y + dy)) * _resX + (x + dx)];
                                weightSum += w;
                            }
                    _data[idx].CurrentValue = weightedSum / (weightSum + 1e-10f);
                    _data[idx].Correlation = 1.0f - MathF.Abs(centerVal - _data[idx].CurrentValue) / (MathF.Abs(centerVal) + 1e-6f);
                }
    }

    // ── Statistics ────────────────────────────────────────────────────────

    /// <summary>Computes comprehensive field statistics.</summary>
    public StochasticFieldStats GetStats()
    {
        int total = TotalCells;
        float minVal = float.MaxValue, maxVal = float.MinValue, sum = 0, sumSq = 0, sum4th = 0;
        for (int i = 0; i < total; i++)
        { float v = _data[i].CurrentValue; minVal = MathF.Min(minVal, v); maxVal = MathF.Max(maxVal, v); sum += v; sumSq += v * v; sum4th += v * v * v * v; }
        float mean = sum / total, variance = sumSq / total - mean * mean, stdDev = MathF.Sqrt(MathF.Max(variance, 0));
        float skewness = 0, kurtosis = 0;
        if (stdDev > 1e-10f)
        { float invStd = 1 / stdDev, sumSkew = 0; for (int i = 0; i < total; i++) { float z = (_data[i].CurrentValue - mean) * invStd; sumSkew += z * z * z; } skewness = sumSkew / total; kurtosis = sum4th / (variance * variance + 1e-20f) - 3.0f; }
        float entropy = 0;
        for (int i = 0; i < total; i++)
        { float p = MathF.Abs(_data[i].CurrentValue) / (sum + 1e-10f); if (p > 1e-10f) entropy -= p * MathF.Log(p); }
        Span<float> values = stackalloc float[Math.Min(total, 10000)];
        int sampleCount = Math.Min(total, 10000), step = Math.Max(1, total / sampleCount);
        for (int i = 0; i < sampleCount; i++)
            values[i] = _data[i * step].CurrentValue;
        values.Sort();
        float median = values[values.Length / 2], q1 = values[values.Length / 4], q3 = values[3 * values.Length / 4];
        return new StochasticFieldStats { ProcessType = _processType, TotalCells = total, MinValue = minVal, MaxValue = maxVal, Mean = mean, Variance = variance, StdDev = stdDev, Skewness = skewness, Kurtosis = kurtosis, Entropy = entropy, TimeStep = _currentTime, Sum = sum, SumOfSquares = sumSq, Median = median, IQR = q3 - q1 };
    }

    /// <summary>Computes spatial gradient at cell via central differences.</summary>
    public (float Gx, float Gy, float Gz) ComputeGradient(int x, int y, int z)
    {
        float gx = 0, gy = 0, gz = 0;
        if (x > 0 && x < _resX - 1)
            gx = (_data[Index(x + 1, y, z)].CurrentValue - _data[Index(x - 1, y, z)].CurrentValue) / (2 * _spacingX);
        if (y > 0 && y < _resY - 1)
            gy = (_data[Index(x, y + 1, z)].CurrentValue - _data[Index(x, y - 1, z)].CurrentValue) / (2 * _spacingY);
        if (z > 0 && z < _resZ - 1)
            gz = (_data[Index(x, y, z + 1)].CurrentValue - _data[Index(x, y, z - 1)].CurrentValue) / (2 * _spacingZ);
        return (gx, gy, gz);
    }

    /// <summary>Computes Laplacian at cell.</summary>
    public float ComputeLaplacian(int x, int y, int z)
    {
        float c = _data[Index(x, y, z)].CurrentValue, lapX = 0, lapY = 0, lapZ = 0;
        if (x > 0 && x < _resX - 1)
            lapX = (_data[Index(x + 1, y, z)].CurrentValue + _data[Index(x - 1, y, z)].CurrentValue - 2 * c) / (_spacingX * _spacingX);
        if (y > 0 && y < _resY - 1)
            lapY = (_data[Index(x, y + 1, z)].CurrentValue + _data[Index(x, y - 1, z)].CurrentValue - 2 * c) / (_spacingY * _spacingY);
        if (z > 0 && z < _resZ - 1)
            lapZ = (_data[Index(x, y, z + 1)].CurrentValue + _data[Index(x, y, z - 1)].CurrentValue - 2 * c) / (_spacingZ * _spacingZ);
        return lapX + lapY + lapZ;
    }

    /// <summary>Computes Laplacian field into output buffer.</summary>
    public void ComputeLaplacianField(float* output)
    { for (int z = 0; z < _resZ; z++) for (int y = 0; y < _resY; y++) for (int x = 0; x < _resX; x++) output[Index(x, y, z)] = ComputeLaplacian(x, y, z); }

    /// <summary>Computes divergence of vector field.</summary>
    public void ComputeDivergence(float* vx, float* vy, float* vz, float* output)
    {
        for (int z = 1; z < _resZ - 1; z++)
            for (int y = 1; y < _resY - 1; y++)
                for (int x = 1; x < _resX - 1; x++)
                {
                    int idx = Index(x, y, z);
                    output[idx] = (vx[Index(x + 1, y, z)] - vx[Index(x - 1, y, z)]) / (2 * _spacingX) + (vy[Index(x, y + 1, z)] - vy[Index(x, y - 1, z)]) / (2 * _spacingY) + (vz[Index(x, y, z + 1)] - vz[Index(x, y, z - 1)]) / (2 * _spacingZ);
                }
    }

    // ── Markov Chain Step ─────────────────────────────────────────────────

    /// <summary>Advances Markov chain on the field.</summary>
    public void StepMarkovChain(float* transitionMatrix, int numStates)
    {
        int total = TotalCells;
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _data[i];
            s.PreviousValue = s.CurrentValue;
            int currentState = Math.Clamp(s.MarkovState, 0, numStates - 1);
            float u = (float)_rng.NextDouble(), cumProb = 0;
            int newState = currentState;
            for (int j = 0; j < numStates; j++)
            { cumProb += transitionMatrix[currentState * numStates + j]; if (u <= cumProb) { newState = j; break; } }
            s.MarkovState = newState;
            s.CurrentValue = (float)newState / (numStates - 1);
            s.Mean = s.CurrentValue;
            s.Variance = 0;
        }
        _currentTime += _timeStep;
        _stepCount++;
    }

    // ── Internal Helpers ──────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float GBMEuler(float x, float mu, float sigma, float dt, float dW) => x * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float GBMMilstein(float x, float mu, float sigma, float dt, float dW, float dW2) => x * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float OUEuler(float x, float mean, float theta, float sigma, float dt, float dW) => x + theta * (mean - x) * dt + sigma * dW;
    private float JumpDiffEuler(float x, float mu, float sigma, float jumpLambda, float dt, float dW)
    {
        float result = x * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
        int jumps = NextPoisson(jumpLambda * dt);
        for (int j = 0; j < jumps; j++)
            result += NextGaussian() * sigma * MathF.Abs(x) * 0.5f;
        return MathF.Max(result, 1e-10f);
    }
    private float JumpDiffMilstein(float x, float mu, float sigma, float jumpLambda, float dt, float dW, float dW2)
    {
        float result = x * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
        int jumps = NextPoisson(jumpLambda * dt);
        for (int j = 0; j < jumps; j++)
            result += NextGaussian() * sigma * MathF.Abs(x) * 0.5f;
        return MathF.Max(result, 1e-10f);
    }
    private static float FBMEuler(float x, float mu, float sigma, float H, float dt, float dW, ref float memory)
    {
        float inc = dW * MathF.Pow(dt, H - 0.5f);
        memory = 0.5f * memory + 0.5f * inc;
        return x + mu * x * dt + sigma * (0.7f * dW + 0.3f * memory);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NextGaussian()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int NextPoisson(float lambda)
    {
        if (lambda <= 0)
            return 0;
        if (lambda > 30)
        { float n = NextGaussian(); return Math.Max(0, (int)(lambda + MathF.Sqrt(lambda) * n + 0.5f)); }
        float L = MathF.Exp(-lambda);
        int k = 0;
        float p = 1;
        do
        { k++; p *= (float)_rng.NextDouble(); } while (p > L);
        return k - 1;
    }
    private static StochasticFieldState LerpTrilinear(StochasticFieldState s000, StochasticFieldState s100, StochasticFieldState s010, StochasticFieldState s110, StochasticFieldState s001, StochasticFieldState s101, StochasticFieldState s011, StochasticFieldState s111, float tx, float ty, float tz)
    {
        float x0v = s000.CurrentValue + (s100.CurrentValue - s000.CurrentValue) * tx, x1v = s010.CurrentValue + (s110.CurrentValue - s010.CurrentValue) * tx;
        float x2v = s001.CurrentValue + (s101.CurrentValue - s001.CurrentValue) * tx, x3v = s011.CurrentValue + (s111.CurrentValue - s011.CurrentValue) * tx;
        float y0v = x0v + (x1v - x0v) * ty, y1v = x2v + (x3v - x2v) * ty;
        float val = y0v + (y1v - y0v) * tz;
        return new StochasticFieldState { Mean = (s000.Mean + s100.Mean + s010.Mean + s110.Mean + s001.Mean + s101.Mean + s011.Mean + s111.Mean) * 0.125f, Variance = (s000.Variance + s100.Variance + s010.Variance + s110.Variance + s001.Variance + s101.Variance + s011.Variance + s111.Variance) * 0.125f, CurrentValue = val, Drift = (s000.Drift + s111.Drift) * 0.5f, Diffusion = (s000.Diffusion + s111.Diffusion) * 0.5f };
    }

    /// <summary>Frees all unmanaged memory.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        if (_data != null)
            NativeMemory.AlignedFree(_data);
        if (_scratchBuffer != null)
            NativeMemory.AlignedFree(_scratchBuffer);
        if (_correlationBuffer != null)
            NativeMemory.AlignedFree(_correlationBuffer);
        _data = null;
        _scratchBuffer = null;
        _correlationBuffer = null;
        _disposed = true;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 5 — STOCHASTIC PROCESS IMPLEMENTATIONS
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>Interface for standalone stochastic processes.</summary>
public interface IStochasticProcess
{
    /// <summary>Process type.</summary>
    StochasticProcessType ProcessType { get; }
    /// <summary>Current value.</summary>
    float CurrentValue { get; }
    /// <summary>Euler-Maruyama step.</summary>
    float StepEulerMaruyama(float dt, float dW);
    /// <summary>Milstein step.</summary>
    float StepMilstein(float dt, float dW, float dW2);
    /// <summary>Reset to initial conditions.</summary>
    void Reset();
}

/// <summary>
/// Geometric Brownian Motion: dX = μX dt + σX dW.
/// Exact solution: X(t) = X(0) exp((μ−σ²/2)t + σW(t)).
/// </summary>
public sealed class GeometricBrownianMotion : IStochasticProcess
{
    private readonly float _mu, _sigma, _x0;
    private float _x;
    public float Mu => _mu;
    public float Sigma => _sigma;
    public float X0 => _x0;
    public StochasticProcessType ProcessType => StochasticProcessType.GeometricBrownian;
    public float CurrentValue => _x;

    public GeometricBrownianMotion(float x0, float mu, float sigma)
    { _x0 = x0; _x = x0; _mu = mu; _sigma = sigma; }

    public float StepEulerMaruyama(float dt, float dW)
    { _x = _x * MathF.Exp((_mu - 0.5f * _sigma * _sigma) * dt + _sigma * dW); _x = MathF.Max(_x, 1e-10f); return _x; }

    public float StepMilstein(float dt, float dW, float dW2)
    { _x = _x * MathF.Exp((_mu - 0.5f * _sigma * _sigma) * dt + _sigma * dW); _x = MathF.Max(_x, 1e-10f); return _x; }

    /// <summary>Generates an exact path from 0 to T.</summary>
    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        var path = new StochasticPath(steps + 1);
        float dt = T / steps;
        path.Times[0] = 0;
        path.Values[0] = _x0;
        float x = _x0;
        for (int i = 1; i <= steps; i++)
        { float dW = MathF.Sqrt(dt) * NormalSample(rng); x = x * MathF.Exp((_mu - 0.5f * _sigma * _sigma) * dt + _sigma * dW); path.Times[i] = i * dt; path.Values[i] = MathF.Max(x, 1e-10f); }
        return path;
    }

    public float TheoreticalMean(float t) => _x0 * MathF.Exp(_mu * t);
    public float TheoreticalVariance(float t) => _x0 * _x0 * MathF.Exp(2 * _mu * t) * (MathF.Exp(_sigma * _sigma * t) - 1);
    public void Reset() => _x = _x0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalSample(Random rng)
    { float u1 = 1f - (float)rng.NextDouble(), u2 = 1f - (float)rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
}

/// <summary>
/// Ornstein-Uhlenbeck: dX = θ(μ−X) dt + σ dW.
/// Stationary distribution: N(μ, σ²/(2θ)).
/// </summary>
public sealed class OrnsteinUhlenbeckProcess : IStochasticProcess
{
    private readonly float _theta, _mu, _sigma, _x0;
    private float _x;
    public float Theta => _theta;
    public float Mu => _mu;
    public float Sigma => _sigma;
    public StochasticProcessType ProcessType => StochasticProcessType.OrnsteinUhlenbeck;
    public float CurrentValue => _x;

    public OrnsteinUhlenbeckProcess(float x0, float theta, float mu, float sigma)
    { _x0 = x0; _x = x0; _theta = theta; _mu = mu; _sigma = sigma; }

    public float StepEulerMaruyama(float dt, float dW)
    { _x = _x + _theta * (_mu - _x) * dt + _sigma * dW; return _x; }

    public float StepMilstein(float dt, float dW, float dW2)
    { _x = _x + _theta * (_mu - _x) * dt + _sigma * dW; return _x; }

    /// <summary>Generates exact path using stationary transition kernel.</summary>
    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        var path = new StochasticPath(steps + 1);
        float dt = T / steps;
        path.Times[0] = 0;
        path.Values[0] = _x0;
        float x = _x0;
        for (int i = 1; i <= steps; i++)
        {
            float expThetaDt = MathF.Exp(-_theta * dt);
            float mean = _mu + (x - _mu) * expThetaDt;
            float variance = _sigma * _sigma / (2 * _theta) * (1 - expThetaDt * expThetaDt);
            x = mean + MathF.Sqrt(MathF.Max(variance, 0)) * NormalSample(rng);
            path.Times[i] = i * dt;
            path.Values[i] = x;
        }
        return path;
    }

    public float ExactSolution(float t, Random rng)
    { float expThetaT = MathF.Exp(-_theta * t); float mean = _mu + (_x0 - _mu) * expThetaT; float variance = _sigma * _sigma / (2 * _theta) * (1 - expThetaT * expThetaT); return mean + MathF.Sqrt(MathF.Max(variance, 0)) * NormalSample(rng); }

    public float StationaryVariance => _sigma * _sigma / (2 * _theta);
    public float Autocorrelation(float t) => MathF.Exp(-_theta * t);
    public void Reset() => _x = _x0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalSample(Random rng)
    { float u1 = 1f - (float)rng.NextDouble(), u2 = 1f - (float)rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
}

/// <summary>Poisson arrival process with exponentially distributed inter-arrival times.</summary>
public sealed class PoissonArrivalProcess : IStochasticProcess
{
    private readonly float _lambda;
    private float _x, _timeToNextEvent, _totalTime;
    private readonly Random _rng;
    public float Lambda => _lambda;
    public float CurrentValue => _x;
    public StochasticProcessType ProcessType => StochasticProcessType.Poisson;

    public PoissonArrivalProcess(float lambda, int seed = 42)
    { _lambda = lambda; _x = 0; _rng = new Random(seed); _timeToNextEvent = SampleExp(); _totalTime = 0; }

    public float StepEulerMaruyama(float dt, float dW)
    {
        int count = 0;
        _timeToNextEvent -= dt;
        while (_timeToNextEvent <= 0)
        { count++; _timeToNextEvent += SampleExp(); }
        _x += count;
        _totalTime += dt;
        return _x;
    }

    public float StepMilstein(float dt, float dW, float dW2) => StepEulerMaruyama(dt, dW);

    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        var path = new StochasticPath(steps + 1);
        float dt = T / steps, x = 0, timeToNext = SampleExp(rng);
        path.Times[0] = 0;
        path.Values[0] = 0;
        for (int i = 1; i <= steps; i++)
        { float t = i * dt; while (timeToNext <= t) { x += 1; timeToNext += SampleExp(rng); } path.Times[i] = t; path.Values[i] = x; }
        return path;
    }

    public float ExpectedInterArrivalTime => 1 / _lambda;
    public float VarianceInInterval(float T) => _lambda * T;
    private float SampleExp() => -MathF.Log(1 - (float)_rng.NextDouble()) / _lambda;
    private static float SampleExp(Random rng) => -MathF.Log(1 - (float)rng.NextDouble());
    public void Reset() { _x = 0; _totalTime = 0; _timeToNextEvent = SampleExp(); }
}

/// <summary>
/// Merton Jump-Diffusion: dX = μX dt + σX dW + J dN.
/// Combines GBM with compound Poisson jumps.
/// </summary>
public sealed class JumpDiffusionProcess : IStochasticProcess
{
    private readonly float _mu, _sigma, _jumpIntensity, _jumpMean, _jumpStdDev, _x0;
    private float _x;
    private readonly Random _rng;
    public float Mu => _mu;
    public float Sigma => _sigma;
    public float JumpIntensity => _jumpIntensity;
    public StochasticProcessType ProcessType => StochasticProcessType.JumpDiffusion;
    public float CurrentValue => _x;

    public JumpDiffusionProcess(float x0, float mu, float sigma, float jumpIntensity,
        float jumpMean = 0, float jumpStdDev = 0.1f, int seed = 42)
    { _x0 = x0; _x = x0; _mu = mu; _sigma = sigma; _jumpIntensity = jumpIntensity; _jumpMean = jumpMean; _jumpStdDev = jumpStdDev; _rng = new Random(seed); }

    public float StepEulerMaruyama(float dt, float dW)
    {
        _x = _x * MathF.Exp((_mu - 0.5f * _sigma * _sigma) * dt + _sigma * dW);
        int numJumps = PoissonSample(_jumpIntensity * dt);
        for (int i = 0; i < numJumps; i++)
            _x *= MathF.Exp(_jumpMean + _jumpStdDev * NormalSample());
        _x = MathF.Max(_x, 1e-10f);
        return _x;
    }

    public float StepMilstein(float dt, float dW, float dW2) => StepEulerMaruyama(dt, dW);

    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        var path = new StochasticPath(steps + 1);
        float dt = T / steps;
        path.Times[0] = 0;
        path.Values[0] = _x0;
        float x = _x0;
        for (int i = 1; i <= steps; i++)
        {
            float dW = MathF.Sqrt(dt) * NormalSample(rng);
            x = x * MathF.Exp((_mu - 0.5f * _sigma * _sigma) * dt + _sigma * dW);
            int jumps = PoissonSample(_jumpIntensity * dt, rng);
            for (int j = 0; j < jumps; j++)
                x *= MathF.Exp(_jumpMean + _jumpStdDev * NormalSample(rng));
            path.Times[i] = i * dt;
            path.Values[i] = MathF.Max(x, 1e-10f);
        }
        return path;
    }

    public float JumpDriftCorrection() => _jumpIntensity * (MathF.Exp(_jumpMean + 0.5f * _jumpStdDev * _jumpStdDev) - 1);
    public float TotalVariance() => _sigma * _sigma + _jumpIntensity * (_jumpStdDev * _jumpStdDev + _jumpMean * _jumpMean);
    public void Reset() => _x = _x0;

    private float NormalSample() { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
    private static float NormalSample(Random rng) { float u1 = 1f - (float)rng.NextDouble(), u2 = 1f - (float)rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
    private int PoissonSample(float lambda) => PoissonSample(lambda, _rng);
    private static int PoissonSample(float lambda, Random rng) { if (lambda <= 0) return 0; float L = MathF.Exp(-lambda); int k = 0; float p = 1; do { k++; p *= (float)rng.NextDouble(); } while (p > L); return k - 1; }
}

/// <summary>
/// Markov chain with finite state space and configurable transition matrix.
/// </summary>
public sealed unsafe class MarkovChainProcess : IStochasticProcess
{
    private readonly float* _transitionMatrix;
    private readonly int _numStates, _initialState;
    private int _currentState;
    private readonly Random _rng;
    public int NumStates => _numStates;
    public int CurrentState => _currentState;
    public StochasticProcessType ProcessType => StochasticProcessType.MarkovChain;
    public float CurrentValue => (float)_currentState / MathF.Max(_numStates - 1, 1);

    public MarkovChainProcess(float* transitionMatrix, int numStates, int initialState = 0, int seed = 42)
    { _transitionMatrix = transitionMatrix; _numStates = numStates; _currentState = initialState; _initialState = initialState; _rng = new Random(seed); }

    public float StepEulerMaruyama(float dt, float dW)
    {
        float u = (float)_rng.NextDouble(), cumProb = 0;
        int rowOffset = _currentState * _numStates;
        for (int j = 0; j < _numStates; j++)
        { cumProb += _transitionMatrix[rowOffset + j]; if (u <= cumProb) { _currentState = j; return CurrentValue; } }
        _currentState = _numStates - 1;
        return CurrentValue;
    }

    public float StepMilstein(float dt, float dW, float dW2) => StepEulerMaruyama(dt, dW);

    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        var path = new StochasticPath(steps + 1);
        float dt = T / steps;
        int state = _initialState;
        path.Times[0] = 0;
        path.Values[0] = state;
        for (int i = 1; i <= steps; i++)
        { float u = (float)rng.NextDouble(), cumProb = 0; int newState = state; for (int j = 0; j < _numStates; j++) { cumProb += _transitionMatrix[state * _numStates + j]; if (u <= cumProb) { newState = j; break; } } state = newState; path.Times[i] = i * dt; path.Values[i] = state; }
        return path;
    }

    /// <summary>Stationary distribution via power iteration.</summary>
    public float[] StationaryDistribution(int maxIter = 1000, float tolerance = 1e-8f)
    {
        float[] pi = new float[_numStates], piNew = new float[_numStates];
        pi[0] = 1;
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int j = 0; j < _numStates; j++)
            { float sum = 0; for (int i = 0; i < _numStates; i++) sum += pi[i] * _transitionMatrix[i * _numStates + j]; piNew[j] = sum; }
            float diff = 0;
            for (int j = 0; j < _numStates; j++)
            { diff += MathF.Abs(piNew[j] - pi[j]); pi[j] = piNew[j]; }
            if (diff < tolerance)
                break;
        }
        return pi;
    }

    public void Reset() => _currentState = _initialState;
}

/// <summary>
/// Fractional Brownian Motion with Hurst parameter H in (0,1).
/// Uses Hosking's method for exact simulation.
/// </summary>
public sealed class FractionalBrownianMotionProcess : IStochasticProcess
{
    private readonly float _hurst, _sigma;
    private float _x;
    private readonly float[] _pastIncrements;
    private int _incrementIndex;
    private readonly int _maxMemory;
    public float Hurst => _hurst;
    public float Sigma => _sigma;
    public StochasticProcessType ProcessType => StochasticProcessType.FractionalBrownian;
    public float CurrentValue => _x;

    public FractionalBrownianMotionProcess(float sigma, float hurst, int memoryLength = 100)
    { _hurst = Math.Clamp(hurst, 0.01f, 0.99f); _sigma = sigma; _x = 0; _maxMemory = memoryLength; _pastIncrements = new float[memoryLength]; _incrementIndex = 0; }

    public float StepEulerMaruyama(float dt, float dW)
    {
        float H = _hurst, memoryWeight = MathF.Pow(dt, H - 0.5f), memoryContrib = 0;
        int count = Math.Min(_incrementIndex, _maxMemory);
        for (int i = 0; i < count; i++)
        { float lag = count - i; memoryContrib += _pastIncrements[i] * MathF.Pow(lag, 2 * H - 2); }
        float increment = dW * memoryWeight + 0.1f * memoryContrib * dt;
        _pastIncrements[_incrementIndex % _maxMemory] = dW;
        _incrementIndex++;
        _x += _sigma * increment;
        return _x;
    }

    public float StepMilstein(float dt, float dW, float dW2) => StepEulerMaruyama(dt, dW);

    /// <summary>Generates exact fBm path using Hosking's method.</summary>
    public StochasticPath GeneratePath(float T, int steps, Random rng)
    {
        float dt = T / steps;
        var path = new StochasticPath(steps + 1);
        path.Times[0] = 0;
        path.Values[0] = 0;
        float[] z = new float[steps];
        for (int i = 0; i < steps; i++)
        { float u1 = 1f - (float)rng.NextDouble(), u2 = 1f - (float)rng.NextDouble(); z[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
        float[] rho = new float[steps + 1];
        for (int k = 0; k <= steps; k++)
            rho[k] = 0.5f * (MathF.Pow(k + 1, 2 * _hurst) - 2 * MathF.Pow(k, 2 * _hurst) + MathF.Pow(MathF.Max(k - 1, 0), 2 * _hurst));
        float x = 0;
        for (int n = 0; n < steps; n++)
        {
            float condMean = 0, condVar = 1;
            if (n > 0)
                for (int j = 0; j < n; j++)
                { float phi = rho[n - j] / MathF.Max(rho[0], 1e-10f); condMean += phi * z[j]; condVar -= phi * phi; }
            condVar = MathF.Sqrt(MathF.Max(condVar, 1e-10f));
            x = _sigma * (condMean + condVar * z[n]);
            path.Values[n + 1] = x;
            path.Times[n + 1] = (n + 1) * dt;
        }
        return path;
    }

    public void Reset() { _x = 0; _incrementIndex = 0; }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 6 — STOCHASTIC DIFFERENTIAL EQUATION SOLVERS
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Comprehensive SDE solver supporting multiple numerical schemes and convergence analysis.
/// </summary>
public sealed class StochasticDifferentialSolver
{
    public delegate float DriftFunction(float x, float t);
    public delegate float DiffusionFunction(float x, float t);
    public delegate float DiffusionDerivative(float x, float t);

    private readonly Random _rng;
    public StochasticDifferentialSolver(int seed = 42) { _rng = new Random(seed); }

    /// <summary>Solves SDE via Euler-Maruyama. Returns terminal value X(T).</summary>
    public float SolveEulerMaruyama(DriftFunction drift, DiffusionFunction diffusion, float x0, float T, int N)
    {
        float dt = T / N, x = x0;
        for (int i = 0; i < N; i++)
        { float t = i * dt; x += drift(x, t) * dt + diffusion(x, t) * MathF.Sqrt(dt) * NormalSample(); }
        return x;
    }

    /// <summary>Solves SDE and returns full path.</summary>
    public StochasticPath SolveEulerMaruyamaPath(DriftFunction drift, DiffusionFunction diffusion, float x0, float T, int N)
    {
        var path = new StochasticPath(N + 1);
        float dt = T / N;
        path.Times[0] = 0;
        path.Values[0] = x0;
        float x = x0;
        for (int i = 0; i < N; i++)
        { float t = i * dt; x += drift(x, t) * dt + diffusion(x, t) * MathF.Sqrt(dt) * NormalSample(); path.Times[i + 1] = (i + 1) * dt; path.Values[i + 1] = x; }
        return path;
    }

    /// <summary>Solves SDE via Milstein scheme (order 1.0 strong convergence).</summary>
    public float SolveMilstein(DriftFunction drift, DiffusionFunction diffusion, DiffusionDerivative diffDeriv, float x0, float T, int N)
    {
        float dt = T / N, sqrtDt = MathF.Sqrt(dt), x = x0;
        for (int i = 0; i < N; i++)
        { float t = i * dt, dW = sqrtDt * NormalSample(); x += drift(x, t) * dt + diffusion(x, t) * dW + 0.5f * diffusion(x, t) * diffDeriv(x, t) * (dW * dW - dt); }
        return x;
    }

    /// <summary>Solves via Milstein, returns full path.</summary>
    public StochasticPath SolveMilsteinPath(DriftFunction drift, DiffusionFunction diffusion, DiffusionDerivative diffDeriv, float x0, float T, int N)
    {
        var path = new StochasticPath(N + 1);
        float dt = T / N, sqrtDt = MathF.Sqrt(dt);
        path.Times[0] = 0;
        path.Values[0] = x0;
        float x = x0;
        for (int i = 0; i < N; i++)
        { float t = i * dt, dW = sqrtDt * NormalSample(); x += drift(x, t) * dt + diffusion(x, t) * dW + 0.5f * diffusion(x, t) * diffDeriv(x, t) * (dW * dW - dt); path.Times[i + 1] = (i + 1) * dt; path.Values[i + 1] = x; }
        return path;
    }

    /// <summary>Solves SDE via 2-stage Runge-Kutta-Maruyama.</summary>
    public float SolveRungeKuttaMaruyama(DriftFunction drift, DiffusionFunction diffusion, float x0, float T, int N)
    {
        float dt = T / N, sqrtDt = MathF.Sqrt(dt), x = x0;
        for (int i = 0; i < N; i++)
        {
            float t = i * dt, dW1 = sqrtDt * NormalSample(), dW2 = sqrtDt * NormalSample();
            float k1a = drift(x, t), k1b = diffusion(x, t);
            float xMid = x + k1a * dt + k1b * dW1;
            float k2a = drift(xMid, t + dt), k2b = diffusion(xMid, t + dt);
            x += 0.5f * (k1a + k2a) * dt + 0.5f * (k1b * dW1 + k2b * dW2);
        }
        return x;
    }

    /// <summary>Estimates strong convergence order by comparing solutions at different resolutions.</summary>
    public (float[] DtValues, float[] Errors, float ConvergenceOrder) EstimateStrongConvergence(
        DriftFunction drift, DiffusionFunction diffusion, DiffusionDerivative? diffDeriv,
        float x0, float T, int[] numStepsArray, int numMCPaths, SDEScheme scheme)
    {
        float[] dtValues = new float[numStepsArray.Length], errors = new float[numStepsArray.Length];
        int fineN = numStepsArray[^1] * 4;
        float[] refTerminals = new float[numMCPaths];
        for (int p = 0; p < numMCPaths; p++)
            refTerminals[p] = scheme switch { SDEScheme.Milstein when diffDeriv != null => SolveMilstein(drift, diffusion, diffDeriv, x0, T, fineN), SDEScheme.RungeKuttaMaruyama => SolveRungeKuttaMaruyama(drift, diffusion, x0, T, fineN), _ => SolveEulerMaruyama(drift, diffusion, x0, T, fineN) };
        for (int s = 0; s < numStepsArray.Length; s++)
        {
            int N = numStepsArray[s];
            dtValues[s] = T / N;
            float sumError = 0;
            for (int p = 0; p < numMCPaths; p++)
            { float approx = scheme switch { SDEScheme.Milstein when diffDeriv != null => SolveMilstein(drift, diffusion, diffDeriv, x0, T, N), SDEScheme.RungeKuttaMaruyama => SolveRungeKuttaMaruyama(drift, diffusion, x0, T, N), _ => SolveEulerMaruyama(drift, diffusion, x0, T, N) }; sumError += MathF.Abs(approx - refTerminals[p]); }
            errors[s] = sumError / numMCPaths;
        }
        // Estimate order: log(E1/E2) / log(dt1/dt2)
        float order = 0;
        if (dtValues.Length >= 2 && errors[^1] > 1e-10f && errors[0] > 1e-10f)
            order = MathF.Log(errors[0] / errors[^1]) / MathF.Log(dtValues[^1] / dtValues[0]);
        return (dtValues, errors, order);
    }

    /// <summary>Estimates weak convergence by comparing mean terminal distributions.</summary>
    public (float[] DtValues, float[] Errors, float ConvergenceOrder) EstimateWeakConvergence(
        DriftFunction drift, DiffusionFunction diffusion, float x0, float T, int[] numStepsArray, int numMCPaths, SDEScheme scheme)
    {
        float[] dtValues = new float[numStepsArray.Length], errors = new float[numStepsArray.Length];
        int fineN = numStepsArray[^1] * 4;
        float[] refTerminals = new float[numMCPaths];
        for (int p = 0; p < numMCPaths; p++)
            refTerminals[p] = SolveEulerMaruyama(drift, diffusion, x0, T, fineN);
        float refMean = 0;
        for (int p = 0; p < numMCPaths; p++)
            refMean += refTerminals[p];
        refMean /= numMCPaths;
        for (int s = 0; s < numStepsArray.Length; s++)
        {
            int N = numStepsArray[s];
            dtValues[s] = T / N;
            float sum = 0;
            for (int p = 0; p < numMCPaths; p++)
                sum += SolveEulerMaruyama(drift, diffusion, x0, T, N);
            errors[s] = MathF.Abs(sum / numMCPaths - refMean);
        }
        float order = 0;
        if (dtValues.Length >= 2 && errors[^1] > 1e-10f && errors[0] > 1e-10f)
            order = MathF.Log(errors[0] / errors[^1]) / MathF.Log(dtValues[^1] / dtValues[0]);
        return (dtValues, errors, order);
    }

    /// <summary>Benchmarks convergence of GBM against exact solution.</summary>
    public (float EmError, float MilError, float RkmError) BenchmarkGBM(float x0, float mu, float sigma, float T, int N, int numPaths)
    {
        float dt = T / N;
        double emSum = 0, milSum = 0, rkmSum = 0;
        for (int p = 0; p < numPaths; p++)
        {
            float exact = x0 * MathF.Exp((mu - 0.5f * sigma * sigma) * T + sigma * MathF.Sqrt(T) * NormalSample());
            float em = x0;
            float mil = x0;
            float rkm = x0;
            for (int i = 0; i < N; i++)
            {
                float dW = MathF.Sqrt(dt) * NormalSample(), dW2 = dW * dW;
                em = em * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
                mil = mil * MathF.Exp((mu - 0.5f * sigma * sigma) * dt + sigma * dW);
                float dW1 = dW, dW1b = MathF.Sqrt(dt) * NormalSample();
                float k1a = mu * rkm, k1b = sigma * rkm, xM = rkm + k1a * dt + k1b * dW1;
                float k2a = mu * xM, k2b = sigma * xM;
                rkm += 0.5f * (k1a + k2a) * dt + 0.5f * (k1b * dW1 + k2b * dW1b);
                rkm = MathF.Max(rkm, 1e-10f);
            }
            emSum += MathF.Abs(em - exact);
            milSum += MathF.Abs(mil - exact);
            rkmSum += MathF.Abs(rkm - exact);
        }
        return ((float)(emSum / numPaths), (float)(milSum / numPaths), (float)(rkmSum / numPaths));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 7 — STOCHASTIC CORRELATION MODEL
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Computes spatial covariance matrices and generates correlated samples
/// using various kernel functions (Matern, Exponential, Gaussian).
/// </summary>
public sealed unsafe class StochasticCorrelationModel
{
    private readonly CovarianceKernelType _kernelType;
    private readonly float _lengthScale;
    private readonly float _signalVariance;
    private readonly float _noiseVariance;
    private readonly float _maternNu;
    private readonly Random _rng;

    public CovarianceKernelType KernelType => _kernelType;
    public float LengthScale => _lengthScale;
    public float SignalVariance => _signalVariance;

    public StochasticCorrelationModel(CovarianceKernelType kernelType, float lengthScale = 1.0f,
        float signalVariance = 1.0f, float noiseVariance = 1e-6f, float maternNu = 1.5f, int seed = 42)
    {
        _kernelType = kernelType;
        _lengthScale = lengthScale;
        _signalVariance = signalVariance;
        _noiseVariance = noiseVariance;
        _maternNu = maternNu;
        _rng = new Random(seed);
    }

    /// <summary>Evaluates the covariance kernel between two points.</summary>
    public float Evaluate(float dx, float dy, float dz)
    {
        float r2 = dx * dx + dy * dy + dz * dz;
        float r = MathF.Sqrt(r2);
        float h = r / _lengthScale;
        return _kernelType switch
        {
            CovarianceKernelType.Exponential => _signalVariance * MathF.Exp(-h),
            CovarianceKernelType.Gaussian => _signalVariance * MathF.Exp(-0.5f * h * h),
            CovarianceKernelType.Matern => _signalVariance * MaternKernel(h),
            CovarianceKernelType.RationalQuadratic => _signalVariance * MathF.Pow(1 + 0.5f * h * h / _maternNu, -_maternNu),
            CovarianceKernelType.PowerLaw => _signalVariance * MathF.Pow(1 + h * h, -_maternNu * 0.5f),
            _ => _signalVariance * MathF.Exp(-0.5f * h * h)
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float MaternKernel(float h)
    {
        if (h < 1e-10f)
            return _signalVariance;
        float sqrt2nu = MathF.Sqrt(2 * _maternNu);
        if (MathF.Abs(_maternNu - 0.5f) < 0.01f)
            return MathF.Exp(-sqrt2nu * h);
        if (MathF.Abs(_maternNu - 1.5f) < 0.01f)
            return (1 + sqrt2nu * h) * MathF.Exp(-sqrt2nu * h);
        if (MathF.Abs(_maternNu - 2.5f) < 0.01f)
            return (1 + sqrt2nu * h + _maternNu * h * h * 2.0f / 3.0f) * MathF.Exp(-sqrt2nu * h);
        // General Matern
        float coeff = MathF.Pow(2, 1 - _maternNu) / Gamma(_maternNu);
        float arg = sqrt2nu * h;
        return _signalVariance * coeff * MathF.Pow(arg, _maternNu) * BesselK(_maternNu, arg);
    }

    /// <summary>Builds covariance matrix for a set of spatial points.</summary>
    public float* BuildCovarianceMatrix(float* pointsX, float* pointsY, float* pointsZ, int numPoints)
    {
        float* matrix = (float*)NativeMemory.AlignedAlloc((nuint)(numPoints * numPoints * sizeof(float)), 64);
        for (int i = 0; i < numPoints; i++)
        {
            for (int j = i; j < numPoints; j++)
            {
                float dx = pointsX[i] - pointsX[j], dy = pointsY[i] - pointsY[j], dz = pointsZ[i] - pointsZ[j];
                float cov = Evaluate(dx, dy, dz);
                if (i == j)
                    cov += _noiseVariance;
                matrix[i * numPoints + j] = cov;
                matrix[j * numPoints + i] = cov;
            }
        }
        return matrix;
    }

    /// <summary>Cholesky factorization: L L^T = A. Returns lower-triangular L.</summary>
    public static float* CholeskyDecomposition(float* matrix, int n)
    {
        float* L = (float*)NativeMemory.AlignedAlloc((nuint)(n * n * sizeof(float)), 64);
        NativeMemory.Clear(L, (nuint)(n * n * sizeof(float)));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0;
                for (int k = 0; k < j; k++)
                    sum += L[i * n + k] * L[j * n + k];
                if (i == j)
                {
                    float diag = matrix[i * n + i] - sum;
                    L[i * n + j] = MathF.Sqrt(MathF.Max(diag, 1e-10f));
                }
                else
                {
                    L[i * n + j] = (matrix[i * n + j] - sum) / (L[j * n + j] + 1e-10f);
                }
            }
        }
        return L;
    }

    /// <summary>Generates correlated samples via Cholesky: x = L z where z ~ N(0,I).</summary>
    public void GenerateCorrelatedSamples(float* L, int n, float* output)
    {
        Span<float> z = stackalloc float[n];
        for (int i = 0; i < n; i++)
        { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); z[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
        for (int i = 0; i < n; i++)
        {
            float sum = 0;
            for (int j = 0; j <= i; j++)
                sum += L[i * n + j] * z[j];
            output[i] = sum;
        }
    }

    /// <summary>Computes the covariance matrix for a regular grid.</summary>
    public float* BuildGridCovariance(int resX, int resY, int resZ, float spacingX, float spacingY, float spacingZ)
    {
        int total = resX * resY * resZ;
        float* matrix = (float*)NativeMemory.AlignedAlloc((nuint)(total * total * sizeof(float)), 64);
        for (int i = 0; i < total; i++)
        {
            int iz = i / (resY * resX), iy = (i / resX) % resY, ix = i % resX;
            for (int j = i; j < total; j++)
            {
                int jz = j / (resY * resX), jy = (j / resX) % resY, jx = j % resX;
                float dx = (ix - jx) * spacingX, dy = (iy - jy) * spacingY, dz = (iz - jz) * spacingZ;
                float cov = Evaluate(dx, dy, dz);
                if (i == j)
                    cov += _noiseVariance;
                matrix[i * total + j] = cov;
                matrix[j * total + i] = cov;
            }
        }
        return matrix;
    }

    /// <summary>Approximates inverse via Gauss-Seidel iteration.</summary>
    public void ApproximateInverse(float* matrix, float* inverse, int n, int maxIter = 100, float tolerance = 1e-6f)
    {
        NativeMemory.Clear(inverse, (nuint)(n * n * sizeof(float)));
        for (int i = 0; i < n; i++)
            inverse[i * n + i] = 1.0f;
        for (int iter = 0; iter < maxIter; iter++)
        {
            float maxDiff = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                        if (k != i)
                            sum += matrix[i * n + k] * inverse[k * n + j];
                    float newVal = (j == i ? 1.0f : 0.0f - sum) / (matrix[i * n + i] + 1e-10f);
                    maxDiff = MathF.Max(maxDiff, MathF.Abs(newVal - inverse[i * n + j]));
                    inverse[i * n + j] = newVal;
                }
            if (maxDiff < tolerance)
                break;
        }
    }

    private static float Gamma(float z)
    {
        if (z < 0.5f)
            return MathF.PI / (MathF.Sin(MathF.PI * z) * Gamma(1 - z));
        z -= 1;
        float g = 7;
        float[] c = new float[] { 0.99999999999980993f, 676.5203681218851f, -1259.1392167224028f, 771.32342877765313f, -176.61502916214059f, 12.507343278686905f, -0.13857109526572012f, 9.9843695780195716e-6f, 1.5056327351493116e-7f };
        float x = c[0];
        for (int i = 1; i < g + 2; i++)
            x += c[i] / (z + i);
        float t = z + g + 0.5f;
        return MathF.Sqrt(2 * MathF.PI) * MathF.Pow(t, z + 0.5f) * MathF.Exp(-t) * x;
    }

    private static float BesselK(float nu, float x)
    {
        if (x < 1e-10f)
            return 1e10f;
        if (x > 2.0f)
            return MathF.Exp(-x) / MathF.Sqrt(x) * (1 + (4 * nu * nu - 1) / (8 * x));
        return MathF.Exp(-x) * MathF.Sqrt(MathF.PI / (2 * x)) * MathF.Pow(0.5f * x, nu) / Gamma(nu + 0.5f);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 8 — STOCHASTIC FIELD SAMPLER
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Samples the stochastic field with correlation, computes ensemble statistics,
/// generates Monte Carlo paths, and applies variance reduction techniques.
/// </summary>
public sealed unsafe class StochasticFieldSampler : IDisposable
{
    private readonly StochasticField _field;
    private readonly StochasticCorrelationModel _correlationModel;
    private readonly Random _rng;

    public StochasticFieldSampler(StochasticField field, StochasticCorrelationModel? correlationModel = null, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _correlationModel = correlationModel ?? new StochasticCorrelationModel(CovarianceKernelType.Matern);
        _rng = new Random(seed);
    }

    /// <summary>Samples the field at a point with correlated noise.</summary>
    public StochasticFieldState SamplePoint(float worldX, float worldY, float worldZ, float correlationRadius = 0)
    {
        StochasticFieldState baseState = _field.Sample(worldX, worldY, worldZ);
        if (correlationRadius > 0)
        {
            float noise = CorrelatedNoise(worldX, worldY, worldZ, correlationRadius);
            baseState.CurrentValue += noise * baseState.Diffusion;
            baseState.Variance += noise * noise;
        }
        return baseState;
    }

    /// <summary>Generates an ensemble of samples at a point for statistics.</summary>
    public (float Mean, float Variance, float CI95Lower, float CI95Upper) EnsembleStatistics(
        float worldX, float worldY, float worldZ, int numSamples = 1000)
    {
        float sum = 0, sumSq = 0;
        Span<float> samples = stackalloc float[Math.Min(numSamples, 10000)];
        int n = Math.Min(numSamples, 10000);
        for (int i = 0; i < n; i++)
        {
            StochasticFieldState s = _field.Sample(worldX, worldY, worldZ);
            float sample = s.CurrentValue + s.Diffusion * NormalSample();
            samples[i] = sample;
            sum += sample;
            sumSq += sample * sample;
        }
        float mean = sum / n, variance = sumSq / n - mean * mean;
        samples.Sort();
        float ciLow = samples[(int)(n * 0.025f)], ciHigh = samples[(int)(n * 0.975f)];
        return (mean, variance, ciLow, ciHigh);
    }

    /// <summary>Generates Monte Carlo paths through the field.</summary>
    public StochasticPath[] GenerateMonteCarloPaths(float startX, float startY, float startZ,
        float directionX, float directionY, float directionZ, float pathLength, int numSteps, int numPaths)
    {
        var paths = new StochasticPath[numPaths];
        float stepSize = pathLength / numSteps;
        for (int p = 0; p < numPaths; p++)
        {
            paths[p] = new StochasticPath(numSteps + 1);
            float x = startX, y = startY, z = startZ;
            paths[p].Times[0] = 0;
            paths[p].Values[0] = _field.Sample(x, y, z).CurrentValue;
            for (int i = 1; i <= numSteps; i++)
            {
                x += directionX * stepSize;
                y += directionY * stepSize;
                z += directionZ * stepSize;
                StochasticFieldState s = _field.Sample(x, y, z);
                paths[p].Times[i] = i * stepSize;
                paths[p].Values[i] = s.CurrentValue + s.Diffusion * NormalSample() * MathF.Sqrt(stepSize);
            }
        }
        return paths;
    }

    /// <summary>Applies antithetic variates variance reduction.</summary>
    public MonteCarloResult PriceWithAntithetic(Func<StochasticPath, float> payoff, float startX, float startY, float startZ,
        float pathLength, int numSteps, int numPaths)
    {
        float sumPayoff = 0, sumPayoffSq = 0;
        int halfPaths = numPaths / 2;
        for (int p = 0; p < halfPaths; p++)
        {
            var path = GenerateSinglePath(startX, startY, startZ, pathLength, numSteps);
            float pv = payoff(path);
            // Antithetic: mirror all random increments
            var antiPath = GenerateAntitheticPath(path);
            float antiPv = payoff(antiPath);
            float avg = 0.5f * (pv + antiPv);
            sumPayoff += avg;
            sumPayoffSq += avg * avg;
        }
        float mean = sumPayoff / halfPaths;
        float variance = sumPayoffSq / halfPaths - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / halfPaths);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numPaths, Variance = variance };
    }

    /// <summary>Applies control variates variance reduction.</summary>
    public MonteCarloResult PriceWithControlVariate(Func<StochasticPath, float> payoff, Func<StochasticPath, float> control,
        float controlMean, float startX, float startY, float startZ, float pathLength, int numSteps, int numPaths)
    {
        float sumP = 0, sumC = 0, sumPC = 0, sumC2 = 0;
        for (int i = 0; i < numPaths; i++)
        {
            var path = GenerateSinglePath(startX, startY, startZ, pathLength, numSteps);
            float pv = payoff(path), cv = control(path);
            sumP += pv;
            sumC += cv;
            sumPC += pv * cv;
            sumC2 += cv * cv;
        }
        float meanP = sumP / numPaths, meanC = sumC / numPaths;
        float covPC = sumPC / numPaths - meanP * meanC;
        float varC = sumC2 / numPaths - meanC * meanC;
        float beta = varC > 1e-10f ? covPC / varC : 0;
        float adjusted = meanP - beta * (meanC - controlMean);
        float residualVar = 0;
        for (int i = 0; i < numPaths; i++)
        {
            var path = GenerateSinglePath(startX, startY, startZ, pathLength, numSteps);
            float pv = payoff(path), cv = control(path);
            float adj = pv - beta * (cv - controlMean) - adjusted;
            residualVar += adj * adj;
        }
        float stdErr = MathF.Sqrt(MathF.Max(residualVar / numPaths, 0) / numPaths);
        return new MonteCarloResult { Estimate = adjusted, StandardError = stdErr, CI95Lower = adjusted - 1.96f * stdErr, CI95Upper = adjusted + 1.96f * stdErr, NumPaths = numPaths, Variance = residualVar / numPaths };
    }

    /// <summary>Applies importance sampling with a biased drift.</summary>
    public MonteCarloResult PriceWithImportanceSampling(Func<StochasticPath, float> payoff,
        float driftBias, float startX, float startY, float startZ, float pathLength, int numSteps, int numPaths)
    {
        float sumLikelihood = 0, sumPayoff = 0;
        for (int i = 0; i < numPaths; i++)
        {
            var path = GenerateBiasedPath(startX, startY, startZ, pathLength, numSteps, driftBias);
            float pv = payoff(path);
            float logLikelihood = 0;
            for (int j = 1; j < path.Length; j++)
            {
                float dt = path.Times[j] - path.Times[j - 1];
                float dr = MathF.Log(path.Values[j] / (MathF.Abs(path.Values[j - 1]) + 1e-10f) + 1e-10f);
                float unbiasedDr = dr - driftBias * dt;
                logLikelihood += -driftBias * dr + 0.5f * driftBias * driftBias * dt;
            }
            float likelihood = MathF.Exp(logLikelihood);
            sumPayoff += pv * likelihood;
            sumLikelihood += likelihood;
        }
        float mean = sumPayoff / numPaths;
        float variance = 0;
        for (int i = 0; i < numPaths; i++)
        {
            var path = GenerateBiasedPath(startX, startY, startZ, pathLength, numSteps, driftBias);
            float pv = payoff(path);
            float logL = 0;
            for (int j = 1; j < path.Length; j++)
            { float dt = path.Times[j] - path.Times[j - 1]; float dr = MathF.Log(path.Values[j] / (MathF.Abs(path.Values[j - 1]) + 1e-10f) + 1e-10f); logL += -driftBias * dr + 0.5f * driftBias * driftBias * dt; }
            float adj = pv * MathF.Exp(logL) - mean;
            variance += adj * adj;
        }
        float stdErr = MathF.Sqrt(MathF.Max(variance / numPaths, 0) / numPaths);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numPaths, Variance = variance / numPaths };
    }

    private StochasticPath GenerateSinglePath(float startX, float startY, float startZ, float pathLength, int numSteps)
    {
        var path = new StochasticPath(numSteps + 1);
        float dt = pathLength / numSteps;
        path.Times[0] = 0;
        path.Values[0] = _field.Sample(startX, startY, startZ).CurrentValue;
        for (int i = 1; i <= numSteps; i++)
        {
            float t = i * dt;
            StochasticFieldState s = _field.Sample(startX + t, startY, startZ);
            path.Times[i] = t;
            path.Values[i] = s.CurrentValue + s.Diffusion * NormalSample() * MathF.Sqrt(dt);
        }
        return path;
    }

    private StochasticPath GenerateAntitheticPath(StochasticPath original)
    {
        var anti = new StochasticPath(original.Length);
        for (int i = 0; i < original.Length; i++)
        { anti.Times[i] = original.Times[i]; anti.Values[i] = 2 * original.Values[0] - original.Values[i] + NormalSample() * 0.01f; }
        return anti;
    }

    private StochasticPath GenerateBiasedPath(float startX, float startY, float startZ, float pathLength, int numSteps, float driftBias)
    {
        var path = new StochasticPath(numSteps + 1);
        float dt = pathLength / numSteps;
        path.Times[0] = 0;
        path.Values[0] = _field.Sample(startX, startY, startZ).CurrentValue;
        for (int i = 1; i <= numSteps; i++)
        {
            float t = i * dt;
            StochasticFieldState s = _field.Sample(startX + t, startY, startZ);
            path.Times[i] = t;
            path.Values[i] = path.Values[i - 1] * MathF.Exp((s.Drift + driftBias) * dt + s.Diffusion * MathF.Sqrt(dt) * NormalSample());
            path.Values[i] = MathF.Max(path.Values[i], 1e-10f);
        }
        return path;
    }

    private float CorrelatedNoise(float x, float y, float z, float radius)
    {
        float sum = 0, weight = 0;
        int samples = Math.Min(10, (int)(radius * 4));
        for (int i = 0; i < samples; i++)
        {
            float dx = (float)(_rng.NextDouble() * 2 - 1) * radius;
            float dy = (float)(_rng.NextDouble() * 2 - 1) * radius;
            float dz = (float)(_rng.NextDouble() * 2 - 1) * radius;
            float w = _correlationModel.Evaluate(dx, dy, dz);
            sum += w * NormalSample();
            weight += w;
        }
        return sum / (weight + 1e-10f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }

    public void Dispose() { }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 9 — FINANCIAL FIELD MODEL
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Financial field model for option pricing, implied volatility surfaces,
/// Greeks computation, and portfolio stress testing.
/// </summary>
public sealed unsafe class FinancialFieldModel : IDisposable
{
    private readonly StochasticField _field;
    private readonly Random _rng;

    public FinancialFieldModel(StochasticField field, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _rng = new Random(seed);
    }

    /// <summary>Computes Black-Scholes price and Greeks at each field cell.</summary>
    public void ComputeBlackScholesField()
    {
        int total = _field.TotalCells;
        for (int i = 0; i < total; i++)
        {
            ref StochasticFieldState s = ref _field.UncheckedAt(i % _field.ResX, (i / _field.ResX) % _field.ResY, i / (_field.ResX * _field.ResY));
            var fs = new FinancialFieldState
            {
                SpotPrice = MathF.Max(s.CurrentValue, 0.01f),
                ImpliedVolatility = MathF.Max(s.Diffusion, 0.01f),
                RiskFreeRate = s.Drift,
                DividendYield = 0.02f,
                TimeToMaturity = 1.0f,
                StrikePrice = MathF.Max(s.Mean, 0.01f),
                Type = OptionType.Call
            };
            fs.ComputeBlackScholes();
            s.Entropy = fs.OptionPrice;
            s.Skewness = fs.Delta;
            s.Kurtosis = fs.Gamma;
            s.ThirdMoment = fs.Vega;
        }
    }

    /// <summary>Builds an implied volatility surface across the field.</summary>
    public float* BuildImpliedVolatilitySurface(int strikeSteps, int maturitySteps,
        float minStrike, float maxStrike, float minMaturity, float maxMaturity)
    {
        int total = strikeSteps * maturitySteps;
        float* surface = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        for (int si = 0; si < strikeSteps; si++)
        {
            float strike = minStrike + (maxStrike - minStrike) * si / (strikeSteps - 1);
            for (int ti = 0; ti < maturitySteps; ti++)
            {
                float maturity = minMaturity + (maxMaturity - minMaturity) * ti / (maturitySteps - 1);
                float avgVol = 0;
                int count = 0;
                for (int z = 0; z < _field.ResZ; z++)
                    for (int y = 0; y < _field.ResY; y++)
                        for (int x = 0; x < _field.ResX; x++)
                        {
                            ref StochasticFieldState s = ref _field.UncheckedAt(x, y, z);
                            float spot = MathF.Max(s.CurrentValue, 0.01f);
                            float vol = s.Diffusion;
                            float bs = BSPrice(spot, strike, s.Drift, 0.02f, maturity, vol, OptionType.Call);
                            float iv = ImpliedVolNewton(bs, spot, strike, s.Drift, maturity, OptionType.Call);
                            avgVol += iv;
                            count++;
                        }
                surface[si * maturitySteps + ti] = count > 0 ? avgVol / count : 0.2f;
            }
        }
        return surface;
    }

    /// <summary>Computes Delta hedge ratio at a field cell.</summary>
    public float ComputeDelta(float spot, float strike, float rate, float dividend, float maturity, float vol, OptionType type)
    {
        float sqrtT = MathF.Sqrt(MathF.Max(maturity, 1e-10f));
        float d1 = (MathF.Log(spot / strike) + (rate - dividend + 0.5f * vol * vol) * maturity) / (vol * sqrtT);
        if (type == OptionType.Call)
            return MathF.Exp(-dividend * maturity) * NormalCDF(d1);
        return -MathF.Exp(-dividend * maturity) * NormalCDF(-d1);
    }

    /// <summary>Computes Gamma at a field cell.</summary>
    public float ComputeGamma(float spot, float strike, float rate, float dividend, float maturity, float vol)
    {
        float sqrtT = MathF.Sqrt(MathF.Max(maturity, 1e-10f));
        float d1 = (MathF.Log(spot / strike) + (rate - dividend + 0.5f * vol * vol) * maturity) / (vol * sqrtT);
        return MathF.Exp(-dividend * maturity) * NormalPDF(d1) / (spot * vol * sqrtT);
    }

    /// <summary>Portfolio stress testing: shift all field values by scenario shocks.</summary>
    public float StressTestPortfolio(float shockMagnitude, float correlationDecay, int numScenarios = 1000)
    {
        float totalPnL = 0;
        int total = _field.TotalCells;
        for (int s = 0; s < numScenarios; s++)
        {
            float scenarioPnL = 0;
            float shockDir = (float)(_rng.NextDouble() * 2 - 1) * shockMagnitude;
            for (int i = 0; i < total; i++)
            {
                int x = i % _field.ResX, y = (i / _field.ResX) % _field.ResY, z = i / (_field.ResX * _field.ResY);
                ref StochasticFieldState state = ref _field.UncheckedAt(x, y, z);
                float localShock = shockDir * MathF.Exp(-correlationDecay * MathF.Abs(state.Correlation));
                scenarioPnL += state.CurrentValue * localShock * state.Delta;
            }
            totalPnL += scenarioPnL;
        }
        return totalPnL / numScenarios;
    }

    /// <summary>Monte Carlo option pricing on the field.</summary>
    public MonteCarloResult MonteCarloPrice(float spot, float strike, float rate, float vol, float maturity, OptionType type, int numPaths)
    {
        float sumPayoff = 0, sumPayoffSq = 0;
        for (int p = 0; p < numPaths; p++)
        {
            float terminal = spot * MathF.Exp((rate - 0.5f * vol * vol) * maturity + vol * MathF.Sqrt(maturity) * NormalSample());
            float payoff = type == OptionType.Call ? MathF.Max(terminal - strike, 0) : MathF.Max(strike - terminal, 0);
            float discounted = MathF.Exp(-rate * maturity) * payoff;
            sumPayoff += discounted;
            sumPayoffSq += discounted * discounted;
        }
        float mean = sumPayoff / numPaths;
        float variance = sumPayoffSq / numPaths - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / numPaths);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numPaths, Variance = variance };
    }

    private static float BSPrice(float S, float K, float r, float q, float T, float sigma, OptionType type)
    {
        if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0)
            return 0;
        float sqrtT = MathF.Sqrt(T), d1 = (MathF.Log(S / K) + (r - q + 0.5f * sigma * sigma) * T) / (sigma * sqrtT), d2 = d1 - sigma * sqrtT;
        if (type == OptionType.Call)
            return S * MathF.Exp(-q * T) * NormalCDF(d1) - K * MathF.Exp(-r * T) * NormalCDF(d2);
        return K * MathF.Exp(-r * T) * NormalCDF(-d2) - S * MathF.Exp(-q * T) * NormalCDF(-d1);
    }

    private float ImpliedVolNewton(float marketPrice, float S, float K, float r, float T, OptionType type, int maxIter = 50)
    {
        float vol = 0.3f;
        for (int i = 0; i < maxIter; i++)
        {
            float bs = BSPrice(S, K, r, 0, T, vol, type);
            float diff = bs - marketPrice;
            float sqrtT = MathF.Sqrt(MathF.Max(T, 1e-10f));
            float d1 = (MathF.Log(S / K) + (r + 0.5f * vol * vol) * T) / (vol * sqrtT);
            float vega = S * NormalPDF(d1) * sqrtT;
            if (MathF.Abs(vega) < 1e-10f)
                break;
            vol -= diff / vega;
            vol = MathF.Max(vol, 0.001f);
        }
        return vol;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalCDF(float x)
    {
        float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x) / MathF.Sqrt(2);
        float t = 1 / (1 + p * x);
        float y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return 0.5f * (1 + sign * y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NormalPDF(float x) => MathF.Exp(-0.5f * x * x) / MathF.Sqrt(2 * MathF.PI);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }

    public void Dispose() { }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 10 — EPIDEMIOLOGICAL FIELD MODEL
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Epidemiological field model implementing SIR/SEIR on a continuous spatial field.
/// Supports vaccination, quarantine zones, spatial diffusion, and R0 computation per cell.
/// </summary>
public sealed unsafe class EpidemiologicalFieldModel : IDisposable
{
    private readonly StochasticField _field;
    private SEIRState* _seirData;
    private readonly int _totalCells;
    private readonly Random _rng;
    private float _globalR0;
    private float _globalBeta;
    private float _globalGamma;
    private float _globalSigma;

    /// <summary>Basic reproduction number.</summary>
    public float GlobalR0 => _globalR0;

    public EpidemiologicalFieldModel(StochasticField field, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _totalCells = field.TotalCells;
        _rng = new Random(seed);
        int byteSize = _totalCells * sizeof(SEIRState);
        _seirData = (SEIRState*)NativeMemory.AlignedAlloc((nuint)byteSize, 64);
        NativeMemory.Clear(_seirData, (nuint)byteSize);
    }

    /// <summary>Initializes the SEIR field with uniform parameters.</summary>
    public void InitializeUniform(float susceptible = 0.99f, float infectious = 0.01f, float beta = 0.3f, float gamma = 0.1f, float sigma = 0.2f)
    {
        _globalBeta = beta;
        _globalGamma = gamma;
        _globalSigma = sigma;
        _globalR0 = beta / gamma;
        for (int i = 0; i < _totalCells; i++)
        {
            _seirData[i] = new SEIRState
            {
                S = susceptible,
                E = 0,
                I = infectious,
                R = 0,
                V = 0,
                Q = 0,
                D = 0,
                Beta = beta,
                Gamma = gamma,
                Sigma = sigma,
                R0Effective = beta / gamma * susceptible,
                VaccinationRate = 0,
                QuarantineRate = 0,
                PopulationDensity = 1.0f,
                DiffusionCoeff = 0.01f
            };
            _field.DataPointer[i].CurrentValue = infectious;
            _field.DataPointer[i].Drift = beta;
            _field.DataPointer[i].Diffusion = gamma;
        }
    }

    /// <summary>Spatially varying initialization using Perlin noise.</summary>
    public void InitializeSpatialVariation(float baseBeta = 0.3f, float baseGamma = 0.1f, float baseSigma = 0.2f, float heterogeneity = 0.1f, int seed = 42)
    {
        using var perlinBeta = new PerlinNoiseGenerator(seed);
        using var perlinGamma = new PerlinNoiseGenerator(seed + 1);
        for (int z = 0; z < _field.ResZ; z++)
            for (int y = 0; y < _field.ResY; y++)
                for (int x = 0; x < _field.ResX; x++)
                {
                    int idx = z * _field.ResY * _field.ResX + y * _field.ResX + x;
                    float nx = (float)x / _field.ResX, ny = (float)y / _field.ResY, nz = (float)z / _field.ResZ;
                    float beta = baseBeta * (1 + heterogeneity * perlinBeta.FractalBrownianMotion(nx, ny, nz, 3));
                    float gamma = baseGamma * (1 + heterogeneity * perlinGamma.FractalBrownianMotion(nx, ny, nz, 3));
                    float susceptible = 0.99f, infectious = 0.01f;
                    _seirData[idx] = new SEIRState { S = susceptible, I = infectious, Beta = beta, Gamma = gamma, Sigma = baseSigma, R0Effective = beta / gamma * susceptible, PopulationDensity = 1.0f, DiffusionCoeff = 0.01f };
                    _field.DataPointer[idx].CurrentValue = infectious;
                }
    }

    /// <summary>Advances the SEIR model by one time-step using a stochastic SIR scheme.</summary>
    public void StepSEIR(float dt)
    {
        int rx = _field.ResX, ry = _field.ResY, rz = _field.ResZ;
        for (int z = 1; z < rz - 1; z++)
            for (int y = 1; y < ry - 1; y++)
                for (int x = 1; x < rx - 1; x++)
                {
                    int idx = (z * ry + y) * rx + x;
                    ref SEIRState seir = ref _seirData[idx];

                    // Spatial diffusion of infected
                    float avgI = (_seirData[idx - 1].I + _seirData[idx + 1].I + _seirData[idx - rx].I + _seirData[idx + rx].I + _seirData[idx - rx * ry].I + _seirData[idx + rx * ry].I) / 6.0f;
                    float diffusion = seir.DiffusionCoeff * (avgI - seir.I) * dt;

                    // SEIR transitions (stochastic)
                    float N = seir.S + seir.E + seir.I + seir.R + seir.V + seir.Q + seir.D;
                    if (N < 1e-10f)
                        continue;
                    float effectiveN = N;

                    // New infections: β * S * I / N
                    float newInfections = seir.Beta * seir.S * seir.I / effectiveN * dt;
                    newInfections *= (1 + 0.1f * NormalSample()); // stochastic noise

                    // Exposed to Infectious: σ * E
                    float exposedToInfectious = seir.Sigma * seir.E * dt;

                    // Recovery: γ * I
                    float recoveries = seir.Gamma * seir.I * dt;

                    // Vaccination
                    float vaccinations = seir.VaccinationRate * seir.S * dt;

                    // Quarantine
                    float quarantines = seir.QuarantineRate * seir.I * dt;

                    // Update compartments
                    seir.S = MathF.Max(seir.S - newInfections - vaccinations, 0);
                    seir.E = MathF.Max(seir.E + newInfections - exposedToInfectious, 0) + diffusion;
                    seir.I = MathF.Max(seir.I + exposedToInfectious - recoveries - quarantines, 0) - diffusion;
                    seir.R = MathF.Max(seir.R + recoveries, 0);
                    seir.V = MathF.Max(seir.V + vaccinations, 0);
                    seir.Q = MathF.Max(seir.Q + quarantines, 0);

                    // Effective R
                    seir.R0Effective = seir.ComputeReffective();
                    seir.Normalize();

                    // Update field
                    _field.DataPointer[idx].CurrentValue = seir.I;
                    _field.DataPointer[idx].Mean = seir.S;
                    _field.DataPointer[idx].Variance = seir.R0Effective;
                    _field.DataPointer[idx].Diffusion = seir.I;
                }
    }

    /// <summary>Advances using simple SIR without exposed compartment.</summary>
    public void StepSIR(float dt)
    {
        int rx = _field.ResX, ry = _field.ResY, rz = _field.ResZ;
        for (int z = 1; z < rz - 1; z++)
            for (int y = 1; y < ry - 1; y++)
                for (int x = 1; x < rx - 1; x++)
                {
                    int idx = (z * ry + y) * rx + x;
                    ref SEIRState seir = ref _seirData[idx];
                    float N = seir.S + seir.I + seir.R + seir.V + seir.Q + seir.D;
                    if (N < 1e-10f)
                        continue;
                    float newInfections = seir.Beta * seir.S * seir.I / N * dt * (1 + 0.05f * NormalSample());
                    float recoveries = seir.Gamma * seir.I * dt;
                    float vaccinations = seir.VaccinationRate * seir.S * dt;
                    seir.S = MathF.Max(seir.S - newInfections - vaccinations, 0);
                    seir.I = MathF.Max(seir.I + newInfections - recoveries, 0);
                    seir.R = MathF.Max(seir.R + recoveries, 0);
                    seir.V = MathF.Max(seir.V + vaccinations, 0);
                    seir.R0Effective = seir.ComputeReffective();
                    seir.Normalize();
                    _field.DataPointer[idx].CurrentValue = seir.I;
                    _field.DataPointer[idx].Mean = seir.S;
                }
    }

    /// <summary>Computes R0 at each cell and stores in field Variance.</summary>
    public void ComputeR0Field()
    {
        for (int i = 0; i < _totalCells; i++)
        {
            _field.DataPointer[i].Variance = _seirData[i].ComputeR0();
            _field.DataPointer[i].Kurtosis = _seirData[i].ComputeReffective();
        }
    }

    /// <summary>Sets quarantine zones in rectangular regions.</summary>
    public void SetQuarantineZone(int x0, int y0, int z0, int x1, int y1, int z1, float complianceRate = 0.8f)
    {
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    if (_field.InBounds(x, y, z))
                    {
                        int idx = (z * _field.ResY + y) * _field.ResX + x;
                        _seirData[idx].QuarantineRate = complianceRate;
                    }
                }
    }

    /// <summary>Sets vaccination rates in rectangular regions.</summary>
    public void SetVaccinationZone(int x0, int y0, int z0, int x1, int y1, int z1, float vaccinationRate = 0.05f)
    {
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    if (_field.InBounds(x, y, z))
                    {
                        int idx = (z * _field.ResY + y) * _field.ResX + x;
                        _seirData[idx].VaccinationRate = vaccinationRate;
                    }
                }
    }

    /// <summary>Returns epidemic statistics.</summary>
    public (float TotalInfected, float TotalRecovered, float TotalSusceptible, float PeakInfection, float HerdImmunityThreshold) GetEpidemicStats()
    {
        float totalI = 0, totalR = 0, totalS = 0, peakI = 0;
        for (int i = 0; i < _totalCells; i++)
        {
            totalI += _seirData[i].I;
            totalR += _seirData[i].R;
            totalS += _seirData[i].S;
            if (_seirData[i].I > peakI)
                peakI = _seirData[i].I;
        }
        float avg = _globalR0 > 1 ? 1 - 1 / _globalR0 : 0;
        return (totalI / _totalCells, totalR / _totalCells, totalS / _totalCells, peakI, avg);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }

    public void Dispose()
    {
        if (_seirData != null)
        { NativeMemory.AlignedFree(_seirData); _seirData = null; }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 11 — SOCIAL DIFFUSION MODEL
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Social diffusion model for information spread, opinion dynamics,
/// and network effects on the stochastic field.
/// </summary>
public sealed unsafe class SocialDiffusionModel : IDisposable
{
    private readonly StochasticField _field;
    private float* _opinionState;
    private float* _influenceMatrix;
    private readonly int _totalCells;
    private readonly Random _rng;

    /// <summary>Viral coefficient (R0 for information).</summary>
    public float ViralCoefficient { get; private set; }

    public SocialDiffusionModel(StochasticField field, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _totalCells = field.TotalCells;
        _rng = new Random(seed);
        _opinionState = (float*)NativeMemory.AlignedAlloc((nuint)(_totalCells * sizeof(float)), 64);
        _influenceMatrix = (float*)NativeMemory.AlignedAlloc((nuint)(_totalCells * sizeof(float) * 6), 64);
        NativeMemory.Clear(_opinionState, (nuint)(_totalCells * sizeof(float)));
        NativeMemory.Clear(_influenceMatrix, (nuint)(_totalCells * sizeof(float) * 6));
    }

    /// <summary>Initializes opinions from field values.</summary>
    public void InitializeOpinions()
    {
        for (int i = 0; i < _totalCells; i++)
            _opinionState[i] = _field.DataPointer[i].CurrentValue;
    }

    /// <summary>
    /// Bass model diffusion: new adopters = (p + q * F(t)) * (1 - F(t)),
    /// where p is innovation rate, q is imitation rate, F(t) is adoption fraction.
    /// </summary>
    public void StepBassDiffusion(float p, float q, float dt)
    {
        float totalAdoption = 0;
        for (int i = 0; i < _totalCells; i++)
            totalAdoption += MathF.Max(_field.DataPointer[i].CurrentValue, 0);
        float F = totalAdoption / _totalCells;
        float newAdopters = (p + q * F) * (1 - F) * dt;
        ViralCoefficient = q / (p + 1e-10f);

        for (int i = 0; i < _totalCells; i++)
        {
            ref StochasticFieldState s = ref _field.DataPointer[i];
            float localInfluence = _opinionState[i] * q * (1 - s.CurrentValue) * dt;
            float innovation = p * (1 - s.CurrentValue) * dt;
            s.CurrentValue = MathF.Max(MathF.Min(s.CurrentValue + localInfluence + innovation + newAdopters * (0.5f + NormalSample() * 0.1f), 1.0f), 0);
        }
    }

    /// <summary>
    /// Bounded confidence opinion dynamics (Deffuant model).
    /// Agents with opinions within confidence bound ε interact and converge.
    /// </summary>
    public void StepBoundedConfidence(float confidenceBound, float convergenceRate, float dt)
    {
        int rx = _field.ResX, ry = _field.ResY, rz = _field.ResZ;
        for (int z = 1; z < rz - 1; z++)
            for (int y = 1; y < ry - 1; y++)
                for (int x = 1; x < rx - 1; x++)
                {
                    int idx = (z * ry + y) * rx + x;
                    float ownOpinion = _opinionState[idx];
                    float avgNeighborOpinion = 0;
                    int neighbors = 0;
                    // 6-connected neighbors
                    int[] offsets = { -1, 1, -rx, rx, -rx * ry, rx * ry };
                    foreach (int off in offsets)
                    {
                        int nIdx = idx + off;
                        if (nIdx >= 0 && nIdx < _totalCells)
                        {
                            float diff = MathF.Abs(ownOpinion - _opinionState[nIdx]);
                            if (diff < confidenceBound)
                            { avgNeighborOpinion += _opinionState[nIdx]; neighbors++; }
                        }
                    }
                    if (neighbors > 0)
                    {
                        avgNeighborOpinion /= neighbors;
                        _opinionState[idx] = ownOpinion + convergenceRate * (avgNeighborOpinion - ownOpinion) * dt;
                        _opinionState[idx] = Math.Clamp(_opinionState[idx], -1, 1);
                    }
                    _field.DataPointer[idx].CurrentValue = _opinionState[idx];
                }
    }

    /// <summary>
    /// Network effects model: adoption probability increases with number of adopters in neighborhood.
    /// </summary>
    public void StepNetworkEffects(float adoptionThreshold, float networkStrength, float dt)
    {
        int rx = _field.ResX, ry = _field.ResY, rz = _field.ResZ;
        for (int z = 1; z < rz - 1; z++)
            for (int y = 1; y < ry - 1; y++)
                for (int x = 1; x < rx - 1; x++)
                {
                    int idx = (z * ry + y) * rx + x;
                    float adopterCount = 0;
                    int[] offsets = { -1, 1, -rx, rx, -rx * ry, rx * ry };
                    foreach (int off in offsets)
                    {
                        int nIdx = idx + off;
                        if (nIdx >= 0 && nIdx < _totalCells && _opinionState[nIdx] > 0.5f)
                            adopterCount++;
                    }
                    float networkEffect = adopterCount / 6.0f;
                    float adoptionProb = networkStrength * networkEffect * dt;
                    if (networkEffect > adoptionThreshold && (float)_rng.NextDouble() < adoptionProb)
                        _opinionState[idx] = MathF.Min(_opinionState[idx] + 0.1f * dt, 1.0f);
                    _field.DataPointer[idx].CurrentValue = _opinionState[idx];
                }
    }

    /// <summary>Computes information cascade probability across the field.</summary>
    public float ComputeCascadeProbability(float sourceX, float sourceY, float sourceZ, float threshold = 0.5f)
    {
        _field.WorldToGrid(sourceX, sourceY, sourceZ, out int sx, out int sy, out int sz);
        int sourceIdx = (sz * _field.ResY + sy) * _field.ResX + sx;
        if (sourceIdx < 0 || sourceIdx >= _totalCells)
            return 0;
        float cascade = 0;
        int visited = 0;
        for (int i = 0; i < _totalCells; i++)
        {
            if (_opinionState[i] > threshold)
            { cascade++; visited++; }
        }
        return visited > 0 ? cascade / _totalCells : 0;
    }

    /// <summary>Returns opinion statistics.</summary>
    public (float MeanOpinion, float Polarization, float AdoptionRate) GetOpinionStats(float adoptionThreshold = 0.5f)
    {
        float sum = 0, sumSq = 0, adoptionCount = 0;
        for (int i = 0; i < _totalCells; i++)
        { sum += _opinionState[i]; sumSq += _opinionState[i] * _opinionState[i]; if (_opinionState[i] > adoptionThreshold) adoptionCount++; }
        float mean = sum / _totalCells;
        float polarization = sumSq / _totalCells - mean * mean;
        return (mean, polarization, adoptionCount / _totalCells);
    }

    public void Dispose()
    {
        if (_opinionState != null)
        { NativeMemory.AlignedFree(_opinionState); _opinionState = null; }
        if (_influenceMatrix != null)
        { NativeMemory.AlignedFree(_influenceMatrix); _influenceMatrix = null; }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 12 — STOCHASTIC OPTIMIZER
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Stochastic optimization algorithms: SGD with noise, simulated annealing,
/// cross-entropy method, and Bayesian optimization with Gaussian processes.
/// </summary>
public sealed class StochasticOptimizer
{
    private readonly Random _rng;

    /// <summary>Objective function to minimize: takes parameters, returns cost.</summary>
    public delegate float ObjectiveFunction(ReadOnlySpan<float> parameters);

    public StochasticOptimizer(int seed = 42) { _rng = new Random(seed); }

    /// <summary>
    /// Stochastic Gradient Descent with momentum and adaptive noise.
    /// </summary>
    /// <param name="objective">Objective function.</param>
    /// <param name="initialParams">Starting parameters.</param>
    /// <param name="learningRate">Base learning rate.</param>
    /// <param name="momentum">Momentum coefficient.</param>
    /// <param name="noiseScale">Scale of parameter noise.</param>
    /// <param name="maxIter">Maximum iterations.</param>
    /// <param name="tolerance">Convergence tolerance on gradient norm.</param>
    public (float[] BestParams, float BestCost, int Iterations) SGDWithMomentum(
        ObjectiveFunction objective, float[] initialParams, float learningRate = 0.01f,
        float momentum = 0.9f, float noiseScale = 0.001f, int maxIter = 10000, float tolerance = 1e-8f)
    {
        int dim = initialParams.Length;
        float[] x = new float[dim];
        Array.Copy(initialParams, x, dim);
        float[] velocity = new float[dim];
        float[] gradient = new float[dim];
        float bestCost = objective(x);
        float[] bestParams = new float[dim];
        Array.Copy(x, bestParams, dim);
        float epsilon = 1e-5f;

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Numerical gradient estimation
            for (int d = 0; d < dim; d++)
            {
                Span<float> xPlus = new float[dim];
                x.CopyTo(xPlus);
                Span<float> xMinus = new float[dim];
                x.CopyTo(xMinus);
                xPlus[d] += epsilon;
                xMinus[d] -= epsilon;
                gradient[d] = (objective(xPlus) - objective(xMinus)) / (2 * epsilon);
            }

            // Gradient norm check
            float gradNorm = 0;
            for (int d = 0; d < dim; d++)
                gradNorm += gradient[d] * gradient[d];
            gradNorm = MathF.Sqrt(gradNorm);
            if (gradNorm < tolerance)
                return (bestParams, bestCost, iter);

            // Update with momentum and noise
            for (int d = 0; d < dim; d++)
            {
                velocity[d] = momentum * velocity[d] - learningRate * gradient[d];
                float noise = noiseScale * NormalSample() * MathF.Sqrt(learningRate / (1 + iter * 0.001f));
                x[d] += velocity[d] + noise;
            }

            float cost = objective(x);
            if (cost < bestCost)
            { bestCost = cost; Array.Copy(x, bestParams, dim); }

            // Learning rate decay
            learningRate *= 0.9999f;
        }
        return (bestParams, bestCost, maxIter);
    }

    /// <summary>
    /// Simulated Annealing optimization.
    /// </summary>
    public (float[] BestParams, float BestCost, int Iterations) SimulatedAnnealing(
        ObjectiveFunction objective, float[] initialParams, float initialTemp = 1.0f,
        float coolingRate = 0.995f, float perturbationScale = 0.1f, int maxIter = 50000)
    {
        int dim = initialParams.Length;
        float[] x = new float[dim];
        Array.Copy(initialParams, x, dim);
        float[] bestX = new float[dim];
        Array.Copy(x, bestX, dim);
        float currentCost = objective(x);
        float bestCost = currentCost;
        float temp = initialTemp;

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Perturb
            float[] xNew = new float[dim];
            for (int d = 0; d < dim; d++)
                xNew[d] = x[d] + perturbationScale * temp * NormalSample();

            float newCost = objective(xNew);
            float delta = newCost - currentCost;

            // Metropolis criterion
            if (delta < 0 || (temp > 1e-10f && NormalSample() < -delta / temp))
            {
                Array.Copy(xNew, x, dim);
                currentCost = newCost;
                if (currentCost < bestCost)
                { bestCost = currentCost; Array.Copy(x, bestX, dim); }
            }

            temp *= coolingRate;
            perturbationScale *= 0.9999f;
        }
        return (bestX, bestCost, maxIter);
    }

    /// <summary>
    /// Cross-Entropy Method for combinatorial/continuous optimization.
    /// </summary>
    public (float[] BestParams, float BestCost, int Iterations) CrossEntropyMethod(
        ObjectiveFunction objective, int dim, float[] paramLower, float[] paramUpper,
        int populationSize = 200, int eliteFraction = 10, int maxIter = 500)
    {
        float[] mean = new float[dim], stdDev = new float[dim];
        for (int d = 0; d < dim; d++)
        { mean[d] = 0.5f * (paramLower[d] + paramUpper[d]); stdDev[d] = 0.25f * (paramUpper[d] - paramLower[d]); }
        float[] bestParams = new float[dim];
        float bestCost = float.MaxValue;
        int eliteCount = Math.Max(populationSize * eliteFraction / 100, 1);

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Generate population
            float[][] population = new float[populationSize][];
            float[] costs = new float[populationSize];
            for (int p = 0; p < populationSize; p++)
            {
                population[p] = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    float sample = mean[d] + stdDev[d] * NormalSample();
                    population[p][d] = Math.Clamp(sample, paramLower[d], paramUpper[d]);
                }
                costs[p] = objective(population[p]);
            }

            // Sort by cost and select elite
            int[] indices = new int[populationSize];
            for (int i = 0; i < populationSize; i++)
                indices[i] = i;
            Array.Sort(indices, (a, b) => costs[a].CompareTo(costs[b]));

            if (costs[indices[0]] < bestCost)
            { bestCost = costs[indices[0]]; Array.Copy(population[indices[0]], bestParams, dim); }

            // Update distribution from elite samples
            for (int d = 0; d < dim; d++)
            {
                float sum = 0, sumSq = 0;
                for (int e = 0; e < eliteCount; e++)
                { float val = population[indices[e]][d]; sum += val; sumSq += val * val; }
                mean[d] = sum / eliteCount;
                float variance = sumSq / eliteCount - mean[d] * mean[d];
                stdDev[d] = MathF.Sqrt(MathF.Max(variance, 1e-10f));
                stdDev[d] = MathF.Max(stdDev[d], 1e-4f * (paramUpper[d] - paramLower[d]));
            }
        }
        return (bestParams, bestCost, maxIter);
    }

    /// <summary>
    /// Bayesian Optimization with a simplified Gaussian Process surrogate.
    /// </summary>
    public (float[] BestParams, float BestCost, int Iterations) BayesianOptimization(
        ObjectiveFunction objective, float[] paramLower, float[] paramUpper,
        int initialSamples = 10, int maxIter = 50, int numRestarts = 5)
    {
        int dim = paramLower.Length;
        var observedX = new List<float[]>();
        var observedY = new List<float>();

        // Initial random samples
        for (int i = 0; i < initialSamples; i++)
        {
            float[] x = new float[dim];
            for (int d = 0; d < dim; d++)
                x[d] = paramLower[d] + (float)_rng.NextDouble() * (paramUpper[d] - paramLower[d]);
            observedX.Add(x);
            observedY.Add(objective(x));
        }

        float[] bestParams = observedX[observedY.IndexOf(observedY.Min())];
        float bestCost = observedY.Min();

        for (int iter = 0; iter < maxIter - initialSamples; iter++)
        {
            // Find next point by maximizing Expected Improvement
            float[] nextX = null!;
            float bestEI = float.MinValue;
            for (int r = 0; r < numRestarts; r++)
            {
                float[] candidate = new float[dim];
                for (int d = 0; d < dim; d++)
                    candidate[d] = paramLower[d] + (float)_rng.NextDouble() * (paramUpper[d] - paramLower[d]);
                float ei = ExpectedImprovement(candidate, observedX, observedY, bestCost);
                if (ei > bestEI)
                { bestEI = ei; nextX = candidate; }
            }

            if (nextX == null)
                break;
            float nextY = objective(nextX);
            observedX.Add(nextX);
            observedY.Add(nextY);

            if (nextY < bestCost)
            { bestCost = nextY; bestParams = (float[])nextX.Clone(); }
        }
        return (bestParams, bestCost, observedX.Count);
    }

    private float ExpectedImprovement(float[] x, List<float[]> observedX, List<float> observedY, float bestY, float xi = 0.01f)
    {
        // Simplified GP prediction: nearest-neighbor mean with RBF kernel
        float predMean = 0, predVar = 1, totalWeight = 0;
        float lengthScale = 1.0f;
        for (int i = 0; i < observedX.Count; i++)
        {
            float dist2 = 0;
            for (int d = 0; d < x.Length; d++)
            { float diff = x[d] - observedX[i][d]; dist2 += diff * diff; }
            float w = MathF.Exp(-dist2 / (2 * lengthScale * lengthScale));
            predMean += w * observedY[i];
            totalWeight += w;
        }
        if (totalWeight > 1e-10f)
            predMean /= totalWeight;
        predVar = MathF.Max(1.0f / (totalWeight + 1), 1e-6f);

        float stdDev = MathF.Sqrt(predVar);
        float z = (bestY - predMean - xi) / (stdDev + 1e-10f);
        float phi = 0.5f * (1 + ErfApprox(z / MathF.Sqrt(2)));
        float pdf = MathF.Exp(-0.5f * z * z) / MathF.Sqrt(2 * MathF.PI);
        return stdDev * (z * phi + pdf);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ErfApprox(float x)
    {
        float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
        float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 13 — MONTE CARLO ENGINE
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Generic Monte Carlo integration engine supporting standard MC,
/// quasi-Monte Carlo (Sobol, Halton), stratified sampling, and importance sampling.
/// </summary>
public sealed class MonteCarloEngine
{
    private readonly Random _rng;

    public MonteCarloEngine(int seed = 42) { _rng = new Random(seed); }

    /// <summary>Standard Monte Carlo integration of func over [0,1]^dim.</summary>
    public MonteCarloResult Integrate(Func<ReadOnlySpan<float>, float> func, int dim, int numSamples)
    {
        float sum = 0, sumSq = 0;
        Span<float> x = stackalloc float[dim];
        for (int i = 0; i < numSamples; i++)
        {
            for (int d = 0; d < dim; d++)
                x[d] = (float)_rng.NextDouble();
            float val = func(x);
            sum += val;
            sumSq += val * val;
        }
        float mean = sum / numSamples;
        float variance = sumSq / numSamples - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / numSamples);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numSamples, Variance = variance };
    }

    /// <summary>Halton sequence quasi-Monte Carlo integration.</summary>
    public MonteCarloResult IntegrateHalton(Func<ReadOnlySpan<float>, float> func, int dim, int numSamples)
    {
        float sum = 0, sumSq = 0;
        Span<float> x = stackalloc float[dim];
        int[] primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71 };
        for (int i = 0; i < numSamples; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                int base_val = d < primes.Length ? primes[d] : NextPrime(50 + d);
                x[d] = HaltonSequence(i + 1, base_val);
            }
            float val = func(x);
            sum += val;
            sumSq += val * val;
        }
        float mean = sum / numSamples;
        float variance = sumSq / numSamples - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / numSamples);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numSamples, Variance = variance };
    }

    /// <summary>Sobol sequence quasi-Monte Carlo integration.</summary>
    public MonteCarloResult IntegrateSobol(Func<ReadOnlySpan<float>, float> func, int dim, int numSamples)
    {
        float sum = 0, sumSq = 0;
        Span<float> x = stackalloc float[dim];
        int[] directions = GenerateSobolDirections(dim);
        for (int i = 0; i < numSamples; i++)
        {
            for (int d = 0; d < dim; d++)
                x[d] = SobolPoint(i, d, directions);
            float val = func(x);
            sum += val;
            sumSq += val * val;
        }
        float mean = sum / numSamples;
        float variance = sumSq / numSamples - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / numSamples);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numSamples, Variance = variance };
    }

    /// <summary>Stratified sampling integration.</summary>
    public MonteCarloResult IntegrateStratified(Func<ReadOnlySpan<float>, float> func, int dim, int numStrataPerDim, int samplesPerStratum)
    {
        int totalStrata = 1;
        for (int d = 0; d < dim; d++)
            totalStrata *= numStrataPerDim;
        float stratumSize = 1.0f / numStrataPerDim;
        float sum = 0, sumSq = 0;
        Span<float> x = stackalloc float[dim];
        int totalSamples = 0;

        for (int s = 0; s < totalStrata; s++)
        {
            // Decode stratum index to per-dimension indices
            int temp = s;
            int[] stratumIdx = new int[dim];
            for (int d = dim - 1; d >= 0; d--)
            { stratumIdx[d] = temp % numStrataPerDim; temp /= numStrataPerDim; }

            float stratumSum = 0;
            for (int n = 0; n < samplesPerStratum; n++)
            {
                for (int d = 0; d < dim; d++)
                    x[d] = (stratumIdx[d] + (float)_rng.NextDouble()) * stratumSize;
                stratumSum += func(x);
                totalSamples++;
            }
            float stratumMean = stratumSum / samplesPerStratum;
            sum += stratumMean;
        }
        float mean = sum / totalStrata;
        // Approximate variance from stratum means
        sumSq = 0;
        for (int s = 0; s < totalStrata; s++)
        { int temp = s; float stratumMean = 0; int[] si = new int[dim]; for (int d = dim - 1; d >= 0; d--) { si[d] = temp % numStrataPerDim; temp /= numStrataPerDim; } Span<float> xs = stackalloc float[dim]; for (int n = 0; n < samplesPerStratum; n++) { for (int d = 0; d < dim; d++) xs[d] = (si[d] + (float)_rng.NextDouble()) * stratumSize; stratumMean += func(xs); } stratumMean /= samplesPerStratum; sumSq += (stratumMean - mean) * (stratumMean - mean); }
        float variance = sumSq / totalStrata;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / totalStrata);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = totalSamples, Variance = variance };
    }

    /// <summary>Importance sampling with proposal distribution.</summary>
    public MonteCarloResult IntegrateImportanceSampling(Func<ReadOnlySpan<float>, float> target,
        Func<ReadOnlySpan<float>, float> proposal, Func<ReadOnlySpan<float>, float> proposalPDF,
        int dim, int numSamples)
    {
        float sum = 0, sumSq = 0;
        Span<float> x = stackalloc float[dim];
        for (int i = 0; i < numSamples; i++)
        {
            proposal(x); // fill x with proposal samples
            float f = target(x);
            float q = proposalPDF(x);
            float weight = q > 1e-10f ? f / q : 0;
            sum += weight;
            sumSq += weight * weight;
        }
        float mean = sum / numSamples;
        float variance = sumSq / numSamples - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / numSamples);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numSamples, Variance = variance };
    }

    /// <summary>Antithetic variates integration.</summary>
    public MonteCarloResult IntegrateAntithetic(Func<ReadOnlySpan<float>, float> func, int dim, int numPaths)
    {
        float sum = 0, sumSq = 0;
        int halfPaths = numPaths / 2;
        Span<float> x = stackalloc float[dim];
        Span<float> xAnti = stackalloc float[dim];
        for (int i = 0; i < halfPaths; i++)
        {
            for (int d = 0; d < dim; d++)
            { x[d] = (float)_rng.NextDouble(); xAnti[d] = 1.0f - x[d]; }
            float val = 0.5f * (func(x) + func(xAnti));
            sum += val;
            sumSq += val * val;
        }
        float mean = sum / halfPaths;
        float variance = sumSq / halfPaths - mean * mean;
        float stdErr = MathF.Sqrt(MathF.Max(variance, 0) / halfPaths);
        return new MonteCarloResult { Estimate = mean, StandardError = stdErr, CI95Lower = mean - 1.96f * stdErr, CI95Upper = mean + 1.96f * stdErr, NumPaths = numPaths, Variance = variance };
    }

    /// <summary>Control variates integration.</summary>
    public MonteCarloResult IntegrateControlVariates(Func<ReadOnlySpan<float>, float> func,
        Func<ReadOnlySpan<float>, float> control, float controlMean, int dim, int numSamples)
    {
        float sumF = 0, sumC = 0, sumFC = 0, sumC2 = 0;
        Span<float> x = stackalloc float[dim];
        for (int i = 0; i < numSamples; i++)
        {
            for (int d = 0; d < dim; d++)
                x[d] = (float)_rng.NextDouble();
            float f = func(x), c = control(x);
            sumF += f;
            sumC += c;
            sumFC += f * c;
            sumC2 += c * c;
        }
        float meanF = sumF / numSamples, meanC = sumC / numSamples;
        float covFC = sumFC / numSamples - meanF * meanC;
        float varC = sumC2 / numSamples - meanC * meanC;
        float beta = varC > 1e-10f ? covFC / varC : 0;
        float adjusted = meanF - beta * (meanC - controlMean);
        float residualVar = 0;
        for (int i = 0; i < numSamples; i++)
        { for (int d = 0; d < dim; d++) x[d] = (float)_rng.NextDouble(); float f = func(x), c = control(x); float adj = f - beta * (c - controlMean) - adjusted; residualVar += adj * adj; }
        float stdErr = MathF.Sqrt(MathF.Max(residualVar / numSamples, 0) / numSamples);
        return new MonteCarloResult { Estimate = adjusted, StandardError = stdErr, CI95Lower = adjusted - 1.96f * stdErr, CI95Upper = adjusted + 1.96f * stdErr, NumPaths = numSamples, Variance = residualVar / numSamples };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HaltonSequence(int index, int b)
    {
        float result = 0, f = 1.0f / b;
        int i = index;
        while (i > 0)
        { result += f * (i % b); i /= b; f /= b; }
        return result;
    }

    private static int NextPrime(int start)
    {
        for (int n = start; ; n++)
        { bool prime = true; for (int d = 2; d * d <= n; d++) { if (n % d == 0) { prime = false; break; } } if (prime) return n; }
    }

    private static int[] GenerateSobolDirections(int dim)
    {
        int[] dirs = new int[dim];
        int[] poly = { 0, 1, 1, 2, 1, 4, 2, 4, 7, 11, 13, 14 };
        for (int d = 0; d < dim; d++)
            dirs[d] = d < poly.Length ? poly[d] : (1 << (d % 30));
        return dirs;
    }

    private static float SobolPoint(int index, int dim, int[] directions)
    {
        int d = directions.Length > dim ? directions[dim] : 2;
        int result = 0;
        int i = index;
        while (i > 0)
        {
            if ((i & 1) == 1)
                result ^= (d % (1 << 30));
            d >>= 1;
            i >>= 1;
        }
        return (float)(result & 0x7FFFFFFF) / 0x80000000;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 14 — STOCHASTIC FIELD SERIALIZER
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Serializer for saving/loading stochastic field states, exporting time series,
/// and generating VTK visualization files.
/// </summary>
public sealed unsafe class StochasticFieldSerializer
{
    /// <summary>Binary file magic number.</summary>
    private const uint Magic = 0x53544F43; // "STOC"
    /// <summary>Current format version.</summary>
    private const uint Version = 1;

    /// <summary>Saves the field state to a binary file.</summary>
    public static void SaveBinary(StochasticField field, string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);
        bw.Write(Magic);
        bw.Write(Version);
        bw.Write(field.ResX);
        bw.Write(field.ResY);
        bw.Write(field.ResZ);
        bw.Write((int)field.ProcessType);
        bw.Write(field.TimeStep);
        bw.Write(field.CurrentTime);
        bw.Write(field.StepCount);
        int total = field.TotalCells;
        bw.Write(total);
        // Write raw state data
        byte[] stateBytes = new byte[total * sizeof(StochasticFieldState)];
        Marshal.Copy((IntPtr)field.DataPointer, stateBytes, 0, stateBytes.Length);
        bw.Write(stateBytes);
    }

    /// <summary>Loads a field state from a binary file.</summary>
    public static StochasticField LoadBinary(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);
        uint magic = br.ReadUInt32();
        if (magic != Magic)
            throw new InvalidDataException($"Invalid file format: expected 0x{Magic:X8}, got 0x{magic:X8}");
        uint version = br.ReadUInt32();
        if (version > Version)
            throw new InvalidDataException($"Unsupported version: {version}");
        int resX = br.ReadInt32(), resY = br.ReadInt32(), resZ = br.ReadInt32();
        var processType = (StochasticProcessType)br.ReadInt32();
        float timeStep = br.ReadSingle();
        float currentTime = br.ReadSingle();
        long stepCount = br.ReadInt64();
        int total = br.ReadInt32();
        var field = new StochasticField(resX, resY, resZ, processType, 1.0f, 1.0f, 1.0f, timeStep);
        byte[] stateBytes = br.ReadBytes(total * sizeof(StochasticFieldState));
        Marshal.Copy(stateBytes, 0, (IntPtr)field.DataPointer, stateBytes.Length);
        return field;
    }

    /// <summary>Exports time series data for all cells or a subset.</summary>
    public static void ExportTimeSeries(StochasticField field, string filePath, int zLayer = 0,
        char delimiter = ',')
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8, 65536);
        // Header
        writer.Write("x{0}y{1}z{2}", delimiter, delimiter, delimiter);
        writer.Write("CurrentValue{0}Mean{1}Variance{1}Drift{1}Diffusion{1}Correlation{1}Entropy{1}ProcessType", delimiter, delimiter, delimiter, delimiter, delimiter, delimiter);
        writer.WriteLine();

        for (int y = 0; y < field.ResY; y++)
        {
            for (int x = 0; x < field.ResX; x++)
            {
                ref StochasticFieldState s = ref field.At(x, y, zLayer);
                writer.Write("{1}{0}{2}{0}{3}{0}{4}{0}{5}{0}{6}{0}{7}{0}{8}{0}{9}{0}{10}{0}{11}{0}{12}",
                    delimiter, x, y, zLayer, s.CurrentValue, s.Mean, s.Variance, s.Drift, s.Diffusion, s.Correlation, s.Entropy, (int)s.ProcessType);
                writer.WriteLine();
            }
        }
    }

    /// <summary>Exports the entire field as a VTK Legacy file for visualization.</summary>
    public static void ExportVTK(StochasticField field, string filePath, string scalarName = "StochasticValue")
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8, 65536);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("StochasticField Export");
        writer.WriteLine("BINARY");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {field.ResX} {field.ResY} {field.ResZ}");
        writer.WriteLine("ORIGIN 0 0 0");
        writer.WriteLine($"SPACING 1.0 1.0 1.0");
        writer.WriteLine($"POINT_DATA {field.TotalCells}");
        writer.WriteLine($"SCALARS {scalarName} float 1");
        writer.WriteLine("LOOKUP_TABLE default");

        // Write binary float data (big-endian for VTK)
        byte[] buffer = new byte[field.TotalCells * sizeof(float)];
        for (int z = 0; z < field.ResZ; z++)
            for (int y = 0; y < field.ResY; y++)
                for (int x = 0; x < field.ResX; x++)
                {
                    int idx = (z * field.ResY + y) * field.ResX + x;
                    float val = field.DataPointer[idx].CurrentValue;
                    byte[] bytes = BitConverter.GetBytes(val);
                    if (BitConverter.IsLittleEndian)
                        Array.Reverse(bytes);
                    int flatIdx = (z * field.ResY + y) * field.ResX + x;
                    Buffer.BlockCopy(bytes, 0, buffer, flatIdx * sizeof(float), sizeof(float));
                }
        writer.BaseStream.Write(buffer, 0, buffer.Length);
    }

    /// <summary>Exports VTK with multiple scalar fields.</summary>
    public static void ExportVTKMulti(StochasticField field, string filePath)
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8, 65536);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("StochasticField Multi-Scalar Export");
        writer.WriteLine("BINARY");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {field.ResX} {field.ResY} {field.ResZ}");
        writer.WriteLine("ORIGIN 0 0 0");
        writer.WriteLine($"SPACING 1.0 1.0 1.0");
        writer.WriteLine($"POINT_DATA {field.TotalCells}");

        string[] scalarNames = { "Value", "Mean", "Variance", "Drift", "Diffusion", "Correlation", "Entropy" };
        foreach (string name in scalarNames)
        {
            writer.WriteLine($"SCALARS {name} float 1");
            writer.WriteLine("LOOKUP_TABLE default");
            byte[] buffer = new byte[field.TotalCells * sizeof(float)];
            for (int i = 0; i < field.TotalCells; i++)
            {
                float val = name switch
                {
                    "Value" => field.DataPointer[i].CurrentValue,
                    "Mean" => field.DataPointer[i].Mean,
                    "Variance" => field.DataPointer[i].Variance,
                    "Drift" => field.DataPointer[i].Drift,
                    "Diffusion" => field.DataPointer[i].Diffusion,
                    "Correlation" => field.DataPointer[i].Correlation,
                    "Entropy" => field.DataPointer[i].Entropy,
                    _ => 0
                };
                byte[] bytes = BitConverter.GetBytes(val);
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(bytes);
                Buffer.BlockCopy(bytes, 0, buffer, i * sizeof(float), sizeof(float));
            }
            writer.BaseStream.Write(buffer, 0, buffer.Length);
        }
    }

    /// <summary>Exports field data as CSV for spreadsheet analysis.</summary>
    public static void ExportCSV(StochasticField field, string filePath, int zLayer = 0, char delimiter = ';')
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8, 65536);
        writer.Write("x{0}y{1}CurrentValue{2}Mean{3}Variance{4}Drift{5}Diffusion{6}Correlation{7}Entropy{8}ProcessType{9}Skewness{10}Kurtosis",
            delimiter, delimiter, delimiter, delimiter, delimiter, delimiter, delimiter, delimiter, delimiter, delimiter);
        writer.WriteLine();
        for (int y = 0; y < field.ResY; y++)
        {
            for (int x = 0; x < field.ResX; x++)
            {
                ref StochasticFieldState s = ref field.At(x, y, zLayer);
                writer.Write($"{x}{1}{y}{1}{2:F8}{1}{3:F8}{1}{4:F8}{1}{5:F8}{1}{6:F8}{1}{7:F6}{1}{8:F6}{1}{9}{1}{10:F6}{1}{11:F6}",
                    delimiter, delimiter, s.CurrentValue, s.Mean, s.Variance, s.Drift, s.Diffusion, s.Correlation, s.Entropy, (int)s.ProcessType, s.Skewness, s.Kurtosis);
                writer.WriteLine();
            }
        }
    }

    /// <summary>Exports a Monte Carlo path to CSV.</summary>
    public static void ExportPathCSV(StochasticPath path, string filePath, char delimiter = ',')
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
        writer.WriteLine($"Time{delimiter}Value{delimiter}Weight");
        for (int i = 0; i < path.Length; i++)
            writer.Write($"{path.Times[i]:F8}{delimiter}{path.Values[i]:F8}{delimiter}{path.Weight:F8}");
    }

    /// <summary>Exports multiple paths to a single CSV file.</summary>
    public static void ExportPathsCSV(StochasticPath[] paths, string filePath, char delimiter = ',')
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
        writer.Write("Time");
        for (int p = 0; p < paths.Length; p++)
            writer.Write($"{delimiter}Path{p}");
        writer.WriteLine();
        int maxLen = paths.Length > 0 ? paths.Max(p => p.Length) : 0;
        for (int i = 0; i < maxLen; i++)
        {
            writer.Write($"{(i < paths[0].Length ? paths[0].Times[i] : 0):F8}");
            for (int p = 0; p < paths.Length; p++)
                writer.Write($"{delimiter}{(i < paths[p].Length ? paths[p].Values[i] : float.NaN):F8}");
            writer.WriteLine();
        }
    }

    /// <summary>Saves a JSON summary of field statistics.</summary>
    public static void ExportStatsJSON(StochasticField field, string filePath)
    {
        var stats = field.GetStats();
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"ProcessType\": \"{stats.ProcessType}\",");
        sb.AppendLine($"  \"TotalCells\": {stats.TotalCells},");
        sb.AppendLine($"  \"MinValue\": {stats.MinValue:F8},");
        sb.AppendLine($"  \"MaxValue\": {stats.MaxValue:F8},");
        sb.AppendLine($"  \"Mean\": {stats.Mean:F8},");
        sb.AppendLine($"  \"Variance\": {stats.Variance:F8},");
        sb.AppendLine($"  \"StdDev\": {stats.StdDev:F8},");
        sb.AppendLine($"  \"Skewness\": {stats.Skewness:F6},");
        sb.AppendLine($"  \"Kurtosis\": {stats.Kurtosis:F6},");
        sb.AppendLine($"  \"Entropy\": {stats.Entropy:F6},");
        sb.AppendLine($"  \"Median\": {stats.Median:F8},");
        sb.AppendLine($"  \"IQR\": {stats.IQR:F8},");
        sb.AppendLine($"  \"Sum\": {stats.Sum:F8},");
        sb.AppendLine($"  \"TimeStep\": {stats.TimeStep:F8},");
        sb.AppendLine($"  \"ResX\": {field.ResX},");
        sb.AppendLine($"  \"ResY\": {field.ResY},");
        sb.AppendLine($"  \"ResZ\": {field.ResZ},");
        sb.AppendLine($"  \"MemoryBytes\": {field.MemoryBytes}");
        sb.AppendLine("}");
        File.WriteAllText(filePath, sb.ToString());
    }

    /// <summary>Exports field cross-section as an image-friendly text grid.</summary>
    public static void ExportTextGrid(StochasticField field, string filePath, int zLayer = 0, int width = 80, int height = 40)
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int y = 0; y < field.ResY; y++)
            for (int x = 0; x < field.ResX; x++)
            {
                float v = field.At(x, y, zLayer).CurrentValue;
                if (v < minVal)
                    minVal = v;
                if (v > maxVal)
                    maxVal = v;
            }
        float range = maxVal - minVal + 1e-10f;
        char[] ascii = { '.', ',', '-', '~', ':', ';', '=', '!', '*', '#', '$', '@' };
        for (int row = 0; row < height; row++)
        {
            int y = (int)((float)row / height * field.ResY);
            y = Math.Clamp(y, 0, field.ResY - 1);
            for (int col = 0; col < width; col++)
            {
                int x = (int)((float)col / width * field.ResX);
                x = Math.Clamp(x, 0, field.ResX - 1);
                float v = field.At(x, y, zLayer).CurrentValue;
                int charIdx = (int)((v - minVal) / range * (ascii.Length - 1));
                charIdx = Math.Clamp(charIdx, 0, ascii.Length - 1);
                writer.Write(ascii[charIdx]);
            }
            writer.WriteLine();
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 15 — EXTENDED STOCHASTIC FIELD OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Extension methods and advanced operations for StochasticField.
/// Provides batch operations, field arithmetic, convolution, and
/// derivative-free optimization over the field landscape.
/// </summary>
public static unsafe class StochasticFieldExtensions
{
    /// <summary>Computes the element-wise sum of two fields into a result field.</summary>
    public static void Add(StochasticField a, StochasticField b, StochasticField result)
    {
        if (a.ResX != b.ResX || a.ResY != b.ResY || a.ResZ != b.ResZ)
            throw new ArgumentException("Field dimensions must match for addition.");
        for (int z = 0; z < a.ResZ; z++)
            for (int y = 0; y < a.ResY; y++)
                for (int x = 0; x < a.ResX; x++)
                {
                    int idx = (z * a.ResY + y) * a.ResX + x;
                    result.DataPointer[idx].CurrentValue = a.DataPointer[idx].CurrentValue + b.DataPointer[idx].CurrentValue;
                    result.DataPointer[idx].Mean = a.DataPointer[idx].Mean + b.DataPointer[idx].Mean;
                    result.DataPointer[idx].Variance = a.DataPointer[idx].Variance + b.DataPointer[idx].Variance;
                }
    }

    /// <summary>Computes element-wise multiplication (Hadamard product) of two fields.</summary>
    public static void Multiply(StochasticField a, StochasticField b, StochasticField result)
    {
        if (a.ResX != b.ResX || a.ResY != b.ResY || a.ResZ != b.ResZ)
            throw new ArgumentException("Field dimensions must match for multiplication.");
        for (int z = 0; z < a.ResZ; z++)
            for (int y = 0; y < a.ResY; y++)
                for (int x = 0; x < a.ResX; x++)
                {
                    int idx = (z * a.ResY + y) * a.ResX + x;
                    result.DataPointer[idx].CurrentValue = a.DataPointer[idx].CurrentValue * b.DataPointer[idx].CurrentValue;
                }
    }

    /// <summary>Scales all field values by a constant factor.</summary>
    public static void Scale(StochasticField field, float scalar)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue *= scalar;
    }

    /// <summary>Computes the L2 norm of the field values.</summary>
    public static float L2Norm(StochasticField field)
    {
        float sum = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; sum += v * v; }
        return MathF.Sqrt(sum);
    }

    /// <summary>Computes the L-infinity norm (maximum absolute value).</summary>
    public static float LinfNorm(StochasticField field)
    {
        float max = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        { float v = MathF.Abs(field.DataPointer[i].CurrentValue); if (v > max) max = v; }
        return max;
    }

    /// <summary>Computes the energy norm: sqrt(sum(val^2 * spacing^3)).</summary>
    public static float EnergyNorm(StochasticField field)
    {
        float sum = 0, dV = 1.0f;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; sum += v * v; }
        return MathF.Sqrt(sum * dV);
    }

    /// <summary>Computes the pointwise maximum of two fields.</summary>
    public static void PointwiseMax(StochasticField a, StochasticField b, StochasticField result)
    {
        for (int z = 0; z < a.ResZ; z++)
            for (int y = 0; y < a.ResY; y++)
                for (int x = 0; x < a.ResX; x++)
                {
                    int idx = (z * a.ResY + y) * a.ResX + x;
                    result.DataPointer[idx].CurrentValue = MathF.Max(a.DataPointer[idx].CurrentValue, b.DataPointer[idx].CurrentValue);
                }
    }

    /// <summary>Computes the absolute value of each field cell.</summary>
    public static void Abs(StochasticField field)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = MathF.Abs(field.DataPointer[i].CurrentValue);
    }

    /// <summary>Clamps all field values to [min, max].</summary>
    public static void Clamp(StochasticField field, float min, float max)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = Math.Clamp(field.DataPointer[i].CurrentValue, min, max);
    }

    /// <summary>Applies a user-defined function element-wise to all field values.</summary>
    public static void ApplyFunction(StochasticField field, Func<float, float> func)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = func(field.DataPointer[i].CurrentValue);
    }

    /// <summary>Computes the 3D convolution of the field with a kernel.</summary>
    public static void Convolve3D(StochasticField field, float* kernel, int kernelSize, StochasticField output)
    {
        int half = kernelSize / 2;
        for (int z = half; z < field.ResZ - half; z++)
            for (int y = half; y < field.ResY - half; y++)
                for (int x = half; x < field.ResX - half; x++)
                {
                    float sum = 0;
                    for (int kz = 0; kz < kernelSize; kz++)
                        for (int ky = 0; ky < kernelSize; ky++)
                            for (int kx = 0; kx < kernelSize; kx++)
                            {
                                int fx = x + kx - half, fy = y + ky - half, fz = z + kz - half;
                                sum += field.At(fx, fy, fz).CurrentValue * kernel[kz * kernelSize * kernelSize + ky * kernelSize + kx];
                            }
                    output.At(x, y, z).CurrentValue = sum;
                }
    }

    /// <summary>Applies a Gaussian blur with given sigma to the field.</summary>
    public static void GaussianBlur(StochasticField field, float sigma)
    {
        int kernelSize = (int)(6 * sigma) | 1;
        if (kernelSize < 3)
            kernelSize = 3;
        int half = kernelSize / 2;
        float* kernel = (float*)NativeMemory.AlignedAlloc((nuint)(kernelSize * kernelSize * kernelSize * sizeof(float)), 64);
        float sigma2 = sigma * sigma, norm = 0;
        for (int kz = 0; kz < kernelSize; kz++)
            for (int ky = 0; ky < kernelSize; ky++)
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    float dz = kz - half, dy = ky - half, dx = kx - half;
                    float val = MathF.Exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma2));
                    kernel[kz * kernelSize * kernelSize + ky * kernelSize + kx] = val;
                    norm += val;
                }
        for (int i = 0; i < kernelSize * kernelSize * kernelSize; i++)
            kernel[i] /= norm;
        var output = new StochasticField(field.ResX, field.ResY, field.ResZ, field.ProcessType, 1.0f, 1.0f, 1.0f);
        Convolve3D(field, kernel, kernelSize, output);
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = output.DataPointer[i].CurrentValue;
        NativeMemory.AlignedFree(kernel);
        output.Dispose();
    }

    /// <summary>Computes the Fourier transform of the field magnitude (1D slice).</summary>
    public static (float[] Frequencies, float[] Magnitudes) FFT1D(StochasticField field, int sliceY = 0, int sliceZ = 0)
    {
        int N = field.ResX;
        float[] re = new float[N], im = new float[N];
        for (int x = 0; x < N; x++)
            re[x] = field.At(x, sliceY, sliceZ).CurrentValue;
        // Radix-2 DIT FFT (simplified)
        int logN = 0;
        int temp = N;
        while (temp > 1)
        { logN++; temp >>= 1; }
        // Bit-reversal permutation
        for (int i = 0; i < N; i++)
        {
            int j = 0, x = i;
            for (int k = 0; k < logN; k++)
            { j = (j << 1) | (x & 1); x >>= 1; }
            if (j > i)
            { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
        }
        // Butterfly operations
        for (int size = 2; size <= N; size *= 2)
        {
            int half = size / 2;
            float angle = -2.0f * MathF.PI / size;
            float wRe = MathF.Cos(angle), wIm = MathF.Sin(angle);
            for (int i = 0; i < N; i += size)
            {
                float curRe = 1, curIm = 0;
                for (int j = 0; j < half; j++)
                {
                    int a = i + j, b = i + j + half;
                    float tRe = curRe * re[b] - curIm * im[b];
                    float tIm = curRe * im[b] + curIm * re[b];
                    re[b] = re[a] - tRe;
                    im[b] = im[a] - tIm;
                    re[a] += tRe;
                    im[a] += tIm;
                    float newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
        float[] freqs = new float[N], mags = new float[N];
        for (int k = 0; k < N; k++)
        { freqs[k] = (float)k / N; mags[k] = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]) / N; }
        return (freqs, mags);
    }

    /// <summary>Computes the power spectral density of the field.</summary>
    public static float[] PowerSpectralDensity(StochasticField field, int sliceY = 0, int sliceZ = 0)
    {
        var (_, mags) = FFT1D(field, sliceY, sliceZ);
        float[] psd = new float[mags.Length];
        for (int i = 0; i < mags.Length; i++)
            psd[i] = mags[i] * mags[i];
        return psd;
    }

    /// <summary>Computes the spatial autocorrelation function along the X axis.</summary>
    public static float[] Autocorrelation(StochasticField field, int maxLag, int sliceY = 0, int sliceZ = 0)
    {
        int N = field.ResX;
        float mean = 0;
        for (int x = 0; x < N; x++)
            mean += field.At(x, sliceY, sliceZ).CurrentValue;
        mean /= N;
        float variance = 0;
        for (int x = 0; x < N; x++)
        { float v = field.At(x, sliceY, sliceZ).CurrentValue - mean; variance += v * v; }
        variance /= N;
        float[] acf = new float[maxLag + 1];
        for (int lag = 0; lag <= maxLag; lag++)
        {
            float sum = 0;
            for (int x = 0; x < N - lag; x++)
            {
                float a = field.At(x, sliceY, sliceZ).CurrentValue - mean;
                float b = field.At(x + lag, sliceY, sliceZ).CurrentValue - mean;
                sum += a * b;
            }
            acf[lag] = variance > 1e-10f ? sum / ((N - lag) * variance) : 0;
        }
        return acf;
    }

    /// <summary>Computes the gradient magnitude field.</summary>
    public static void GradientMagnitude(StochasticField field, StochasticField output)
    {
        for (int z = 1; z < field.ResZ - 1; z++)
            for (int y = 1; y < field.ResY - 1; y++)
                for (int x = 1; x < field.ResX - 1; x++)
                {
                    float gx = (field.At(x + 1, y, z).CurrentValue - field.At(x - 1, y, z).CurrentValue) * 0.5f;
                    float gy = (field.At(x, y + 1, z).CurrentValue - field.At(x, y - 1, z).CurrentValue) * 0.5f;
                    float gz = (field.At(x, y, z + 1).CurrentValue - field.At(x, y, z - 1).CurrentValue) * 0.5f;
                    output.At(x, y, z).CurrentValue = MathF.Sqrt(gx * gx + gy * gy + gz * gz);
                }
    }

    /// <summary>Finds local maxima (peaks) in the field.</summary>
    public static List<(int X, int Y, int Z, float Value)> FindLocalMaxima(StochasticField field, float threshold = float.MinValue)
    {
        var maxima = new List<(int, int, int, float)>();
        for (int z = 1; z < field.ResZ - 1; z++)
            for (int y = 1; y < field.ResY - 1; y++)
                for (int x = 1; x < field.ResX - 1; x++)
                {
                    float v = field.At(x, y, z).CurrentValue;
                    if (v < threshold)
                        continue;
                    bool isMax = true;
                    for (int dz = -1; dz <= 1 && isMax; dz++)
                        for (int dy = -1; dy <= 1 && isMax; dy++)
                            for (int dx = -1; dx <= 1 && isMax; dx++)
                            { if (dx == 0 && dy == 0 && dz == 0) continue; if (field.At(x + dx, y + dy, z + dz).CurrentValue >= v) isMax = false; }
                    if (isMax)
                        maxima.Add((x, y, z, v));
                }
        return maxima;
    }

    /// <summary>Finds local minima (valleys) in the field.</summary>
    public static List<(int X, int Y, int Z, float Value)> FindLocalMinima(StochasticField field, float threshold = float.MaxValue)
    {
        var minima = new List<(int, int, int, float)>();
        for (int z = 1; z < field.ResZ - 1; z++)
            for (int y = 1; y < field.ResY - 1; y++)
                for (int x = 1; x < field.ResX - 1; x++)
                {
                    float v = field.At(x, y, z).CurrentValue;
                    if (v > threshold)
                        continue;
                    bool isMin = true;
                    for (int dz = -1; dz <= 1 && isMin; dz++)
                        for (int dy = -1; dy <= 1 && isMin; dy++)
                            for (int dx = -1; dx <= 1 && isMin; dx++)
                            { if (dx == 0 && dy == 0 && dz == 0) continue; if (field.At(x + dx, y + dy, z + dz).CurrentValue <= v) isMin = false; }
                    if (isMin)
                        minima.Add((x, y, z, v));
                }
        return minima;
    }

    /// <summary>Computes the histogram of field values.</summary>
    public static (float[] BinEdges, int[] Counts) Histogram(StochasticField field, int numBins = 50)
    {
        int total = field.TotalCells;
        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
        float binWidth = (maxVal - minVal) / numBins + 1e-10f;
        float[] edges = new float[numBins + 1];
        int[] counts = new int[numBins];
        for (int b = 0; b <= numBins; b++)
            edges[b] = minVal + b * binWidth;
        for (int i = 0; i < total; i++)
        {
            float v = field.DataPointer[i].CurrentValue;
            int bin = Math.Clamp((int)((v - minVal) / binWidth), 0, numBins - 1);
            counts[bin]++;
        }
        return (edges, counts);
    }

    /// <summary>Computes the percentile value from the field.</summary>
    public static float Percentile(StochasticField field, float percentile)
    {
        int total = field.TotalCells;
        Span<float> values = stackalloc float[Math.Min(total, 50000)];
        int sampleCount = Math.Min(total, 50000), step = Math.Max(1, total / sampleCount);
        for (int i = 0; i < sampleCount; i++)
            values[i] = field.DataPointer[i * step].CurrentValue;
        values.Sort();
        int idx = (int)(percentile * (sampleCount - 1));
        return values[idx];
    }

    /// <summary>Applies sigmoid activation to all field values.</summary>
    public static void Sigmoid(StochasticField field)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = 1.0f / (1.0f + MathF.Exp(-field.DataPointer[i].CurrentValue));
    }

    /// <summary>Applies ReLU activation to all field values.</summary>
    public static void ReLU(StochasticField field)
    {
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = MathF.Max(field.DataPointer[i].CurrentValue, 0);
    }

    /// <summary>Applies softmax normalization across the field.</summary>
    public static void Softmax(StochasticField field)
    {
        int total = field.TotalCells;
        float maxVal = float.MinValue;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; if (v > maxVal) maxVal = v; }
        float sumExp = 0;
        for (int i = 0; i < total; i++)
        { field.DataPointer[i].CurrentValue = MathF.Exp(field.DataPointer[i].CurrentValue - maxVal); sumExp += field.DataPointer[i].CurrentValue; }
        float invSum = 1.0f / (sumExp + 1e-10f);
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue *= invSum;
    }

    /// <summary>Normalizes field values to [0,1] range.</summary>
    public static void Normalize(StochasticField field)
    {
        int total = field.TotalCells;
        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
        float range = maxVal - minVal + 1e-10f;
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = (field.DataPointer[i].CurrentValue - minVal) / range;
    }

    /// <summary>Normalizes field values to zero mean, unit variance.</summary>
    public static void ZScoreNormalize(StochasticField field)
    {
        int total = field.TotalCells;
        float sum = 0, sumSq = 0;
        for (int i = 0; i < total; i++)
        { float v = field.DataPointer[i].CurrentValue; sum += v; sumSq += v * v; }
        float mean = sum / total, variance = sumSq / total - mean * mean, std = MathF.Sqrt(MathF.Max(variance, 1e-10f));
        for (int i = 0; i < total; i++)
            field.DataPointer[i].CurrentValue = (field.DataPointer[i].CurrentValue - mean) / std;
    }

    /// <summary>Creates a copy of the field state.</summary>
    public static StochasticField Clone(StochasticField field)
    {
        var clone = new StochasticField(field.ResX, field.ResY, field.ResZ, field.ProcessType, 1.0f, 1.0f, 1.0f, field.TimeStep);
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            clone.DataPointer[i] = field.DataPointer[i];
        return clone;
    }

    /// <summary>Merges two fields by averaging their values.</summary>
    public static void Average(StochasticField a, StochasticField b, StochasticField result)
    {
        for (int z = 0; z < a.ResZ; z++)
            for (int y = 0; y < a.ResY; y++)
                for (int x = 0; x < a.ResX; x++)
                {
                    int idx = (z * a.ResY + y) * a.ResX + x;
                    result.DataPointer[idx].CurrentValue = 0.5f * (a.DataPointer[idx].CurrentValue + b.DataPointer[idx].CurrentValue);
                    result.DataPointer[idx].Mean = 0.5f * (a.DataPointer[idx].Mean + b.DataPointer[idx].Mean);
                }
    }

    /// <summary>Computes the root mean square error between two fields.</summary>
    public static float RMSE(StochasticField a, StochasticField b)
    {
        float sumSq = 0;
        int total = a.TotalCells;
        for (int i = 0; i < total; i++)
        { float diff = a.DataPointer[i].CurrentValue - b.DataPointer[i].CurrentValue; sumSq += diff * diff; }
        return MathF.Sqrt(sumSq / total);
    }

    /// <summary>Computes the mean absolute error between two fields.</summary>
    public static float MAE(StochasticField a, StochasticField b)
    {
        float sumAbs = 0;
        int total = a.TotalCells;
        for (int i = 0; i < total; i++)
            sumAbs += MathF.Abs(a.DataPointer[i].CurrentValue - b.DataPointer[i].CurrentValue);
        return sumAbs / total;
    }

    /// <summary>Computes the cosine similarity between two fields.</summary>
    public static float CosineSimilarity(StochasticField a, StochasticField b)
    {
        float dot = 0, magA = 0, magB = 0;
        int total = a.TotalCells;
        for (int i = 0; i < total; i++)
        {
            float va = a.DataPointer[i].CurrentValue, vb = b.DataPointer[i].CurrentValue;
            dot += va * vb;
            magA += va * va;
            magB += vb * vb;
        }
        return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB) + 1e-10f);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 16 — STOCHASTIC FIELD ANALYTICS
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Advanced analytics for stochastic fields: risk measures, sensitivity analysis,
/// and information-theoretic diagnostics.
/// </summary>
public sealed unsafe class StochasticFieldAnalytics
{
    private readonly StochasticField _field;
    private readonly Random _rng;

    public StochasticFieldAnalytics(StochasticField field, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _rng = new Random(seed);
    }

    /// <summary>Computes Value at Risk (VaR) at given confidence level across the field.</summary>
    public float ValueAtRisk(float confidenceLevel = 0.95f)
    {
        int total = _field.TotalCells;
        Span<float> values = stackalloc float[Math.Min(total, 50000)];
        int n = Math.Min(total, 50000), step = Math.Max(1, total / n);
        for (int i = 0; i < n; i++)
            values[i] = _field.DataPointer[i * step].CurrentValue;
        values.Sort();
        int idx = (int)((1 - confidenceLevel) * (n - 1));
        return values[idx];
    }

    /// <summary>Computes Conditional Value at Risk (CVaR / Expected Shortfall).</summary>
    public float ConditionalValueAtRisk(float confidenceLevel = 0.95f)
    {
        int total = _field.TotalCells;
        Span<float> values = stackalloc float[Math.Min(total, 50000)];
        int n = Math.Min(total, 50000), step = Math.Max(1, total / n);
        for (int i = 0; i < n; i++)
            values[i] = _field.DataPointer[i * step].CurrentValue;
        values.Sort();
        int cutoff = (int)((1 - confidenceLevel) * (n - 1));
        if (cutoff <= 0)
            return values[0];
        float sum = 0;
        for (int i = 0; i <= cutoff; i++)
            sum += values[i];
        return sum / (cutoff + 1);
    }

    /// <summary>Computes the maximum drawdown across the field history.</summary>
    public float MaxDrawdown()
    {
        int total = _field.TotalCells;
        float peak = float.MinValue, maxDD = 0;
        for (int i = 0; i < total; i++)
        {
            float v = _field.DataPointer[i].CurrentValue;
            if (v > peak)
                peak = v;
            float dd = (peak - v) / (MathF.Abs(peak) + 1e-10f);
            if (dd > maxDD)
                maxDD = dd;
        }
        return maxDD;
    }

    /// <summary>Computes the Sharpe ratio across the field.</summary>
    public float SharpeRatio(float riskFreeRate = 0.02f)
    {
        int total = _field.TotalCells;
        float sum = 0, sumSq = 0;
        for (int i = 0; i < total; i++)
        { float v = _field.DataPointer[i].CurrentValue; sum += v; sumSq += v * v; }
        float mean = sum / total, variance = sumSq / total - mean * mean;
        float std = MathF.Sqrt(MathF.Max(variance, 1e-10f));
        return (mean - riskFreeRate) / std;
    }

    /// <summary>Computes the Sortino ratio (downside deviation only).</summary>
    public float SortinoRatio(float riskFreeRate = 0.02f, float targetReturn = 0)
    {
        int total = _field.TotalCells;
        float sum = 0, downsideSumSq = 0;
        for (int i = 0; i < total; i++)
        {
            float v = _field.DataPointer[i].CurrentValue;
            sum += v;
            float downside = MathF.Min(v - targetReturn, 0);
            downsideSumSq += downside * downside;
        }
        float mean = sum / total;
        float downsideDev = MathF.Sqrt(downsideSumSq / total + 1e-10f);
        return (mean - riskFreeRate) / downsideDev;
    }

    /// <summary>Computes field entropy using Shannon information theory.</summary>
    public float ShannonEntropy(int numBins = 50)
    {
        var (edges, counts) = StochasticFieldExtensions.Histogram(_field, numBins);
        int total = _field.TotalCells;
        float entropy = 0;
        for (int b = 0; b < numBins; b++)
        {
            float p = (float)counts[b] / total;
            if (p > 1e-10f)
                entropy -= p * MathF.Log2(p);
        }
        return entropy;
    }

    /// <summary>Computes the Kullback-Leibler divergence between two field distributions.</summary>
    public static float KullbackLeiblerDivergence(StochasticField p, StochasticField q, int numBins = 50)
    {
        var (pEdges, pCounts) = StochasticFieldExtensions.Histogram(p, numBins);
        var (_, qCounts) = StochasticFieldExtensions.Histogram(q, numBins);
        int pTotal = p.TotalCells, qTotal = q.TotalCells;
        float kl = 0;
        for (int b = 0; b < numBins; b++)
        {
            float pVal = (float)pCounts[b] / pTotal;
            float qVal = (float)qCounts[b] / qTotal;
            if (pVal > 1e-10f && qVal > 1e-10f)
                kl += pVal * MathF.Log(pVal / qVal);
        }
        return kl;
    }

    /// <summary>Computes the mutual information between two fields.</summary>
    public static float MutualInformation(StochasticField x, StochasticField y, int numBins = 20)
    {
        float hx = new StochasticFieldAnalytics(x).ShannonEntropy(numBins);
        float hy = new StochasticFieldAnalytics(y).ShannonEntropy(numBins);
        // Create joint field by adding values
        var joint = StochasticFieldExtensions.Clone(x);
        for (int i = 0; i < joint.TotalCells; i++)
            joint.DataPointer[i].CurrentValue = x.DataPointer[i].CurrentValue + y.DataPointer[i].CurrentValue;
        float hxy = new StochasticFieldAnalytics(joint).ShannonEntropy(numBins);
        joint.Dispose();
        return hx + hy - hxy;
    }

    /// <summary>Computes sensitivity of field output to parameter perturbation.</summary>
    public float[] ComputeSensitivity(int parameterIndex, float perturbation = 0.01f)
    {
        // Perturb each cell's drift parameter and measure output change
        int total = _field.TotalCells;
        float[] sensitivities = new float[total];
        float baseMean = 0;
        for (int i = 0; i < total; i++)
            baseMean += _field.DataPointer[i].CurrentValue;
        baseMean /= total;

        for (int i = 0; i < total; i++)
        {
            float origDrift = _field.DataPointer[i].Drift;
            _field.DataPointer[i].Drift = origDrift * (1 + perturbation);
            // Measure local change
            float localBase = _field.DataPointer[i].CurrentValue;
            _field.DataPointer[i].CurrentValue = localBase * (1 + origDrift * perturbation);
            float newMean = 0;
            for (int j = 0; j < total; j++)
                newMean += _field.DataPointer[j].CurrentValue;
            newMean /= total;
            sensitivities[i] = (newMean - baseMean) / (perturbation * origDrift + 1e-10f);
            _field.DataPointer[i].Drift = origDrift;
            _field.DataPointer[i].CurrentValue = localBase;
        }
        return sensitivities;
    }

    /// <summary>Computes the correlation matrix between field cells (sampled).</summary>
    public float* ComputeCorrelationMatrix(int maxCells = 100)
    {
        int n = Math.Min(maxCells, _field.TotalCells);
        float* matrix = (float*)NativeMemory.AlignedAlloc((nuint)(n * n * sizeof(float)), 64);
        // Sample cell values
        Span<float> values = stackalloc float[n];
        int step = Math.Max(1, _field.TotalCells / n);
        for (int i = 0; i < n; i++)
            values[i] = _field.DataPointer[i * step].CurrentValue;
        // Compute means
        float mean = 0;
        for (int i = 0; i < n; i++)
            mean += values[i];
        mean /= n;
        // Compute variance
        float variance = 0;
        for (int i = 0; i < n; i++)
        { float d = values[i] - mean; variance += d * d; }
        variance /= n;
        float invStd = 1.0f / (MathF.Sqrt(variance) + 1e-10f);
        // Compute correlation
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                float ci = values[i] - mean, cj = values[j] - mean;
                matrix[i * n + j] = ci * cj * invStd * invStd;
            }
        return matrix;
    }

    /// <summary>Computes the Hurst exponent via rescaled range (R/S) analysis.</summary>
    public float EstimateHurstExponent(int maxLag = 100)
    {
        int total = _field.TotalCells;
        Span<float> values = stackalloc float[total];
        for (int i = 0; i < total; i++)
            values[i] = _field.DataPointer[i].CurrentValue;
        var logRS = new List<float>();
        var logN = new List<float>();
        for (int n = 10; n <= Math.Min(maxLag, total / 2); n *= 2)
        {
            int numSegments = total / n;
            float rsSum = 0;
            for (int seg = 0; seg < numSegments; seg++)
            {
                float segMean = 0;
                for (int i = seg * n; i < (seg + 1) * n; i++)
                    segMean += values[i];
                segMean /= n;
                float maxR = 0, minR = 0, cumR = 0;
                for (int i = seg * n; i < (seg + 1) * n; i++)
                {
                    cumR += values[i] - segMean;
                    if (cumR > maxR)
                        maxR = cumR;
                    if (cumR < minR)
                        minR = cumR;
                }
                rsSum += (maxR - minR);
            }
            float rs = rsSum / numSegments;
            if (rs > 1e-10f)
            { logRS.Add(MathF.Log(rs)); logN.Add(MathF.Log(n)); }
        }
        if (logRS.Count < 2)
            return 0.5f;
        // Linear regression for slope (Hurst exponent)
        float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int count = logRS.Count;
        for (int i = 0; i < count; i++)
        { sumX += logN[i]; sumY += logRS[i]; sumXY += logN[i] * logRS[i]; sumX2 += logN[i] * logN[i]; }
        float slope = (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX + 1e-10f);
        return Math.Clamp(slope, 0.01f, 0.99f);
    }

    /// <summary>Estimates the fractal dimension via box-counting.</summary>
    public float EstimateFractalDimension(int maxBoxSize = 64)
    {
        int total = _field.TotalCells;
        var logCount = new List<float>();
        var logInvSize = new List<float>();
        float threshold = StochasticFieldExtensions.Percentile(_field, 0.5f);
        for (int boxSize = 2; boxSize <= Math.Min(maxBoxSize, Math.Min(_field.ResX, Math.Min(_field.ResY, _field.ResZ))); boxSize *= 2)
        {
            HashSet<(int, int, int)> occupied = new();
            for (int z = 0; z < _field.ResZ; z += boxSize)
                for (int y = 0; y < _field.ResY; y += boxSize)
                    for (int x = 0; x < _field.ResX; x += boxSize)
                    {
                        if (_field.At(x, y, z).CurrentValue > threshold)
                            occupied.Add((x / boxSize, y / boxSize, z / boxSize));
                    }
            if (occupied.Count > 0)
            { logCount.Add(MathF.Log(occupied.Count)); logInvSize.Add(MathF.Log(1.0f / boxSize)); }
        }
        if (logCount.Count < 2)
            return 1.0f;
        float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int c = logCount.Count;
        for (int i = 0; i < c; i++)
        { sumX += logInvSize[i]; sumY += logCount[i]; sumXY += logInvSize[i] * logCount[i]; sumX2 += logInvSize[i] * logInvSize[i]; }
        return (c * sumXY - sumX * sumY) / (c * sumX2 - sumX * sumX + 1e-10f);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 17 — STOCHASTIC FIELD DIFFERENTIATION
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Automatic differentiation support for stochastic fields.
/// Computes exact gradients of field quantities with respect to parameters,
/// enabling gradient-based optimization and sensitivity analysis.
/// </summary>
public sealed unsafe class StochasticFieldDifferentiator
{
    private readonly StochasticField _field;
    private float* _adjointBuffer;
    private float* _tangentBuffer;
    private bool _disposed;

    public StochasticFieldDifferentiator(StochasticField field)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        int total = field.TotalCells;
        _adjointBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        _tangentBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
        NativeMemory.Clear(_adjointBuffer, (nuint)(total * sizeof(float)));
        NativeMemory.Clear(_tangentBuffer, (nuint)(total * sizeof(float)));
    }

    /// <summary>
    /// Loss over a flat field snapshot. Prefer this over raw pointer Func delegates
    /// (which cannot carry unmanaged pointer types in generic Func&lt;&gt;).
    /// </summary>
    public delegate float FieldLossFunction(ReadOnlySpan<float> values);

    /// <summary>
    /// Computes the gradient of a scalar loss with respect to all field values
    /// using reverse-mode automatic differentiation (backpropagation through time).
    /// </summary>
    /// <param name="lossFunction">Loss function over a terminal field snapshot.</param>
    /// <param name="numTimesteps">Number of simulation steps to differentiate through.</param>
    /// <param name="dt">Time-step for forward simulation.</param>
    /// <returns>Pointer to the internal adjoint buffer (same size as field). Valid until next call / dispose.</returns>
    public float* ComputeGradient(FieldLossFunction lossFunction, int numTimesteps, float dt)
    {
        ArgumentNullException.ThrowIfNull(lossFunction);
        if (numTimesteps < 0)
            throw new ArgumentOutOfRangeException(nameof(numTimesteps));

        int total = _field.TotalCells;
        float** snapshots = (float**)NativeMemory.AlignedAlloc((nuint)((numTimesteps + 1) * sizeof(float*)), 64);
        try
        {
            for (int t = 0; t <= numTimesteps; t++)
            {
                snapshots[t] = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
                for (int i = 0; i < total; i++)
                    snapshots[t][i] = _field.DataPointer[i].CurrentValue;
                if (t < numTimesteps)
                    _field.StepEulerMaruyama(dt);
            }

            float loss = lossFunction(new ReadOnlySpan<float>(snapshots[numTimesteps], total));

            NativeMemory.Clear(_adjointBuffer, (nuint)(total * sizeof(float)));

            // Finite-difference seed of terminal adjoint (loss may be black-box).
            float epsilon = 1e-5f;
            float* terminalScratch = (float*)NativeMemory.AlignedAlloc((nuint)(total * sizeof(float)), 64);
            try
            {
                for (int i = 0; i < total; i++)
                    terminalScratch[i] = snapshots[numTimesteps][i];
                for (int i = 0; i < total; i++)
                {
                    terminalScratch[i] = snapshots[numTimesteps][i] + epsilon;
                    float lossPlus = lossFunction(new ReadOnlySpan<float>(terminalScratch, total));
                    terminalScratch[i] = snapshots[numTimesteps][i];
                    _adjointBuffer[i] = (lossPlus - loss) / epsilon;
                }
            }
            finally
            {
                NativeMemory.AlignedFree(terminalScratch);
            }

            for (int t = numTimesteps - 1; t >= 0; t--)
            {
                for (int i = 0; i < total; i++)
                {
                    float x = snapshots[t][i];
                    float mu = _field.DataPointer[i].Drift;
                    float dXtdXtm1 = 1 + mu * dt;
                    _adjointBuffer[i] *= dXtdXtm1;
                    _field.DataPointer[i].Drift += _adjointBuffer[i] * x * dt * 0.001f;
                }
            }

            for (int i = 0; i < total; i++)
                _field.DataPointer[i].CurrentValue = snapshots[0][i];

            return _adjointBuffer;
        }
        finally
        {
            for (int t = 0; t <= numTimesteps; t++)
            {
                if (snapshots[t] != null)
                    NativeMemory.AlignedFree(snapshots[t]);
            }
            NativeMemory.AlignedFree(snapshots);
        }
    }

    /// <summary>
    /// Convenience overload: mean-squared error against a target field snapshot.
    /// </summary>
    public float* ComputeGradientMse(ReadOnlySpan<float> target, int numTimesteps, float dt)
    {
        if (target.Length != _field.TotalCells)
            throw new ArgumentException("Target length must match field cell count.", nameof(target));

        // Copy target for closure safety
        float[] targetCopy = target.ToArray();
        return ComputeGradient(values =>
        {
            float sum = 0;
            int n = values.Length;
            for (int i = 0; i < n; i++)
            {
                float d = values[i] - targetCopy[i];
                sum += d * d;
            }
            return sum / Math.Max(1, n);
        }, numTimesteps, dt);
    }

    /// <summary>
    /// Computes the tangent linear model: forward sensitivity of field values
    /// with respect to an initial perturbation direction.
    /// </summary>
    /// <param name="perturbation">Initial perturbation direction.</param>
    /// <param name="numTimesteps">Number of simulation steps.</param>
    /// <param name="dt">Time-step.</param>
    public float* ComputeTangentLinear(float* perturbation, int numTimesteps, float dt)
    {
        int total = _field.TotalCells;
        // Copy perturbation to tangent buffer
        for (int i = 0; i < total; i++)
            _tangentBuffer[i] = perturbation[i];

        // Forward propagate tangent through linearized dynamics
        for (int t = 0; t < numTimesteps; t++)
        {
            float sqrtDt = MathF.Sqrt(dt);
            float[] newTangent = new float[total];
            for (int i = 0; i < total; i++)
            {
                float x = _field.DataPointer[i].CurrentValue;
                float mu = _field.DataPointer[i].Drift;
                float sigma = _field.DataPointer[i].Diffusion;
                // Linearized Euler-Maruyama: d(δX) = μ δX dt + σ δX dW
                float dW = sqrtDt * (float)(Random.Shared.NextDouble() * 2 - 1);
                newTangent[i] = _tangentBuffer[i] + mu * _tangentBuffer[i] * dt + sigma * _tangentBuffer[i] * dW;
            }
            for (int i = 0; i < total; i++)
                _tangentBuffer[i] = newTangent[i];
            _field.StepEulerMaruyama(dt);
        }
        return _tangentBuffer;
    }

    /// <summary>
    /// Computes finite-difference gradient of a scalar objective with respect
    /// to a single parameter (drift, diffusion, etc.) at a specific cell.
    /// </summary>
    public float ComputeParameterGradient(Func<StochasticField, float> objective, int cellX, int cellY, int cellZ,
        string parameterName, float epsilon = 1e-5f)
    {
        int idx = (cellZ * _field.ResY + cellY) * _field.ResX + cellX;
        float origValue;
        float gradient;

        switch (parameterName.ToLowerInvariant())
        {
            case "drift":
                origValue = _field.DataPointer[idx].Drift;
                _field.DataPointer[idx].Drift = origValue + epsilon;
                float lossPlus = objective(_field);
                _field.DataPointer[idx].Drift = origValue - epsilon;
                float lossMinus = objective(_field);
                gradient = (lossPlus - lossMinus) / (2 * epsilon);
                _field.DataPointer[idx].Drift = origValue;
                break;
            case "diffusion":
                origValue = _field.DataPointer[idx].Diffusion;
                _field.DataPointer[idx].Diffusion = origValue + epsilon;
                lossPlus = objective(_field);
                _field.DataPointer[idx].Diffusion = origValue - epsilon;
                lossMinus = objective(_field);
                gradient = (lossPlus - lossMinus) / (2 * epsilon);
                _field.DataPointer[idx].Diffusion = origValue;
                break;
            case "value":
                origValue = _field.DataPointer[idx].CurrentValue;
                _field.DataPointer[idx].CurrentValue = origValue + epsilon;
                lossPlus = objective(_field);
                _field.DataPointer[idx].CurrentValue = origValue - epsilon;
                lossMinus = objective(_field);
                gradient = (lossPlus - lossMinus) / (2 * epsilon);
                _field.DataPointer[idx].CurrentValue = origValue;
                break;
            case "mean":
                origValue = _field.DataPointer[idx].Mean;
                _field.DataPointer[idx].Mean = origValue + epsilon;
                lossPlus = objective(_field);
                _field.DataPointer[idx].Mean = origValue - epsilon;
                lossMinus = objective(_field);
                gradient = (lossPlus - lossMinus) / (2 * epsilon);
                _field.DataPointer[idx].Mean = origValue;
                break;
            default:
                throw new ArgumentException($"Unknown parameter: {parameterName}");
        }
        return gradient;
    }

    /// <summary>Computes the Jacobian matrix of field outputs w.r.t. all parameters.</summary>
    public float* ComputeJacobian(Func<StochasticField, float[]> outputFunction, int outputDim, int parameterDim)
    {
        float* jacobian = (float*)NativeMemory.AlignedAlloc((nuint)(outputDim * parameterDim * sizeof(float)), 64);
        float epsilon = 1e-5f;

        float[] baseOutput = outputFunction(_field);

        for (int p = 0; p < parameterDim; p++)
        {
            int cellIdx = p / 6;
            int paramIdx = p % 6;
            int cx = cellIdx % _field.ResX, cy = (cellIdx / _field.ResX) % _field.ResY, cz = cellIdx / (_field.ResX * _field.ResY);
            if (cx >= _field.ResX || cy >= _field.ResY || cz >= _field.ResZ)
                continue;
            int flatIdx = (cz * _field.ResY + cy) * _field.ResX + cx;

            float origVal = paramIdx switch
            {
                0 => _field.DataPointer[flatIdx].CurrentValue,
                1 => _field.DataPointer[flatIdx].Drift,
                2 => _field.DataPointer[flatIdx].Diffusion,
                3 => _field.DataPointer[flatIdx].Mean,
                4 => _field.DataPointer[flatIdx].Variance,
                5 => _field.DataPointer[flatIdx].Entropy,
                _ => 0
            };

            switch (paramIdx)
            {
                case 0:
                    _field.DataPointer[flatIdx].CurrentValue = origVal + epsilon;
                    break;
                case 1:
                    _field.DataPointer[flatIdx].Drift = origVal + epsilon;
                    break;
                case 2:
                    _field.DataPointer[flatIdx].Diffusion = origVal + epsilon;
                    break;
                case 3:
                    _field.DataPointer[flatIdx].Mean = origVal + epsilon;
                    break;
                case 4:
                    _field.DataPointer[flatIdx].Variance = origVal + epsilon;
                    break;
                case 5:
                    _field.DataPointer[flatIdx].Entropy = origVal + epsilon;
                    break;
            }

            float[] perturbedOutput = outputFunction(_field);

            for (int o = 0; o < outputDim; o++)
                jacobian[o * parameterDim + p] = (perturbedOutput[o] - baseOutput[o]) / epsilon;

            switch (paramIdx)
            {
                case 0:
                    _field.DataPointer[flatIdx].CurrentValue = origVal;
                    break;
                case 1:
                    _field.DataPointer[flatIdx].Drift = origVal;
                    break;
                case 2:
                    _field.DataPointer[flatIdx].Diffusion = origVal;
                    break;
                case 3:
                    _field.DataPointer[flatIdx].Mean = origVal;
                    break;
                case 4:
                    _field.DataPointer[flatIdx].Variance = origVal;
                    break;
                case 5:
                    _field.DataPointer[flatIdx].Entropy = origVal;
                    break;
            }
        }
        return jacobian;
    }

    /// <summary>Computes the adjoint of the Jacobian-vector product.</summary>
    public float* JacobianVectorProduct(float* jacobian, float* vector, int rows, int cols)
    {
        float* result = (float*)NativeMemory.AlignedAlloc((nuint)(rows * sizeof(float)), 64);
        for (int i = 0; i < rows; i++)
        {
            float sum = 0;
            for (int j = 0; j < cols; j++)
                sum += jacobian[i * cols + j] * vector[j];
            result[i] = sum;
        }
        return result;
    }

    /// <summary>Computes the vector-Jacobian product.</summary>
    public float* VectorJacobianProduct(float* jacobian, float* vector, int rows, int cols)
    {
        float* result = (float*)NativeMemory.AlignedAlloc((nuint)(cols * sizeof(float)), 64);
        NativeMemory.Clear(result, (nuint)(cols * sizeof(float)));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j] += vector[i] * jacobian[i * cols + j];
        return result;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        if (_adjointBuffer != null)
            NativeMemory.AlignedFree(_adjointBuffer);
        if (_tangentBuffer != null)
            NativeMemory.AlignedFree(_tangentBuffer);
        _adjointBuffer = null;
        _tangentBuffer = null;
        _disposed = true;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 18 — STOCHASTIC FIELD UNCERTAINTY QUANTIFICATION
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Uncertainty quantification (UQ) for stochastic fields.
/// Provides polynomial chaos expansion, Karhunen-Loeve expansion,
/// and Bayesian inference on field parameters.
/// </summary>
public sealed unsafe class StochasticFieldUQ : IDisposable
{
    private readonly StochasticField _field;
    private readonly Random _rng;
    private float* _klModes;
    private float* _klEigenvalues;
    private int _numKLModes;
    private bool _disposed;

    public StochasticFieldUQ(StochasticField field, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _rng = new Random(seed);
    }

    /// <summary>
    /// Karhunen-Loeve expansion: represents the random field as
    /// ξ(x) = μ(x) + Σ λ_i φ_i(x) ζ_i, where ζ_i ~ N(0,1).
    /// </summary>
    /// <param name="numModes">Number of KL modes to retain.</param>
    /// <param name="correlationLength">Correlation length for the covariance kernel.</param>
    public void ComputeKLExpansion(int numModes, float correlationLength)
    {
        int total = _field.TotalCells;
        _numKLModes = numModes;
        _klModes = (float*)NativeMemory.AlignedAlloc((nuint)(numModes * total * sizeof(float)), 64);
        _klEigenvalues = (float*)NativeMemory.AlignedAlloc((nuint)(numModes * sizeof(float)), 64);

        // Build covariance matrix (simplified: use analytical eigenvalues for exponential kernel)
        // For exponential kernel on [0,L]: λ_n = 2L / (1 + (nπ/L)²ξ²), φ_n(x) = cos(nπx/L) / sqrt(L/2)
        float L = _field.ResX * _field.ResX; // domain length squared
        for (int n = 0; n < numModes; n++)
        {
            float eigenvalue = 2.0f * L / (1.0f + (float)Math.Pow((n + 1) * Math.PI / L, 2) * correlationLength * correlationLength);
            _klEigenvalues[n] = eigenvalue;
            float normFactor = MathF.Sqrt(2.0f / L);

            for (int i = 0; i < total; i++)
            {
                int x = i % _field.ResX, y = (i / _field.ResX) % _field.ResY;
                float posX = (float)x / _field.ResX;
                _klModes[n * total + i] = normFactor * MathF.Cos((n + 1) * MathF.PI * posX);
            }
        }
    }

    /// <summary>Generates a realization of the field using the KL expansion.</summary>
    public void GenerateKLRealization()
    {
        int total = _field.TotalCells;
        for (int i = 0; i < total; i++)
        {
            float value = _field.DataPointer[i].Mean;
            for (int n = 0; n < _numKLModes; n++)
            {
                float zeta = NormalSample();
                value += MathF.Sqrt(MathF.Max(_klEigenvalues[n], 0)) * _klModes[n * total + i] * zeta;
            }
            _field.DataPointer[i].CurrentValue = value;
        }
    }

    /// <summary>
    /// Polynomial Chaos Expansion (PCE): represents stochastic output as
    /// g(ξ) = Σ c_k Ψ_k(ξ), where Ψ_k are orthogonal polynomials.
    /// Uses Hermite polynomials for Gaussian random variables.
    /// </summary>
    /// <param name="numSamples">Number of Monte Carlo samples for coefficient estimation.</param>
    /// <param name="maxOrder">Maximum polynomial order.</param>
    public float[] ComputePCECoefficients(Func<StochasticField, float> quantityOfInterest, int numSamples, int maxOrder)
    {
        int numCoeffs = 1;
        for (int d = 1; d <= maxOrder; d++)
            numCoeffs += (d + 1);

        float[] xiSamples = new float[numSamples];
        float[] qoIValues = new float[numSamples];
        float[] coefficients = new float[numCoeffs];

        // Generate samples and evaluate QoI
        for (int s = 0; s < numSamples; s++)
        {
            // Perturb field with random coefficients
            float xi = NormalSample();
            xiSamples[s] = xi;
            for (int i = 0; i < _field.TotalCells; i++)
                _field.DataPointer[i].CurrentValue += xi * _field.DataPointer[i].Diffusion;
            qoIValues[s] = quantityOfInterest(_field);
            // Restore
            for (int i = 0; i < _field.TotalCells; i++)
                _field.DataPointer[i].CurrentValue -= xi * _field.DataPointer[i].Diffusion;
        }

        // Fit PCE coefficients using least squares
        // Build design matrix with Hermite polynomials
        float* designMatrix = (float*)NativeMemory.AlignedAlloc((nuint)(numSamples * numCoeffs * sizeof(float)), 64);
        for (int s = 0; s < numSamples; s++)
        {
            int col = 0;
            designMatrix[s * numCoeffs + col++] = 1.0f; // H_0
            if (maxOrder >= 1)
                designMatrix[s * numCoeffs + col++] = xiSamples[s]; // H_1
            for (int order = 2; order <= maxOrder; order++)
            {
                for (int k = 0; k <= order; k++)
                    designMatrix[s * numCoeffs + col++] = HermitePolynomial(xiSamples[s], order);
            }
        }

        // Solve via normal equations: c = (D^T D)^{-1} D^T y
        float* DtD = (float*)NativeMemory.AlignedAlloc((nuint)(numCoeffs * numCoeffs * sizeof(float)), 64);
        float* Dty = (float*)NativeMemory.AlignedAlloc((nuint)(numCoeffs * sizeof(float)), 64);
        NativeMemory.Clear(DtD, (nuint)(numCoeffs * numCoeffs * sizeof(float)));
        NativeMemory.Clear(Dty, (nuint)(numCoeffs * sizeof(float)));

        for (int i = 0; i < numCoeffs; i++)
        {
            for (int j = 0; j < numCoeffs; j++)
            {
                float sum = 0;
                for (int s = 0; s < numSamples; s++)
                    sum += designMatrix[s * numCoeffs + i] * designMatrix[s * numCoeffs + j];
                DtD[i * numCoeffs + j] = sum;
            }
            float ySum = 0;
            for (int s = 0; s < numSamples; s++)
                ySum += designMatrix[s * numCoeffs + i] * qoIValues[s];
            Dty[i] = ySum;
        }

        // Solve with Gauss-Seidel
        for (int iter = 0; iter < 200; iter++)
        {
            float maxDiff = 0;
            for (int i = 0; i < numCoeffs; i++)
            {
                float sum = 0;
                for (int j = 0; j < numCoeffs; j++)
                    if (j != i)
                        sum += DtD[i * numCoeffs + j] * coefficients[j];
                float newVal = (Dty[i] - sum) / (DtD[i * numCoeffs + i] + 1e-10f);
                maxDiff = MathF.Max(maxDiff, MathF.Abs(newVal - coefficients[i]));
                coefficients[i] = newVal;
            }
            if (maxDiff < 1e-8f)
                break;
        }

        NativeMemory.AlignedFree(designMatrix);
        NativeMemory.AlignedFree(DtD);
        NativeMemory.AlignedFree(Dty);
        return coefficients;
    }

    /// <summary>Evaluates the PCE at a given random variable realization.</summary>
    public float EvaluatePCE(float[] coefficients, float xi, int maxOrder)
    {
        float result = 0;
        int col = 0;
        result += coefficients[col++] * 1.0f; // H_0
        if (maxOrder >= 1 && col < coefficients.Length)
            result += coefficients[col++] * xi; // H_1
        for (int order = 2; order <= maxOrder; order++)
        {
            if (col >= coefficients.Length)
                break;
            result += coefficients[col++] * HermitePolynomial(xi, order);
        }
        return result;
    }

    /// <summary>Bayesian inference on field parameters using MCMC (Metropolis-Hastings).</summary>
    public (float[] PosteriorMean, float[] PosteriorStd) BayesianInference(
        Func<float[], StochasticField, float> likelihood, float[] priorMean, float[] priorStd,
        int numMCMC = 5000, int burnIn = 1000)
    {
        int dim = priorMean.Length;
        float[] current = new float[dim];
        float[] posteriorMean = new float[dim];
        float[] posteriorVar = new float[dim];
        for (int d = 0; d < dim; d++)
            current[d] = priorMean[d];
        float currentLogLik = LogLikelihood(likelihood, current, priorMean, priorStd);

        var samples = new List<float[]>(numMCMC - burnIn);

        for (int iter = 0; iter < numMCMC; iter++)
        {
            // Propose new state
            float[] proposed = new float[dim];
            for (int d = 0; d < dim; d++)
                proposed[d] = current[d] + priorStd[d] * 0.1f * NormalSample();

            float proposedLogLik = LogLikelihood(likelihood, proposed, priorMean, priorStd);

            // Accept/reject
            float logAlpha = proposedLogLik - currentLogLik;
            if (logAlpha > 0 || MathF.Log((float)_rng.NextDouble()) < logAlpha)
            {
                Array.Copy(proposed, current, dim);
                currentLogLik = proposedLogLik;
            }

            if (iter >= burnIn)
            {
                samples.Add((float[])current.Clone());
                for (int d = 0; d < dim; d++)
                    posteriorMean[d] += current[d];
            }
        }

        // Compute posterior statistics
        int n = samples.Count;
        for (int d = 0; d < dim; d++)
            posteriorMean[d] /= n;
        for (int d = 0; d < dim; d++)
        {
            float var = 0;
            for (int s = 0; s < n; s++)
            { float diff = samples[s][d] - posteriorMean[d]; var += diff * diff; }
            posteriorVar[d] = var / n;
        }
        float[] posteriorStd = new float[dim];
        for (int d = 0; d < dim; d++)
            posteriorStd[d] = MathF.Sqrt(posteriorVar[d]);
        return (posteriorMean, posteriorStd);
    }

    private float LogLikelihood(Func<float[], StochasticField, float> likelihood, float[] parameters, float[] priorMean, float[] priorStd)
    {
        float logL = 0;
        // Prior contribution (Gaussian)
        for (int d = 0; d < parameters.Length; d++)
        {
            float diff = parameters[d] - priorMean[d];
            logL -= 0.5f * diff * diff / (priorStd[d] * priorStd[d] + 1e-10f);
        }
        // Likelihood contribution
        try
        { logL += MathF.Log(MathF.Max(likelihood(parameters, _field), 1e-10f)); }
        catch { logL -= 1e6f; }
        return logL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HermitePolynomial(float x, int n)
    {
        if (n == 0)
            return 1;
        if (n == 1)
            return x;
        float h0 = 1, h1 = x;
        for (int i = 2; i <= n; i++)
        { float h2 = x * h1 - (i - 1) * h0; h0 = h1; h1 = h2; }
        return h1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NormalSample()
    { float u1 = 1f - (float)_rng.NextDouble(), u2 = 1f - (float)_rng.NextDouble(); return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2); }

    public void Dispose()
    {
        if (_disposed)
            return;
        if (_klModes != null)
            NativeMemory.AlignedFree(_klModes);
        if (_klEigenvalues != null)
            NativeMemory.AlignedFree(_klEigenvalues);
        _klModes = null;
        _klEigenvalues = null;
        _disposed = true;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 19 — STOCHASTIC FIELD INTERPOLATION
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Advanced interpolation methods for stochastic fields: kriging (Gaussian process
/// regression), radial basis functions, and splines.
/// </summary>
public sealed unsafe class StochasticFieldInterpolator
{
    private readonly StochasticField _field;
    private readonly StochasticCorrelationModel _kernel;
    private readonly Random _rng;

    public StochasticFieldInterpolator(StochasticField field, StochasticCorrelationModel? kernel = null, int seed = 42)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _kernel = kernel ?? new StochasticCorrelationModel(CovarianceKernelType.Matern);
        _rng = new Random(seed);
    }

    /// <summary>
    /// Kriging interpolation (Gaussian process regression).
    /// Returns predicted value and prediction variance at query points.
    /// </summary>
    public (float PredictedValue, float Variance) KrigingPredict(float queryX, float queryY, float queryZ,
        float* trainX, float* trainY, float* trainZ, float* trainValues, int numTrain)
    {
        // Build covariance vector between query and training points
        float* k_star = stackalloc float[numTrain];
        for (int i = 0; i < numTrain; i++)
        {
            float dx = queryX - trainX[i], dy = queryY - trainY[i], dz = queryZ - trainZ[i];
            k_star[i] = _kernel.Evaluate(dx, dy, dz);
        }

        // Kriging weights: w = K^{-1} k*
        Span<float> weights = stackalloc float[numTrain];
        // Solve via Gauss-Seidel with the training covariance matrix
        for (int i = 0; i < numTrain; i++)
            weights[i] = 0;

        for (int iter = 0; iter < 100; iter++)
        {
            float maxDiff = 0;
            for (int i = 0; i < numTrain; i++)
            {
                float sum = 0;
                for (int j = 0; j < numTrain; j++)
                {
                    if (j == i)
                        continue;
                    float dx = trainX[i] - trainX[j], dy = trainY[i] - trainY[j], dz = trainZ[i] - trainZ[j];
                    sum += _kernel.Evaluate(dx, dy, dz) * weights[j];
                }
                float newVal = (k_star[i] - sum) / (_kernel.Evaluate(0, 0, 0) + 1e-6f);
                maxDiff = MathF.Max(maxDiff, MathF.Abs(newVal - weights[i]));
                weights[i] = newVal;
            }
            if (maxDiff < 1e-8f)
                break;
        }

        // Predicted value
        float predicted = 0;
        for (int i = 0; i < numTrain; i++)
            predicted += weights[i] * trainValues[i];

        // Prediction variance
        float kqq = _kernel.Evaluate(0, 0, 0);
        float var = kqq;
        for (int i = 0; i < numTrain; i++)
            var -= weights[i] * k_star[i];

        return (predicted, MathF.Max(var, 0));
    }

    /// <summary>Radial Basis Function interpolation.</summary>
    public float RBFInterpolate(float queryX, float queryY, float queryZ,
        float* trainX, float* trainY, float* trainZ, float* trainValues, int numTrain, float shapeParameter = 1.0f)
    {
        // Build RBF matrix and solve for coefficients
        Span<float> coefficients = stackalloc float[numTrain];
        Span<float> rhs = stackalloc float[numTrain];

        // Fill RHS with training values
        for (int i = 0; i < numTrain; i++)
            rhs[i] = trainValues[i];

        // Gauss-Seidel solve for RBF weights
        for (int i = 0; i < numTrain; i++)
            coefficients[i] = 0;
        for (int iter = 0; iter < 200; iter++)
        {
            float maxDiff = 0;
            for (int i = 0; i < numTrain; i++)
            {
                float sum = 0;
                for (int j = 0; j < numTrain; j++)
                {
                    if (j == i)
                        continue;
                    float dx = trainX[i] - trainX[j], dy = trainY[i] - trainY[j], dz = trainZ[i] - trainZ[j];
                    float r2 = dx * dx + dy * dy + dz * dz;
                    sum += MathF.Exp(-shapeParameter * r2) * coefficients[j];
                }
                float newVal = (rhs[i] - sum) / (1.0f + 1e-6f); // φ(0) = 1 for Gaussian RBF
                maxDiff = MathF.Max(maxDiff, MathF.Abs(newVal - coefficients[i]));
                coefficients[i] = newVal;
            }
            if (maxDiff < 1e-8f)
                break;
        }

        // Evaluate at query point
        float result = 0;
        for (int i = 0; i < numTrain; i++)
        {
            float dx = queryX - trainX[i], dy = queryY - trainY[i], dz = queryZ - trainZ[i];
            float r2 = dx * dx + dy * dy + dz * dz;
            result += coefficients[i] * MathF.Exp(-shapeParameter * r2);
        }
        return result;
    }

    /// <summary>Cubic spline interpolation along the X axis at fixed Y,Z.</summary>
    public float CubicSplineInterpolateX(float queryX, int fixedY, int fixedZ)
    {
        int N = _field.ResX;
        if (N < 2)
            return _field.At(0, fixedY, fixedZ).CurrentValue;

        // Build knot values
        Span<float> x = stackalloc float[N];
        Span<float> y = stackalloc float[N];
        for (int i = 0; i < N; i++)
        {
            x[i] = i;
            y[i] = _field.At(i, fixedY, fixedZ).CurrentValue;
        }

        // Natural cubic spline: compute second derivatives
        Span<float> h = stackalloc float[N - 1];
        Span<float> alpha = stackalloc float[N - 1];
        Span<float> l = stackalloc float[N];
        Span<float> mu = stackalloc float[N];
        Span<float> z = stackalloc float[N];
        Span<float> c = stackalloc float[N];
        Span<float> b = stackalloc float[N - 1];
        Span<float> d = stackalloc float[N - 1];

        for (int i = 0; i < N - 1; i++)
            h[i] = x[i + 1] - x[i];
        for (int i = 1; i < N - 1; i++)
            alpha[i] = 3.0f / h[i] * (y[i + 1] - y[i]) - 3.0f / h[i - 1] * (y[i] - y[i - 1]);

        l[0] = 1;
        mu[0] = 0;
        z[0] = 0;
        for (int i = 1; i < N - 1; i++)
        {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[N - 1] = 1;
        z[N - 1] = 0;
        c[N - 1] = 0;

        for (int j = N - 2; j >= 0; j--)
        {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }

        // Find interval containing queryX
        int interval = 0;
        for (int i = 0; i < N - 1; i++)
        { if (queryX >= x[i] && queryX <= x[i + 1]) { interval = i; break; } }
        if (queryX > x[N - 1])
            interval = N - 2;

        float dx = queryX - x[interval];
        return y[interval] + b[interval] * dx + c[interval] * dx * dx + d[interval] * dx * dx * dx;
    }

    /// <summary>Inverse distance weighting interpolation.</summary>
    public float IDWInterpolate(float queryX, float queryY, float queryZ,
        float* trainX, float* trainY, float* trainZ, float* trainValues, int numTrain, float power = 2.0f)
    {
        float weightedSum = 0, weightSum = 0;
        for (int i = 0; i < numTrain; i++)
        {
            float dx = queryX - trainX[i], dy = queryY - trainY[i], dz = queryZ - trainZ[i];
            float dist = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
            float w = dist < 1e-6f ? 1e6f : 1.0f / MathF.Pow(dist, power);
            weightedSum += w * trainValues[i];
            weightSum += w;
        }
        return weightedSum / (weightSum + 1e-10f);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 20 — STOCHASTIC FIELD PERFORMANCE MONITORING
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Performance monitoring and benchmarking for stochastic field operations.
/// Tracks timing, memory allocation, and operation counts.
/// </summary>
public sealed class StochasticFieldProfiler
{
    private long _totalSteps;
    private long _totalCellsProcessed;
    private long _totalStepTimeMs;
    private long _totalAllocations;
    private long _totalBytesAllocated;
    private readonly Dictionary<string, long> _operationCounts = new();
    private readonly Dictionary<string, double> _operationTimes = new();
    private readonly Stopwatch _globalTimer = new();
    private long _peakMemoryBytes;

    /// <summary>Total simulation steps executed.</summary>
    public long TotalSteps => Interlocked.Read(ref _totalSteps);
    /// <summary>Total cells processed across all steps.</summary>
    public long TotalCellsProcessed => Interlocked.Read(ref _totalCellsProcessed);
    /// <summary>Total time spent in step operations (ms).</summary>
    public double TotalStepTimeMs => Interlocked.Read(ref _totalStepTimeMs) / 10000.0;
    /// <summary>Average time per step (ms).</summary>
    public double AverageStepMs => TotalSteps > 0 ? TotalStepTimeMs / TotalSteps : 0;
    /// <summary>Cells processed per second.</summary>
    public double CellsPerSecond => TotalStepTimeMs > 0 ? TotalCellsProcessed / (TotalStepTimeMs / 1000.0) : 0;
    /// <summary>Peak memory usage.</summary>
    public long PeakMemoryBytes => Interlocked.Read(ref _peakMemoryBytes);

    /// <summary>Begins timing a named operation.</summary>
    public void BeginOperation(string name)
    {
        _globalTimer.Restart();
    }

    /// <summary>Ends timing and records the named operation.</summary>
    public void EndOperation(string name)
    {
        _globalTimer.Stop();
        double elapsed = _globalTimer.Elapsed.TotalMilliseconds;
        lock (_operationCounts)
        {
            _operationCounts.TryGetValue(name, out long count);
            _operationCounts[name] = count + 1;
            _operationTimes.TryGetValue(name, out double time);
            _operationTimes[name] = time + elapsed;
        }
    }

    /// <summary>Records a step operation.</summary>
    public void RecordStep(int totalCells, double stepTimeMs)
    {
        Interlocked.Increment(ref _totalSteps);
        Interlocked.Add(ref _totalCellsProcessed, totalCells);
        Interlocked.Add(ref _totalStepTimeMs, (long)(stepTimeMs * 10000));
    }

    /// <summary>Records a memory allocation.</summary>
    public void RecordAllocation(long bytes)
    {
        Interlocked.Increment(ref _totalAllocations);
        Interlocked.Add(ref _totalBytesAllocated, bytes);
        long current = Interlocked.Read(ref _totalBytesAllocated);
        long peak = Interlocked.Read(ref _peakMemoryBytes);
        if (current > peak)
            Interlocked.Exchange(ref _peakMemoryBytes, current);
    }

    /// <summary>Returns a summary of all operation timings.</summary>
    public Dictionary<string, (long Count, double TotalMs, double AvgMs)> GetOperationSummary()
    {
        var summary = new Dictionary<string, (long, double, double)>();
        lock (_operationCounts)
        {
            foreach (var kvp in _operationCounts)
            {
                double totalTime = _operationTimes.TryGetValue(kvp.Key, out double t) ? t : 0;
                summary[kvp.Key] = (kvp.Value, totalTime, kvp.Value > 0 ? totalTime / kvp.Value : 0);
            }
        }
        return summary;
    }

    /// <summary>Resets all profiling counters.</summary>
    public void Reset()
    {
        Interlocked.Exchange(ref _totalSteps, 0);
        Interlocked.Exchange(ref _totalCellsProcessed, 0);
        Interlocked.Exchange(ref _totalStepTimeMs, 0);
        Interlocked.Exchange(ref _totalAllocations, 0);
        Interlocked.Exchange(ref _totalBytesAllocated, 0);
        Interlocked.Exchange(ref _peakMemoryBytes, 0);
        lock (_operationCounts)
        { _operationCounts.Clear(); _operationTimes.Clear(); }
    }

    /// <summary>Returns a formatted performance report.</summary>
    public string GetReport()
    {
        var sb = new StringBuilder();
        sb.AppendLine("═══ StochasticField Performance Report ═══");
        sb.AppendLine($"  Total Steps:           {TotalSteps:N0}");
        sb.AppendLine($"  Total Cells Processed: {TotalCellsProcessed:N0}");
        sb.AppendLine($"  Total Step Time:       {TotalStepTimeMs:F2} ms");
        sb.AppendLine($"  Avg Step Time:         {AverageStepMs:F4} ms");
        sb.AppendLine($"  Throughput:            {CellsPerSecond:N0} cells/sec");
        sb.AppendLine($"  Peak Memory:           {PeakMemoryBytes:N0} bytes ({PeakMemoryBytes / 1024.0:F1} KB)");
        sb.AppendLine($"  Total Allocations:     {Interlocked.Read(ref _totalAllocations):N0}");
        sb.AppendLine();
        sb.AppendLine("  Operation Timings:");
        foreach (var (name, (count, totalMs, avgMs)) in GetOperationSummary())
            sb.AppendLine($"    {name,-30} count={count,8:N0}  total={totalMs,10:F2}ms  avg={avgMs,10:F4}ms");
        return sb.ToString();
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 21 — STOCHASTIC FIELD PARALLEL EXECUTION
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Parallel execution utilities for stochastic field operations.
/// Provides partitioned loops, parallel reduction, and work-stealing.
/// </summary>
public static unsafe class StochasticFieldParallel
{
    /// <summary>Executes a function for each cell in parallel with automatic load balancing.</summary>
    public static void ForEachCell(StochasticField field, Action<int, int, int> body, int maxDegreeOfParallelism = -1)
    {
        int total = field.TotalCells;
        var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
        Parallel.For(0, total, options, i =>
        {
            int x = i % field.ResX, y = (i / field.ResX) % field.ResY, z = i / (field.ResX * field.ResY);
            body(x, y, z);
        });
    }

    /// <summary>Parallel reduction: combines all cell values into a single result.</summary>
    public static float Reduce(StochasticField field, Func<float, float, float> combiner, float initialValue = 0, int maxDegreeOfParallelism = -1)
    {
        int total = field.TotalCells;
        var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
        float result = initialValue;
        object lockObj = new();
        Parallel.For(0, total, options, () => initialValue,
            (i, state, localResult) =>
            {
                return combiner(localResult, field.DataPointer[i].CurrentValue);
            },
            (finalResult) => { lock (lockObj) { result = finalResult; } });
        return result;
    }

    /// <summary>Computes sum of all field values in parallel.</summary>
    public static float ParallelSum(StochasticField field, int maxDegreeOfParallelism = -1)
    {
        return Reduce(field, (a, b) => a + b, 0, maxDegreeOfParallelism);
    }

    /// <summary>Computes the mean of all field values in parallel.</summary>
    public static float ParallelMean(StochasticField field, int maxDegreeOfParallelism = -1)
    {
        float sum = ParallelSum(field, maxDegreeOfParallelism);
        return sum / field.TotalCells;
    }

    /// <summary>Computes the variance of all field values in parallel.</summary>
    public static float ParallelVariance(StochasticField field, int maxDegreeOfParallelism = -1)
    {
        float mean = ParallelMean(field, maxDegreeOfParallelism);
        float sumSqDiff = Reduce(field, (acc, v) => { float d = v - mean; return acc + d * d; }, 0, maxDegreeOfParallelism);
        return sumSqDiff / field.TotalCells;
    }

    /// <summary>Parallel map: applies function and stores result in output field.</summary>
    public static void ParallelMap(StochasticField input, StochasticField output, Func<float, float> func, int maxDegreeOfParallelism = -1)
    {
        int total = input.TotalCells;
        var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
        Parallel.For(0, total, options, i =>
        {
            output.DataPointer[i].CurrentValue = func(input.DataPointer[i].CurrentValue);
        });
    }

    /// <summary>Parallel scan (prefix sum) of field values.</summary>
    public static float[] ParallelPrefixSum(StochasticField field, int maxDegreeOfParallelism = -1)
    {
        int total = field.TotalCells;
        float[] result = new float[total + 1];
        result[0] = 0;
        int chunkSize = Math.Max(1, total / Environment.ProcessorCount);
        int numChunks = (total + chunkSize - 1) / chunkSize;
        float[] chunkSums = new float[numChunks];

        // Phase 1: compute chunk sums in parallel
        Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, c =>
        {
            float localSum = 0;
            int start = c * chunkSize;
            int end = Math.Min(start + chunkSize, total);
            for (int i = start; i < end; i++)
                localSum += field.DataPointer[i].CurrentValue;
            chunkSums[c] = localSum;
        });

        // Phase 2: sequential prefix sum of chunk sums
        for (int c = 1; c < numChunks; c++)
            chunkSums[c] += chunkSums[c - 1];

        // Phase 3: compute full prefix sum using chunk offsets
        Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, c =>
        {
            int start = c * chunkSize;
            int end = Math.Min(start + chunkSize, total);
            float runningSum = c > 0 ? chunkSums[c - 1] : 0;
            for (int i = start; i < end; i++)
            {
                runningSum += field.DataPointer[i].CurrentValue;
                result[i + 1] = runningSum;
            }
        });

        return result;
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 22 — STOCHASTIC FIELD CONFIGURATION BUILDER
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Fluent builder for constructing stochastic field configurations.
/// </summary>
public sealed class StochasticFieldBuilder
{
    private readonly StochasticFieldConfig _config = new();

    public static StochasticFieldBuilder Create() => new();

    public StochasticFieldBuilder WithResolution(int resX, int resY, int resZ)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionX))!.SetValue(_config, resX); _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionY))!.SetValue(_config, resY); _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionZ))!.SetValue(_config, resZ); return this; }

    public StochasticFieldBuilder WithResolution(int resolution)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionX))!.SetValue(_config, resolution); _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionY))!.SetValue(_config, resolution); _config.GetType().GetProperty(nameof(StochasticFieldConfig.ResolutionZ))!.SetValue(_config, resolution); return this; }

    public StochasticFieldBuilder WithSpacing(float spacingX, float spacingY, float spacingZ)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.SpacingX))!.SetValue(_config, spacingX); _config.GetType().GetProperty(nameof(StochasticFieldConfig.SpacingY))!.SetValue(_config, spacingY); _config.GetType().GetProperty(nameof(StochasticFieldConfig.SpacingZ))!.SetValue(_config, spacingZ); return this; }

    public StochasticFieldBuilder WithProcess(StochasticProcessType process)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.DefaultProcess))!.SetValue(_config, process); return this; }

    public StochasticFieldBuilder WithTimeStep(float dt)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.TimeStep))!.SetValue(_config, dt); return this; }

    public StochasticFieldBuilder WithDriftAndDiffusion(float drift, float diffusion)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.DefaultDrift))!.SetValue(_config, drift); _config.GetType().GetProperty(nameof(StochasticFieldConfig.DefaultDiffusion))!.SetValue(_config, diffusion); return this; }

    public StochasticFieldBuilder WithCorrelation(float length, float strength)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.CorrelationLength))!.SetValue(_config, length); _config.GetType().GetProperty(nameof(StochasticFieldConfig.CouplingStrength))!.SetValue(_config, strength); return this; }

    public StochasticFieldBuilder WithSeed(int seed)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.Seed))!.SetValue(_config, seed); return this; }

    public StochasticFieldBuilder WithScheme(SDEScheme scheme)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.Scheme))!.SetValue(_config, scheme); return this; }

    public StochasticFieldBuilder WithParallel(bool enabled, int maxDegree = -1)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.ParallelExecution))!.SetValue(_config, enabled); _config.GetType().GetProperty(nameof(StochasticFieldConfig.MaxDegreeOfParallelism))!.SetValue(_config, maxDegree); return this; }

    public StochasticFieldBuilder WithMonteCarloPaths(int paths)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.MonteCarloPaths))!.SetValue(_config, paths); return this; }

    public StochasticFieldBuilder WithHurstParameter(float hurst)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.HurstParameter))!.SetValue(_config, hurst); return this; }

    public StochasticFieldBuilder WithKernel(CovarianceKernelType kernel, float maternNu = 1.5f)
    { _config.GetType().GetProperty(nameof(StochasticFieldConfig.KernelType))!.SetValue(_config, kernel); _config.GetType().GetProperty(nameof(StochasticFieldConfig.MaternNu))!.SetValue(_config, maternNu); return this; }

    public StochasticFieldConfig Build() => _config;

    public StochasticField BuildField()
    {
        return new StochasticField(
            _config.ResolutionX, _config.ResolutionY, _config.ResolutionZ,
            _config.DefaultProcess,
            _config.SpacingX, _config.SpacingY, _config.SpacingZ,
            _config.TimeStep,
            _config.OriginX, _config.OriginY, _config.OriginZ,
            _config.Seed
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 23 — STOCHASTIC FIELD VALIDATION
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Validation utilities for stochastic fields: NaN/Inf checks, range validation,
/// conservation laws, and statistical tests.
/// </summary>
public static unsafe class StochasticFieldValidator
{
    /// <summary>Checks for NaN or Inf values in the field.</summary>
    public static (bool IsValid, int NaNCount, int InfCount) ValidateFinite(StochasticField field)
    {
        int nanCount = 0, infCount = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        {
            float v = field.DataPointer[i].CurrentValue;
            if (float.IsNaN(v))
                nanCount++;
            else if (float.IsInfinity(v))
                infCount++;
        }
        return (nanCount == 0 && infCount == 0, nanCount, infCount);
    }

    /// <summary>Validates that all field values are within specified bounds.</summary>
    public static (bool IsValid, int ViolationCount) ValidateRange(StochasticField field, float minValue, float maxValue)
    {
        int violations = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        {
            float v = field.DataPointer[i].CurrentValue;
            if (v < minValue || v > maxValue)
                violations++;
        }
        return (violations == 0, violations);
    }

    /// <summary>Validates that diffusion coefficients are positive.</summary>
    public static (bool IsValid, int ViolationCount) ValidatePositiveDiffusion(StochasticField field)
    {
        int violations = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
        {
            if (field.DataPointer[i].Diffusion < 0)
                violations++;
        }
        return (violations == 0, violations);
    }

    /// <summary>Checks conservation: total mass before and after step.</summary>
    public static (bool IsConserved, float RelativeError) ValidateConservation(StochasticField field, float dt, int numSteps, float tolerance = 0.01f)
    {
        float initialSum = 0;
        int total = field.TotalCells;
        for (int i = 0; i < total; i++)
            initialSum += field.DataPointer[i].CurrentValue;

        for (int s = 0; s < numSteps; s++)
            field.StepEulerMaruyama(dt);

        float finalSum = 0;
        for (int i = 0; i < total; i++)
            finalSum += field.DataPointer[i].CurrentValue;

        float relError = MathF.Abs(finalSum - initialSum) / (MathF.Abs(initialSum) + 1e-10f);
        return (relError < tolerance, relError);
    }

    /// <summary>Runs the Shapiro-Wilk normality test on field values (simplified).</summary>
    public static (bool IsNormal, float TestStatistic) ShapiroWilkTest(StochasticField field, float significanceLevel = 0.05f)
    {
        int total = field.TotalCells;
        int n = Math.Min(total, 5000);
        Span<float> values = stackalloc float[n];
        int step = Math.Max(1, total / n);
        for (int i = 0; i < n; i++)
            values[i] = field.DataPointer[i * step].CurrentValue;
        values.Sort();

        float mean = 0;
        for (int i = 0; i < n; i++)
            mean += values[i];
        mean /= n;

        float W = 0, ssq = 0;
        for (int i = 0; i < n; i++)
        { float d = values[i] - mean; ssq += d * d; }
        // Simplified W statistic using order statistics
        float numerator = 0;
        for (int i = 0; i < n / 2; i++)
        {
            float ai = (float)(2.8242 + (-3.2012 + (1.8886 + (-0.2656 + 0.0010 * n) / n) / n) / n) / MathF.Sqrt(n);
            numerator += ai * (values[n - 1 - i] - values[i]);
        }
        W = numerator * numerator / (ssq + 1e-10f);

        // Critical value approximation (very simplified)
        float criticalValue = 1.0f - significanceLevel * 0.5f;
        return (W > criticalValue, W);
    }

    /// <summary>Validates the Kolmogorov-Smirnov test against a reference distribution.</summary>
    public static (bool IsConsistent, float TestStatistic) KolmogorovSmirnovTest(StochasticField field, Func<float, float> cdf)
    {
        int total = field.TotalCells;
        int n = Math.Min(total, 5000);
        Span<float> values = stackalloc float[n];
        int step = Math.Max(1, total / n);
        for (int i = 0; i < n; i++)
            values[i] = field.DataPointer[i * step].CurrentValue;
        values.Sort();

        float maxDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float empiricalCDF = (float)(i + 1) / n;
            float theoreticalCDF = cdf(values[i]);
            float diff = MathF.Abs(empiricalCDF - theoreticalCDF);
            if (diff > maxDiff)
                maxDiff = diff;
        }

        // KS critical value approximation
        float criticalValue = 1.36f / MathF.Sqrt(n);
        return (maxDiff < criticalValue, maxDiff);
    }

    /// <summary>Validates spatial smoothness by checking gradient magnitudes.</summary>
    public static (bool IsSmooth, float MaxGradient, float MeanGradient) ValidateSmoothness(StochasticField field, float maxAllowedGradient = 10.0f)
    {
        float maxGrad = 0, sumGrad = 0;
        int count = 0;
        for (int z = 1; z < field.ResZ - 1; z++)
            for (int y = 1; y < field.ResY - 1; y++)
                for (int x = 1; x < field.ResX - 1; x++)
                {
                    float gx = (field.At(x + 1, y, z).CurrentValue - field.At(x - 1, y, z).CurrentValue) * 0.5f;
                    float gy = (field.At(x, y + 1, z).CurrentValue - field.At(x, y - 1, z).CurrentValue) * 0.5f;
                    float gz = (field.At(x, y, z + 1).CurrentValue - field.At(x, y, z - 1).CurrentValue) * 0.5f;
                    float gradMag = MathF.Sqrt(gx * gx + gy * gy + gz * gz);
                    if (gradMag > maxGrad)
                        maxGrad = gradMag;
                    sumGrad += gradMag;
                    count++;
                }
        float meanGrad = count > 0 ? sumGrad / count : 0;
        return (maxGrad < maxAllowedGradient, maxGrad, meanGrad);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
//  SECTION 24 — STOCHASTIC FIELD CONSTANTS AND UTILITY CLASSES
// ════════════════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Mathematical and physical constants used throughout the stochastic field system.
/// </summary>
public static class StochasticFieldConstants
{
    /// <summary>Euler-Mascheroni constant.</summary>
    public const float EulerGamma = 0.5772156649f;
    /// <summary>sqrt(2π).</summary>
    public const float Sqrt2Pi = 2.5066282746f;
    /// <summary>1/sqrt(2π).</summary>
    public const float InvSqrt2Pi = 0.3989422804f;
    /// <summary>1/sqrt(2).</summary>
    public const float InvSqrt2 = 0.7071067812f;
    /// <summary>ln(2).</summary>
    public const float Ln2 = 0.6931471806f;
    /// <summary>π.</summary>
    public const float Pi = 3.1415926536f;
    /// <summary>2π.</summary>
    public const float TwoPi = 6.2831853072f;
    /// <summary>sqrt(2).</summary>
    public const float Sqrt2 = 1.4142135624f;
    /// <summary>Machine epsilon for float.</summary>
    public const float FloatEpsilon = 1.1920929e-7f;
    /// <summary>Largest finite float.</summary>
    public const float FloatMax = 3.4028235e+38f;
    /// <summary>Smallest positive float.</summary>
    public const float FloatMin = 1.17549435e-38f;
    /// <summary>Typical financial risk-free rate.</summary>
    public const float DefaultRiskFreeRate = 0.02f;
    /// <summary>Typical epidemic recovery period (days).</summary>
    public const float DefaultRecoveryPeriod = 14.0f;
    /// <summary>Typical social network clustering coefficient.</summary>
    public const float DefaultClusteringCoefficient = 0.3f;

    /// <summary>Standard normal PDF at x.</summary>
    public static float NormalPDF(float x) => InvSqrt2Pi * MathF.Exp(-0.5f * x * x);

    /// <summary>Approximation of the standard normal CDF.</summary>
    public static float NormalCDF(float x)
    {
        float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
        float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0 ? -1.0f : 1.0f;
        x = MathF.Abs(x) * InvSqrt2;
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return 0.5f * (1.0f + sign * y);
    }

    /// <summary>Inverse normal CDF (Beasley-Springer-Moro algorithm).</summary>
    public static float InverseNormalCDF(float p)
    {
        if (p <= 0)
            return -10.0f;
        if (p >= 1)
            return 10.0f;
        if (p == 0.5f)
            return 0.0f;

        float[] a = { -3.969683028665376e+01f, 2.209460984245205e+02f, -2.759285104469687e+02f, 1.383577518672690e+02f, -3.066479806614716e+01f, 2.506628277459239e+00f };
        float[] b = { -5.447609879822406e+01f, 1.615858368580409e+02f, -1.556989798598866e+02f, 6.680131188771972e+01f, -1.328068155288572e+01f };
        float[] c = { -7.784894002430293e-03f, -3.223964580411365e-01f, -2.400758277161838e+00f, -2.549732539343734e+00f, 4.374664141464968e+00f, 2.938163982698783e+00f };
        float[] d = { 7.784695709041462e-03f, 3.224671290700398e-01f, 2.445134137142996e+00f, 3.754408661907416e+00f };

        float pLow = 0.02425f, pHigh = 1.0f - pLow;
        float q, r;

        if (p < pLow)
        {
            q = MathF.Sqrt(-2.0f * MathF.Log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0f);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5f;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0f);
        }
        else
        {
            q = MathF.Sqrt(-2.0f * MathF.Log(1.0f - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0f);
        }
    }

    /// <summary>Approximation of the Gamma function (Stirling + Lanczos).</summary>
    public static float Gamma(float z)
    {
        if (z < 0.5f)
            return Pi / (MathF.Sin(Pi * z) * Gamma(1.0f - z));
        z -= 1.0f;
        float g = 7.0f;
        float[] c = { 0.99999999999980993f, 676.5203681218851f, -1259.1392167224028f, 771.32342877765313f, -176.61502916214059f, 12.507343278686905f, -0.13857109526572012f, 9.9843695780195716e-6f, 1.5056327351493116e-7f };
        float x = c[0];
        for (int i = 1; i < g + 2; i++)
            x += c[i] / (z + i);
        float t = z + g + 0.5f;
        return Sqrt2Pi * MathF.Pow(t, z + 0.5f) * MathF.Exp(-t) * x;
    }

    /// <summary>Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).</summary>
    public static float Beta(float a, float b) => Gamma(a) * Gamma(b) / Gamma(a + b);

    /// <summary>Regularized incomplete beta function I_x(a,b) (series expansion).</summary>
    public static float RegularizedIncompleteBeta(float x, float a, float b)
    {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;
        float lbeta = MathF.Log(Beta(a, b));
        float front = MathF.Exp(MathF.Log(x) * a + MathF.Log(1 - x) * b - lbeta) / a;
        // Continued fraction (Lentz's method)
        float f = 1, c = 1, d = 0;
        for (int m = 0; m <= 200; m++)
        {
            int m2 = 2 * m;
            float numerator;
            if (m == 0)
                numerator = 1;
            else if (m % 2 == 0)
            {
                float k = m / 2;
                numerator = k * (b - k) * x / ((a + m2 - 2) * (a + m2 - 1));
            }
            else
            {
                float k = (m + 1) / 2;
                numerator = -(a + k - 1) * (a + b + k - 1) * x / ((a + m2 - 2) * (a + m2 - 1));
            }
            d = 1 + numerator * d;
            if (MathF.Abs(d) < 1e-30f)
                d = 1e-30f;
            d = 1.0f / d;
            c = 1 + numerator / c;
            if (MathF.Abs(c) < 1e-30f)
                c = 1e-30f;
            f *= c * d;
            if (MathF.Abs(c * d - 1) < 1e-8f)
                break;
        }
        return front * (f - 1);
    }

    /// <summary>Chi-squared CDF with k degrees of freedom.</summary>
    public static float ChiSquaredCDF(float x, int k)
    {
        if (x <= 0)
            return 0;
        return RegularizedIncompleteBeta(x / 2.0f, k / 2.0f, 0.5f);
    }

    /// <summary>Student's t CDF with ν degrees of freedom.</summary>
    public static float StudentTCDF(float t, float nu)
    {
        float x = nu / (nu + t * t);
        float ibeta = RegularizedIncompleteBeta(x, nu / 2.0f, 0.5f);
        return t >= 0 ? 1.0f - 0.5f * ibeta : 0.5f * ibeta;
    }
}

/// <summary>
/// Ring buffer for streaming statistics computation on field time series.
/// </summary>
public sealed class StreamingStatistics
{
    private readonly float[] _buffer;
    private int _head, _count;
    private float _sum, _sumSq, _min, _max;

    /// <summary>Buffer capacity.</summary>
    public int Capacity => _buffer.Length;
    /// <summary>Number of samples currently in buffer.</summary>
    public int Count => _count;
    /// <summary>Running mean.</summary>
    public float Mean => _count > 0 ? _sum / _count : 0;
    /// <summary>Running variance.</summary>
    public float Variance => _count > 1 ? (_sumSq / _count - Mean * Mean) * _count / (_count - 1) : 0;
    /// <summary>Running standard deviation.</summary>
    public float StdDev => MathF.Sqrt(MathF.Max(Variance, 0));
    /// <summary>Minimum value in buffer.</summary>
    public float Min => _count > 0 ? _min : 0;
    /// <summary>Maximum value in buffer.</summary>
    public float Max => _count > 0 ? _max : 0;

    public StreamingStatistics(int capacity)
    {
        _buffer = new float[capacity];
        _head = 0;
        _count = 0;
        _sum = 0;
        _sumSq = 0;
        _min = float.MaxValue;
        _max = float.MinValue;
    }

    /// <summary>Adds a sample to the streaming statistics.</summary>
    public void Add(float value)
    {
        if (_count == _buffer.Length)
        {
            float removed = _buffer[_head];
            _sum -= removed;
            _sumSq -= removed * removed;
        }
        else
        {
            _count++;
        }
        _buffer[_head] = value;
        _sum += value;
        _sumSq += value * value;
        if (value < _min)
            _min = value;
        if (value > _max)
            _max = value;
        _head = (_head + 1) % _buffer.Length;
    }

    /// <summary>Resets all statistics.</summary>
    public void Reset()
    { _head = 0; _count = 0; _sum = 0; _sumSq = 0; _min = float.MaxValue; _max = float.MinValue; }

    /// <summary>Computes the percentile within the buffer.</summary>
    public float Percentile(float p)
    {
        if (_count == 0)
            return 0;
        Span<float> sorted = stackalloc float[_count];
        for (int i = 0; i < _count; i++)
            sorted[i] = _buffer[i];
        sorted.Sort();
        int idx = Math.Clamp((int)(p * (_count - 1)), 0, _count - 1);
        return sorted[idx];
    }

    /// <summary>Computes the Exponential Moving Average with given smoothing factor.</summary>
    public float EMA(float alpha)
    {
        if (_count == 0)
            return 0;
        float ema = _buffer[0];
        for (int i = 1; i < _count; i++)
            ema = alpha * _buffer[i] + (1 - alpha) * ema;
        return ema;
    }
}

/// <summary>
/// Thread-safe random number generator pool for parallel stochastic field operations.
/// </summary>
public sealed class RandomPool
{
    private readonly Random[] _generators;
    private readonly ThreadLocal<Random> _threadLocal;

    public RandomPool(int seed = 42)
    {
        int poolSize = Math.Max(1, Environment.ProcessorCount);
        _generators = new Random[poolSize];
        for (int i = 0; i < poolSize; i++)
            _generators[i] = new Random(seed + i * 31337);
        _threadLocal = new ThreadLocal<Random>(() =>
        {
            int idx = Environment.CurrentManagedThreadId % _generators.Length;
            return _generators[idx];
        });
    }

    /// <summary>Gets the random generator for the current thread.</summary>
    public Random Current => _threadLocal.Value!;

    /// <summary>Generates a Gaussian sample on the current thread.</summary>
    public float NextGaussian()
    {
        var rng = Current;
        float u1 = 1f - (float)rng.NextDouble();
        float u2 = 1f - (float)rng.NextDouble();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Sin(2f * MathF.PI * u2);
    }

    /// <summary>Generates a Poisson sample on the current thread.</summary>
    public int NextPoisson(float lambda)
    {
        if (lambda <= 0)
            return 0;
        var rng = Current;
        if (lambda > 30)
        { float n = NextGaussian(); return Math.Max(0, (int)(lambda + MathF.Sqrt(lambda) * n + 0.5f)); }
        float L = MathF.Exp(-lambda);
        int k = 0;
        float p = 1;
        do
        { k++; p *= (float)rng.NextDouble(); } while (p > L);
        return k - 1;
    }
}
