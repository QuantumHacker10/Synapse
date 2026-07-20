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

// SECTION 9: STOCHASTIC STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Etat stochastique d'un point — porte par le StochasticField pour
/// les phenomenes aleatoires (epidemie, marche, rumeur, turbulence).
/// Contient les moments statistiques et les parametres du processus sous-jacent.
///
/// MEMORY LAYOUT: 128 bytes.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 128, Pack = 32)]
public struct StochasticState
{
    [FieldOffset(0)] public double Mean;                  // Moyenne du processus
    [FieldOffset(8)] public double Variance;              // Variance (sigma^2)
    [FieldOffset(16)] public double Drift;                // Derive (mu)
    [FieldOffset(24)] public double Diffusion;            // Coefficient de diffusion (sigma)
    [FieldOffset(32)] public double JumpIntensity;        // Intensite des sauts (lambda)
    [FieldOffset(40)] public double JumpMean;             // Moyenne des sauts
    [FieldOffset(48)] public double JumpVariance;         // Variance des sauts
    [FieldOffset(56)] public double CorrelationTime;      // Temps de correlation (tau)
    [FieldOffset(64)] public double LongTermMean;         // Moyenne a long terme (pour O-U)
    [FieldOffset(72)] public double MeanReversion;        // Taux de retour a la moyenne (theta)
    [FieldOffset(80)] public double Entropy;              // Entropie de la distribution
    [FieldOffset(88)] public double MutualInformation;    // Information mutuelle
    [FieldOffset(96)] public StochasticProcess ProcessType; // Type de processus
    [FieldOffset(100)] public int _pad0;
    [FieldOffset(104)] public double Volatility;          // Volatilite (annualisee)
    [FieldOffset(112)] public double Skewness;            // Asymetrie de la distribution
    [FieldOffset(120)] public double Kurtosis;            // Aplatissement (excedent)

    public static StochasticState Default => new() { ProcessType = StochasticProcess.Deterministic, Diffusion = 1.0 };
    public readonly double StandardDeviation() => Math.Sqrt(Math.Max(0, Variance));
    public readonly double CoefficientOfVariation() => Mean != 0 ? StandardDeviation() / Math.Abs(Mean) : 0;
    public readonly double ConfidenceInterval95 => 1.96 * StandardDeviation();
    public readonly double ConfidenceInterval99 => 2.576 * StandardDeviation();

    /// <summary>Evolution d'Ornstein-Uhlenbeck : dx = theta*(mu-x)*dt + sigma*dW.</summary>
    public readonly StochasticState StepOU(double dt) => new() { Mean = LongTermMean + (Mean - LongTermMean) * Math.Exp(-MeanReversion * dt), Variance = Diffusion * Diffusion / (2 * MeanReversion) * (1 - Math.Exp(-2 * MeanReversion * dt)), Drift = Drift, Diffusion = Diffusion, JumpIntensity = JumpIntensity, JumpMean = JumpMean, JumpVariance = JumpVariance, CorrelationTime = CorrelationTime, LongTermMean = LongTermMean, MeanReversion = MeanReversion, ProcessType = ProcessType, Volatility = Volatility };
    /// <summary>Evolution de Geometric Brownian Motion : dS = mu*S*dt + sigma*S*dW.</summary>
    public readonly StochasticState StepGBM(double dt) { double drift = (Drift - 0.5 * Diffusion * Diffusion) * dt; double vol = Diffusion * Math.Sqrt(dt); return new() { Mean = Mean * Math.Exp(drift), Variance = Variance * Math.Exp(2 * Drift + vol * vol) * (Math.Exp(vol * vol) - 1), Drift = Drift, Diffusion = Diffusion, ProcessType = ProcessType, Volatility = Volatility }; }

    public static StochasticState Lerp(StochasticState a, StochasticState b, double t) { double u = 1 - t; return new StochasticState { Mean = a.Mean * u + b.Mean * t, Variance = a.Variance * u + b.Variance * t, Drift = a.Drift * u + b.Drift * t, Diffusion = a.Diffusion * u + b.Diffusion * t, JumpIntensity = a.JumpIntensity * u + b.JumpIntensity * t, CorrelationTime = a.CorrelationTime * u + b.CorrelationTime * t, LongTermMean = a.LongTermMean * u + b.LongTermMean * t, MeanReversion = a.MeanReversion * u + b.MeanReversion * t, Entropy = a.Entropy * u + b.Entropy * t, ProcessType = t < 0.5 ? a.ProcessType : b.ProcessType, Volatility = a.Volatility * u + b.Volatility * t }; }
}

// ═══════════════════════════════════════════════════════════════════════════════
