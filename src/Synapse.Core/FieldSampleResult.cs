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

// SECTION 11: FIELD SAMPLE RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Resultat d'une evaluation de champ pour un point donne.
/// Contient toutes les informations pour le rendu, l'inspection et la journalisation.
/// Combine l'etat physique, l'etat stochastique, le gradient, et les metadonnees.
///
/// MEMORY LAYOUT: ~480 bytes.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 480, Pack = 32)]
public unsafe struct FieldSampleResult
{
    [FieldOffset(0)] public PhysicsState State;
    [FieldOffset(256)] public StochasticState Stochastic;
    [FieldOffset(384)] public FieldGradient Gradient;
    [FieldOffset(384 + 192 - 4)] public float SdfValue;       // Distance signee au champ SDF
    [FieldOffset(384 + 192 - 3)] public float Confidence;     // Confiance du reseau neuronal
    [FieldOffset(384 + 192 - 2)] public ushort ActiveLod;     // Niveau de detail actif (LOD)
    [FieldOffset(384 + 192 - 1)] public byte Representation;  // 0=poly, 1=neuralSDF, 2=physics
    [FieldOffset(384 + 192)] public byte Flags;

    public static readonly FieldSampleResult Default = default;
    public readonly bool IsEmpty => State.Norm < 1e-20 && Stochastic.Mean == 0;
    public readonly double TotalEnergy => State.KineticEnergy + State.InternalEnergy;
    public readonly double TotalEntropy => State.Entropy + Stochastic.Entropy;
    public readonly Vector3D NetFlux => State.HeatFlux + State.Velocity * State.Density;
}

// ═══════════════════════════════════════════════════════════════════════════════
