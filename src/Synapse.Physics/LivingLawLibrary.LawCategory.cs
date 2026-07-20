// ============================================================
// LawLibraryRegistry.cs - Synapse Omnia Reference Physics Law Library
// The canonical registry of physical laws consumed by LivingLawCompiler.
// C# 14, unsafe code, NativeAOT compatible.
// ============================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;

namespace Synapse.Physics;


//  SECTION 1 - Core Records and Enums

/// <summary>Classifies a law into a broad scientific domain.</summary>
public enum LawCategory : byte
{
    Mechanics,
    Thermodynamics,
    Electromagnetism,
    FluidDynamics,
    QuantumMechanics,
    Optics,
    Gravitation,
    Chemistry,
    Biology,
    Finance,
    Epidemiology,
    MaterialScience,
    Acoustics,
    Plasma,
    Nuclear,
    Astrophysics,
    Climate,
    Neuroscience
}
