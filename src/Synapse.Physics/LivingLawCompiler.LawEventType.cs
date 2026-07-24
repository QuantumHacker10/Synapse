// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    // =========================================================================
    // LawEventSystem — event system for compiler lifecycle
    // =========================================================================

    /// <summary>Events that can occur during compilation and law processing.</summary>
    public enum LawEventType
    {
        CompilationStarted, CompilationCompleted, CompilationFailed,
        HotReloadTriggered, HotReloadCompleted,
        VersionCreated, VersionRolledBack, VersionForked, VersionMerged,
        LawModified, LawApplied, ValidationCompleted, ValidationFailed,
        CacheHit, CacheMiss, CacheEviction,
        SimulationStarted, SimulationCompleted, SimulationFailed,
        LawInvented, LawImported, LawExported
    }
}
