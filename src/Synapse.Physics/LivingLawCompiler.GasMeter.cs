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
    // BytecodeInterpreter — stack-based VM for executing bytecode
    // =========================================================================

    /// <summary>Gas metering configuration for preventing infinite loops.</summary>
    public sealed class GasMeter
    {
        private long _gasRemaining;
        public long GasRemaining => _gasRemaining;
        public long MaxGas { get; }
        public long GasUsed => MaxGas - _gasRemaining;

        public GasMeter(long maxGas = 1_000_000)
        {
            MaxGas = maxGas;
            _gasRemaining = maxGas;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Consume(int amount)
        {
            _gasRemaining -= amount;
            return _gasRemaining >= 0;
        }

        public void Reset() => _gasRemaining = MaxGas;
    }
}
