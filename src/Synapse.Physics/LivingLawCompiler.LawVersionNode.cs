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
    // LawVersionTree — version tree for law modifications
    // =========================================================================

    /// <summary>A node in the law version tree.</summary>
    public sealed class LawVersionNode
    {
        public string VersionId { get; set; } = Guid.NewGuid().ToString("N")[..8];
        public string Expression { get; set; } = "";
        public string Description { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public LawVersionNode? Parent { get; set; }
        public List<LawVersionNode> Children { get; } = new();
        public LawBytecode? CompiledBytecode { get; set; }
        public bool IsActive { get; set; }
        public Dictionary<string, string> Metadata { get; } = new();
    }
}
