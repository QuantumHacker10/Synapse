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

    /// <summary>Collects and manages diagnostic messages during compilation.</summary>
    public sealed class LawDiagnostics
    {
        private readonly List<DiagnosticMessage> _messages = new();
        private readonly object _lock = new();

        public int ErrorCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Error || m.Severity == DiagnosticSeverity.Critical); }
        }

        public int WarningCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Warning); }
        }

        public int InfoCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Info); }
        }

        public bool HasErrors => ErrorCount > 0;

        public IReadOnlyList<DiagnosticMessage> Messages
        {
            get { lock (_lock) return _messages.ToList(); }
        }

        public void Report(DiagnosticSeverity severity, string code, string message, int line = 0, int column = 0, string? expression = null, string? suggestion = null)
        {
            lock (_lock)
            {
                _messages.Add(new DiagnosticMessage
                {
                    Severity = severity,
                    Code = code,
                    Message = message,
                    Line = line,
                    Column = column,
                    Expression = expression,
                    Suggestion = suggestion
                });
            }
        }

        public void Info(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Info, code, message, suggestion: suggestion);

        public void Warn(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Warning, code, message, suggestion: suggestion);

        public void Error(string code, string message, int line = 0, int column = 0, string? suggestion = null) =>
            Report(DiagnosticSeverity.Error, code, message, line, column, suggestion: suggestion);

        public void Critical(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Critical, code, message, suggestion: suggestion);

        public void Clear() { lock (_lock) { _messages.Clear(); } }

        public IReadOnlyList<DiagnosticMessage> GetBySeverity(DiagnosticSeverity severity)
        {
            lock (_lock)
                return _messages.Where(m => m.Severity == severity).ToList();
        }

        public IReadOnlyList<DiagnosticMessage> GetByCode(string code)
        {
            lock (_lock)
                return _messages.Where(m => m.Code == code).ToList();
        }

        public string FormatReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"=== Diagnostics Report ({_messages.Count} messages) ===");
            sb.AppendLine($"Errors: {ErrorCount}, Warnings: {WarningCount}, Info: {InfoCount}");
            sb.AppendLine();

            foreach (var msg in _messages.OrderByDescending(m => m.Severity))
            {
                sb.AppendLine(msg.ToString());
            }

            return sb.ToString();
        }

        public string FormatErrorsOnly()
        {
            var sb = new StringBuilder();
            var errors = GetBySeverity(DiagnosticSeverity.Error).Concat(GetBySeverity(DiagnosticSeverity.Critical));
            foreach (var err in errors)
                sb.AppendLine(err.ToString());
            return sb.ToString();
        }

        public static LawDiagnostics FromCompilationResult(CompilationResult result)
        {
            var diag = new LawDiagnostics();
            if (!result.Success)
            {
                foreach (var error in result.Errors)
                    diag.Error("COMP001", error);
            }
            foreach (var warning in result.Warnings)
                diag.Warn("COMP002", warning);
            return diag;
        }

        public static LawDiagnostics FromValidationResult(ValidationResult result)
        {
            var diag = new LawDiagnostics();
            if (!result.IsValid)
            {
                foreach (var error in result.Errors)
                    diag.Error("VAL001", error);
            }
            foreach (var warning in result.Warnings)
                diag.Warn("VAL002", warning);
            if (!result.DimensionallyConsistent)
                diag.Error("VAL003", "Expression is not dimensionally consistent");
            return diag;
        }
    }
}
