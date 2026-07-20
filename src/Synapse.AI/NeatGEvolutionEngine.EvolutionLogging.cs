// =============================================================================
// NeatGEvolutionEngine.cs - NEAT-G Evolution Engine Core
// GDNN.Engine - Geometric Deep Neural Network Engine
// Copyright (c) 2024. All rights reserved.
// =============================================================================
// This file is the heart of the G-DNN Engine implementing the NEAT-G
// (NeuroEvolution of Augmented Topologies - Geometric) algorithm.
// It provides comprehensive evolutionary optimization for neural network
// architectures with geometric awareness, semantic crossover, manifold-based
// speciation, and swarm evolution capabilities.
// =============================================================================

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using GDNN.Core.NEAT.Models;
using Synapse.Infrastructure.Logging;

namespace GDNN.Core.NEAT
{

    /// <summary>
    /// Structured logging for the NEAT-G evolution engine.
    /// Provides severity levels, categorized messages, and optional file output.
    /// </summary>
    public sealed class EvolutionLogger
    {
        private readonly ConcurrentQueue<LogEntry> _logEntries;
        private readonly int _maxEntries;
        private readonly object _writeLock;
        private LogLevel _minimumLevel;

        /// <summary>
        /// Initializes a new instance of the EvolutionLogger class.
        /// </summary>
        /// <param name="minimumLevel">Minimum log level to record.</param>
        /// <param name="maxEntries">Maximum log entries to retain.</param>
        public EvolutionLogger(LogLevel minimumLevel = LogLevel.Info, int maxEntries = 50000)
        {
            _minimumLevel = minimumLevel;
            _maxEntries = maxEntries;
            _logEntries = new ConcurrentQueue<LogEntry>();
            _writeLock = new object();
        }

        /// <summary>Gets the current minimum log level.</summary>
        public LogLevel MinimumLevel
        {
            get => _minimumLevel;
            set => _minimumLevel = value;
        }

        /// <summary>Gets the number of log entries.</summary>
        public int EntryCount => _logEntries.Count;

        /// <summary>
        /// Logs a debug message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogDebug(string message, string? category = null)
        {
            Log(LogLevel.Debug, message, category);
        }

        /// <summary>
        /// Logs an informational message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogInfo(string message, string? category = null)
        {
            Log(LogLevel.Info, message, category);
        }

        /// <summary>
        /// Logs a warning message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogWarning(string message, string? category = null)
        {
            Log(LogLevel.Warning, message, category);
        }

        /// <summary>
        /// Logs an error message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="exception">Optional exception.</param>
        /// <param name="category">Optional category.</param>
        public void LogError(string message, Exception? exception = null, string? category = null)
        {
            Log(LogLevel.Error, message, category, exception);
        }

        /// <summary>
        /// Logs a message at the specified level.
        /// </summary>
        /// <param name="level">Log level.</param>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        /// <param name="exception">Optional exception.</param>
        public void Log(LogLevel level, string message, string? category = null, Exception? exception = null)
        {
            if (level < _minimumLevel)
                return;

            var entry = new LogEntry
            {
                Timestamp = DateTime.UtcNow,
                Level = level,
                Message = message,
                Category = category ?? "General",
                Exception = exception?.ToString(),
                ThreadId = Environment.CurrentManagedThreadId
            };

            _logEntries.Enqueue(entry);

            while (_logEntries.Count > _maxEntries)
            {
                _logEntries.TryDequeue(out _);
            }
        }

        /// <summary>
        /// Gets log entries filtered by level and category.
        /// </summary>
        /// <param name="level">Minimum level (null for all).</param>
        /// <param name="category">Category filter (null for all).</param>
        /// <param name="count">Maximum entries to return.</param>
        public IReadOnlyList<LogEntry> GetEntries(
            LogLevel? level = null,
            string? category = null,
            int count = 100)
        {
            return _logEntries
                .Where(e => (!level.HasValue || e.Level >= level.Value) &&
                           (category == null || e.Category == category))
            .Take(count)
            .ToList()
            .AsReadOnly();
        }

        /// <summary>
        /// Exports all log entries as a formatted string.
        /// </summary>
        /// <param name="format">Log format.</param>
        public string Export(LogFormat format = LogFormat.Text)
        {
            var entries = _logEntries.ToList();

            return format switch
            {
                LogFormat.Text => ExportAsText(entries),
                LogFormat.Json => ExportAsJson(entries),
                LogFormat.Csv => ExportAsCsv(entries),
                _ => ExportAsText(entries)
            };
        }

        /// <summary>
        /// Clears all log entries.
        /// </summary>
        public void Clear()
        {
            while (_logEntries.TryDequeue(out _))
            { }
        }

        /// <summary>
        /// Gets summary statistics for the log.
        /// </summary>
        public LogSummary GetSummary()
        {
            var entries = _logEntries.ToList();
            return new LogSummary
            {
                TotalEntries = entries.Count,
                DebugCount = entries.Count(e => e.Level == LogLevel.Debug),
                InfoCount = entries.Count(e => e.Level == LogLevel.Info),
                WarningCount = entries.Count(e => e.Level == LogLevel.Warning),
                ErrorCount = entries.Count(e => e.Level == LogLevel.Error),
                Categories = entries.Select(e => e.Category).Distinct().ToList().AsReadOnly(),
                FirstEntry = entries.Count > 0 ? entries[0].Timestamp : DateTime.MinValue,
                LastEntry = entries.Count > 0 ? entries[^1].Timestamp : DateTime.MinValue
            };
        }

        private string ExportAsText(List<LogEntry> entries)
        {
            var sb = new StringBuilder();
            foreach (var entry in entries)
            {
                sb.AppendLine($"[{entry.Timestamp:HH:mm:ss.fff}] [{entry.Level}] [{entry.Category}] {entry.Message}");
                if (entry.Exception != null)
                    sb.AppendLine($"  Exception: {entry.Exception}");
            }
            return sb.ToString();
        }

        private string ExportAsJson(List<LogEntry> entries)
        {
            return JsonSerializer.Serialize(entries, new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            });
        }

        private string ExportAsCsv(List<LogEntry> entries)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Timestamp,Level,Category,Message,ThreadId");
            foreach (var entry in entries)
            {
                string msg = entry.Message.Replace("\"", "\"\"");
                sb.AppendLine($"\"{entry.Timestamp:O}\",{entry.Level},{entry.Category},\"{msg}\",{entry.ThreadId}");
            }
            return sb.ToString();
        }
    }

    /// <summary>
    /// Log severity levels.
    /// </summary>
    public enum LogLevel
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }

    /// <summary>
    /// Log output formats.
    /// </summary>
    public enum LogFormat
    {
        Text,
        Json,
        Csv
    }

    /// <summary>
    /// A single log entry.
    /// </summary>
    public sealed class LogEntry
    {
        public DateTime Timestamp { get; init; }
        public LogLevel Level { get; init; }
        public string Message { get; init; } = string.Empty;
        public string Category { get; init; } = string.Empty;
        public string? Exception { get; init; }
        public int ThreadId { get; init; }
    }

    /// <summary>
    /// Summary statistics for logs.
    /// </summary>
    public sealed class LogSummary
    {
        public int TotalEntries { get; init; }
        public int DebugCount { get; init; }
        public int InfoCount { get; init; }
        public int WarningCount { get; init; }
        public int ErrorCount { get; init; }
        public IReadOnlyList<string> Categories { get; init; } = Array.Empty<string>();
        public DateTime FirstEntry { get; init; }
        public DateTime LastEntry { get; init; }
    }

}
