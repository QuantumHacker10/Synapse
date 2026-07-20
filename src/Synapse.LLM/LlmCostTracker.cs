// Multi-provider LLM pipeline for Synapse (split from HybridLlmRouter.cs).

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Timers;
using GDNN.Scene;
using Synapse.Infrastructure.Logging;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Tracks LLM usage costs across providers, enforces budgets, generates reports,
    /// and provides cost optimization suggestions.
    /// </summary>
    public sealed class LlmCostTracker : IDisposable
    {
        private readonly ConcurrentBag<CostEntry> _entries;
        private readonly CostTrackerConfig _config;
        private readonly object _lock = new();
        private decimal _currentDailyCost;
        private decimal _currentMonthlyCost;
        private int _currentDailyTokens;
        private int _currentMonthlyTokens;
        private DateTimeOffset _lastDailyReset;
        private DateTimeOffset _lastMonthlyReset;
        private bool _disposed;

        /// <summary>Current daily cost in USD.</summary>
        public decimal CurrentDailyCost => _currentDailyCost;

        /// <summary>Current monthly cost in USD.</summary>
        public decimal CurrentMonthlyCost => _currentMonthlyCost;

        /// <summary>Total cost entries tracked.</summary>
        public int EntryCount => _entries.Count;

        /// <summary>
        /// Initializes a new cost tracker.
        /// </summary>
        /// <param name="config">Configuration for budget tracking.</param>
        public LlmCostTracker(CostTrackerConfig? config = null)
        {
            _config = config ?? new CostTrackerConfig();
            _entries = new ConcurrentBag<CostEntry>();
            _lastDailyReset = GetDayStart(DateTimeOffset.UtcNow);
            _lastMonthlyReset = GetMonthStart(DateTimeOffset.UtcNow);
        }

        /// <summary>
        /// Tracks cost for a completed request.
        /// </summary>
        /// <param name="entry">Cost entry to record.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public Task TrackAsync(CostEntry entry, CancellationToken cancellationToken = default)
        {
            if (_disposed)
                return Task.CompletedTask;

            _entries.Add(entry);

            lock (_lock)
            {
                CheckAndResetPeriods();
                _currentDailyCost += entry.CostUsd;
                _currentMonthlyCost += entry.CostUsd;
                _currentDailyTokens += entry.InputTokens + entry.OutputTokens;
                _currentMonthlyTokens += entry.InputTokens + entry.OutputTokens;
            }

            if (_config.PersistToDisk && !string.IsNullOrEmpty(_config.StoragePath))
            {
                PersistEntry(entry);
            }

            return Task.CompletedTask;
        }

        /// <summary>
        /// Estimates the cost of a request before sending it.
        /// </summary>
        /// <param name="provider">Provider name.</param>
        /// <param name="model">Model name.</param>
        /// <param name="inputTokens">Estimated input tokens.</param>
        /// <param name="outputTokens">Estimated output tokens.</param>
        /// <returns>Estimated cost in USD.</returns>
        public decimal EstimateCost(string provider, string model, int inputTokens, int outputTokens)
        {
            var pricing = GetModelPricing(model);
            decimal inputCost = (decimal)inputTokens / 1_000_000m * pricing.InputCostPer1M;
            decimal outputCost = (decimal)outputTokens / 1_000_000m * pricing.OutputCostPer1M;
            return inputCost + outputCost;
        }

        /// <summary>
        /// Checks whether a request would exceed budget limits.
        /// </summary>
        /// <param name="estimatedCost">Estimated cost in USD.</param>
        /// <returns>True if within budget.</returns>
        public bool CheckBudget(decimal estimatedCost)
        {
            if (!_config.EnforceLimits)
                return true;

            lock (_lock)
            {
                CheckAndResetPeriods();

                if (_config.PerRequestCeilingUsd.HasValue &&
                    estimatedCost > _config.PerRequestCeilingUsd.Value)
                    return false;

                if (_config.DailyBudgetUsd.HasValue &&
                    _currentDailyCost + estimatedCost > _config.DailyBudgetUsd.Value)
                    return false;

                if (_config.MonthlyBudgetUsd.HasValue &&
                    _currentMonthlyCost + estimatedCost > _config.MonthlyBudgetUsd.Value)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets daily usage report.
        /// </summary>
        /// <param name="date">Date to report on.</param>
        /// <returns>Usage report.</returns>
        public UsageReport GetDailyReport(DateTimeOffset? date = null)
        {
            var targetDate = date ?? DateTimeOffset.UtcNow;
            var dayStart = GetDayStart(targetDate);
            var dayEnd = dayStart.AddDays(1);

            var entries = _entries
                .Where(e => e.Timestamp >= dayStart && e.Timestamp < dayEnd)
                .ToList();

            return BuildReport(entries, "Daily", dayStart, dayEnd);
        }

        /// <summary>
        /// Gets monthly usage report.
        /// </summary>
        /// <param name="date">Date in the target month.</param>
        /// <returns>Usage report.</returns>
        public UsageReport GetMonthlyReport(DateTimeOffset? date = null)
        {
            var targetDate = date ?? DateTimeOffset.UtcNow;
            var monthStart = GetMonthStart(targetDate);
            var monthEnd = monthStart.AddMonths(1);

            var entries = _entries
                .Where(e => e.Timestamp >= monthStart && e.Timestamp < monthEnd)
                .ToList();

            return BuildReport(entries, "Monthly", monthStart, monthEnd);
        }

        /// <summary>
        /// Gets usage breakdown by provider.
        /// </summary>
        /// <returns>Provider usage dictionary.</returns>
        public IReadOnlyDictionary<string, ProviderUsage> GetProviderBreakdown()
        {
            var breakdown = new Dictionary<string, ProviderUsage>(StringComparer.OrdinalIgnoreCase);

            foreach (var entry in _entries)
            {
                if (!breakdown.TryGetValue(entry.Provider, out var usage))
                {
                    usage = new ProviderUsage { Provider = entry.Provider };
                    breakdown[entry.Provider] = usage;
                }

                breakdown[entry.Provider] = usage with
                {
                    TotalCostUsd = usage.TotalCostUsd + entry.CostUsd,
                    TotalInputTokens = usage.TotalInputTokens + entry.InputTokens,
                    TotalOutputTokens = usage.TotalOutputTokens + entry.OutputTokens,
                    RequestCount = usage.RequestCount + 1
                };
            }

            return breakdown;
        }

        /// <summary>
        /// Exports cost data to CSV format.
        /// </summary>
        /// <returns>CSV string.</returns>
        public string ExportCsv()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Timestamp,Provider,Model,InputTokens,OutputTokens,CostUsd,TaskType");

            foreach (var entry in _entries.OrderBy(e => e.Timestamp))
            {
                sb.AppendLine($"{entry.Timestamp:O},{entry.Provider},{entry.Model}," +
                    $"{entry.InputTokens},{entry.OutputTokens},{entry.CostUsd:F6},{entry.TaskType}");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Exports cost data to JSON format.
        /// </summary>
        /// <returns>JSON string.</returns>
        public string ExportJson()
        {
            return JsonSerializer.Serialize(_entries.OrderBy(e => e.Timestamp).ToList(),
                new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>
        /// Gets cost optimization suggestions based on usage patterns.
        /// </summary>
        /// <returns>List of suggestions.</returns>
        public IReadOnlyList<string> GetOptimizationSuggestions()
        {
            var suggestions = new List<string>();
            var breakdown = GetProviderBreakdown();

            foreach (var (provider, usage) in breakdown)
            {
                if (usage.RequestCount > 100 && usage.TotalCostUsd > 1.0m)
                {
                    var avgCostPerRequest = usage.TotalCostUsd / usage.RequestCount;
                    if (avgCostPerRequest > 0.01m)
                    {
                        suggestions.Add(
                            $"Provider '{provider}' averages ${avgCostPerRequest:F4} per request. " +
                            $"Consider using a cheaper model for simple tasks.");
                    }
                }
            }

            var dailyReport = GetDailyReport();
            if (dailyReport.TotalCostUsd > (_config.DailyBudgetUsd ?? decimal.MaxValue) * 0.8m)
            {
                suggestions.Add("Daily budget is approaching 80% usage. Consider reducing request volume.");
            }

            return suggestions;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private void CheckAndResetPeriods()
        {
            var now = DateTimeOffset.UtcNow;
            var currentDayStart = GetDayStart(now);
            if (currentDayStart > _lastDailyReset)
            {
                _currentDailyCost = 0;
                _currentDailyTokens = 0;
                _lastDailyReset = currentDayStart;
            }

            var currentMonthStart = GetMonthStart(now);
            if (currentMonthStart > _lastMonthlyReset)
            {
                _currentMonthlyCost = 0;
                _currentMonthlyTokens = 0;
                _lastMonthlyReset = currentMonthStart;
            }
        }

        private static DateTimeOffset GetDayStart(DateTimeOffset dt)
        {
            return new DateTimeOffset(dt.Year, dt.Month, dt.Day, 0, 0, 0, dt.Offset);
        }

        private static DateTimeOffset GetMonthStart(DateTimeOffset dt)
        {
            return new DateTimeOffset(dt.Year, dt.Month, 1, 0, 0, 0, dt.Offset);
        }

        private static (decimal InputCostPer1M, decimal OutputCostPer1M) GetModelPricing(string model)
        {
            return model switch
            {
                var m when m.Contains("gpt-4o-mini") => (0.15m, 0.60m),
                var m when m.Contains("gpt-4o") => (2.50m, 10.00m),
                var m when m.Contains("gpt-4-turbo") => (10.00m, 30.00m),
                var m when m.Contains("claude-sonnet") => (3.00m, 15.00m),
                var m when m.Contains("claude-haiku") => (0.80m, 4.00m),
                var m when m.Contains("gemini-flash") => (0.075m, 0.30m),
                var m when m.Contains("gemini-pro") => (1.25m, 5.00m),
                _ => (0m, 0m)
            };
        }

        private static UsageReport BuildReport(
            List<CostEntry> entries, string period,
            DateTimeOffset start, DateTimeOffset end)
        {
            var byProvider = entries.GroupBy(e => e.Provider)
                .ToDictionary(g => g.Key, g => g.Sum(e => e.CostUsd));
            var byModel = entries.GroupBy(e => e.Model)
                .ToDictionary(g => g.Key, g => g.Sum(e => e.CostUsd));

            return new UsageReport
            {
                Period = period,
                StartDate = start,
                EndDate = end,
                TotalCostUsd = entries.Sum(e => e.CostUsd),
                TotalInputTokens = entries.Sum(e => e.InputTokens),
                TotalOutputTokens = entries.Sum(e => e.OutputTokens),
                TotalRequests = entries.Count,
                CostByProvider = byProvider,
                CostByModel = byModel
            };
        }

        private void PersistEntry(CostEntry entry)
        {
            try
            {
                if (string.IsNullOrEmpty(_config.StoragePath))
                    return;
                if (!Directory.Exists(_config.StoragePath))
                    Directory.CreateDirectory(_config.StoragePath);

                var filePath = Path.Combine(_config.StoragePath, $"costs_{DateTimeOffset.UtcNow:yyyyMMdd}.jsonl");
                var json = JsonSerializer.Serialize(entry);
                File.AppendAllText(filePath, json + Environment.NewLine);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("LlmCostTracker", "Best-effort cost persistence failed.", ex);
            }
        }
    }

    /// <summary>
    /// Usage report for a time period.
    /// </summary>
    public record UsageReport
    {
        /// <summary>Report period type (Daily, Monthly).</summary>
        public string Period { get; init; } = string.Empty;

        /// <summary>Start of the reporting period.</summary>
        public DateTimeOffset StartDate { get; init; }

        /// <summary>End of the reporting period.</summary>
        public DateTimeOffset EndDate { get; init; }

        /// <summary>Total cost in USD.</summary>
        public decimal TotalCostUsd { get; init; }

        /// <summary>Total input tokens.</summary>
        public int TotalInputTokens { get; init; }

        /// <summary>Total output tokens.</summary>
        public int TotalOutputTokens { get; init; }

        /// <summary>Total requests.</summary>
        public int TotalRequests { get; init; }

        /// <summary>Cost breakdown by provider.</summary>
        public IReadOnlyDictionary<string, decimal> CostByProvider { get; init; } =
            new Dictionary<string, decimal>();

        /// <summary>Cost breakdown by model.</summary>
        public IReadOnlyDictionary<string, decimal> CostByModel { get; init; } =
            new Dictionary<string, decimal>();
    }

    /// <summary>
    /// Usage statistics for a single provider.
    /// </summary>
    public record ProviderUsage
    {
        /// <summary>Provider name.</summary>
        public string Provider { get; init; } = string.Empty;

        /// <summary>Total cost in USD.</summary>
        public decimal TotalCostUsd { get; init; }

        /// <summary>Total input tokens.</summary>
        public int TotalInputTokens { get; init; }

        /// <summary>Total output tokens.</summary>
        public int TotalOutputTokens { get; init; }

        /// <summary>Total request count.</summary>
        public int RequestCount { get; init; }
    }
}
