using System;
// ============================================================
// FILE: AsyncPipeline.cs
// PATH: Streaming/AsyncPipeline.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Streaming
{
    /// <summary>
    /// Represents the status of a pipeline stage.
    /// </summary>
    public enum PipelineStageStatus : byte
    {
        /// <summary>Stage has not started.</summary>
        Pending = 0,

        /// <summary>Stage is currently executing.</summary>
        Running = 1,

        /// <summary>Stage completed successfully.</summary>
        Completed = 2,

        /// <summary>Stage failed with an error.</summary>
        Failed = 3,

        /// <summary>Stage was cancelled.</summary>
        Cancelled = 4,

        /// <summary>Stage was skipped (e.g., optional stage not needed).</summary>
        Skipped = 5,

        /// <summary>Stage is retrying after a failure.</summary>
        Retrying = 6
    }

    /// <summary>
    /// Represents the overall status of a pipeline job.
    /// </summary>
    public enum PipelineJobStatus : byte
    {
        /// <summary>Job has been created but not yet started.</summary>
        Created = 0,

        /// <summary>Job is currently running.</summary>
        Running = 1,

        /// <summary>Job completed all stages successfully.</summary>
        Completed = 2,

        /// <summary>Job failed at one or more stages.</summary>
        Failed = 3,

        /// <summary>Job was cancelled.</summary>
        Cancelled = 4,

        /// <summary>Job is waiting to be scheduled.</summary>
        Waiting = 5
    }

    /// <summary>
    /// Non-generic pipeline stage contract for object-typed pipelines.
    /// </summary>
    public interface IObjectPipelineStage
    {
        string Name { get; }
        int Order { get; }
        bool IsRequired { get; }
        int TimeoutMs { get; }
        int MaxRetries { get; }
        Task<object> ProcessAsync(object input, CancellationToken cancellationToken);
        bool ValidateInput(object input);
        bool ValidateOutput(object output);
    }

    /// <summary>
    /// Defines a stage in the processing pipeline.
    /// </summary>
    /// <typeparam name="TInput">Input type for this stage.</typeparam>
    /// <typeparam name="TOutput">Output type from this stage.</typeparam>
    public interface IPipelineStage<TInput, TOutput>
    {
        /// <summary>Unique name for this stage.</summary>
        string Name { get; }

        /// <summary>Order index for sequential execution.</summary>
        int Order { get; }

        /// <summary>Whether this stage is required or can be skipped.</summary>
        bool IsRequired { get; }

        /// <summary>Timeout for this stage in milliseconds (0 = no timeout).</summary>
        int TimeoutMs { get; }

        /// <summary>Maximum number of retry attempts.</summary>
        int MaxRetries { get; }

        /// <summary>Processes the input and returns the output.</summary>
        Task<TOutput> ProcessAsync(TInput input, CancellationToken cancellationToken);

        /// <summary>Validates the input before processing.</summary>
        bool ValidateInput(TInput input);

        /// <summary>Validates the output after processing.</summary>
        bool ValidateOutput(TOutput output);
    }

    /// <summary>
    /// Represents a pipeline stage as a delegate.
    /// </summary>
    public sealed class DelegateStage<TInput, TOutput> : IPipelineStage<TInput, TOutput>, IObjectPipelineStage
    {
        private readonly Func<TInput, CancellationToken, Task<TOutput>> _processor;
        private Func<TInput, bool>? _inputValidator;
        private Func<TOutput, bool>? _outputValidator;

        /// <inheritdoc/>
        public string Name { get; init; } = $"Stage_{typeof(TInput).Name}_to_{typeof(TOutput).Name}";

        /// <inheritdoc/>
        public int Order { get; init; }

        /// <inheritdoc/>
        public bool IsRequired { get; init; } = true;

        /// <inheritdoc/>
        public int TimeoutMs { get; init; } = 30000;

        /// <inheritdoc/>
        public int MaxRetries { get; init; } = 3;

        /// <summary>
        /// Initializes a new delegate stage.
        /// </summary>
        public DelegateStage(Func<TInput, CancellationToken, Task<TOutput>> processor)
        {
            _processor = processor ?? throw new ArgumentNullException(nameof(processor));
        }

        /// <summary>
        /// Sets the input validator.
        /// </summary>
        public DelegateStage<TInput, TOutput> WithInputValidator(Func<TInput, bool> validator)
        {
            _inputValidator = validator;
            return this;
        }

        /// <summary>
        /// Sets the output validator.
        /// </summary>
        public DelegateStage<TInput, TOutput> WithOutputValidator(Func<TOutput, bool> validator)
        {
            _outputValidator = validator;
            return this;
        }

        /// <inheritdoc/>
        public async Task<TOutput> ProcessAsync(TInput input, CancellationToken cancellationToken)
        {
            return await _processor(input, cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public bool ValidateInput(TInput input)
        {
            return _inputValidator?.Invoke(input) ?? true;
        }

        /// <inheritdoc/>
        public bool ValidateOutput(TOutput output)
        {
            return _outputValidator?.Invoke(output) ?? true;
        }

        Task<object> IObjectPipelineStage.ProcessAsync(object input, CancellationToken cancellationToken) =>
            ProcessAsync((TInput)input, cancellationToken).ContinueWith(t => (object)t.Result, cancellationToken);

        bool IObjectPipelineStage.ValidateInput(object input) => ValidateInput((TInput)input);

        bool IObjectPipelineStage.ValidateOutput(object output) => ValidateOutput((TOutput)output);
    }

    /// <summary>
    /// Tracks the status and timing of a single stage execution.
    /// </summary>
    public sealed class StageExecutionRecord
    {
        /// <summary>Stage name.</summary>
        public string StageName { get; init; }

        /// <summary>Current status.</summary>
        public PipelineStageStatus Status { get; set; } = PipelineStageStatus.Pending;

        /// <summary>Start time.</summary>
        public DateTime? StartTime { get; set; }

        /// <summary>End time.</summary>
        public DateTime? EndTime { get; set; }

        /// <summary>Duration in milliseconds.</summary>
        public double DurationMs => EndTime.HasValue && StartTime.HasValue
            ? (EndTime.Value - StartTime.Value).TotalMilliseconds
            : 0;

        /// <summary>Number of retry attempts.</summary>
        public int RetryCount { get; set; }

        /// <summary>Error message if failed.</summary>
        public string? ErrorMessage { get; set; }

        /// <summary>Exception if failed.</summary>
        public Exception? Exception { get; set; }

        /// <summary>Whether timeout was reached.</summary>
        public bool TimedOut { get; set; }
    }

    /// <summary>
    /// Represents a complete pipeline job with all stage records.
    /// </summary>
    public sealed class PipelineJob
    {
        /// <summary>Unique job identifier.</summary>
        public string JobId { get; init; }

        /// <summary>Human-readable job name.</summary>
        public string? Name { get; set; }

        /// <summary>Current job status.</summary>
        public PipelineJobStatus Status { get; set; } = PipelineJobStatus.Created;

        /// <summary>Stage execution records in order.</summary>
        public List<StageExecutionRecord> Stages { get; init; } = new();

        /// <summary>Job creation time.</summary>
        public DateTime CreatedTime { get; init; } = DateTime.UtcNow;

        /// <summary>Job start time.</summary>
        public DateTime? StartTime { get; set; }

        /// <summary>Job completion time.</summary>
        public DateTime? EndTime { get; set; }

        /// <summary>Total job duration in milliseconds.</summary>
        public double TotalDurationMs => EndTime.HasValue && StartTime.HasValue
            ? (EndTime.Value - StartTime.Value).TotalMilliseconds
            : 0;

        /// <summary>Current stage index being processed.</summary>
        public int CurrentStageIndex { get; set; }

        /// <summary>Cancellation token source for this job.</summary>
        public CancellationTokenSource? CancellationTokenSource { get; set; }

        /// <summary>Job priority.</summary>
        public int Priority { get; set; }

        /// <summary>Custom user data.</summary>
        public Dictionary<string, object> UserData { get; init; } = new();

        /// <summary>Gets a summary of the job execution.</summary>
        public string GetSummary()
        {
            int completed = Stages.Count(s => s.Status == PipelineStageStatus.Completed);
            int failed = Stages.Count(s => s.Status == PipelineStageStatus.Failed);
            return $"Job {JobId} ({Name ?? "unnamed"}): {Status} - " +
                   $"{completed}/{Stages.Count} stages completed, {failed} failed - " +
                   $"{TotalDurationMs:F1}ms";
        }
    }

    /// <summary>
    /// Metrics collected during pipeline execution.
    /// </summary>
    public sealed class PipelineMetrics
    {
        private long _totalJobsProcessed;
        private long _totalJobsCompleted;
        private long _totalJobsFailed;
        private long _totalStagesExecuted;
        private long _totalRetries;
        private long _totalTimeouts;

        /// <summary>Total jobs processed.</summary>
        public long TotalJobsProcessed
        {
            get => Interlocked.Read(ref _totalJobsProcessed);
            set => Interlocked.Exchange(ref _totalJobsProcessed, value);
        }

        /// <summary>Total jobs that completed successfully.</summary>
        public long TotalJobsCompleted
        {
            get => Interlocked.Read(ref _totalJobsCompleted);
            set => Interlocked.Exchange(ref _totalJobsCompleted, value);
        }

        /// <summary>Total jobs that failed.</summary>
        public long TotalJobsFailed
        {
            get => Interlocked.Read(ref _totalJobsFailed);
            set => Interlocked.Exchange(ref _totalJobsFailed, value);
        }

        /// <summary>Total individual stage executions.</summary>
        public long TotalStagesExecuted
        {
            get => Interlocked.Read(ref _totalStagesExecuted);
            set => Interlocked.Exchange(ref _totalStagesExecuted, value);
        }

        /// <summary>Total retry attempts across all stages.</summary>
        public long TotalRetries
        {
            get => Interlocked.Read(ref _totalRetries);
            set => Interlocked.Exchange(ref _totalRetries, value);
        }

        /// <summary>Total timeouts across all stages.</summary>
        public long TotalTimeouts
        {
            get => Interlocked.Read(ref _totalTimeouts);
            set => Interlocked.Exchange(ref _totalTimeouts, value);
        }

        /// <summary>Average job duration in milliseconds.</summary>
        public double AverageJobDurationMs { get; set; }

        /// <summary>Average stage duration in milliseconds.</summary>
        public double AverageStageDurationMs { get; set; }

        /// <summary>Job success rate (0.0-1.0).</summary>
        public double SuccessRate
        {
            get
            {
                long completed = TotalJobsCompleted;
                long total = TotalJobsCompleted + TotalJobsFailed;
                return total > 0 ? (double)completed / total : 0.0;
            }
        }

        /// <summary>Active jobs currently in progress.</summary>
        public int ActiveJobs { get; set; }

        /// <summary>Queued jobs waiting to be processed.</summary>
        public int QueuedJobs { get; set; }

        /// <summary>Resets all metrics.</summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _totalJobsProcessed, 0);
            Interlocked.Exchange(ref _totalJobsCompleted, 0);
            Interlocked.Exchange(ref _totalJobsFailed, 0);
            Interlocked.Exchange(ref _totalStagesExecuted, 0);
            Interlocked.Exchange(ref _totalRetries, 0);
            Interlocked.Exchange(ref _totalTimeouts, 0);
            AverageJobDurationMs = 0;
            AverageStageDurationMs = 0;
            ActiveJobs = 0;
            QueuedJobs = 0;
        }

        internal void IncrementJobsProcessed() => Interlocked.Increment(ref _totalJobsProcessed);
        internal void IncrementJobsCompleted() => Interlocked.Increment(ref _totalJobsCompleted);
        internal void IncrementJobsFailed() => Interlocked.Increment(ref _totalJobsFailed);
        internal void IncrementStagesExecuted() => Interlocked.Increment(ref _totalStagesExecuted);
        internal void IncrementRetries() => Interlocked.Increment(ref _totalRetries);
        internal void IncrementTimeouts() => Interlocked.Increment(ref _totalTimeouts);
    }

    /// <summary>
    /// Configuration for the async pipeline.
    /// </summary>
    public sealed class PipelineConfig
    {
        /// <summary>Maximum concurrent jobs.</summary>
        public int MaxConcurrentJobs { get; set; } = 4;

        /// <summary>Default timeout for stages in milliseconds.</summary>
        public int DefaultStageTimeoutMs { get; set; } = 30000;

        /// <summary>Default maximum retries per stage.</summary>
        public int DefaultMaxRetries { get; set; } = 3;

        /// <summary>Base delay between retries in milliseconds.</summary>
        public int RetryBaseDelayMs { get; set; } = 100;

        /// <summary>Maximum retry delay in milliseconds.</summary>
        public int RetryMaxDelayMs { get; set; } = 5000;

        /// <summary>Metrics reporting interval in seconds.</summary>
        public float MetricsReportingIntervalSeconds { get; set; } = 5.0f;

        /// <summary>Whether to enable pipeline monitoring.</summary>
        public bool EnableMonitoring { get; set; } = true;
    }

    /// <summary>
    /// Event arguments for pipeline stage progress.
    /// </summary>
    public sealed class PipelineStageEventArgs : EventArgs
    {
        /// <summary>Job identifier.</summary>
        public string JobId { get; init; }

        /// <summary>Stage name.</summary>
        public string StageName { get; init; }

        /// <summary>Current stage status.</summary>
        public PipelineStageStatus Status { get; init; }

        /// <summary>Stage index (0-based).</summary>
        public int StageIndex { get; init; }

        /// <summary>Total number of stages.</summary>
        public int TotalStages { get; init; }

        /// <summary>Progress as a percentage (0-100).</summary>
        public float ProgressPercent { get; init; }

        /// <summary>Error message if status is Failed.</summary>
        public string? ErrorMessage { get; init; }

        /// <summary>Exception if status is Failed.</summary>
        public Exception? Exception { get; init; }
    }

    /// <summary>
    /// Asynchronous pipeline for multi-stage asset processing with timeout,
    /// retry logic, monitoring, and metrics collection.
    /// </summary>
    public sealed class AsyncPipeline : IDisposable
    {
        private readonly PipelineConfig _config;
        private readonly PipelineMetrics _metrics;
        private readonly ConcurrentDictionary<string, PipelineJob> _jobs;
        private readonly SemaphoreSlim _jobSemaphore;
        private readonly CancellationTokenSource _shutdownCts;
        private readonly Task _monitoringTask;
        private bool _disposed;

        private static int GetStageIndex(IReadOnlyList<IObjectPipelineStage> stages, IObjectPipelineStage stage)
        {
            for (int i = 0; i < stages.Count; i++)
            {
                if (ReferenceEquals(stages[i], stage))
                    return i;
            }
            return -1;
        }

        /// <summary>Raised when a stage status changes.</summary>
        public event EventHandler<PipelineStageEventArgs>? StageProgress;

        /// <summary>Raised when metrics are updated.</summary>
        public event EventHandler<PipelineMetrics>? MetricsUpdated;

        /// <summary>Gets the current pipeline metrics.</summary>
        public PipelineMetrics Metrics => _metrics;

        /// <summary>Gets the current configuration.</summary>
        public PipelineConfig Configuration => _config;

        /// <summary>Gets the number of active jobs.</summary>
        public int ActiveJobCount => _jobs.Count(kvp =>
            kvp.Value.Status == PipelineJobStatus.Running);

        /// <summary>
        /// Initializes a new instance of the <see cref="AsyncPipeline"/> class.
        /// </summary>
        /// <param name="config">Pipeline configuration.</param>
        public AsyncPipeline(PipelineConfig? config = null)
        {
            _config = config ?? new PipelineConfig();
            _metrics = new PipelineMetrics();
            _jobs = new ConcurrentDictionary<string, PipelineJob>();
            _jobSemaphore = new SemaphoreSlim(_config.MaxConcurrentJobs, _config.MaxConcurrentJobs);
            _shutdownCts = new CancellationTokenSource();

            if (_config.EnableMonitoring)
            {
                _monitoringTask = Task.Run(() => MonitoringLoopAsync(_shutdownCts.Token));
            }
        }

        /// <summary>
        /// Creates and enqueues a multi-stage pipeline job.
        /// </summary>
        /// <param name="stages">Ordered list of stages to execute.</param>
        /// <param name="initialInput">Initial input for the first stage.</param>
        /// <param name="jobName">Optional human-readable job name.</param>
        /// <param name="priority">Job priority (higher = more important).</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The created pipeline job.</returns>
        public PipelineJob CreateJob<TInput>(
            IReadOnlyList<IObjectPipelineStage> stages,
            TInput initialInput,
            string? jobName = null,
            int priority = 0,
            CancellationToken cancellationToken = default)
        {
            var jobId = Guid.NewGuid().ToString("N")[..12];
            var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            var job = new PipelineJob
            {
                JobId = jobId,
                Name = jobName,
                Priority = priority,
                CancellationTokenSource = cts,
                StartTime = DateTime.UtcNow,
                Status = PipelineJobStatus.Waiting,
                UserData = { ["InitialInput"] = initialInput! }
            };

            foreach (var stage in stages)
            {
                job.Stages.Add(new StageExecutionRecord
                {
                    StageName = stage.Name
                });
            }

            _jobs[jobId] = job;
            _metrics.IncrementJobsProcessed();

            _ = Task.Run(() => ExecuteJobAsync(job, stages, cts.Token), cts.Token);

            return job;
        }

        /// <summary>
        /// Creates and executes a typed pipeline with automatic type conversions.
        /// </summary>
        /// <typeparam name="TInput">Initial input type.</typeparam>
        /// <typeparam name="TOutput">Final output type.</typeparam>
        /// <param name="stages">Ordered list of typed stages.</param>
        /// <param name="initialInput">Initial input value.</param>
        /// <param name="jobName">Optional job name.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Task with the final output.</returns>
        public async Task<TOutput> ExecuteTypedAsync<TInput, TOutput>(
            IReadOnlyList<IObjectPipelineStage> stages,
            TInput initialInput,
            string? jobName = null,
            CancellationToken cancellationToken = default)
        {
            object current = initialInput!;

            foreach (var stage in stages)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var record = new StageExecutionRecord
                {
                    StageName = stage.Name,
                    StartTime = DateTime.UtcNow,
                    Status = PipelineStageStatus.Running
                };

                OnStageProgress(new PipelineStageEventArgs
                {
                    JobId = "typed",
                    StageName = stage.Name,
                    Status = PipelineStageStatus.Running,
                    StageIndex = GetStageIndex(stages, stage),
                    TotalStages = stages.Count,
                    ProgressPercent = (float)(GetStageIndex(stages, stage)) / stages.Count * 100
                });

                try
                {
                    if (!stage.ValidateInput(current))
                    {
                        throw new InvalidOperationException($"Input validation failed for stage {stage.Name}");
                    }

                    using var timeoutCts = new CancellationTokenSource(
                        stage.TimeoutMs > 0 ? stage.TimeoutMs : Timeout.Infinite);
                    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
                        cancellationToken, timeoutCts.Token);

                    current = await stage.ProcessAsync(current, linkedCts.Token).ConfigureAwait(false);

                    if (!stage.ValidateOutput(current))
                    {
                        throw new InvalidOperationException($"Output validation failed for stage {stage.Name}");
                    }

                    record.Status = PipelineStageStatus.Completed;
                    record.EndTime = DateTime.UtcNow;

                    OnStageProgress(new PipelineStageEventArgs
                    {
                        JobId = "typed",
                        StageName = stage.Name,
                        Status = PipelineStageStatus.Completed,
                        StageIndex = GetStageIndex(stages, stage),
                        TotalStages = stages.Count,
                        ProgressPercent = (float)(GetStageIndex(stages, stage) + 1) / stages.Count * 100
                    });
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    record.Status = PipelineStageStatus.Cancelled;
                    record.EndTime = DateTime.UtcNow;
                    throw;
                }
                catch (OperationCanceledException)
                {
                    record.Status = PipelineStageStatus.Failed;
                    record.TimedOut = true;
                    record.ErrorMessage = "Stage timed out";
                    record.EndTime = DateTime.UtcNow;

                    _metrics.IncrementTimeouts();

                    OnStageProgress(new PipelineStageEventArgs
                    {
                        JobId = "typed",
                        StageName = stage.Name,
                        Status = PipelineStageStatus.Failed,
                        StageIndex = GetStageIndex(stages, stage),
                        TotalStages = stages.Count,
                        ErrorMessage = "Timeout"
                    });

                    if (stage.IsRequired)
                        throw;
                }
                catch (Exception ex)
                {
                    record.Status = PipelineStageStatus.Failed;
                    record.ErrorMessage = ex.Message;
                    record.Exception = ex;
                    record.EndTime = DateTime.UtcNow;

                    OnStageProgress(new PipelineStageEventArgs
                    {
                        JobId = "typed",
                        StageName = stage.Name,
                        Status = PipelineStageStatus.Failed,
                        StageIndex = GetStageIndex(stages, stage),
                        TotalStages = stages.Count,
                        ErrorMessage = ex.Message,
                        Exception = ex
                    });

                    if (stage.IsRequired)
                        throw;
                }

                _metrics.IncrementStagesExecuted();
            }

            return (TOutput)current!;
        }

        /// <summary>
        /// Executes a pipeline job with all configured stages.
        /// </summary>
        private async Task ExecuteJobAsync(
            PipelineJob job,
            IReadOnlyList<IObjectPipelineStage> stages,
            CancellationToken cancellationToken)
        {
            await _jobSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            _metrics.ActiveJobs++;

            try
            {
                job.Status = PipelineJobStatus.Running;
                job.StartTime = DateTime.UtcNow;
                object current = job.UserData.TryGetValue("InitialInput", out var input) ? input! : null!;

                for (int i = 0; i < stages.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var stage = stages[i];
                    var record = job.Stages[i];
                    job.CurrentStageIndex = i;

                    record.Status = PipelineStageStatus.Running;
                    record.StartTime = DateTime.UtcNow;

                    OnStageProgress(new PipelineStageEventArgs
                    {
                        JobId = job.JobId,
                        StageName = stage.Name,
                        Status = PipelineStageStatus.Running,
                        StageIndex = i,
                        TotalStages = stages.Count,
                        ProgressPercent = (float)i / stages.Count * 100
                    });

                    bool success = await ExecuteStageWithRetryAsync(
                        stage, record, job, current, cancellationToken).ConfigureAwait(false);

                    if (!success && stage.IsRequired)
                    {
                        job.Status = PipelineJobStatus.Failed;
                        job.EndTime = DateTime.UtcNow;
                        _metrics.IncrementJobsFailed();
                        return;
                    }
                }

                job.Status = PipelineJobStatus.Completed;
                job.EndTime = DateTime.UtcNow;
                _metrics.IncrementJobsCompleted();
            }
            catch (OperationCanceledException)
            {
                job.Status = PipelineJobStatus.Cancelled;
                job.EndTime = DateTime.UtcNow;
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AsyncPipeline", $"Pipeline job '{job.JobId}' failed.", ex);
                job.Status = PipelineJobStatus.Failed;
                job.EndTime = DateTime.UtcNow;
                _metrics.IncrementJobsFailed();
            }
            finally
            {
                _metrics.ActiveJobs--;
                _jobSemaphore.Release();
            }
        }

        /// <summary>
        /// Executes a single stage with retry logic and timeout handling.
        /// </summary>
        private async Task<bool> ExecuteStageWithRetryAsync(
            IObjectPipelineStage stage,
            StageExecutionRecord record,
            PipelineJob job,
            object currentInput,
            CancellationToken cancellationToken)
        {
            int maxAttempts = Math.Max(1, stage.MaxRetries + 1);

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                if (cancellationToken.IsCancellationRequested)
                    return false;

                if (attempt > 0)
                {
                    record.RetryCount = attempt;
                    record.Status = PipelineStageStatus.Retrying;
                    _metrics.IncrementRetries();

                    int delay = Math.Min(
                        _config.RetryBaseDelayMs * (1 << (attempt - 1)),
                        _config.RetryMaxDelayMs);
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }

                record.Status = PipelineStageStatus.Running;
                record.StartTime = DateTime.UtcNow;
                record.TimedOut = false;
                record.ErrorMessage = null;
                record.Exception = null;

                try
                {
                    if (!stage.ValidateInput(currentInput))
                    {
                        throw new InvalidOperationException(
                            $"Input validation failed for stage {stage.Name}");
                    }

                    using var timeoutCts = new CancellationTokenSource(
                        stage.TimeoutMs > 0 ? stage.TimeoutMs : _config.DefaultStageTimeoutMs);
                    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
                        cancellationToken, timeoutCts.Token);

                    var result = await stage.ProcessAsync(currentInput, linkedCts.Token)
                        .ConfigureAwait(false);

                    if (!stage.ValidateOutput(result))
                    {
                        throw new InvalidOperationException(
                            $"Output validation failed for stage {stage.Name}");
                    }

                    record.Status = PipelineStageStatus.Completed;
                    record.EndTime = DateTime.UtcNow;
                    currentInput = result;

                    OnStageProgress(new PipelineStageEventArgs
                    {
                        JobId = job.JobId,
                        StageName = stage.Name,
                        Status = PipelineStageStatus.Completed,
                        StageIndex = job.CurrentStageIndex,
                        TotalStages = job.Stages.Count,
                        ProgressPercent = (float)(job.CurrentStageIndex + 1) / job.Stages.Count * 100
                    });

                    _metrics.IncrementStagesExecuted();
                    return true;
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    record.Status = PipelineStageStatus.Cancelled;
                    record.EndTime = DateTime.UtcNow;
                    return false;
                }
                catch (OperationCanceledException)
                {
                    record.Status = PipelineStageStatus.Failed;
                    record.TimedOut = true;
                    record.ErrorMessage = $"Stage timed out after {stage.TimeoutMs}ms";
                    record.EndTime = DateTime.UtcNow;
                    _metrics.IncrementTimeouts();
                }
                catch (Exception ex)
                {
                    record.Status = PipelineStageStatus.Failed;
                    record.ErrorMessage = ex.Message;
                    record.Exception = ex;
                    record.EndTime = DateTime.UtcNow;
                }
            }

            OnStageProgress(new PipelineStageEventArgs
            {
                JobId = job.JobId,
                StageName = stage.Name,
                Status = PipelineStageStatus.Failed,
                StageIndex = job.CurrentStageIndex,
                TotalStages = job.Stages.Count,
                ErrorMessage = record.ErrorMessage,
                Exception = record.Exception
            });

            return false;
        }

        /// <summary>
        /// Gets the status of a specific job.
        /// </summary>
        /// <param name="jobId">Job identifier.</param>
        /// <returns>Pipeline job if found, null otherwise.</returns>
        public PipelineJob? GetJob(string jobId)
        {
            return _jobs.TryGetValue(jobId, out var job) ? job : null;
        }

        /// <summary>
        /// Gets all active (running or waiting) jobs.
        /// </summary>
        /// <returns>Collection of active jobs.</returns>
        public IReadOnlyCollection<PipelineJob> GetActiveJobs()
        {
            return _jobs.Values
                .Where(j => j.Status is PipelineJobStatus.Running or PipelineJobStatus.Waiting)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Cancels a specific job.
        /// </summary>
        /// <param name="jobId">Job identifier.</param>
        /// <returns>True if the job was found and cancelled.</returns>
        public bool CancelJob(string jobId)
        {
            if (_jobs.TryGetValue(jobId, out var job))
            {
                job.CancellationTokenSource?.Cancel();
                job.Status = PipelineJobStatus.Cancelled;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Cancels all active jobs.
        /// </summary>
        public void CancelAllJobs()
        {
            foreach (var job in _jobs.Values)
            {
                if (job.Status is PipelineJobStatus.Running or PipelineJobStatus.Waiting)
                {
                    job.CancellationTokenSource?.Cancel();
                    job.Status = PipelineJobStatus.Cancelled;
                }
            }
        }

        /// <summary>
        /// Cleans up completed jobs older than the specified age.
        /// </summary>
        /// <param name="maxAge">Maximum age of completed jobs to keep.</param>
        /// <returns>Number of jobs removed.</returns>
        public int CleanupCompletedJobs(TimeSpan maxAge)
        {
            var cutoff = DateTime.UtcNow - maxAge;
            var toRemove = _jobs.Values
                .Where(j => j.Status is PipelineJobStatus.Completed
                    or PipelineJobStatus.Failed
                    or PipelineJobStatus.Cancelled
                    && j.EndTime.HasValue && j.EndTime.Value < cutoff)
                .Select(j => j.JobId)
                .ToList();

            int removed = 0;
            foreach (var jobId in toRemove)
            {
                if (_jobs.TryRemove(jobId, out var job))
                {
                    job.CancellationTokenSource?.Dispose();
                    removed++;
                }
            }

            return removed;
        }

        /// <summary>
        /// Monitoring loop that periodically reports metrics.
        /// </summary>
        private async Task MonitoringLoopAsync(CancellationToken cancellationToken)
        {
            var interval = TimeSpan.FromSeconds(_config.MetricsReportingIntervalSeconds);

            while (!cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(interval, cancellationToken).ConfigureAwait(false);

                _metrics.QueuedJobs = _jobs.Count(kvp =>
                    kvp.Value.Status == PipelineJobStatus.Waiting);
                _metrics.ActiveJobs = _jobs.Count(kvp =>
                    kvp.Value.Status == PipelineJobStatus.Running);

                var completedJobs = _jobs.Values
                    .Where(j => j.Status == PipelineJobStatus.Completed && j.TotalDurationMs > 0)
                    .ToList();

                if (completedJobs.Count > 0)
                {
                    _metrics.AverageJobDurationMs = completedJobs.Average(j => j.TotalDurationMs);
                }

                var completedStages = _jobs.Values
                    .SelectMany(j => j.Stages)
                    .Where(s => s.Status == PipelineStageStatus.Completed && s.DurationMs > 0)
                    .ToList();

                if (completedStages.Count > 0)
                {
                    _metrics.AverageStageDurationMs = completedStages.Average(s => s.DurationMs);
                }

                MetricsUpdated?.Invoke(this, _metrics);
            }
        }

        /// <summary>
        /// Raises the StageProgress event.
        /// </summary>
        private void OnStageProgress(PipelineStageEventArgs e)
        {
            StageProgress?.Invoke(this, e);
        }

        /// <summary>
        /// Disposes the pipeline and cancels all operations.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            try
            {
                _shutdownCts.Cancel();
            }
            catch (ObjectDisposedException)
            {
            }

            if (_config.EnableMonitoring)
                WaitQuietly(_monitoringTask, TimeSpan.FromSeconds(3));

            foreach (var job in _jobs.Values)
            {
                try
                {
                    job.CancellationTokenSource?.Cancel();
                    job.CancellationTokenSource?.Dispose();
                }
                catch (ObjectDisposedException)
                {
                }
            }

            _jobSemaphore.Dispose();
            _shutdownCts.Dispose();
        }

        private static void WaitQuietly(Task? task, TimeSpan timeout)
        {
            if (task == null)
                return;
            try
            {
                task.Wait(timeout);
            }
            catch (AggregateException ex) when (ex.InnerExceptions.All(e =>
                e is OperationCanceledException or TaskCanceledException))
            {
            }
            catch (OperationCanceledException)
            {
            }
        }
    }

    /// <summary>
    /// Predefined pipeline stages for the neural asset streaming workflow.
    /// </summary>
    public static class StreamingPipelineStages
    {
        /// <summary>
        /// Creates the download stage.
        /// </summary>
        public static DelegateStage<string, byte[]> CreateDownloadStage()
        {
            const int maxBytes = 64 * 1024 * 1024;
            return new DelegateStage<string, byte[]>(async (url, ct) =>
            {
                var uri = Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(url, allowLoopbackHttp: false);
                using var client = Synapse.Core.Security.UrlSecurity.CreateSafeHttpClient(TimeSpan.FromSeconds(30));
                using var response = await client.GetAsync(uri, System.Net.Http.HttpCompletionOption.ResponseHeadersRead, ct)
                    .ConfigureAwait(false);
                response.EnsureSuccessStatusCode();
                if (response.Content.Headers.ContentLength is long declared && declared > maxBytes)
                    throw new InvalidDataException($"Download exceeds {maxBytes} byte limit.");

                await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                using var ms = new System.IO.MemoryStream();
                var buffer = new byte[81920];
                int read;
                long total = 0;
                while ((read = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), ct).ConfigureAwait(false)) > 0)
                {
                    total += read;
                    if (total > maxBytes)
                        throw new InvalidDataException($"Download exceeds {maxBytes} byte limit.");
                    ms.Write(buffer, 0, read);
                }
                return ms.ToArray();
            })
            {
                Name = "Download",
                Order = 0,
                IsRequired = true,
                TimeoutMs = 60000,
                MaxRetries = 3
            };
        }

        /// <summary>
        /// Creates the decompression stage.
        /// </summary>
        public static DelegateStage<byte[], byte[]> CreateDecompressStage(CompressionUtils compression)
        {
            return new DelegateStage<byte[], byte[]>((data, ct) =>
            {
                var result = compression.Decompress(data);
                return Task.FromResult(result);
            })
            {
                Name = "Decompress",
                Order = 1,
                IsRequired = true,
                TimeoutMs = 30000,
                MaxRetries = 2
            };
        }

        /// <summary>
        /// Creates the decode stage for converting raw bytes to NeuralAsset.
        /// </summary>
        public static DelegateStage<byte[], Core.NeuralNetwork.NeuralAsset> CreateDecodeStage()
        {
            return new DelegateStage<byte[], Core.NeuralNetwork.NeuralAsset>(async (data, ct) =>
            {
                using var ms = new System.IO.MemoryStream(data);
                return await Core.NeuralNetwork.NeuralAsset.DeserializeAsync(ms, ct).ConfigureAwait(false);
            })
            {
                Name = "Decode",
                Order = 2,
                IsRequired = true,
                TimeoutMs = 15000,
                MaxRetries = 1
            };
        }

        /// <summary>
        /// Creates the GPU upload stage.
        /// Validates and CPU-stages neural weights for device upload.
        /// Real VkBuffer upload remains device-bound in the render backend.
        /// </summary>
        public static DelegateStage<Core.NeuralNetwork.NeuralAsset, Core.NeuralNetwork.NeuralAsset>
            CreateGpuUploadStage()
        {
            return new DelegateStage<Core.NeuralNetwork.NeuralAsset,
                Core.NeuralNetwork.NeuralAsset>((asset, ct) =>
            {
                ct.ThrowIfCancellationRequested();
                ArgumentNullException.ThrowIfNull(asset);

                if (asset.CompressedWeights.Length == 0 &&
                    asset.UncompressedWeights is { Length: > 0 })
                {
                    asset.Compress();
                }

                if (asset.CompressedWeights.Length == 0 &&
                    (asset.UncompressedWeights == null || asset.UncompressedWeights.Length == 0) &&
                    asset.LODTiers.Count == 0)
                {
                    throw new InvalidDataException(
                        "GpuUpload: neural asset has no weight payload to stage.");
                }

                // Integrity: hash payload; do not assume Brotli — CompressedWeights may be
                // Brotli floats or MicroMLP FP8 depending on producer.
                if (asset.CompressedWeights.Length > 0)
                {
                    if (string.IsNullOrWhiteSpace(asset.Metadata.ContentHash))
                    {
                        asset.Metadata.ContentHash = Convert.ToHexString(
                            System.Security.Cryptography.SHA256.HashData(asset.CompressedWeights));
                    }
                    else
                    {
                        var actual = Convert.ToHexString(
                            System.Security.Cryptography.SHA256.HashData(asset.CompressedWeights));
                        if (!string.Equals(actual, asset.Metadata.ContentHash, StringComparison.OrdinalIgnoreCase))
                        {
                            throw new InvalidDataException(
                                "GpuUpload: CompressedWeights content hash mismatch.");
                        }
                    }
                }

                asset.IsGpuUploadPrepared = true;
                return Task.FromResult(asset);
            })
            {
                Name = "GpuUpload",
                Order = 3,
                IsRequired = true,
                TimeoutMs = 10000,
                MaxRetries = 2
            };
        }

        /// <summary>
        /// Creates the full streaming pipeline with all standard stages.
        /// </summary>
        public static IReadOnlyList<IObjectPipelineStage> CreateFullPipeline(
            CompressionUtils compression)
        {
            var download = CreateDownloadStage();
            var decompress = CreateDecompressStage(compression);
            var decode = CreateDecodeStage();
            var upload = CreateGpuUploadStage();

            return new List<IObjectPipelineStage>
            {
                (IObjectPipelineStage)download,
                (IObjectPipelineStage)decompress,
                (IObjectPipelineStage)decode,
                (IObjectPipelineStage)upload
            };
        }
    }
}
