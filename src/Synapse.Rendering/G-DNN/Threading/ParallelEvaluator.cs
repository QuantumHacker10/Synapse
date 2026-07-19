using System;
// ============================================================
// FILE: ParallelEvaluator.cs
// PATH: Threading/ParallelEvaluator.cs
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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;

namespace GDNN.Threading
{
    /// <summary>
    /// Delegate for parallel iteration with unsafe state pointer.
    /// </summary>
    public unsafe delegate void ForEachAction(int iteration, int threadId, void* state);

    /// <summary>
    /// Work partitioning strategies for distributing iterations across threads.
    /// </summary>
    public enum PartitionStrategy
    {
        /// <summary>Each thread processes a contiguous block of iterations.</summary>
        Block = 0,

        /// <summary>Iterations are distributed in round-robin fashion across threads.</summary>
        Cyclic = 1,

        /// <summary>Threads dynamically steal work from a shared range based on current load.</summary>
        Guided = 2,

        /// <summary>Each thread processes interleaved iterations with stride equal to thread count.</summary>
        Striped = 3
    }

    /// <summary>
    /// Aggregated performance statistics for a parallel evaluation run.
    /// </summary>
    public readonly struct EvaluationStatistics
    {
        /// <summary>Total number of iterations completed.</summary>
        public long TotalIterations { get; init; }

        /// <summary>Elapsed wall-clock time for the entire evaluation.</summary>
        public TimeSpan ElapsedTime { get; init; }

        /// <summary>Iterations per second throughput.</summary>
        public double IterationsPerSecond { get; init; }

        /// <summary>Number of threads used during evaluation.</summary>
        public int ThreadCount { get; init; }

        /// <summary>Peak memory allocated during evaluation, in bytes.</summary>
        public long PeakMemoryBytes { get; init; }

        /// <summary>Number of work-stealing attempts that resulted in a steal.</summary>
        public long SuccessfulSteals { get; init; }

        /// <summary>Number of failed work-stealing attempts.</summary>
        public long FailedSteals { get; init; }

        /// <summary>Load balance ratio (min/avg iterations per thread). 1.0 is perfectly balanced.</summary>
        public double BalanceRatio { get; init; }

        /// <summary>Overhead fraction (1 - useful_time / total_time).</summary>
        public double OverheadFraction { get; init; }

        public override string ToString() =>
            $"[Eval: {TotalIterations:N0} iters in {ElapsedTime.TotalMilliseconds:F2}ms, " +
            $"{IterationsPerSecond:N0} it/s, {ThreadCount} threads, " +
            $"balance={BalanceRatio:P1}, overhead={OverheadFraction:P1}]";
    }

    /// <summary>
    /// Delegate for progress reporting during parallel evaluation.
    /// </summary>
    /// <param name="completed">Number of iterations completed so far.</param>
    /// <param name="total">Total number of iterations.</param>
    /// <param name="elapsed">Elapsed time since evaluation started.</param>
    public delegate void ProgressHandler(long completed, long total, TimeSpan elapsed);

    /// <summary>
    /// Delegate for evaluating a single iteration of a neural network.
    /// </summary>
    /// <param name="iteration">The iteration index.</param>
    /// <param name="threadId">The thread performing the evaluation.</param>
    /// <param name="state">Optional user-defined state.</param>
    /// <returns>A numeric result for this iteration.</returns>
    public unsafe delegate float EvaluateFunc(int iteration, int threadId, void* state);

    /// <summary>
    /// Delegate for reducing two partial results into one.
    /// </summary>
    public delegate float ReduceFunc(float a, float b);

    /// <summary>
    /// Provides parallel evaluation of neural networks across multiple threads with
    /// configurable work partitioning, load balancing, result aggregation, cancellation,
    /// progress reporting, and performance statistics.
    /// </summary>
    public sealed class ParallelEvaluator : IDisposable
    {
        private const int PROGRESS_REPORT_INTERVAL_MS = 50;
        private const int GUIDED_MIN_CHUNK = 64;
        private const double GUIDED_DECAY = 0.5;
        private const int SPIN_WAIT_CYCLES = 128;

        private int _threadCount;
        private volatile bool _disposed;
        private CancellationTokenSource? _activeCts;
        private long _completedIterations;
        private long _activeThreadCount;
        private double[]? _threadTimings;
        private long[]? _threadIterationCounts;

        /// <summary>
        /// Gets or sets the number of worker threads. Defaults to <see cref="Environment.ProcessorCount"/>.
        /// </summary>
        public int ThreadCount
        {
            get => _threadCount;
            set
            {
                if (value < 1)
                    throw new ArgumentOutOfRangeException(nameof(value));
                _threadCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the default partition strategy used for parallel loops.
        /// </summary>
        public PartitionStrategy Strategy { get; set; } = PartitionStrategy.Guided;

        /// <summary>
        /// Gets or sets the minimum chunk size for guided partitioning.
        /// </summary>
        public int GuidedMinChunkSize { get; set; } = GUIDED_MIN_CHUNK;

        /// <summary>
        /// Gets or sets the progress reporting interval. Defaults to 50ms.
        /// </summary>
        public TimeSpan ProgressInterval { get; set; } = TimeSpan.FromMilliseconds(PROGRESS_REPORT_INTERVAL_MS);

        /// <summary>
        /// Gets the last computed evaluation statistics.
        /// </summary>
        public EvaluationStatistics LastStatistics { get; private set; }

        /// <summary>
        /// Event raised periodically to report evaluation progress.
        /// </summary>
        public event ProgressHandler? Progress;

        /// <summary>
        /// Gets whether an evaluation is currently running.
        /// </summary>
        public bool IsRunning => Interlocked.CompareExchange(ref _activeThreadCount, 0, 0) > 0;

        /// <summary>
        /// Gets the current number of completed iterations across all threads.
        /// </summary>
        public long CompletedIterations => Interlocked.Read(ref _completedIterations);

        /// <summary>
        /// Initializes a new instance of the <see cref="ParallelEvaluator"/> class
        /// using the optimal number of threads for the current system.
        /// </summary>
        public ParallelEvaluator()
        {
            _threadCount = Environment.ProcessorCount;
        }

        /// <summary>
        /// Initializes a new instance with the specified thread count.
        /// </summary>
        /// <param name="threadCount">Number of worker threads.</param>
        public ParallelEvaluator(int threadCount)
        {
            if (threadCount < 1)
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            _threadCount = threadCount;
        }

        /// <summary>
        /// Evaluates the given function across the range [0, totalIterations) in parallel
        /// using the configured partition strategy and returns an aggregated result.
        /// </summary>
        /// <param name="evaluate">The evaluation function to invoke for each iteration.</param>
        /// <param name="totalIterations">Total number of iterations.</param>
        /// <param name="reduce">Reduction function to combine partial results.</param>
        /// <param name="seed">Initial value for reduction.</param>
        /// <param name="state">Optional user-defined state passed to the evaluate function.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>The aggregated result.</returns>
        public unsafe float Evaluate(
            EvaluateFunc evaluate,
            int totalIterations,
            ReduceFunc reduce,
            float seed = 0f,
            void* state = null,
            CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (evaluate == null)
                throw new ArgumentNullException(nameof(evaluate));
            if (reduce == null)
                throw new ArgumentNullException(nameof(reduce));
            if (totalIterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(totalIterations));

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            _activeCts = cts;

            var sw = Stopwatch.StartNew();
            long memoryBefore = GC.GetTotalMemory(false);
            var perThreadResults = new float[_threadCount];
            var perThreadCounts = new long[_threadCount];
            _threadTimings = new double[_threadCount];
            _threadIterationCounts = new long[_threadCount];
            Interlocked.Exchange(ref _completedIterations, 0);
            Interlocked.Exchange(ref _activeThreadCount, 0);

            bool progressRaised = Progress != null;
            var progressCts = progressRaised
                ? CancellationTokenSource.CreateLinkedTokenSource(cts.Token)
                : null;

            Thread? progressThread = null;
            if (progressRaised && progressCts != null)
            {
                progressThread = new Thread(() => ProgressReporterLoop(totalIterations, progressCts.Token))
                {
                    IsBackground = true,
                    Name = "GDNN-ProgressReporter"
                };
                progressThread.Start();
            }

            try
            {
                var threads = new Thread[_threadCount];
                for (int t = 0; t < _threadCount; t++)
                {
                    int localT = t;
                    threads[t] = new Thread(() =>
                    {
                        Interlocked.Increment(ref _activeThreadCount);
                        try
                        {
                            var threadSw = Stopwatch.StartNew();
                            ThreadLocalEvaluate(
                                evaluate, totalIterations, localT, state,
                                ref perThreadResults[localT], ref perThreadCounts[localT],
                                cts.Token);
                            threadSw.Stop();
                            _threadTimings![localT] = threadSw.Elapsed.TotalSeconds;
                            _threadIterationCounts![localT] = perThreadCounts[localT];
                        }
                        finally
                        {
                            Interlocked.Decrement(ref _activeThreadCount);
                        }
                    })
                    {
                        IsBackground = true,
                        Name = $"GDNN-Worker-{localT}"
                    };
                }

                for (int t = 0; t < _threadCount; t++)
                    threads[t].Start();

                for (int t = 0; t < _threadCount; t++)
                    threads[t].Join();
            }
            finally
            {
                progressCts?.Cancel();
                progressThread?.Join(TimeSpan.FromSeconds(1));
                progressCts?.Dispose();
                _activeCts = null;
            }

            float result = seed;
            for (int t = 0; t < _threadCount; t++)
                result = reduce(result, perThreadResults[t]);

            sw.Stop();
            long memoryAfter = GC.GetTotalMemory(false);

            long totalCompleted = 0;
            for (int t = 0; t < _threadCount; t++)
                totalCompleted += perThreadCounts[t];

            double totalTime = _threadTimings.Sum();
            double maxTime = _threadTimings.Max();
            double balanceRatio = maxTime > 0 ? totalTime / (maxTime * _threadCount) : 1.0;

            LastStatistics = new EvaluationStatistics
            {
                TotalIterations = totalCompleted,
                ElapsedTime = sw.Elapsed,
                IterationsPerSecond = sw.Elapsed.TotalSeconds > 0
                    ? totalCompleted / sw.Elapsed.TotalSeconds
                    : 0,
                ThreadCount = _threadCount,
                PeakMemoryBytes = memoryAfter - memoryBefore,
                BalanceRatio = balanceRatio,
                OverheadFraction = totalTime > 0 ? 1.0 - (sw.Elapsed.TotalSeconds * _threadCount / totalTime) : 0
            };

            return result;
        }

        /// <summary>
        /// Executes an action in parallel across the range [0, totalIterations) with no result aggregation.
        /// </summary>
        /// <param name="action">The action to invoke for each iteration.</param>
        /// <param name="totalIterations">Total number of iterations.</param>
        /// <param name="state">Optional user-defined state.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        public unsafe void ForEach(
            ForEachAction action,
            int totalIterations,
            void* state = null,
            CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            if (totalIterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(totalIterations));

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            _activeCts = cts;
            Interlocked.Exchange(ref _completedIterations, 0);
            Interlocked.Exchange(ref _activeThreadCount, 0);

            var threads = new Thread[_threadCount];
            for (int t = 0; t < _threadCount; t++)
            {
                int localT = t;
                threads[t] = new Thread(() =>
                {
                    Interlocked.Increment(ref _activeThreadCount);
                    try
                    {
                        ThreadLocalForEach(action, totalIterations, localT, state, cts.Token);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _activeThreadCount);
                    }
                })
                {
                    IsBackground = true,
                    Name = $"GDNN-ForEach-{localT}"
                };
            }

            for (int t = 0; t < _threadCount; t++)
                threads[t].Start();
            for (int t = 0; t < _threadCount; t++)
                threads[t].Join();

            _activeCts = null;
        }

        /// <summary>
        /// Executes a parallel for-each using a partition strategy and returns per-thread statistics.
        /// </summary>
        /// <param name="startInclusive">Start of the range (inclusive).</param>
        /// <param name="endExclusive">End of the range (exclusive).</param>
        /// <param name="action">Action to execute per iteration.</param>
        /// <param name="strategy">Partition strategy override.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        public unsafe void PartitionedForEach(
            int startInclusive,
            int endExclusive,
            Action<int, int> action,
            PartitionStrategy? strategy = null,
            CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            if (endExclusive <= startInclusive)
                throw new ArgumentOutOfRangeException(nameof(endExclusive));

            var effectiveStrategy = strategy ?? Strategy;
            int total = endExclusive - startInclusive;

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            Interlocked.Exchange(ref _completedIterations, 0);
            Interlocked.Exchange(ref _activeThreadCount, 0);

            var threads = new Thread[_threadCount];
            for (int t = 0; t < _threadCount; t++)
            {
                int localT = t;
                threads[t] = new Thread(() =>
                {
                    Interlocked.Increment(ref _activeThreadCount);
                    try
                    {
                        switch (effectiveStrategy)
                        {
                            case PartitionStrategy.Block:
                                ExecuteBlockPartition(startInclusive, total, localT, action, cts.Token);
                                break;
                            case PartitionStrategy.Cyclic:
                                ExecuteCyclicPartition(startInclusive, total, localT, action, cts.Token);
                                break;
                            case PartitionStrategy.Guided:
                                ExecuteGuidedPartition(startInclusive, total, localT, action, cts.Token);
                                break;
                            case PartitionStrategy.Striped:
                                ExecuteStripedPartition(startInclusive, total, localT, action, cts.Token);
                                break;
                        }
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _activeThreadCount);
                    }
                })
                {
                    IsBackground = true,
                    Name = $"GDNN-Partitioned-{effectiveStrategy}-{localT}"
                };
            }

            for (int t = 0; t < _threadCount; t++)
                threads[t].Start();
            for (int t = 0; t < _threadCount; t++)
                threads[t].Join();

            _activeCts = null;
        }

        /// <summary>
        /// Attempts to cancel the currently running evaluation.
        /// </summary>
        public void Cancel()
        {
            _activeCts?.Cancel();
        }

        /// <summary>
        /// Computes optimal chunk sizes for the given total iterations and thread count.
        /// </summary>
        /// <param name="totalIterations">Total iterations.</param>
        /// <param name="strategy">Partition strategy.</param>
        /// <returns>An array of chunk sizes, one per thread.</returns>
        public int[] ComputePartitionSizes(int totalIterations, PartitionStrategy strategy)
        {
            int[] sizes = new int[_threadCount];

            switch (strategy)
            {
                case PartitionStrategy.Block:
                    int blockSize = totalIterations / _threadCount;
                    int remainder = totalIterations % _threadCount;
                    for (int t = 0; t < _threadCount; t++)
                        sizes[t] = blockSize + (t < remainder ? 1 : 0);
                    break;

                case PartitionStrategy.Cyclic:
                    for (int t = 0; t < _threadCount; t++)
                    {
                        int count = 0;
                        for (int i = t; i < totalIterations; i += _threadCount)
                            count++;
                        sizes[t] = count;
                    }
                    break;

                case PartitionStrategy.Guided:
                    int remaining = totalIterations;
                    for (int t = 0; t < _threadCount - 1 && remaining > 0; t++)
                    {
                        int chunk = Math.Max(GuidedMinChunkSize, remaining / (_threadCount - t) / 2);
                        chunk = Math.Min(chunk, remaining);
                        sizes[t] = chunk;
                        remaining -= chunk;
                    }
                    sizes[_threadCount - 1] = remaining;
                    break;

                case PartitionStrategy.Striped:
                    for (int t = 0; t < _threadCount; t++)
                        sizes[t] = (totalIterations - t + _threadCount - 1) / _threadCount;
                    break;
            }

            return sizes;
        }

        /// <summary>
        /// Estimates memory requirements for a parallel evaluation with the given parameters.
        /// </summary>
        /// <param name="totalIterations">Number of iterations.</param>
        /// <param name="bytesPerIteration">Estimated working set per iteration.</param>
        /// <returns>Estimated total memory in bytes.</returns>
        public long EstimateMemoryRequirement(int totalIterations, int bytesPerIteration)
        {
            long perThreadBuffer = (long)bytesPerIteration * GuidedMinChunkSize;
            long totalWorkingSet = perThreadBuffer * _threadCount;
            long metadataOverhead = _threadCount * 256;
            return totalWorkingSet + metadataOverhead;
        }

        private unsafe void ThreadLocalEvaluate(
            EvaluateFunc evaluate,
            int totalIterations,
            int threadId,
            void* state,
            ref float result,
            ref long iterCount,
            CancellationToken token)
        {
            var partition = GetPartitionRange(totalIterations, threadId);
            float localResult = 0f;
            long localCount = 0;

            for (int i = partition.Start; i < partition.End; i++)
            {
                token.ThrowIfCancellationRequested();
                localResult = evaluate(i, threadId, state);
                localCount++;
                Interlocked.Increment(ref _completedIterations);
            }

            result = localResult;
            iterCount = localCount;
        }

        private unsafe void ThreadLocalForEach(
            ForEachAction action,
            int totalIterations,
            int threadId,
            void* state,
            CancellationToken token)
        {
            var partition = GetPartitionRange(totalIterations, threadId);

            for (int i = partition.Start; i < partition.End; i++)
            {
                token.ThrowIfCancellationRequested();
                action(i, threadId, state);
                Interlocked.Increment(ref _completedIterations);
            }
        }

        private (int Start, int End) GetPartitionRange(int totalIterations, int threadId)
        {
            return Strategy switch
            {
                PartitionStrategy.Block => GetBlockRange(totalIterations, threadId),
                PartitionStrategy.Cyclic => (threadId, totalIterations),
                PartitionStrategy.Guided => (0, 0),
                PartitionStrategy.Striped => (threadId, totalIterations),
                _ => GetBlockRange(totalIterations, threadId)
            };
        }

        private (int Start, int End) GetBlockRange(int totalIterations, int threadId)
        {
            int blockSize = totalIterations / _threadCount;
            int remainder = totalIterations % _threadCount;
            int start = threadId * blockSize + Math.Min(threadId, remainder);
            int end = start + blockSize + (threadId < remainder ? 1 : 0);
            return (start, end);
        }

        private unsafe void ExecuteBlockPartition(
            int startInclusive, int total, int threadId,
            Action<int, int> action, CancellationToken token)
        {
            int blockSize = total / _threadCount;
            int remainder = total % _threadCount;
            int start = startInclusive + threadId * blockSize + Math.Min(threadId, remainder);
            int count = blockSize + (threadId < remainder ? 1 : 0);

            for (int i = 0; i < count; i++)
            {
                token.ThrowIfCancellationRequested();
                action(start + i, threadId);
                Interlocked.Increment(ref _completedIterations);
            }
        }

        private unsafe void ExecuteCyclicPartition(
            int startInclusive, int total, int threadId,
            Action<int, int> action, CancellationToken token)
        {
            for (int i = threadId; i < total; i += _threadCount)
            {
                token.ThrowIfCancellationRequested();
                action(startInclusive + i, threadId);
                Interlocked.Increment(ref _completedIterations);
            }
        }

        private unsafe void ExecuteGuidedPartition(
            int startInclusive, int total, int threadId,
            Action<int, int> action, CancellationToken token)
        {
            int nextIndex = 0;
            int remaining = total;

            while (remaining > 0)
            {
                int chunk = Math.Max(GuidedMinChunkSize, remaining / (_threadCount * 2));
                int actualIndex = Interlocked.Add(ref nextIndex, chunk) - chunk;
                actualIndex = Math.Min(actualIndex, total);

                int end = Math.Min(actualIndex + chunk, total);

                for (int i = actualIndex; i < end && i < total; i++)
                {
                    token.ThrowIfCancellationRequested();
                    action(startInclusive + i, threadId);
                    Interlocked.Increment(ref _completedIterations);
                }

                remaining = total - Math.Min(actualIndex + chunk, total);
            }
        }

        private unsafe void ExecuteStripedPartition(
            int startInclusive, int total, int threadId,
            Action<int, int> action, CancellationToken token)
        {
            for (int i = threadId; i < total; i += _threadCount)
            {
                token.ThrowIfCancellationRequested();
                action(startInclusive + i, threadId);
                Interlocked.Increment(ref _completedIterations);
            }
        }

        private void ProgressReporterLoop(int totalIterations, CancellationToken token)
        {
            var sw = Stopwatch.StartNew();
            long lastReported = 0;

            while (!token.IsCancellationRequested)
            {
                Thread.Sleep(ProgressInterval);

                if (token.IsCancellationRequested)
                    break;

                long completed = Interlocked.Read(ref _completedIterations);
                if (completed > lastReported)
                {
                    Progress?.Invoke(completed, totalIterations, sw.Elapsed);
                    lastReported = completed;
                }
            }

            long finalCompleted = Interlocked.Read(ref _completedIterations);
            Progress?.Invoke(finalCompleted, totalIterations, sw.Elapsed);
        }

        /// <summary>
        /// Releases all resources used by the parallel evaluator.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Cancel();
            _activeCts?.Dispose();
            _activeCts = null;
        }
    }
}
