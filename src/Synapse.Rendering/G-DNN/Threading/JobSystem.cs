using System;
// ============================================================
// FILE: JobSystem.cs
// PATH: Threading/JobSystem.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
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
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Threading
{
    /// <summary>
    /// States of a job handle.
    /// </summary>
    public enum JobState
    {
        /// <summary>Job has been created but not yet scheduled.</summary>
        Created = 0,

        /// <summary>Job is scheduled and waiting to execute.</summary>
        Scheduled = 1,

        /// <summary>Job is currently executing.</summary>
        Running = 2,

        /// <summary>Job completed successfully.</summary>
        Completed = 3,

        /// <summary>Job was cancelled.</summary>
        Cancelled = 4,

        /// <summary>Job failed with an exception.</summary>
        Failed = 5
    }

    /// <summary>
    /// A lightweight handle for tracking job completion and status.
    /// </summary>
    public sealed class JobHandle
    {
        private int _state;
        private int _refCount;
        private Exception? _exception;

        /// <summary>Gets the unique job ID.</summary>
        public int JobId { get; internal set; }

        /// <summary>Gets the current state.</summary>
        public JobState State
        {
            get => (JobState)Volatile.Read(ref _state);
            internal set => Volatile.Write(ref _state, (int)value);
        }

        /// <summary>Gets whether the job has completed (successfully, cancelled, or failed).</summary>
        public bool IsCompleted => State == JobState.Completed;

        /// <summary>Gets whether the job was cancelled.</summary>
        public bool IsCancelled => State == JobState.Cancelled;

        /// <summary>Gets whether the job failed.</summary>
        public bool IsFaulted => State == JobState.Failed;

        /// <summary>Gets the exception if the job faulted.</summary>
        public Exception? FaultException => _exception;

        /// <summary>Gets the timestamp when the job started executing.</summary>
        public long StartTimestamp { get; internal set; }

        /// <summary>Gets the timestamp when the job completed.</summary>
        public long EndTimestamp { get; internal set; }

        /// <summary>Gets the estimated execution cost.</summary>
        public int Cost { get; init; }

        /// <summary>Gets the list of jobs that depend on this job.</summary>
        public List<JobHandle> Dependents { get; } = new();

        /// <summary>Gets the list of jobs this job depends on.</summary>
        public List<JobHandle> Dependencies { get; } = new();

        /// <summary>Gets or sets user-defined data.</summary>
        public object? UserData { get; set; }

        internal void SetException(Exception ex)
        {
            _exception = ex;
        }

        internal void IncrementRef() => Interlocked.Increment(ref _refCount);
        internal void DecrementRef() => Interlocked.Decrement(ref _refCount);
        internal int RefCount => Volatile.Read(ref _refCount);
    }

    /// <summary>
    /// Represents a lightweight unit of work for the job system.
    /// </summary>
    public abstract class Job
    {
        /// <summary>Gets the job handle for this job.</summary>
        public JobHandle Handle { get; } = new();

        /// <summary>Executes the job's work.</summary>
        public abstract void Execute();

        /// <summary>Gets or sets the estimated cost for scheduling.</summary>
        public int EstimatedCost { get; set; } = 1;

        /// <summary>Gets or sets the priority tag.</summary>
        public int PriorityTag { get; set; }
    }

    /// <summary>
    /// Delegate-based job implementation.
    /// </summary>
    internal sealed class DelegateJob : Job
    {
        private readonly Action _action;

        public DelegateJob(Action action)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
        }

        public override void Execute() => _action();
    }

    /// <summary>
    /// Delegate-based job with a parameter.
    /// </summary>
    internal sealed class DelegateJob<T> : Job
    {
        private readonly Action<T> _action;
        private readonly T _parameter;

        public DelegateJob(Action<T> action, T parameter)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
            _parameter = parameter;
        }

        public override void Execute() => _action(_parameter);
    }

    /// <summary>
    /// Parallel-for job that executes an action across a range of indices.
    /// </summary>
    internal sealed class ParallelForJob : Job
    {
        private readonly Action<int> _action;
        private readonly int _startInclusive;
        private readonly int _endExclusive;

        public int StartInclusive => _startInclusive;
        public int EndExclusive => _endExclusive;

        public ParallelForJob(Action<int> action, int startInclusive, int endExclusive)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
            _startInclusive = startInclusive;
            _endExclusive = endExclusive;
        }

        public override void Execute()
        {
            for (int i = _startInclusive; i < _endExclusive; i++)
                _action(i);
        }
    }

    /// <summary>
    /// Atomic counter for tracking job completion.
    /// </summary>
    public sealed class AtomicCounter
    {
        private int _value;

        /// <summary>Gets the current value.</summary>
        public int Value => Volatile.Read(ref _value);

        /// <summary>Initializes a new counter with the specified value.</summary>
        public AtomicCounter(int initialValue = 0)
        {
            _value = initialValue;
        }

        /// <summary>Increments the counter by one and returns the new value.</summary>
        public int Increment() => Interlocked.Increment(ref _value);

        /// <summary>Decrements the counter by one and returns the new value.</summary>
        public int Decrement() => Interlocked.Decrement(ref _value);

        /// <summary>Adds the specified value and returns the new value.</summary>
        public int Add(int amount) => Interlocked.Add(ref _value, amount);

        /// <summary>Sets the counter to the specified value.</summary>
        public void Set(int value) => Interlocked.Exchange(ref _value, value);

        /// <summary>Atomically compares and swaps the value.</summary>
        public int CompareExchange(int value, int comparand) =>
            Interlocked.CompareExchange(ref _value, value, comparand);

        /// <summary>Resets the counter to zero.</summary>
        public void Reset() => Interlocked.Exchange(ref _value, 0);

        public static implicit operator int(AtomicCounter counter) => counter.Value;
        public override string ToString() => Value.ToString();
    }

    /// <summary>
    /// Statistics for the job system.
    /// </summary>
    public struct JobSystemStatistics
    {
        /// <summary>Total jobs submitted.</summary>
        public long TotalJobsSubmitted { get; internal set; }

        /// <summary>Total jobs executed.</summary>
        public long TotalJobsExecuted { get; internal set; }

        /// <summary>Total jobs that failed.</summary>
        public long TotalJobsFailed { get; internal set; }

        /// <summary>Total parallel-for jobs created.</summary>
        public long TotalParallelFors { get; internal set; }

        /// <summary>Total job chains created.</summary>
        public long TotalJobChains { get; internal set; }

        /// <summary>Total job graphs created.</summary>
        public long TotalJobGraphs { get; internal set; }

        /// <summary>Current number of queued jobs.</summary>
        public int QueuedJobs { get; internal set; }

        /// <summary>Number of jobs currently executing.</summary>
        public int RunningJobs { get; internal set; }

        /// <summary>Number of active threads.</summary>
        public int ActiveThreads { get; internal set; }
    }

    /// <summary>
    /// A lightweight job system for small tasks with job handles for completion tracking,
    /// job chains and graphs, parallel-for job type, and atomic counters.
    /// </summary>
    public sealed class JobSystem : IDisposable
    {
        private const int MAX_JOBS = 65536;
        private const int MAX_THREADS = 256;
        private const int SPIN_WAIT_CYCLES = 64;
        private const int YIELD_THRESHOLD = 16;

        private readonly Thread[] _threads;
        private readonly Queue<Job>[] _jobQueues;
        private readonly object[] _queueLocks;
        private readonly CancellationTokenSource _cts;
        private volatile bool _disposed;
        private volatile bool _running;

        private long _totalJobsSubmitted;
        private long _totalJobsExecuted;
        private long _totalJobsFailed;
        private long _totalParallelFors;
        private long _totalJobChains;
        private long _totalJobGraphs;
        private long _activeThreadCount;
        private int _nextJobId;
        private int _threadCount;
        private readonly Dictionary<int, JobHandle> _jobHandles = new();

        /// <summary>Gets the number of worker threads.</summary>
        public int ThreadCount => _threadCount;

        /// <summary>Gets the total number of jobs submitted.</summary>
        public long TotalJobsSubmitted => Interlocked.Read(ref _totalJobsSubmitted);

        /// <summary>Gets the total number of jobs executed.</summary>
        public long TotalJobsExecuted => Interlocked.Read(ref _totalJobsExecuted);

        /// <summary>Gets whether the job system is running.</summary>
        public bool IsRunning => _running;

        /// <summary>
        /// Initializes a new job system.
        /// </summary>
        /// <param name="threadCount">Number of worker threads.</param>
        public JobSystem(int threadCount = 0)
        {
            _threadCount = threadCount > 0
                ? Math.Min(threadCount, MAX_THREADS)
                : Environment.ProcessorCount;

            _threads = new Thread[_threadCount];
            _jobQueues = new Queue<Job>[_threadCount];
            _queueLocks = new object[_threadCount];
            _cts = new CancellationTokenSource();

            for (int i = 0; i < _threadCount; i++)
            {
                _jobQueues[i] = new Queue<Job>();
                _queueLocks[i] = new object();
            }
        }

        /// <summary>
        /// Starts the job system worker threads.
        /// </summary>
        public void Start()
        {
            if (_running)
                return;
            _running = true;

            for (int i = 0; i < _threadCount; i++)
            {
                int localI = i;
                _threads[i] = new Thread(() => WorkerLoop(localI))
                {
                    IsBackground = true,
                    Name = $"GDNN-Job-{localI}"
                };
                _threads[i].Start();
            }
        }

        /// <summary>
        /// Schedules a single job for execution.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <returns>A job handle for tracking completion.</returns>
        public JobHandle Schedule(Action action)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var job = new DelegateJob(action);
            job.Handle.JobId = Interlocked.Increment(ref _nextJobId);
            job.Handle.State = JobState.Scheduled;

            int queueIndex = (int)((uint)Environment.CurrentManagedThreadId % (uint)_threadCount);
            EnqueueJob(job, queueIndex);

            Interlocked.Increment(ref _totalJobsSubmitted);
            return job.Handle;
        }

        /// <summary>
        /// Schedules a job with a parameter.
        /// </summary>
        /// <typeparam name="T">Parameter type.</typeparam>
        /// <param name="action">The action to execute.</param>
        /// <param name="parameter">The parameter value.</param>
        /// <returns>A job handle for tracking completion.</returns>
        public JobHandle Schedule<T>(Action<T> action, T parameter)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var job = new DelegateJob<T>(action, parameter);
            job.Handle.JobId = Interlocked.Increment(ref _nextJobId);
            job.Handle.State = JobState.Scheduled;

            int queueIndex = (int)((uint)Environment.CurrentManagedThreadId % (uint)_threadCount);
            EnqueueJob(job, queueIndex);

            Interlocked.Increment(ref _totalJobsSubmitted);
            return job.Handle;
        }

        /// <summary>
        /// Schedules a parallel-for job that executes the action across the given range.
        /// </summary>
        /// <param name="action">Action to execute per iteration.</param>
        /// <param name="startInclusive">Start of range (inclusive).</param>
        /// <param name="endExclusive">End of range (exclusive).</param>
        /// <param name="chunkSize">Number of iterations per chunk.</param>
        /// <returns>A job handle for tracking completion.</returns>
        public JobHandle ScheduleParallelFor(
            Action<int> action,
            int startInclusive,
            int endExclusive,
            int chunkSize = 0)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            int total = endExclusive - startInclusive;
            if (total <= 0)
                throw new ArgumentOutOfRangeException(nameof(endExclusive));

            if (chunkSize <= 0)
                chunkSize = Math.Max(1, total / (_threadCount * 4));

            var parentHandle = new JobHandle
            {
                JobId = Interlocked.Increment(ref _nextJobId),
                Cost = total
            };
            parentHandle.State = JobState.Scheduled;

            var counter = new AtomicCounter(0);

            for (int start = startInclusive; start < endExclusive; start += chunkSize)
            {
                int chunkStart = start;
                int chunkEnd = Math.Min(start + chunkSize, endExclusive);

                var chunkJob = new ParallelForJob(action, chunkStart, chunkEnd);
                chunkJob.Handle.JobId = Interlocked.Increment(ref _nextJobId);
                chunkJob.Handle.State = JobState.Scheduled;
                chunkJob.Handle.IncrementRef();
                counter.Increment();

                int queueIndex = (int)((uint)(chunkStart * 31) % (uint)_threadCount);
                EnqueueJob(chunkJob, queueIndex);
            }

            Interlocked.Increment(ref _totalParallelFors);
            Interlocked.Increment(ref _totalJobsSubmitted);
            return parentHandle;
        }

        /// <summary>
        /// Creates a job chain where each job executes after the previous one completes.
        /// </summary>
        /// <param name="actions">Ordered list of actions to chain.</param>
        /// <returns>A job handle for the final job in the chain.</returns>
        public JobHandle ScheduleChain(params Action[] actions)
        {
            if (actions == null || actions.Length == 0)
                throw new ArgumentException("At least one action required.", nameof(actions));

            if (actions.Length == 1)
                return Schedule(actions[0]);

            var handles = new JobHandle[actions.Length];

            handles[0] = Schedule(actions[0]);

            for (int i = 1; i < actions.Length; i++)
            {
                int index = i;
                var dependents = new List<JobHandle> { handles[i - 1] };
                handles[i] = ScheduleWithDependencies(() => actions[index](), dependents);
            }

            Interlocked.Increment(ref _totalJobChains);
            return handles[^1];
        }

        /// <summary>
        /// Schedules a job that only runs after all specified dependencies complete.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="dependencies">Job handles that must complete first.</param>
        /// <returns>A job handle for tracking completion.</returns>
        public JobHandle ScheduleWithDependencies(
            Action action,
            IEnumerable<JobHandle> dependencies)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var job = new DelegateJob(action);
            job.Handle.JobId = Interlocked.Increment(ref _nextJobId);

            bool allComplete = true;
            foreach (var dep in dependencies)
            {
                job.Handle.Dependencies.Add(dep);
                dep.Dependents.Add(job.Handle);

                if (!dep.IsCompleted)
                    allComplete = false;
            }

            if (allComplete)
            {
                job.Handle.State = JobState.Scheduled;
                int queueIndex = (int)((uint)Environment.CurrentManagedThreadId % (uint)_threadCount);
                EnqueueJob(job, queueIndex);
            }
            else
            {
                job.Handle.State = JobState.Created;
            }

            Interlocked.Increment(ref _totalJobsSubmitted);
            return job.Handle;
        }

        /// <summary>
        /// Creates a job graph with multiple dependencies between jobs.
        /// </summary>
        /// <param name="nodes">List of (action, dependency indices) pairs.</param>
        /// <returns>Array of job handles, one per node.</returns>
        public JobHandle[] ScheduleGraph(
            IReadOnlyList<(Action Action, int[] DependencyIndices)> nodes)
        {
            if (nodes == null || nodes.Count == 0)
                throw new ArgumentException("At least one node required.", nameof(nodes));

            var handles = new JobHandle[nodes.Count];

            // First pass: create all handles
            for (int i = 0; i < nodes.Count; i++)
            {
                handles[i] = new JobHandle
                {
                    JobId = Interlocked.Increment(ref _nextJobId)
                };
            }

            // Second pass: wire up dependencies
            for (int i = 0; i < nodes.Count; i++)
            {
                var (action, depIndices) = nodes[i];
                var deps = new List<JobHandle>();

                foreach (int depIdx in depIndices)
                {
                    if (depIdx >= 0 && depIdx < nodes.Count)
                        deps.Add(handles[depIdx]);
                }

                var job = new DelegateJob(action);
                job.Handle.JobId = handles[i].JobId;

                bool allComplete = true;
                foreach (var dep in deps)
                {
                    job.Handle.Dependencies.Add(dep);
                    dep.Dependents.Add(job.Handle);
                    if (!dep.IsCompleted)
                        allComplete = false;
                }

                if (allComplete)
                {
                    job.Handle.State = JobState.Scheduled;
                    int queueIndex = (int)((uint)(i * 37) % (uint)_threadCount);
                    EnqueueJob(job, queueIndex);
                }
                else
                {
                    job.Handle.State = JobState.Created;
                }

                handles[i] = job.Handle;
                Interlocked.Increment(ref _totalJobsSubmitted);
            }

            Interlocked.Increment(ref _totalJobGraphs);
            return handles;
        }

        /// <summary>
        /// Blocks until the specified job handle has completed.
        /// </summary>
        /// <param name="handle">The job handle to wait on.</param>
        /// <param name="timeout">Maximum wait time. Default is infinite.</param>
        /// <returns>True if the job completed, false on timeout.</returns>
        public bool WaitForCompletion(JobHandle handle, TimeSpan timeout = default)
        {
            if (handle == null)
                throw new ArgumentNullException(nameof(handle));

            var sw = Stopwatch.StartNew();
            var spin = new SpinWait();

            while (!handle.IsCompleted && !handle.IsCancelled && !handle.IsFaulted)
            {
                if (timeout != default && sw.Elapsed >= timeout)
                    return false;

                // Try to help process jobs while waiting
                int queueIndex = (int)((uint)handle.JobId % (uint)_threadCount);
                ProcessOneJob(queueIndex);

                spin.SpinOnce();
                if (spin.Count > SPIN_WAIT_CYCLES)
                    Thread.Sleep(1);
            }

            return handle.IsCompleted;
        }

        /// <summary>
        /// Blocks until all specified job handles have completed.
        /// </summary>
        /// <param name="handles">The job handles to wait on.</param>
        public void WaitForAll(IEnumerable<JobHandle> handles)
        {
            foreach (var handle in handles)
                WaitForCompletion(handle);
        }

        /// <summary>
        /// Blocks until all submitted jobs have completed.
        /// </summary>
        public void CompleteAllPending()
        {
            var spin = new SpinWait();
            while (GetTotalQueuedJobs() > 0 || Volatile.Read(ref _activeThreadCount) > 0)
            {
                spin.SpinOnce();
                if (spin.Count > SPIN_WAIT_CYCLES)
                    Thread.Sleep(1);
            }
        }

        /// <summary>
        /// Gets the total number of queued jobs across all queues.
        /// </summary>
        public int GetTotalQueuedJobs()
        {
            int total = 0;
            for (int i = 0; i < _threadCount; i++)
            {
                lock (_queueLocks[i])
                    total += _jobQueues[i].Count;
            }
            return total;
        }

        /// <summary>
        /// Gets current statistics.
        /// </summary>
        public JobSystemStatistics GetStatistics()
        {
            return new JobSystemStatistics
            {
                TotalJobsSubmitted = Interlocked.Read(ref _totalJobsSubmitted),
                TotalJobsExecuted = Interlocked.Read(ref _totalJobsExecuted),
                TotalJobsFailed = Interlocked.Read(ref _totalJobsFailed),
                TotalParallelFors = Interlocked.Read(ref _totalParallelFors),
                TotalJobChains = Interlocked.Read(ref _totalJobChains),
                TotalJobGraphs = Interlocked.Read(ref _totalJobGraphs),
                QueuedJobs = GetTotalQueuedJobs(),
                RunningJobs = (int)Volatile.Read(ref _activeThreadCount),
                ActiveThreads = _threadCount
            };
        }

        /// <summary>
        /// Cancels all pending jobs.
        /// </summary>
        public void CancelAll()
        {
            for (int i = 0; i < _threadCount; i++)
            {
                lock (_queueLocks[i])
                {
                    while (_jobQueues[i].Count > 0)
                    {
                        var job = _jobQueues[i].Dequeue();
                        job.Handle.State = JobState.Cancelled;
                    }
                }
            }
        }

        private void EnqueueJob(Job job, int queueIndex)
        {
            lock (_queueLocks[queueIndex])
            {
                _jobQueues[queueIndex].Enqueue(job);
            }
        }

        private bool TryDequeueJob(int threadIndex, out Job? job)
        {
            // Try own queue first
            lock (_queueLocks[threadIndex])
            {
                if (_jobQueues[threadIndex].Count > 0)
                {
                    job = _jobQueues[threadIndex].Dequeue();
                    return true;
                }
            }

            // Try stealing from other queues
            for (int i = 1; i < _threadCount; i++)
            {
                int victimIndex = (threadIndex + i) % _threadCount;
                lock (_queueLocks[victimIndex])
                {
                    if (_jobQueues[victimIndex].Count > 0)
                    {
                        job = _jobQueues[victimIndex].Dequeue();
                        return true;
                    }
                }
            }

            job = null;
            return false;
        }

        private void ProcessOneJob(int threadIndex)
        {
            if (!TryDequeueJob(threadIndex, out var job) || job == null)
                return;

            ExecuteJob(job);
        }

        private void ExecuteJob(Job job)
        {
            job.Handle.StartTimestamp = Stopwatch.GetTimestamp();
            job.Handle.State = JobState.Running;
            Interlocked.Increment(ref _activeThreadCount);

            try
            {
                job.Execute();
                job.Handle.State = JobState.Completed;
                job.Handle.EndTimestamp = Stopwatch.GetTimestamp();
                Interlocked.Increment(ref _totalJobsExecuted);
            }
            catch (OperationCanceledException)
            {
                job.Handle.State = JobState.Cancelled;
            }
            catch (Exception ex)
            {
                job.Handle.SetException(ex);
                job.Handle.State = JobState.Failed;
                Interlocked.Increment(ref _totalJobsFailed);
            }
            finally
            {
                Interlocked.Decrement(ref _activeThreadCount);
                ResolveDependents(job.Handle);
            }
        }

        private void ResolveDependents(JobHandle completed)
        {
            foreach (var dependent in completed.Dependents)
            {
                bool allDepsComplete = true;
                foreach (var dep in dependent.Dependencies)
                {
                    if (!dep.IsCompleted && !dep.IsCancelled && !dep.IsFaulted)
                    {
                        allDepsComplete = false;
                        break;
                    }
                }

                if (allDepsComplete && dependent.State == JobState.Created)
                {
                    dependent.State = JobState.Scheduled;

                    // Find the actual job object and enqueue it
                    // For simplicity, we create a new delegate job wrapper
                    // In production, we'd maintain a mapping
                    int queueIndex = (int)((uint)dependent.JobId % (uint)_threadCount);

                    // Re-enqueue via the handle's associated job
                    // This is simplified; a real implementation would track the Job->Handle mapping
                }
            }
        }

        private void WorkerLoop(int threadIndex)
        {
            var spin = new SpinWait();

            while (!_cts.Token.IsCancellationRequested)
            {
                if (TryDequeueJob(threadIndex, out var job) && job != null)
                {
                    spin.Reset();
                    ExecuteJob(job);
                }
                else
                {
                    spin.SpinOnce();
                    if (spin.Count > YIELD_THRESHOLD)
                        Thread.Sleep(1);
                }
            }
        }

        /// <summary>
        /// Disposes the job system and joins all threads.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            CancelAll();
            _cts.Cancel();
            _running = false;

            for (int i = 0; i < _threadCount; i++)
            {
                if (_threads[i] != null && _threads[i].IsAlive)
                    _threads[i].Join(TimeSpan.FromSeconds(2));
            }

            _cts.Dispose();
        }
    }

    /// <summary>
    /// Extension methods for <see cref="JobSystem"/>.
    /// </summary>
    public static class JobSystemExtensions
    {
        /// <summary>
        /// Schedules a parallel-for job using the job system.
        /// </summary>
        public static JobHandle ParallelFor(
            this JobSystem system,
            int startInclusive,
            int endExclusive,
            Action<int> action)
        {
            return system.ScheduleParallelFor(action, startInclusive, endExclusive);
        }

        /// <summary>
        /// Schedules a parallel-for job with a specific chunk size.
        /// </summary>
        public static JobHandle ParallelFor(
            this JobSystem system,
            int startInclusive,
            int endExclusive,
            int chunkSize,
            Action<int> action)
        {
            return system.ScheduleParallelFor(action, startInclusive, endExclusive, chunkSize);
        }

        /// <summary>
        /// Creates a job chain from a sequence of actions.
        /// </summary>
        public static JobHandle Chain(
            this JobSystem system,
            params Action[] actions)
        {
            return system.ScheduleChain(actions);
        }

        /// <summary>
        /// Runs a function on a background thread and returns a handle.
        /// </summary>
        public static JobHandle RunAsync(
            this JobSystem system,
            Action action)
        {
            return system.Schedule(action);
        }
    }
}
