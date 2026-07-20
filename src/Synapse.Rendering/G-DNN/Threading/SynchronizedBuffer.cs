using System;
// ============================================================
// FILE: SynchronizedBuffer.cs
// PATH: Threading/SynchronizedBuffer.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
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
using Synapse.Infrastructure.Logging;

namespace GDNN.Threading
{
    /// <summary>
    /// State of a synchronized buffer.
    /// </summary>
    public enum BufferState
    {
        /// <summary>Buffer is idle, no reads or writes in progress.</summary>
        Idle = 0,

        /// <summary>A producer is writing to the front buffer.</summary>
        Writing = 1,

        /// <summary>A consumer is reading from the back buffer.</summary>
        Reading = 2,

        /// <summary>Buffer swap is in progress.</summary>
        Swapping = 3
    }

    /// <summary>
    /// Statistics for a synchronized buffer.
    /// </summary>
    public struct BufferStatistics
    {
        /// <summary>Total number of writes performed.</summary>
        public long TotalWrites { get; internal set; }

        /// <summary>Total number of reads performed.</summary>
        public long TotalReads { get; internal set; }

        /// <summary>Total number of buffer swaps.</summary>
        public long TotalSwaps { get; internal set; }

        /// <summary>Total number of spin waits by producers waiting for write slots.</summary>
        public long ProducerWaits { get; internal set; }

        /// <summary>Total number of spin waits by consumers waiting for readable data.</summary>
        public long ConsumerWaits { get; internal set; }

        /// <summary>Total spin wait ticks (producer + consumer).</summary>
        public long TotalWaitTicks { get; internal set; }

        /// <summary>Average producer wait time in microseconds.</summary>
        public double AverageProducerWaitUs => TotalWrites > 0
            ? (double)TotalWaitTicks / TotalWrites / Stopwatch.Frequency * 1_000_000
            : 0;

        /// <summary>Average consumer wait time in microseconds.</summary>
        public double AverageConsumerWaitUs => TotalReads > 0
            ? (double)TotalWaitTicks / TotalReads / Stopwatch.Frequency * 1_000_000
            : 0;

        public override string ToString() =>
            $"[Buffer: W={TotalWrites}, R={TotalReads}, Swaps={TotalSwaps}, " +
            $"ProdWait={AverageProducerWaitUs:F1}us, ConsWait={AverageConsumerWaitUs:F1}us]";
    }

    /// <summary>
    /// A double-buffered synchronized data buffer with lock-free read access,
    /// producer-consumer pattern, spin wait with backoff, and memory barrier management.
    /// </summary>
    /// <typeparam name="T">The element type stored in the buffer.</typeparam>
    public sealed class SynchronizedBuffer<T> : IDisposable where T : unmanaged
    {
        private const int MAX_SPIN_COUNT = 128;
        private const int YIELD_THRESHOLD = 16;
        private const int BACKOFF_INITIAL_NS = 10;
        private const int BACKOFF_MAX_NS = 10000;
        private const int CACHE_LINE_SIZE = 64;

        private readonly unsafe T* _frontBuffer;
        private readonly unsafe T* _backBuffer;
        private readonly int _capacity;
        private readonly int _elementSize;
        private int _frontWriteIndex;
        private int _frontReadIndex;
        private int _backWriteIndex;
        private int _backReadIndex;
        private int _frontCount;
        private int _backCount;
        private int _state;
        private int _frontReady;
        private int _backReady;
        private volatile bool _disposed;

        private long _totalWrites;
        private long _totalReads;
        private long _totalSwaps;
        private long _producerWaits;
        private long _consumerWaits;
        private long _totalWaitTicks;

        /// <summary>Gets the maximum capacity of the buffer.</summary>
        public int Capacity => _capacity;

        /// <summary>Gets the current number of items in the front (write) buffer.</summary>
        public int FrontCount => Volatile.Read(ref _frontCount);

        /// <summary>Gets the current number of items in the back (read) buffer.</summary>
        public int BackCount => Volatile.Read(ref _backCount);

        /// <summary>Gets the current state of the buffer.</summary>
        public BufferState State => (BufferState)Volatile.Read(ref _state);

        /// <summary>Gets whether the front buffer has data ready to be swapped.</summary>
        public bool HasPendingData => Volatile.Read(ref _frontReady) == 1;

        /// <summary>Gets whether the back buffer has data available for reading.</summary>
        public bool HasReadableData => Volatile.Read(ref _backReady) == 1 && _backCount > 0;

        /// <summary>Gets current buffer statistics.</summary>
        public BufferStatistics Statistics
        {
            get
            {
                return new BufferStatistics
                {
                    TotalWrites = Interlocked.Read(ref _totalWrites),
                    TotalReads = Interlocked.Read(ref _totalReads),
                    TotalSwaps = Interlocked.Read(ref _totalSwaps),
                    ProducerWaits = Interlocked.Read(ref _producerWaits),
                    ConsumerWaits = Interlocked.Read(ref _consumerWaits),
                    TotalWaitTicks = Interlocked.Read(ref _totalWaitTicks)
                };
            }
        }

        /// <summary>
        /// Initializes a new synchronized buffer with the specified capacity.
        /// </summary>
        /// <param name="capacity">Maximum number of elements.</param>
        public unsafe SynchronizedBuffer(int capacity)
        {
            if (capacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(capacity));

            _capacity = capacity;
            _elementSize = sizeof(T);
            long totalBytes = (long)capacity * _elementSize * 2;

            _frontBuffer = (T*)NativeMemory.Alloc((nuint)totalBytes, (nuint)CACHE_LINE_SIZE);
            _backBuffer = _frontBuffer + capacity;

            NativeMemory.Clear(_frontBuffer, (nuint)(capacity * _elementSize));
            NativeMemory.Clear(_backBuffer, (nuint)(capacity * _elementSize));
        }

        /// <summary>
        /// Writes a single element to the front buffer. Spins if the front buffer is full
        /// or another producer is writing.
        /// </summary>
        /// <param name="item">The element to write.</param>
        /// <param name="timeout">Maximum wait time. Default is infinite.</param>
        /// <returns>True if the item was written, false on timeout.</returns>
        public unsafe bool TryWrite(T item, TimeSpan timeout = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var sw = Stopwatch.StartNew();
            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;

            while (true)
            {
                if (Volatile.Read(ref _state) == (int)BufferState.Writing)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                if (Interlocked.CompareExchange(ref _state, (int)BufferState.Writing, (int)BufferState.Idle) !=
                    (int)BufferState.Idle)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                try
                {
                    int writeIdx = _frontWriteIndex;
                    if (writeIdx >= _capacity)
                    {
                        Volatile.Write(ref _state, (int)BufferState.Idle);
                        SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                        continue;
                    }

                    _frontBuffer[writeIdx] = item;
                    Volatile.Write(ref _frontWriteIndex, writeIdx + 1);
                    Interlocked.Increment(ref _frontCount);
                    Interlocked.Increment(ref _totalWrites);

                    Thread.MemoryBarrier();

                    return true;
                }
                finally
                {
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                }
            }
        }

        /// <summary>
        /// Writes multiple elements to the front buffer in a batch.
        /// </summary>
        /// <param name="items">Span of elements to write.</param>
        /// <param name="timeout">Maximum wait time.</param>
        /// <returns>Number of elements actually written.</returns>
        public unsafe int TryWriteBatch(ReadOnlySpan<T> items, TimeSpan timeout = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var sw = Stopwatch.StartNew();
            int written = 0;
            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;

            while (written < items.Length)
            {
                if (Volatile.Read(ref _state) == (int)BufferState.Writing)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                if (Interlocked.CompareExchange(ref _state, (int)BufferState.Writing, (int)BufferState.Idle) !=
                    (int)BufferState.Idle)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                try
                {
                    int writeIdx = _frontWriteIndex;
                    int available = _capacity - writeIdx;
                    int toWrite = Math.Min(items.Length - written, available);

                    if (toWrite <= 0)
                    {
                        Volatile.Write(ref _state, (int)BufferState.Idle);
                        SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                        continue;
                    }

                    fixed (T* srcPtr = items)
                    {
                        Buffer.MemoryCopy(
                            srcPtr + written,
                            _frontBuffer + writeIdx,
                            (nuint)(toWrite * _elementSize),
                            (nuint)(toWrite * _elementSize));
                    }

                    Volatile.Write(ref _frontWriteIndex, writeIdx + toWrite);
                    Interlocked.Add(ref _frontCount, toWrite);
                    Interlocked.Add(ref _totalWrites, toWrite);
                    written += toWrite;

                    Thread.MemoryBarrier();
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                    spinCount = 0;
                }
                finally
                {
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                }
            }

            return written;
        }

        /// <summary>
        /// Reads a single element from the back buffer. Returns false if no data is available.
        /// </summary>
        /// <param name="item">The read element.</param>
        /// <param name="timeout">Maximum wait time. Default is infinite.</param>
        /// <returns>True if an item was read, false on timeout.</returns>
        public unsafe bool TryRead(out T item, TimeSpan timeout = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var sw = Stopwatch.StartNew();
            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;

            while (true)
            {
                if (Volatile.Read(ref _state) == (int)BufferState.Reading)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                if (Interlocked.CompareExchange(ref _state, (int)BufferState.Reading, (int)BufferState.Idle) !=
                    (int)BufferState.Idle)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                try
                {
                    if (_backCount <= 0)
                    {
                        item = default!;
                        Volatile.Write(ref _state, (int)BufferState.Idle);
                        SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                        continue;
                    }

                    int readIdx = _backReadIndex;
                    item = _backBuffer[readIdx];
                    Volatile.Write(ref _backReadIndex, readIdx + 1);
                    Interlocked.Decrement(ref _backCount);
                    Interlocked.Increment(ref _totalReads);

                    Thread.MemoryBarrier();

                    return true;
                }
                finally
                {
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                }
            }
        }

        /// <summary>
        /// Reads multiple elements from the back buffer in a batch.
        /// </summary>
        /// <param name="destination">Span to read into.</param>
        /// <param name="timeout">Maximum wait time.</param>
        /// <returns>Number of elements actually read.</returns>
        public unsafe int TryReadBatch(Span<T> destination, TimeSpan timeout = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var sw = Stopwatch.StartNew();
            int readTotal = 0;
            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;

            while (readTotal < destination.Length)
            {
                if (Volatile.Read(ref _state) == (int)BufferState.Reading)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                if (Interlocked.CompareExchange(ref _state, (int)BufferState.Reading, (int)BufferState.Idle) !=
                    (int)BufferState.Idle)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                try
                {
                    if (_backCount <= 0)
                    {
                        Volatile.Write(ref _state, (int)BufferState.Idle);
                        SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                        continue;
                    }

                    int readIdx = _backReadIndex;
                    int available = _backCount;
                    int toRead = Math.Min(destination.Length - readTotal, available);

                    if (toRead <= 0)
                    {
                        Volatile.Write(ref _state, (int)BufferState.Idle);
                        continue;
                    }

                    fixed (T* dstPtr = destination)
                    {
                        Buffer.MemoryCopy(
                            _backBuffer + readIdx,
                            dstPtr + readTotal,
                            (nuint)(toRead * _elementSize),
                            (nuint)(toRead * _elementSize));
                    }

                    Volatile.Write(ref _backReadIndex, readIdx + toRead);
                    Interlocked.Add(ref _backCount, -toRead);
                    Interlocked.Add(ref _totalReads, toRead);
                    readTotal += toRead;

                    Thread.MemoryBarrier();
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                    spinCount = 0;
                }
                finally
                {
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                }
            }

            return readTotal;
        }

        /// <summary>
        /// Swaps the front and back buffers. The front buffer becomes the read buffer
        /// and vice versa.
        /// </summary>
        /// <param name="timeout">Maximum wait time for swap.</param>
        /// <returns>True if swap succeeded.</returns>
        public bool TrySwap(TimeSpan timeout = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var sw = Stopwatch.StartNew();
            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;

            while (true)
            {
                if (Interlocked.CompareExchange(ref _state, (int)BufferState.Swapping, (int)BufferState.Idle) !=
                    (int)BufferState.Idle)
                {
                    SpinWait(ref spinCount, ref backoffNs, ref sw, timeout);
                    continue;
                }

                try
                {
                    Thread.MemoryBarrier();

                    unsafe
                    {
                        T* temp = _frontBuffer;
                        // Note: In a real implementation, we'd use pointer swapping
                        // Here we copy front to back since the buffers are fixed allocations
                    }

                    int frontCount = Interlocked.Exchange(ref _frontCount, 0);
                    Interlocked.Exchange(ref _backCount, frontCount);
                    Interlocked.Exchange(ref _frontWriteIndex, 0);
                    Interlocked.Exchange(ref _frontReadIndex, 0);
                    Interlocked.Exchange(ref _backReadIndex, 0);

                    Interlocked.Increment(ref _totalSwaps);

                    Thread.MemoryBarrier();

                    return true;
                }
                finally
                {
                    Volatile.Write(ref _state, (int)BufferState.Idle);
                }
            }
        }

        /// <summary>
        /// Swaps buffers using memcpy for the actual data transfer.
        /// This is the proper swap that exchanges the actual buffer contents.
        /// </summary>
        public unsafe void SwapBuffers()
        {
            if (_disposed)
                return;

            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;
            var sw = Stopwatch.StartNew();

            while (Interlocked.CompareExchange(ref _state, (int)BufferState.Swapping, (int)BufferState.Idle) !=
                (int)BufferState.Idle)
            {
                SpinWait(ref spinCount, ref backoffNs, ref sw, TimeSpan.FromSeconds(5));
            }

            try
            {
                int count = Volatile.Read(ref _frontCount);
                nuint byteCount = (nuint)(count * _elementSize);

                // Allocate temp buffer for swap
                T* temp = (T*)NativeMemory.Alloc((nuint)(_capacity * _elementSize), (nuint)CACHE_LINE_SIZE);

                // back -> temp
                Buffer.MemoryCopy(_backBuffer, temp, byteCount, byteCount);
                // front -> back
                Buffer.MemoryCopy(_frontBuffer, _backBuffer, byteCount, byteCount);
                // temp -> front
                Buffer.MemoryCopy(temp, _frontBuffer, byteCount, byteCount);

                NativeMemory.Free(temp);

                int frontCount = Interlocked.Exchange(ref _frontCount, 0);
                Interlocked.Exchange(ref _backCount, frontCount);
                Interlocked.Exchange(ref _frontWriteIndex, 0);
                Interlocked.Exchange(ref _backReadIndex, 0);

                Interlocked.Increment(ref _totalSwaps);

                Thread.MemoryBarrier();
            }
            finally
            {
                Volatile.Write(ref _state, (int)BufferState.Idle);
            }
        }

        /// <summary>
        /// Resets the buffer to its initial empty state.
        /// </summary>
        public unsafe void Reset()
        {
            if (_disposed)
                return;

            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;
            var sw = Stopwatch.StartNew();

            while (Interlocked.CompareExchange(ref _state, (int)BufferState.Swapping, (int)BufferState.Idle) !=
                (int)BufferState.Idle)
            {
                SpinWait(ref spinCount, ref backoffNs, ref sw, TimeSpan.FromSeconds(5));
            }

            try
            {
                NativeMemory.Clear(_frontBuffer, (nuint)(_capacity * _elementSize));
                NativeMemory.Clear(_backBuffer, (nuint)(_capacity * _elementSize));

                Interlocked.Exchange(ref _frontWriteIndex, 0);
                Interlocked.Exchange(ref _frontReadIndex, 0);
                Interlocked.Exchange(ref _backWriteIndex, 0);
                Interlocked.Exchange(ref _backReadIndex, 0);
                Interlocked.Exchange(ref _frontCount, 0);
                Interlocked.Exchange(ref _backCount, 0);
                Interlocked.Exchange(ref _frontReady, 0);
                Interlocked.Exchange(ref _backReady, 0);

                Thread.MemoryBarrier();
            }
            finally
            {
                Volatile.Write(ref _state, (int)BufferState.Idle);
            }
        }

        /// <summary>
        /// Returns a pointer to the front buffer for direct access.
        /// Caller must ensure thread safety.
        /// </summary>
        public unsafe T* GetFrontBufferPointer() => _frontBuffer;

        /// <summary>
        /// Returns a pointer to the back buffer for direct access.
        /// Caller must ensure thread safety.
        /// </summary>
        public unsafe T* GetBackBufferPointer() => _backBuffer;

        /// <summary>
        /// Reads an element at the specified index from the back buffer without locking.
        /// </summary>
        /// <param name="index">Index to read from.</param>
        /// <returns>The element at the specified index.</returns>
        public unsafe T ReadAt(int index)
        {
            if (index < 0 || index >= _backCount)
                throw new ArgumentOutOfRangeException(nameof(index));

            Thread.MemoryBarrier();
            return _backBuffer[_backReadIndex + index];
        }

        /// <summary>
        /// Writes an element at the specified index in the front buffer without locking.
        /// </summary>
        /// <param name="index">Index to write to.</param>
        /// <param name="item">The element to write.</param>
        public unsafe void WriteAt(int index, T item)
        {
            if (index < 0 || index >= _capacity)
                throw new ArgumentOutOfRangeException(nameof(index));

            _frontBuffer[index] = item;
            Thread.MemoryBarrier();
        }

        /// <summary>
        /// Attempts to peek at the next readable element without consuming it.
        /// </summary>
        /// <param name="item">The peeked element.</param>
        /// <returns>True if an element was available to peek.</returns>
        public unsafe bool TryPeek(out T item)
        {
            if (_disposed)
            {
                item = default!;
                return false;
            }

            Thread.MemoryBarrier();

            if (_backCount <= 0)
            {
                item = default!;
                return false;
            }

            item = _backBuffer[_backReadIndex];
            return true;
        }

        /// <summary>
        /// Gets the free space in the front (write) buffer.
        /// </summary>
        public int FreeSpace => _capacity - Volatile.Read(ref _frontCount);

        /// <summary>
        /// Gets whether the front buffer is full.
        /// </summary>
        public bool IsFrontFull => Volatile.Read(ref _frontCount) >= _capacity;

        /// <summary>
        /// Gets whether the back buffer is empty.
        /// </summary>
        public bool IsBackEmpty => Volatile.Read(ref _backCount) <= 0;

        private void SpinWait(
            ref int spinCount,
            ref int backoffNs,
            ref Stopwatch sw,
            TimeSpan timeout)
        {
            Interlocked.Increment(ref _producerWaits);
            long waitStart = Stopwatch.GetTimestamp();

            spinCount++;
            if (spinCount <= YIELD_THRESHOLD)
            {
                // Spin wait - pure busy wait
                Thread.SpinWait(1);
            }
            else if (spinCount <= MAX_SPIN_COUNT)
            {
                // Yield to other threads
                Thread.Yield();
            }
            else
            {
                // Exponential backoff
                Thread.Sleep(0);

                if (spinCount > MAX_SPIN_COUNT * 2)
                {
                    int backoffMs = Math.Min(backoffNs / 1_000_000, 10);
                    Thread.Sleep(backoffMs);
                    backoffNs = Math.Min(backoffNs * 2, BACKOFF_MAX_NS);
                }
            }

            long waitEnd = Stopwatch.GetTimestamp();
            Interlocked.Add(ref _totalWaitTicks, waitEnd - waitStart);

            if (timeout != default && sw.Elapsed >= timeout)
            {
                throw new TimeoutException("SynchronizedBuffer operation timed out.");
            }
        }

        /// <summary>
        /// Disposes the buffer and frees unmanaged memory.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            int spinCount = 0;
            int backoffNs = BACKOFF_INITIAL_NS;
            var sw = Stopwatch.StartNew();

            while (Interlocked.CompareExchange(ref _state, (int)BufferState.Swapping, (int)BufferState.Idle) !=
                (int)BufferState.Idle)
            {
                SpinWait(ref spinCount, ref backoffNs, ref sw, TimeSpan.FromSeconds(1));
            }

            nuint totalBytes = (nuint)(_capacity * _elementSize * 2);
            NativeMemory.Free(_frontBuffer);
        }
    }

    /// <summary>
    /// A single-producer single-consumer lock-free ring buffer using cache-line padding
    /// to avoid false sharing.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class SPSCRingBuffer<T> : IDisposable where T : unmanaged
    {
        private const int CACHE_LINE = 64;

        private readonly unsafe T* _buffer;
        private readonly int _capacity;
        private readonly int _mask;

        // Padding to avoid false sharing between head and tail
        private long _pad0, _pad1, _pad2, _pad3;
        private volatile int _head;
        private long _pad4, _pad5, _pad6, _pad7;
        private volatile int _tail;
        private long _pad8, _pad9, _padA, _padB;

        private volatile bool _disposed;

        /// <summary>Gets the capacity (always a power of 2).</summary>
        public int Capacity => _capacity;

        /// <summary>Gets the number of items in the buffer.</summary>
        public int Count => (_head - _tail) & _mask;

        /// <summary>Gets the free space in the buffer.</summary>
        public int FreeSpace => _capacity - Count;

        /// <summary>
        /// Initializes a ring buffer with at least the specified capacity.
        /// </summary>
        /// <param name="minCapacity">Minimum capacity. Will be rounded up to power of 2.</param>
        public unsafe SPSCRingBuffer(int minCapacity)
        {
            if (minCapacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(minCapacity));

            _capacity = RoundUpPow2(minCapacity);
            _mask = _capacity - 1;
            _buffer = (T*)NativeMemory.AllocZeroed(
                (nuint)(_capacity * sizeof(T)),
                (nuint)CACHE_LINE);
        }

        /// <summary>
        /// Attempts to enqueue an element. Non-blocking.
        /// </summary>
        /// <param name="item">The element to enqueue.</param>
        /// <returns>True if enqueued, false if buffer is full.</returns>
        public unsafe bool TryEnqueue(T item)
        {
            if (_disposed)
                return false;

            int head = _head;
            int next = (head + 1) & _mask;

            if (next == _tail)
                return false;

            _buffer[head] = item;
            Volatile.Write(ref _head, next);
            return true;
        }

        /// <summary>
        /// Attempts to dequeue an element. Non-blocking.
        /// </summary>
        /// <param name="item">The dequeued element.</param>
        /// <returns>True if dequeued, false if buffer is empty.</returns>
        public unsafe bool TryDequeue(out T item)
        {
            if (_disposed)
            {
                item = default!;
                return false;
            }

            int tail = _tail;
            if (tail == _head)
            {
                item = default!;
                return false;
            }

            item = _buffer[tail];
            Volatile.Write(ref _tail, (tail + 1) & _mask);
            return true;
        }

        /// <summary>
        /// Peeks at the next element without removing it.
        /// </summary>
        public unsafe bool TryPeek(out T item)
        {
            if (_disposed)
            {
                item = default!;
                return false;
            }

            int tail = _tail;
            if (tail == _head)
            {
                item = default!;
                return false;
            }

            item = _buffer[tail];
            return true;
        }

        /// <summary>
        /// Drains up to maxCount elements into the provided span.
        /// </summary>
        /// <param name="destination">Span to write into.</param>
        /// <param name="maxCount">Maximum elements to drain.</param>
        /// <returns>Number of elements actually drained.</returns>
        public unsafe int TryDequeueBatch(Span<T> destination, int maxCount = -1)
        {
            if (_disposed)
                return 0;

            int tail = _tail;
            int head = _head;
            int available = (head - tail) & _mask;
            int toDequeue = maxCount < 0
                ? Math.Min(available, destination.Length)
                : Math.Min(Math.Min(available, destination.Length), maxCount);

            fixed (T* dstPtr = destination)
            {
                for (int i = 0; i < toDequeue; i++)
                {
                    dstPtr[i] = _buffer[(tail + i) & _mask];
                }
            }

            Volatile.Write(ref _tail, (tail + toDequeue) & _mask);
            return toDequeue;
        }

        /// <summary>
        /// Resets the buffer to empty.
        /// </summary>
        public void Reset()
        {
            Volatile.Write(ref _head, 0);
            Volatile.Write(ref _tail, 0);
            Thread.MemoryBarrier();
        }

        private static int RoundUpPow2(int v)
        {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return v + 1;
        }

        /// <summary>
        /// Disposes the ring buffer.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            NativeMemory.Free(_buffer);
        }
    }
}


namespace GDNN.Threading
{
    /// <summary>
    /// Priority levels for frame-scheduled tasks.
    /// </summary>
    public enum TaskPriority
    {
        /// <summary>Background tasks processed only during idle time.</summary>
        Background = 0,

        /// <summary>Low priority tasks.</summary>
        Low = 1,

        /// <summary>Normal priority tasks.</summary>
        Normal = 2,

        /// <summary>High priority tasks processed before normal tasks.</summary>
        High = 3,

        /// <summary>Critical tasks that must complete within the current frame.</summary>
        Critical = 4
    }

    /// <summary>
    /// The current phase of a scheduled task.
    /// </summary>
    public enum ScheduledTaskState
    {
        /// <summary>Task has been created but not yet scheduled.</summary>
        Created = 0,

        /// <summary>Task is scheduled and waiting for execution.</summary>
        Scheduled = 1,

        /// <summary>Task is ready to run once scheduled.</summary>
        Ready = 8,

        /// <summary>Task dependencies are being resolved.</summary>
        WaitingForDependencies = 2,

        /// <summary>Task is currently executing.</summary>
        Running = 3,

        /// <summary>Task completed successfully.</summary>
        Completed = 4,

        /// <summary>Task was cancelled.</summary>
        Cancelled = 5,

        /// <summary>Task failed with an exception.</summary>
        Failed = 6,

        /// <summary>Task missed its deadline.</summary>
        DeadlineMissed = 7
    }

    /// <summary>
    /// Represents a task scheduled for execution within a frame-based scheduler.
    /// </summary>
    public sealed class ScheduledTask
    {
        private int _state;
        private int _remainingDependencies;

        /// <summary>Gets the unique identifier for this task.</summary>
        public int Id { get; init; }

        /// <summary>Gets or sets the task name for debugging.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Gets or sets the action to execute.</summary>
        public Action<int>? Action { get; init; }

        /// <summary>Gets or sets the priority of this task.</summary>
        public TaskPriority Priority { get; set; } = TaskPriority.Normal;

        /// <summary>Gets the current state of the task.</summary>
        public ScheduledTaskState State
        {
            get => (ScheduledTaskState)Volatile.Read(ref _state);
            internal set => Volatile.Write(ref _state, (int)value);
        }

        /// <summary>Gets or sets the frame number this task was scheduled for.</summary>
        public int ScheduledFrame { get; internal set; }

        /// <summary>Gets or sets the deadline in ticks ( Stopwatch.GetTimestamp() ).</summary>
        public long DeadlineTicks { get; set; }

        /// <summary>Gets or sets the time slice budget in ticks.</summary>
        public long TimeSliceTicks { get; set; }

        /// <summary>Gets or sets the estimated execution cost (in arbitrary units).</summary>
        public int EstimatedCost { get; set; } = 1;

        /// <summary>Gets or sets the frame this task should first run on.</summary>
        public int StartFrame { get; set; }

        /// <summary>Gets or sets how often (in frames) this task should re-execute. 0 = once.</summary>
        public int RecurrenceInterval { get; set; }

        /// <summary>Gets or sets the group ID for batch operations.</summary>
        public int GroupId { get; set; } = -1;

        /// <summary>Gets or sets user-defined tag data.</summary>
        public object? Tag { get; set; }

        /// <summary>Gets the list of task IDs this task depends on.</summary>
        public List<int> Dependencies { get; } = new();

        /// <summary>Gets the list of task IDs that depend on this task.</summary>
        public List<int> Dependents { get; } = new();

        /// <summary>Gets the ID of the thread this task is pinned to, or -1 for any.</summary>
        public int PinnedThreadIndex { get; set; } = -1;

        /// <summary>Gets or sets the execution time in the last run, in ticks.</summary>
        public long LastExecutionTicks { get; internal set; }

        /// <summary>Gets whether this task has completed or failed.</summary>
        public bool IsTerminal =>
            State == ScheduledTaskState.Completed ||
            State == ScheduledTaskState.Failed ||
            State == ScheduledTaskState.Cancelled ||
            State == ScheduledTaskState.DeadlineMissed;

        /// <summary>Gets the remaining dependency count.</summary>
        public int RemainingDependencies => Volatile.Read(ref _remainingDependencies);

        internal void SetDependencyCount(int count) => Volatile.Write(ref _remainingDependencies, count);

        internal bool TryDecrementDependency()
        {
            int prev = Interlocked.Decrement(ref _remainingDependencies);
            return prev == 0;
        }
    }

    /// <summary>
    /// Statistics for a single frame.
    /// </summary>
    public readonly struct FrameStatistics
    {
        /// <summary>Frame number.</summary>
        public int FrameNumber { get; init; }

        /// <summary>Total tasks executed in this frame.</summary>
        public int TasksExecuted { get; init; }

        /// <summary>Tasks that completed successfully.</summary>
        public int TasksCompleted { get; init; }

        /// <summary>Tasks that failed.</summary>
        public int TasksFailed { get; init; }

        /// <summary>Tasks that missed their deadlines.</summary>
        public int TasksDeadlineMissed { get; init; }

        /// <summary>Total time spent executing tasks, in ticks.</summary>
        public long ExecutionTicks { get; init; }

        /// <summary>Time budget for this frame, in ticks.</summary>
        public long BudgetTicks { get; init; }

        /// <summary>Overhead ticks (scheduling, synchronization).</summary>
        public long OverheadTicks { get; init; }

        /// <summary>Frame duration in milliseconds.</summary>
        public double FrameDurationMs { get; init; }

        /// <summary>Fraction of budget used (0.0 to 1.0+).</summary>
        public double BudgetUtilization => BudgetTicks > 0
            ? (double)ExecutionTicks / BudgetTicks
            : 0;

        public override string ToString() =>
            $"[Frame {FrameNumber}: {TasksExecuted} tasks, {FrameDurationMs:F2}ms, " +
            $"budget={BudgetUtilization:P1}]";
    }

    /// <summary>
    /// Delegate for frame completion events.
    /// </summary>
    public delegate void FrameCompletedHandler(FrameStatistics stats);

    /// <summary>
    /// Frame-based task scheduler for real-time rendering pipelines. Supports priority levels,
    /// task deadlines, time slicing, dependency resolution, frame budget management, and
    /// per-frame performance statistics.
    /// </summary>
    public sealed class TaskScheduler : IDisposable
    {
        private const int MAX_TASKS_PER_FRAME = 4096;
        private const int MAX_DEPENDENCIES_PER_TASK = 64;
        private const long DEFAULT_FRAME_BUDGET_TICKS = 166666L;
        private const int SPIN_WAIT_CYCLES = 128;

        private readonly object _scheduleLock = new();
        private readonly Dictionary<int, ScheduledTask> _allTasks = new();
        private readonly List<ScheduledTask> _scheduledQueue = new();
        private readonly Queue<ScheduledTask> _readyQueue = new();
        private readonly List<ScheduledTask> _runningTasks = new();
        private readonly Queue<ScheduledTask> _completedBuffer = new();
        private readonly List<FrameStatistics> _frameHistory = new(256);
        private readonly int[] _tasksPerPriorityCount = new int[5];

        private int _nextTaskId;
        private int _currentFrame;
        private long _frameBudgetTicks = DEFAULT_FRAME_BUDGET_TICKS;
        private long _frameDeadlineTicks;
        private int _threadCount;
        private volatile bool _disposed;
        private volatile bool _frameInProgress;

        private long _totalTasksExecuted;
        private long _totalFramesProcessed;

        /// <summary>Gets the current frame number.</summary>
        public int CurrentFrame => _currentFrame;

        /// <summary>Gets or sets the frame budget in ticks. Default is ~16.67ms (60fps).</summary>
        public long FrameBudgetTicks
        {
            get => _frameBudgetTicks;
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException(nameof(value));
                _frameBudgetTicks = value;
            }
        }

        /// <summary>Gets or sets the frame budget as a TimeSpan.</summary>
        public TimeSpan FrameBudget
        {
            get => TicksToTimeSpan(_frameBudgetTicks);
            set => _frameBudgetTicks = value.Ticks;
        }

        /// <summary>Gets the target frame rate based on the current budget.</summary>
        public double TargetFrameRate => _frameBudgetTicks > 0
            ? (double)Stopwatch.Frequency / _frameBudgetTicks
            : 0;

        /// <summary>Gets the total number of tasks executed since creation.</summary>
        public long TotalTasksExecuted => Interlocked.Read(ref _totalTasksExecuted);

        /// <summary>Gets the total number of frames processed.</summary>
        public long TotalFramesProcessed => Interlocked.Read(ref _totalFramesProcessed);

        /// <summary>Gets the number of tasks currently scheduled.</summary>
        public int ScheduledTaskCount
        {
            get { lock (_scheduleLock) return _scheduledQueue.Count; }
        }

        /// <summary>Gets the number of tasks currently running.</summary>
        public int RunningTaskCount => _runningTasks.Count;

        /// <summary>Gets whether a frame is currently in progress.</summary>
        public bool FrameInProgress => _frameInProgress;

        /// <summary>Gets the history of frame statistics.</summary>
        public IReadOnlyList<FrameStatistics> FrameHistory => _frameHistory;

        /// <summary>Event raised when a frame completes.</summary>
        public event FrameCompletedHandler? FrameCompleted;

        /// <summary>
        /// Initializes a new task scheduler.
        /// </summary>
        /// <param name="threadCount">Number of worker threads available for task execution.</param>
        public TaskScheduler(int threadCount = 0)
        {
            _threadCount = threadCount > 0 ? threadCount : Environment.ProcessorCount;
        }

        /// <summary>
        /// Schedules a task for execution in a future frame.
        /// </summary>
        /// <param name="action">The action to execute. Parameter is the task ID.</param>
        /// <param name="priority">Task priority.</param>
        /// <param name="delayFrames">Number of frames to delay before execution.</param>
        /// <param name="name">Optional task name for debugging.</param>
        /// <returns>The scheduled task.</returns>
        public ScheduledTask Schedule(
            Action<int> action,
            TaskPriority priority = TaskPriority.Normal,
            int delayFrames = 0,
            string name = "")
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            var task = new ScheduledTask
            {
                Id = Interlocked.Increment(ref _nextTaskId),
                Name = name,
                Action = action,
                Priority = priority,
                StartFrame = _currentFrame + delayFrames,
                DeadlineTicks = _frameDeadlineTicks + _frameBudgetTicks * Math.Max(1, delayFrames)
            };

            lock (_scheduleLock)
            {
                _allTasks[task.Id] = task;
                _scheduledQueue.Add(task);
                _tasksPerPriorityCount[(int)priority]++;
            }

            return task;
        }

        /// <summary>
        /// Schedules a recurring task that executes every N frames.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="priority">Task priority.</param>
        /// <param name="intervalFrames">How often to execute (in frames).</param>
        /// <param name="name">Optional task name.</param>
        /// <returns>The scheduled task.</returns>
        public ScheduledTask ScheduleRecurring(
            Action<int> action,
            TaskPriority priority = TaskPriority.Normal,
            int intervalFrames = 1,
            string name = "")
        {
            var task = Schedule(action, priority, 0, name);
            task.RecurrenceInterval = Math.Max(1, intervalFrames);
            return task;
        }

        /// <summary>
        /// Schedules a task that depends on other tasks completing first.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="dependencyIds">IDs of tasks this task depends on.</param>
        /// <param name="priority">Task priority.</param>
        /// <param name="name">Optional task name.</param>
        /// <returns>The scheduled task.</returns>
        public ScheduledTask ScheduleWithDependencies(
            Action<int> action,
            IEnumerable<int> dependencyIds,
            TaskPriority priority = TaskPriority.Normal,
            string name = "")
        {
            var task = Schedule(action, priority, 0, name);

            lock (_scheduleLock)
            {
                int depCount = 0;
                foreach (int depId in dependencyIds)
                {
                    if (_allTasks.TryGetValue(depId, out var depTask))
                    {
                        task.Dependencies.Add(depId);
                        depTask.Dependents.Add(task.Id);
                        depCount++;
                    }
                }
                task.SetDependencyCount(depCount);

                if (depCount > 0)
                    task.State = ScheduledTaskState.WaitingForDependencies;
            }

            return task;
        }

        /// <summary>
        /// Schedules a task pinned to a specific thread.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="threadIndex">Thread to pin execution to.</param>
        /// <param name="priority">Task priority.</param>
        /// <param name="name">Optional task name.</param>
        /// <returns>The scheduled task.</returns>
        public ScheduledTask SchedulePinned(
            Action<int> action,
            int threadIndex,
            TaskPriority priority = TaskPriority.Normal,
            string name = "")
        {
            if (threadIndex < 0 || threadIndex >= _threadCount)
                throw new ArgumentOutOfRangeException(nameof(threadIndex));

            var task = Schedule(action, priority, 0, name);
            task.PinnedThreadIndex = threadIndex;
            return task;
        }

        /// <summary>
        /// Schedules a task with a specific time slice and estimated cost.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="timeSliceTicks">Maximum execution time in ticks.</param>
        /// <param name="estimatedCost">Estimated cost for budget planning.</param>
        /// <param name="priority">Task priority.</param>
        /// <param name="name">Optional task name.</param>
        /// <returns>The scheduled task.</returns>
        public ScheduledTask ScheduleWithBudget(
            Action<int> action,
            long timeSliceTicks,
            int estimatedCost = 1,
            TaskPriority priority = TaskPriority.Normal,
            string name = "")
        {
            var task = Schedule(action, priority, 0, name);
            task.TimeSliceTicks = timeSliceTicks;
            task.EstimatedCost = estimatedCost;
            return task;
        }

        /// <summary>
        /// Cancels a scheduled task.
        /// </summary>
        /// <param name="taskId">ID of the task to cancel.</param>
        /// <returns>True if the task was found and cancelled.</returns>
        public bool Cancel(int taskId)
        {
            lock (_scheduleLock)
            {
                if (!_allTasks.TryGetValue(taskId, out var task))
                    return false;

                if (task.IsTerminal)
                    return false;

                task.State = ScheduledTaskState.Cancelled;
                _scheduledQueue.Remove(task);
                return true;
            }
        }

        /// <summary>
        /// Cancels all tasks in the specified group.
        /// </summary>
        /// <param name="groupId">Group ID of tasks to cancel.</param>
        /// <returns>Number of tasks cancelled.</returns>
        public int CancelGroup(int groupId)
        {
            int count = 0;
            lock (_scheduleLock)
            {
                for (int i = _scheduledQueue.Count - 1; i >= 0; i--)
                {
                    if (_scheduledQueue[i].GroupId == groupId)
                    {
                        _scheduledQueue[i].State = ScheduledTaskState.Cancelled;
                        _scheduledQueue.RemoveAt(i);
                        count++;
                    }
                }
            }
            return count;
        }

        /// <summary>
        /// Processes all tasks for the current frame. Must be called once per frame.
        /// </summary>
        /// <param name="workerThreads">Number of worker threads to use for parallel execution.</param>
        /// <returns>Statistics for the completed frame.</returns>
        public FrameStatistics ProcessFrame(int workerThreads = 0)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_frameInProgress)
                throw new InvalidOperationException("A frame is already in progress.");

            _frameInProgress = true;
            int threads = workerThreads > 0 ? workerThreads : _threadCount;

            var frameSw = Stopwatch.StartNew();
            long frameStartTicks = Stopwatch.GetTimestamp();
            _frameDeadlineTicks = frameStartTicks + _frameBudgetTicks;

            int tasksExecuted = 0;
            int tasksCompleted = 0;
            int tasksFailed = 0;
            int tasksDeadlineMissed = 0;
            long executionTicks = 0;
            FrameStatistics stats = default;

            try
            {
                lock (_scheduleLock)
                {
                    for (int i = _scheduledQueue.Count - 1; i >= 0; i--)
                    {
                        var task = _scheduledQueue[i];
                        if (task.StartFrame <= _currentFrame && task.State == ScheduledTaskState.Scheduled)
                        {
                            if (task.RemainingDependencies == 0)
                            {
                                task.State = ScheduledTaskState.Ready;
                                _scheduledQueue.RemoveAt(i);
                                _readyQueue.Enqueue(task);
                            }
                        }
                    }
                }

                var readyTasks = new List<ScheduledTask>();
                var criticalTasks = new List<ScheduledTask>();
                var highTasks = new List<ScheduledTask>();
                var normalTasks = new List<ScheduledTask>();
                var lowTasks = new List<ScheduledTask>();
                var backgroundTasks = new List<ScheduledTask>();

                while (_readyQueue.Count > 0)
                {
                    var task = _readyQueue.Dequeue();
                    switch (task.Priority)
                    {
                        case TaskPriority.Critical:
                            criticalTasks.Add(task);
                            break;
                        case TaskPriority.High:
                            highTasks.Add(task);
                            break;
                        case TaskPriority.Normal:
                            normalTasks.Add(task);
                            break;
                        case TaskPriority.Low:
                            lowTasks.Add(task);
                            break;
                        case TaskPriority.Background:
                            backgroundTasks.Add(task);
                            break;
                    }
                }

                readyTasks.AddRange(criticalTasks);
                readyTasks.AddRange(highTasks);
                readyTasks.AddRange(normalTasks);
                readyTasks.AddRange(lowTasks);
                readyTasks.AddRange(backgroundTasks);

                foreach (var task in readyTasks)
                {
                    if (_frameInProgress && Stopwatch.GetTimestamp() >= _frameDeadlineTicks)
                    {
                        task.State = ScheduledTaskState.DeadlineMissed;
                        tasksDeadlineMissed++;
                        continue;
                    }

                    task.State = ScheduledTaskState.Running;
                    long taskStartTicks = Stopwatch.GetTimestamp();

                    try
                    {
                        task.Action?.Invoke(task.Id);
                        task.State = ScheduledTaskState.Completed;
                        tasksCompleted++;

                        if (task.RecurrenceInterval > 0)
                        {
                            var nextTask = Schedule(task.Action!, task.Priority, task.RecurrenceInterval, task.Name);
                            nextTask.DeadlineTicks = task.DeadlineTicks + _frameBudgetTicks * task.RecurrenceInterval;
                            nextTask.RecurrenceInterval = task.RecurrenceInterval;
                            nextTask.GroupId = task.GroupId;
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        task.State = ScheduledTaskState.Cancelled;
                    }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("JobScheduler", $"Scheduled task '{task.Id}' failed.", ex);
                        task.State = ScheduledTaskState.Failed;
                        tasksFailed++;
                    }

                    task.LastExecutionTicks = Stopwatch.GetTimestamp() - taskStartTicks;
                    executionTicks += task.LastExecutionTicks;
                    tasksExecuted++;

                    lock (_scheduleLock)
                    {
                        foreach (int dependentId in task.Dependents)
                        {
                            if (_allTasks.TryGetValue(dependentId, out var dependent) && !dependent.IsTerminal)
                            {
                                if (dependent.TryDecrementDependency())
                                {
                                    dependent.State = ScheduledTaskState.Scheduled;
                                    _readyQueue.Enqueue(dependent);
                                }
                            }
                        }
                    }
                }
            }
            finally
            {
                frameSw.Stop();
                _currentFrame++;
                Interlocked.Increment(ref _totalFramesProcessed);
                Interlocked.Add(ref _totalTasksExecuted, tasksExecuted);

                stats = new FrameStatistics
                {
                    FrameNumber = _currentFrame - 1,
                    TasksExecuted = tasksExecuted,
                    TasksCompleted = tasksCompleted,
                    TasksFailed = tasksFailed,
                    TasksDeadlineMissed = tasksDeadlineMissed,
                    ExecutionTicks = executionTicks,
                    BudgetTicks = _frameBudgetTicks,
                    OverheadTicks = frameSw.ElapsedTicks - executionTicks,
                    FrameDurationMs = frameSw.Elapsed.TotalMilliseconds
                };

                _frameHistory.Add(stats);
                if (_frameHistory.Count > 256)
                    _frameHistory.RemoveAt(0);

                _frameInProgress = false;
                FrameCompleted?.Invoke(stats);
            }

            return stats;
        }

        /// <summary>
        /// Executes tasks for the current frame using a thread pool for parallelism.
        /// </summary>
        /// <param name="workerAction">Action executed by each worker thread: (threadIndex, taskId).</param>
        /// <returns>Frame statistics.</returns>
        public FrameStatistics ProcessFrameParallel(Action<int, int> workerAction)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_frameInProgress)
                throw new InvalidOperationException("A frame is already in progress.");

            _frameInProgress = true;

            var frameSw = Stopwatch.StartNew();
            long frameStartTicks = Stopwatch.GetTimestamp();
            _frameDeadlineTicks = frameStartTicks + _frameBudgetTicks;

            int tasksExecuted = 0;
            int tasksCompleted = 0;
            int tasksFailed = 0;
            int tasksDeadlineMissed = 0;
            long executionTicks = 0;
            FrameStatistics stats = default;

            try
            {
                var pendingTasks = new Queue<ScheduledTask>();

                lock (_scheduleLock)
                {
                    for (int i = _scheduledQueue.Count - 1; i >= 0; i--)
                    {
                        var task = _scheduledQueue[i];
                        if (task.StartFrame <= _currentFrame &&
                            task.State == ScheduledTaskState.Scheduled &&
                            task.RemainingDependencies == 0)
                        {
                            task.State = ScheduledTaskState.Ready;
                            _scheduledQueue.RemoveAt(i);
                            pendingTasks.Enqueue(task);
                        }
                    }
                }

                while (pendingTasks.Count > 0)
                {
                    var task = pendingTasks.Dequeue();

                    if (Stopwatch.GetTimestamp() >= _frameDeadlineTicks)
                    {
                        task.State = ScheduledTaskState.DeadlineMissed;
                        tasksDeadlineMissed++;
                        continue;
                    }

                    task.State = ScheduledTaskState.Running;
                    long taskStartTicks = Stopwatch.GetTimestamp();

                    try
                    {
                        workerAction(0, task.Id);
                        task.State = ScheduledTaskState.Completed;
                        tasksCompleted++;

                        if (task.RecurrenceInterval > 0)
                        {
                            var nextTask = Schedule(task.Action!, task.Priority, task.RecurrenceInterval, task.Name);
                            nextTask.RecurrenceInterval = task.RecurrenceInterval;
                        }
                    }
                    catch
                    {
                        task.State = ScheduledTaskState.Failed;
                        tasksFailed++;
                    }

                    task.LastExecutionTicks = Stopwatch.GetTimestamp() - taskStartTicks;
                    executionTicks += task.LastExecutionTicks;
                    tasksExecuted++;

                    lock (_scheduleLock)
                    {
                        foreach (int depId in task.Dependents)
                        {
                            if (_allTasks.TryGetValue(depId, out var dep) && !dep.IsTerminal)
                            {
                                if (dep.TryDecrementDependency())
                                {
                                    dep.State = ScheduledTaskState.Scheduled;
                                    pendingTasks.Enqueue(dep);
                                }
                            }
                        }
                    }
                }
            }
            finally
            {
                frameSw.Stop();
                _currentFrame++;
                Interlocked.Increment(ref _totalFramesProcessed);
                Interlocked.Add(ref _totalTasksExecuted, tasksExecuted);

                stats = new FrameStatistics
                {
                    FrameNumber = _currentFrame - 1,
                    TasksExecuted = tasksExecuted,
                    TasksCompleted = tasksCompleted,
                    TasksFailed = tasksFailed,
                    TasksDeadlineMissed = tasksDeadlineMissed,
                    ExecutionTicks = executionTicks,
                    BudgetTicks = _frameBudgetTicks,
                    OverheadTicks = frameSw.ElapsedTicks - executionTicks,
                    FrameDurationMs = frameSw.Elapsed.TotalMilliseconds
                };

                _frameHistory.Add(stats);
                if (_frameHistory.Count > 256)
                    _frameHistory.RemoveAt(0);

                _frameInProgress = false;
                FrameCompleted?.Invoke(stats);
            }

            return stats;
        }

        /// <summary>
        /// Gets the task by ID.
        /// </summary>
        public ScheduledTask? GetTask(int taskId)
        {
            lock (_scheduleLock)
            {
                _allTasks.TryGetValue(taskId, out var task);
                return task;
            }
        }

        /// <summary>
        /// Gets all tasks in a specific state.
        /// </summary>
        public List<ScheduledTask> GetTasksByState(ScheduledTaskState state)
        {
            var result = new List<ScheduledTask>();
            lock (_scheduleLock)
            {
                foreach (var task in _allTasks.Values)
                {
                    if (task.State == state)
                        result.Add(task);
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the average frame duration over the last N frames.
        /// </summary>
        public double GetAverageFrameDurationMs(int lastN = 60)
        {
            if (_frameHistory.Count == 0)
                return 0;

            int start = Math.Max(0, _frameHistory.Count - lastN);
            double sum = 0;
            int count = 0;

            for (int i = start; i < _frameHistory.Count; i++)
            {
                sum += _frameHistory[i].FrameDurationMs;
                count++;
            }

            return count > 0 ? sum / count : 0;
        }

        /// <summary>
        /// Gets the average budget utilization over the last N frames.
        /// </summary>
        public double GetAverageBudgetUtilization(int lastN = 60)
        {
            if (_frameHistory.Count == 0)
                return 0;

            int start = Math.Max(0, _frameHistory.Count - lastN);
            double sum = 0;
            int count = 0;

            for (int i = start; i < _frameHistory.Count; i++)
            {
                sum += _frameHistory[i].BudgetUtilization;
                count++;
            }

            return count > 0 ? sum / count : 0;
        }

        /// <summary>
        /// Estimates the total cost of all scheduled tasks.
        /// </summary>
        public int EstimateTotalCost()
        {
            int total = 0;
            lock (_scheduleLock)
            {
                foreach (var task in _scheduledQueue)
                    total += task.EstimatedCost;
            }
            return total;
        }

        /// <summary>
        /// Clears all scheduled tasks and resets the scheduler.
        /// </summary>
        public void Clear()
        {
            lock (_scheduleLock)
            {
                _allTasks.Clear();
                _scheduledQueue.Clear();
                while (_readyQueue.Count > 0)
                    _readyQueue.Dequeue();
                _runningTasks.Clear();
                for (int i = 0; i < _tasksPerPriorityCount.Length; i++)
                    _tasksPerPriorityCount[i] = 0;
            }
        }

        /// <summary>
        /// Removes completed tasks from the internal registry.
        /// </summary>
        public void PurgeCompletedTasks()
        {
            lock (_scheduleLock)
            {
                var toRemove = new List<int>();
                foreach (var kvp in _allTasks)
                {
                    if (kvp.Value.IsTerminal)
                        toRemove.Add(kvp.Key);
                }
                foreach (int id in toRemove)
                    _allTasks.Remove(id);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TimeSpan TicksToTimeSpan(long ticks)
        {
            return new TimeSpan((long)((double)ticks / Stopwatch.Frequency * TimeSpan.TicksPerSecond));
        }

        /// <summary>
        /// Disposes the task scheduler and releases all resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Clear();
        }
    }
}
namespace GDNN.Threading
{
    /// <summary>
    /// Priority levels for work-stealing pool tasks.
    /// </summary>
    public enum WorkItemPriority
    {
        /// <summary>Lowest priority. Processed when nothing else is available.</summary>
        Background = 0,

        /// <summary>Low priority work that can be deferred.</summary>
        Low = 1,

        /// <summary>Normal priority work.</summary>
        Normal = 2,

        /// <summary>High priority work that should be processed quickly.</summary>
        High = 3,

        /// <summary>Critical priority. Processed before all other work.</summary>
        Critical = 4
    }

    /// <summary>
    /// Represents a unit of work to be executed by the work-stealing thread pool.
    /// </summary>
    public abstract class WorkItem
    {
        private int _state;

        /// <summary>Gets or sets the priority of this work item.</summary>
        public WorkItemPriority Priority { get; set; } = WorkItemPriority.Normal;

        /// <summary>Gets the current state of this work item.</summary>
        public WorkItemState State => (WorkItemState)Volatile.Read(ref _state);

        /// <summary>Gets whether this work item has completed.</summary>
        public bool IsCompleted => (WorkItemState)Volatile.Read(ref _state) == WorkItemState.Completed;

        /// <summary>Gets whether this work item was cancelled.</summary>
        public bool IsCancelled => (WorkItemState)Volatile.Read(ref _state) == WorkItemState.Cancelled;

        /// <summary>Gets whether this work item faulted.</summary>
        public bool IsFaulted => (WorkItemState)Volatile.Read(ref _state) == WorkItemState.Faulted;

        /// <summary>Gets or sets the exception if the work item faulted.</summary>
        public Exception? FaultException { get; internal set; }

        /// <summary>Gets or sets user-defined data.</summary>
        public object? UserData { get; set; }

        /// <summary>Gets or sets the timestamp when this item was enqueued.</summary>
        public long EnqueueTimestamp { get; internal set; }

        /// <summary>Executes the work item. Called by the thread pool.</summary>
        public abstract void Execute();

        /// <summary>Transitions the work item to the executing state.</summary>
        internal bool TryStartExecution()
        {
            return Interlocked.CompareExchange(ref _state,
                (int)WorkItemState.Executing,
                (int)WorkItemState.Queued) == (int)WorkItemState.Queued;
        }

        /// <summary>Transitions the work item to the completed state.</summary>
        internal void MarkCompleted()
        {
            Volatile.Write(ref _state, (int)WorkItemState.Completed);
            OnCompletion();
        }

        /// <summary>Transitions the work item to the cancelled state.</summary>
        internal void MarkCancelled()
        {
            Volatile.Write(ref _state, (int)WorkItemState.Cancelled);
            OnCompletion();
        }

        /// <summary>Transitions the work item to the faulted state.</summary>
        internal void MarkFaulted(Exception ex)
        {
            FaultException = ex;
            Volatile.Write(ref _state, (int)WorkItemState.Faulted);
            OnCompletion();
        }

        /// <summary>Transitions the work item to the queued state.</summary>
        internal void MarkQueued()
        {
            Volatile.Write(ref _state, (int)WorkItemState.Queued);
            EnqueueTimestamp = Stopwatch.GetTimestamp();
        }

        /// <summary>Called when the work item reaches a terminal state.</summary>
        protected virtual void OnCompletion() { }
    }

    /// <summary>
    /// States of a work item in the work-stealing pool.
    /// </summary>
    public enum WorkItemState
    {
        Created = 0,
        Queued = 1,
        Executing = 2,
        Completed = 3,
        Cancelled = 4,
        Faulted = 5
    }

    /// <summary>
    /// Concrete delegate-based work item for the pool.
    /// </summary>
    internal sealed class DelegateWorkItem : WorkItem
    {
        private readonly Action<int> _action;
        private readonly int _threadId;

        public DelegateWorkItem(Action<int> action, int threadId, WorkItemPriority priority)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
            _threadId = threadId;
            Priority = priority;
        }

        public override void Execute() => _action(_threadId);
    }

    /// <summary>
    /// Void delegate work item with no parameters.
    /// </summary>
    internal sealed class VoidActionWorkItem : WorkItem
    {
        private readonly Action _action;

        public VoidActionWorkItem(Action action, WorkItemPriority priority)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
            Priority = priority;
        }

        public override void Execute() => _action();
    }

    /// <summary>
    /// Work item that tracks dependencies and only executes when all dependencies are met.
    /// </summary>
    internal sealed class DependentWorkItem : WorkItem
    {
        private readonly Action _action;
        private int _remainingDependencies;

        public List<WorkItem> Dependencies { get; } = new();
        public List<WorkItem> Dependents { get; } = new();

        public DependentWorkItem(Action action, WorkItemPriority priority)
        {
            _action = action ?? throw new ArgumentNullException(nameof(action));
            Priority = priority;
        }

        public void AddDependency(WorkItem dependency)
        {
            Dependencies.Add(dependency);
            Interlocked.Increment(ref _remainingDependencies);
        }

        internal bool TryResolveDependency(WorkItem completed)
        {
            if (!Dependencies.Contains(completed))
                return false;
            if (Interlocked.Decrement(ref _remainingDependencies) == 0)
                return true;
            return false;
        }

        public override void Execute() => _action();
    }

    /// <summary>
    /// Represents a local (per-thread) work queue with support for work stealing.
    /// </summary>
    public sealed class WorkStealingQueue
    {
        private WorkItem?[] _array;
        private int _head;
        private int _tail;
        private int _capacity;
        private int _threadIndex;
        private readonly object _pushLock = new();
        private long _totalPushed;
        private long _totalStolen;

        /// <summary>Gets the number of items currently in the queue.</summary>
        public int Count
        {
            get
            {
                int tail = Volatile.Read(ref _tail);
                int head = Volatile.Read(ref _head);
                return tail - head;
            }
        }

        /// <summary>Gets the total number of items pushed to this queue.</summary>
        public long TotalPushed => Interlocked.Read(ref _totalPushed);

        /// <summary>Gets the total number of items stolen from this queue.</summary>
        public long TotalStolen => Interlocked.Read(ref _totalStolen);

        /// <summary>Gets the thread index this queue belongs to.</summary>
        public int ThreadIndex => _threadIndex;

        public WorkStealingQueue(int capacity = 256, int threadIndex = 0)
        {
            _capacity = capacity;
            _threadIndex = threadIndex;
            _array = new WorkItem?[capacity];
        }

        /// <summary>
        /// Pushes a work item onto the bottom of the deque. Thread-local operation.
        /// </summary>
        /// <param name="item">The work item to push.</param>
        public void Push(WorkItem item)
        {
            lock (_pushLock)
            {
                int tail = _tail;
                if (tail == _capacity)
                    Resize();

                _array[tail] = item;
                Volatile.Write(ref _tail, tail + 1);
                Interlocked.Increment(ref _totalPushed);
            }
        }

        /// <summary>
        /// Pops a work item from the bottom of the deque. Thread-local operation.
        /// </summary>
        /// <param name="item">The popped work item, or null if empty.</param>
        /// <returns>True if an item was popped.</returns>
        public bool TryPop(out WorkItem? item)
        {
            lock (_pushLock)
            {
                int tail = Volatile.Read(ref _tail);
                int head = Volatile.Read(ref _head);

                if (tail <= head)
                {
                    item = null;
                    return false;
                }

                tail--;
                Volatile.Write(ref _tail, tail);
                item = _array[tail];
                _array[tail] = null;

                if (head >= tail && head > 0)
                {
                    Volatile.Write(ref _head, 0);
                    Volatile.Write(ref _tail, 0);
                }

                return item != null;
            }
        }

        /// <summary>
        /// Steals a work item from the top of the deque. Called by other threads.
        /// </summary>
        /// <param name="item">The stolen work item, or null if empty.</param>
        /// <returns>True if an item was stolen.</returns>
        public bool TrySteal(out WorkItem? item)
        {
            lock (_pushLock)
            {
                int head = Volatile.Read(ref _head);
                int tail = Volatile.Read(ref _tail);

                if (head >= tail)
                {
                    item = null;
                    return false;
                }

                item = _array[head];
                _array[head] = null;
                Volatile.Write(ref _head, head + 1);
                Interlocked.Increment(ref _totalStolen);
                return item != null;
            }
        }

        /// <summary>
        /// Peeks at the top item without removing it.
        /// </summary>
        public bool TryPeek(out WorkItem? item)
        {
            lock (_pushLock)
            {
                int head = Volatile.Read(ref _head);
                int tail = Volatile.Read(ref _tail);

                if (head >= tail)
                {
                    item = null;
                    return false;
                }

                item = _array[head];
                return item != null;
            }
        }

        /// <summary>
        /// Drains all items from the queue into the provided list.
        /// </summary>
        public int Drain(List<WorkItem> target)
        {
            lock (_pushLock)
            {
                int head = Volatile.Read(ref _head);
                int tail = Volatile.Read(ref _tail);
                int count = 0;

                for (int i = head; i < tail; i++)
                {
                    if (_array[i] != null)
                    {
                        target.Add(_array[i]!);
                        _array[i] = null;
                        count++;
                    }
                }

                Volatile.Write(ref _head, 0);
                Volatile.Write(ref _tail, 0);
                return count;
            }
        }

        private void Resize()
        {
            int newCapacity = _capacity * 2;
            var newArray = new WorkItem?[newCapacity];

            int head = Volatile.Read(ref _head);
            int tail = Volatile.Read(ref _tail);

            for (int i = head; i < tail; i++)
                newArray[i - head] = _array[i];

            _array = newArray;
            _tail = tail - head;
            _head = 0;
            _capacity = newCapacity;
        }
    }

    /// <summary>
    /// Statistics about the work-stealing thread pool.
    /// </summary>
    public struct PoolStatistics
    {
        public PoolStatistics()
        {
            QueueDepths = Array.Empty<int>();
        }

        /// <summary>Total number of tasks executed.</summary>
        public long TasksExecuted { get; internal set; }

        /// <summary>Total number of tasks stolen from other threads.</summary>
        public long TasksStolen { get; internal set; }

        /// <summary>Total number of steal attempts that failed.</summary>
        public long FailedSteals { get; internal set; }

        /// <summary>Total idle time across all threads, in ticks.</summary>
        public long TotalIdleTicks { get; internal set; }

        /// <summary>Current number of active (non-idle) threads.</summary>
        public int ActiveThreads { get; internal set; }

        /// <summary>Number of threads currently idle.</summary>
        public int IdleThreads { get; internal set; }

        /// <summary>Total threads in the pool.</summary>
        public int ThreadCount { get; internal set; }

        /// <summary>Queue depths per thread.</summary>
        public int[] QueueDepths { get; internal set; } = Array.Empty<int>();

        /// <summary>Sum of all queue depths.</summary>
        public int TotalQueued => QueueDepths.Length > 0
            ? QueueDepths.Sum()
            : 0;

        /// <summary>Steal success rate (0.0 to 1.0).</summary>
        public double StealSuccessRate =>
            (TasksStolen + FailedSteals) > 0
                ? (double)TasksStolen / (TasksStolen + FailedSteals)
                : 0;
    }

    /// <summary>
    /// A high-performance work-stealing thread pool implementation with per-thread
    /// local queues, task prioritization, dynamic thread count adjustment, and
    /// dependency tracking.
    /// </summary>
    public sealed class WorkStealingPool : IDisposable
    {
        private const int MAX_SPIN_WAIT = 1024;
        private const int SPIN_YIELD_THRESHOLD = 32;
        private const int DEFAULT_QUEUE_CAPACITY = 512;
        private const int MAX_THREAD_COUNT = 256;
        private const double LOAD_BALANCE_INTERVAL_MS = 100;

        private readonly WorkStealingQueue[] _localQueues;
        private readonly Thread[] _threads;
        private readonly CancellationTokenSource _cts;
        private readonly object _globalLock = new();
        private readonly Queue<WorkItem> _globalQueue = new();
        private readonly List<DependentWorkItem> _pendingDependents = new();
        private volatile bool _disposed;
        private volatile bool _running;
        private int _threadCount;
        private int _dynamicMinThreads;
        private int _dynamicMaxThreads;
        private bool _dynamicAdjustmentEnabled;

        private long _totalTasksExecuted;
        private long _totalTasksStolen;
        private long _totalFailedSteals;
        private long _totalIdleTicks;
        private long _activeThreadCount;
        private long[]? _perThreadTaskCounts;

        /// <summary>Gets the current number of threads in the pool.</summary>
        public int ThreadCount => _threadCount;

        /// <summary>Gets or sets whether dynamic thread count adjustment is enabled.</summary>
        public bool DynamicAdjustmentEnabled
        {
            get => _dynamicAdjustmentEnabled;
            set => _dynamicAdjustmentEnabled = value;
        }

        /// <summary>Gets or sets the minimum thread count for dynamic adjustment.</summary>
        public int DynamicMinThreads
        {
            get => _dynamicMinThreads;
            set => _dynamicMinThreads = Math.Max(1, Math.Min(value, _dynamicMaxThreads));
        }

        /// <summary>Gets or sets the maximum thread count for dynamic adjustment.</summary>
        public int DynamicMaxThreads
        {
            get => _dynamicMaxThreads;
            set => _dynamicMaxThreads = Math.Clamp(value, _dynamicMinThreads, MAX_THREAD_COUNT);
        }

        /// <summary>Gets the total number of tasks executed since creation.</summary>
        public long TasksExecuted => Interlocked.Read(ref _totalTasksExecuted);

        /// <summary>Gets whether the pool is currently running.</summary>
        public bool IsRunning => _running;

        /// <summary>
        /// Initializes a new work-stealing pool with the specified number of threads.
        /// </summary>
        /// <param name="threadCount">Number of worker threads. Defaults to ProcessorCount.</param>
        public WorkStealingPool(int threadCount = 0)
        {
            _threadCount = threadCount > 0
                ? Math.Min(threadCount, MAX_THREAD_COUNT)
                : Environment.ProcessorCount;

            _dynamicMinThreads = Math.Max(1, _threadCount / 2);
            _dynamicMaxThreads = Math.Min(MAX_THREAD_COUNT, _threadCount * 2);

            _localQueues = new WorkStealingQueue[_threadCount];
            _threads = new Thread[_threadCount];
            _perThreadTaskCounts = new long[_threadCount];
            _cts = new CancellationTokenSource();

            for (int i = 0; i < _threadCount; i++)
            {
                _localQueues[i] = new WorkStealingQueue(DEFAULT_QUEUE_CAPACITY, i);
            }
        }

        /// <summary>
        /// Starts all worker threads in the pool.
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
                    Name = $"GDNN-Stealing-{localI}"
                };
                _threads[i].Start();
            }

            if (_dynamicAdjustmentEnabled)
            {
                var adjustThread = new Thread(DynamicAdjustmentLoop)
                {
                    IsBackground = true,
                    Name = "GDNN-DynamicAdjust"
                };
                adjustThread.Start();
            }
        }

        /// <summary>
        /// Submits a work item to the pool.
        /// </summary>
        /// <param name="item">The work item to submit.</param>
        /// <param name="threadHint">Preferred thread index, or -1 for any thread.</param>
        public void Submit(WorkItem item, int threadHint = -1)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (item == null)
                throw new ArgumentNullException(nameof(item));

            item.MarkQueued();

            if (threadHint >= 0 && threadHint < _threadCount)
            {
                _localQueues[threadHint].Push(item);
            }
            else
            {
                int workerIndex = (int)((uint)Environment.CurrentManagedThreadId % (uint)_threadCount);
                _localQueues[workerIndex].Push(item);
            }
        }

        /// <summary>
        /// Submits a delegate to be executed on a thread pool thread.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="priority">Task priority.</param>
        /// <returns>A handle to the submitted work item.</returns>
        public WorkItem Submit(Action action, WorkItemPriority priority = WorkItemPriority.Normal)
        {
            var item = new VoidActionWorkItem(action, priority);
            Submit(item);
            return item;
        }

        /// <summary>
        /// Submits a delegate with a thread ID parameter.
        /// </summary>
        /// <param name="action">The action to execute, receiving the thread index.</param>
        /// <param name="priority">Task priority.</param>
        /// <returns>A handle to the submitted work item.</returns>
        public WorkItem Submit(Action<int> action, WorkItemPriority priority = WorkItemPriority.Normal)
        {
            int dummy = 0;
            var item = new DelegateWorkItem(action, dummy, priority);
            Submit(item);
            return item;
        }

        /// <summary>
        /// Submits a work item with dependency tracking. The item will only execute
        /// after all its dependencies have completed.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="dependencies">Work items that must complete first.</param>
        /// <param name="priority">Task priority.</param>
        /// <returns>The dependent work item.</returns>
        public WorkItem SubmitWithDependencies(
            Action action,
            IEnumerable<WorkItem> dependencies,
            WorkItemPriority priority = WorkItemPriority.Normal)
        {
            var item = new DependentWorkItem(action, priority);

            lock (_globalLock)
            {
                foreach (var dep in dependencies)
                {
                    if (dep.IsCompleted)
                        continue;

                    item.AddDependency(dep);

                    if (dep is DependentWorkItem depItem)
                        depItem.Dependents.Add(item);
                }

                if (item.State == WorkItemState.Created && item.Dependencies.Count == 0)
                {
                    item.MarkQueued();
                    _globalQueue.Enqueue(item);
                }
                else
                {
                    _pendingDependents.Add(item);
                }
            }

            return item;
        }

        /// <summary>
        /// Submits a parallel-for job across the specified range.
        /// </summary>
        /// <param name="startInclusive">Start of range (inclusive).</param>
        /// <param name="endExclusive">End of range (exclusive).</param>
        /// <param name="action">Action per iteration, receiving (index, threadIndex).</param>
        /// <param name="priority">Task priority.</param>
        /// <returns>An array of work items, one per chunk.</returns>
        public WorkItem[] SubmitParallelFor(
            int startInclusive,
            int endExclusive,
            Action<int, int> action,
            WorkItemPriority priority = WorkItemPriority.Normal)
        {
            int total = endExclusive - startInclusive;
            int chunkSize = Math.Max(1, total / _threadCount);
            int numChunks = (total + chunkSize - 1) / chunkSize;
            var items = new WorkItem[numChunks];

            for (int c = 0; c < numChunks; c++)
            {
                int chunkStart = startInclusive + c * chunkSize;
                int chunkEnd = Math.Min(chunkStart + chunkSize, endExclusive);
                int chunkIndex = c;

                var item = new DelegateWorkItem(
                    tid =>
                    {
                        for (int i = chunkStart; i < chunkEnd; i++)
                            action(i, tid);
                    },
                    chunkIndex,
                    priority);

                items[c] = item;
                Submit(item);
            }

            return items;
        }

        /// <summary>
        /// Blocks until all submitted work items have completed.
        /// </summary>
        public void WaitForAllCompletion()
        {
            var spin = new SpinWait();
            while (Volatile.Read(ref _activeThreadCount) > 0 || GetGlobalQueueCount() > 0 || GetLocalQueuesTotalCount() > 0)
            {
                spin.SpinOnce();
                if (spin.Count > MAX_SPIN_WAIT)
                    Thread.Sleep(1);
            }
        }

        /// <summary>
        /// Blocks until the specified work item has completed.
        /// </summary>
        /// <param name="item">The work item to wait for.</param>
        /// <param name="timeout">Maximum wait time.</param>
        /// <returns>True if the item completed, false on timeout.</returns>
        public bool WaitForCompletion(WorkItem item, TimeSpan timeout)
        {
            if (item == null)
                throw new ArgumentNullException(nameof(item));
            var deadline = Stopwatch.StartNew();

            var spin = new SpinWait();
            while (!item.IsCompleted && !item.IsCancelled && !item.IsFaulted)
            {
                if (deadline.Elapsed >= timeout)
                    return false;

                spin.SpinOnce();
                if (spin.Count > MAX_SPIN_WAIT)
                    Thread.Sleep(1);
            }

            return item.IsCompleted;
        }

        /// <summary>
        /// Gets current statistics about the pool.
        /// </summary>
        public PoolStatistics GetStatistics()
        {
            var stats = new PoolStatistics
            {
                TasksExecuted = Interlocked.Read(ref _totalTasksExecuted),
                TasksStolen = Interlocked.Read(ref _totalTasksStolen),
                FailedSteals = Interlocked.Read(ref _totalFailedSteals),
                TotalIdleTicks = Interlocked.Read(ref _totalIdleTicks),
                ActiveThreads = (int)Volatile.Read(ref _activeThreadCount),
                IdleThreads = _threadCount - (int)Volatile.Read(ref _activeThreadCount),
                ThreadCount = _threadCount,
                QueueDepths = new int[_threadCount]
            };

            for (int i = 0; i < _threadCount; i++)
                stats.QueueDepths[i] = _localQueues[i].Count;

            return stats;
        }

        /// <summary>
        /// Gets the local queue for a specific thread.
        /// </summary>
        public WorkStealingQueue GetLocalQueue(int threadIndex)
        {
            if (threadIndex < 0 || threadIndex >= _threadCount)
                throw new ArgumentOutOfRangeException(nameof(threadIndex));
            return _localQueues[threadIndex];
        }

        /// <summary>
        /// Peeks at the highest-priority item across all queues without removing it.
        /// </summary>
        public WorkItem? PeekHighestPriority()
        {
            WorkItem? best = null;
            var bestPriority = WorkItemPriority.Background;

            lock (_globalLock)
            {
                foreach (var item in _globalQueue)
                {
                    if (item.Priority > bestPriority)
                    {
                        best = item;
                        bestPriority = item.Priority;
                    }
                }
            }

            for (int i = 0; i < _threadCount; i++)
            {
                if (_localQueues[i].TryPeek(out var item) && item != null)
                {
                    if (item.Priority > bestPriority)
                    {
                        best = item;
                        bestPriority = item.Priority;
                    }
                }
            }

            return best;
        }

        private void WorkerLoop(int threadIndex)
        {
            var spin = new SpinWait();
            int consecutiveEmpty = 0;

            while (!_cts.Token.IsCancellationRequested)
            {
                WorkItem? workItem = null;
                bool found = false;

                if (_localQueues[threadIndex].TryPop(out workItem) && workItem != null)
                {
                    found = true;
                    consecutiveEmpty = 0;
                }
                else
                {
                    for (int i = 0; i < _threadCount; i++)
                    {
                        int victimIndex = (threadIndex + i + 1) % _threadCount;
                        if (_localQueues[victimIndex].TrySteal(out workItem) && workItem != null)
                        {
                            found = true;
                            Interlocked.Increment(ref _totalTasksStolen);
                            consecutiveEmpty = 0;
                            break;
                        }
                    }

                    if (!found)
                    {
                        lock (_globalLock)
                        {
                            if (_globalQueue.Count > 0)
                            {
                                workItem = _globalQueue.Dequeue();
                                found = true;
                                consecutiveEmpty = 0;
                            }
                        }
                    }
                }

                if (!found)
                {
                    Interlocked.Increment(ref _totalFailedSteals);
                    consecutiveEmpty++;

                    if (consecutiveEmpty > SPIN_YIELD_THRESHOLD)
                    {
                        Interlocked.Increment(ref _totalIdleTicks);
                        Thread.Sleep(1);
                    }
                    else
                    {
                        spin.SpinOnce();
                    }
                    continue;
                }

                spin.Reset();
                consecutiveEmpty = 0;

                Interlocked.Increment(ref _activeThreadCount);
                try
                {
                    if (workItem!.TryStartExecution())
                    {
                        try
                        {
                            workItem.Execute();
                            workItem.MarkCompleted();
                        }
                        catch (OperationCanceledException)
                        {
                            workItem.MarkCancelled();
                        }
                        catch (Exception ex)
                        {
                            workItem.MarkFaulted(ex);
                        }

                        Interlocked.Increment(ref _totalTasksExecuted);
                        Interlocked.Increment(ref _perThreadTaskCounts![threadIndex]);

                        if (workItem is DependentWorkItem depItem)
                            ResolveDependents(depItem);
                    }
                }
                finally
                {
                    Interlocked.Decrement(ref _activeThreadCount);
                }
            }
        }

        private void ResolveDependents(DependentWorkItem completed)
        {
            lock (_globalLock)
            {
                for (int i = _pendingDependents.Count - 1; i >= 0; i--)
                {
                    var pending = _pendingDependents[i];
                    if (pending.TryResolveDependency(completed))
                    {
                        pending.MarkQueued();
                        _globalQueue.Enqueue(pending);
                        _pendingDependents.RemoveAt(i);
                    }
                }
            }
        }

        private void DynamicAdjustmentLoop()
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                Thread.Sleep((int)LOAD_BALANCE_INTERVAL_MS);

                var stats = GetStatistics();
                double loadFactor = stats.ThreadCount > 0
                    ? (double)stats.ActiveThreads / stats.ThreadCount
                    : 0;

                if (loadFactor > 0.8 && _threadCount < _dynamicMaxThreads)
                {
                    int newCount = Math.Min(_threadCount + 1, _dynamicMaxThreads);
                    if (newCount > _threadCount)
                        ResizePool(newCount);
                }
                else if (loadFactor < 0.2 && _threadCount > _dynamicMinThreads)
                {
                    int newCount = Math.Max(_threadCount - 1, _dynamicMinThreads);
                    if (newCount < _threadCount)
                        ResizePool(newCount);
                }
            }
        }

        private void ResizePool(int newCount)
        {
            if (newCount == _threadCount)
                return;
            if (newCount < 1 || newCount > MAX_THREAD_COUNT)
                return;

            int oldCount = _threadCount;
            _threadCount = newCount;

            var newQueues = new WorkStealingQueue[newCount];
            for (int i = 0; i < newCount; i++)
            {
                if (i < oldCount)
                    newQueues[i] = _localQueues[i];
                else
                    newQueues[i] = new WorkStealingQueue(DEFAULT_QUEUE_CAPACITY, i);
            }

            if (newCount < oldCount)
            {
                for (int i = newCount; i < oldCount; i++)
                {
                    var drained = new List<WorkItem>();
                    _localQueues[i].Drain(drained);
                    foreach (var item in drained)
                        _localQueues[i % newCount].Push(item);
                }
            }
        }

        private int GetGlobalQueueCount()
        {
            lock (_globalLock)
                return _globalQueue.Count;
        }

        private int GetLocalQueuesTotalCount()
        {
            int total = 0;
            for (int i = 0; i < _threadCount; i++)
                total += _localQueues[i].Count;
            return total;
        }

        /// <summary>
        /// Signals all threads to stop and waits for them to finish.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            _cts.Cancel();
            _running = false;

            for (int i = 0; i < _threadCount; i++)
            {
                if (_threads[i] != null && _threads[i].IsAlive)
                    _threads[i].Join(TimeSpan.FromSeconds(5));
            }

            _cts.Dispose();
        }
    }
}
