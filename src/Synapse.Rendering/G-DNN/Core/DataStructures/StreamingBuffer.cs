using System;
// ============================================================
// FILE: StreamingBuffer.cs
// PATH: Core/DataStructures/StreamingBuffer.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO;
using System.IO.Compression;
using System.IO.MemoryMappedFiles;
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

namespace GDNN.Core.DataStructures;

/// <summary>
/// A high-performance ring buffer for streaming neural asset data to the GPU.
/// Supports double-buffering for async uploads, lock-free producer-consumer patterns,
/// and optional memory-mapped file backing for large datasets.
/// </summary>
/// <typeparam name="T">The element type stored in the buffer. Must be unmanaged.</typeparam>
public sealed unsafe class StreamingBuffer<T> : IDisposable
    where T : unmanaged
{
    private readonly T* _bufferA;
    private readonly T* _bufferB;
    private T* _frontBuffer;
    private T* _backBuffer;

    private readonly int _capacity;
    private readonly int _elementSize;
    private readonly int _alignment;
    private long _writePosition;
    private long _readPosition;
    private long _flushedPosition;
    private int _activeBuffer;

    private readonly object _swapLock = new();
    private int _isSwapping;
    private bool _disposed;
    private bool _usingMemoryMapped;

    private MemoryMappedFile? _mappedFile;
    private MemoryMappedViewAccessor? _mappedAccessor;
    private long _mappedViewOffset;

    /// <summary>Capacity in elements.</summary>
    public int Capacity => _capacity;

    /// <summary>Total buffer size in bytes per buffer.</summary>
    public int SizeInBytes => _capacity * _elementSize;

    /// <summary>Total memory usage (both buffers).</summary>
    public long TotalMemoryUsage => (long)SizeInBytes * 2;

    /// <summary>Number of elements currently available for reading.</summary>
    public long AvailableToRead => Volatile.Read(ref _flushedPosition) - Volatile.Read(ref _readPosition);

    /// <summary>Number of elements available for writing.</summary>
    public long AvailableToWrite => _capacity - (Volatile.Read(ref _writePosition) - Volatile.Read(ref _readPosition));

    /// <summary>Current write position in elements.</summary>
    public long WritePosition => Volatile.Read(ref _writePosition);

    /// <summary>Current read position in elements.</summary>
    public long ReadPosition => Volatile.Read(ref _readPosition);

    /// <summary>Flushed position (available for consumer).</summary>
    public long FlushedPosition => Volatile.Read(ref _flushedPosition);

    /// <summary>Whether the buffer is empty.</summary>
    public bool IsEmpty => AvailableToRead <= 0;

    /// <summary>Whether the buffer is full.</summary>
    public bool IsFull => AvailableToWrite <= 0;

    /// <summary>The memory alignment in bytes.</summary>
    public int Alignment => _alignment;

    /// <summary>The active buffer index (0 or 1).</summary>
    public int ActiveBuffer => _activeBuffer;

    /// <summary>Whether a swap is in progress.</summary>
    public bool IsSwapping => Volatile.Read(ref _isSwapping) == 1;

    /// <summary>Whether the buffer is memory-mapped.</summary>
    public bool IsMemoryMapped => _usingMemoryMapped;

    /// <summary>
    /// Initializes a new streaming buffer with native memory allocation.
    /// </summary>
    /// <param name="capacity">Number of elements.</param>
    /// <param name="alignment">Memory alignment in bytes (default 64 for cache line).</param>
    public StreamingBuffer(int capacity, int alignment = 64)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(capacity, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(alignment, sizeof(T));

        _capacity = capacity;
        _elementSize = sizeof(T);
        _alignment = alignment;
        _writePosition = 0;
        _readPosition = 0;
        _flushedPosition = 0;
        _activeBuffer = 0;
        _usingMemoryMapped = false;

        _bufferA = (T*)NativeMemory.AlignedAlloc((nuint)(capacity * _elementSize), (nuint)alignment);
        _bufferB = (T*)NativeMemory.AlignedAlloc((nuint)(capacity * _elementSize), (nuint)alignment);

        NativeMemory.Clear(_bufferA, (nuint)(capacity * _elementSize));
        NativeMemory.Clear(_bufferB, (nuint)(capacity * _elementSize));

        _frontBuffer = _bufferA;
        _backBuffer = _bufferB;
    }

    /// <summary>
    /// Initializes with memory-mapped file backing for very large datasets.
    /// </summary>
    /// <param name="filePath">Path to the backing file.</param>
    /// <param name="capacity">Number of elements.</param>
    /// <param name="alignment">Memory alignment.</param>
    public StreamingBuffer(string filePath, int capacity, int alignment = 64)
    {
        ArgumentOutOfRangeException.ThrowIfNullOrWhiteSpace(filePath);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(capacity, 0);

        _capacity = capacity;
        _elementSize = sizeof(T);
        _alignment = alignment;
        _writePosition = 0;
        _readPosition = 0;
        _flushedPosition = 0;
        _activeBuffer = 0;
        _usingMemoryMapped = true;

        long fileSize = (long)capacity * _elementSize * 2;

        _mappedFile = MemoryMappedFile.CreateFromFile(
            filePath, FileMode.Create, null, fileSize, MemoryMappedFileAccess.ReadWrite);
        _mappedAccessor = _mappedFile.CreateViewAccessor(0, fileSize, MemoryMappedFileAccess.ReadWrite);

        bool addedRef = false;
        try
        {
            _mappedAccessor.SafeMemoryMappedViewHandle.DangerousAddRef(ref addedRef);
            nint handle = _mappedAccessor.SafeMemoryMappedViewHandle.DangerousGetHandle();

            _bufferA = (T*)handle;
            _bufferB = (T*)(handle + capacity * _elementSize);

            NativeMemory.Clear(_bufferA, (nuint)(capacity * _elementSize));
            NativeMemory.Clear(_bufferB, (nuint)(capacity * _elementSize));

            _frontBuffer = _bufferA;
            _backBuffer = _bufferB;
        }
        finally
        {
            if (addedRef)
                _mappedAccessor.SafeMemoryMappedViewHandle.DangerousRelease();
        }
    }

    /// <summary>
    /// Writes a single element to the buffer.
    /// </summary>
    /// <param name="item">The element to write.</param>
    /// <returns>True if written successfully; false if buffer is full.</returns>
    public bool TryWrite(in T item)
    {
        long writePos = Volatile.Read(ref _writePosition);
        long readPos = Volatile.Read(ref _readPosition);

        if (writePos - readPos >= _capacity)
            return false;

        int index = (int)(writePos % _capacity);
        _frontBuffer[index] = item;

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _writePosition, writePos + 1);
        return true;
    }

    /// <summary>
    /// Writes multiple elements to the buffer.
    /// </summary>
    /// <param name="data">Span of elements to write.</param>
    /// <returns>Number of elements actually written.</returns>
    public int Write(ReadOnlySpan<T> data)
    {
        long writePos = Volatile.Read(ref _writePosition);
        long readPos = Volatile.Read(ref _readPosition);

        long available = _capacity - (writePos - readPos);
        int toWrite = (int)Math.Min(data.Length, available);

        if (toWrite <= 0)
            return 0;

        int startIndex = (int)(writePos % _capacity);

        if (startIndex + toWrite <= _capacity)
        {
            data.Slice(0, toWrite).CopyTo(new Span<T>(_frontBuffer + startIndex, toWrite));
        }
        else
        {
            int firstPart = _capacity - startIndex;
            data.Slice(0, firstPart).CopyTo(new Span<T>(_frontBuffer + startIndex, firstPart));
            data.Slice(firstPart, toWrite - firstPart).CopyTo(new Span<T>(_frontBuffer, toWrite - firstPart));
        }

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _writePosition, writePos + toWrite);
        return toWrite;
    }

    /// <summary>
    /// Writes elements from a native pointer.
    /// </summary>
    /// <param name="source">Source pointer.</param>
    /// <param name="count">Number of elements to write.</param>
    /// <returns>Number of elements actually written.</returns>
    public int Write(T* source, int count)
    {
        return Write(new ReadOnlySpan<T>(source, count));
    }

    /// <summary>
    /// Reads a single element from the buffer.
    /// </summary>
    /// <param name="item">The element read.</param>
    /// <returns>True if an element was available; false if empty.</returns>
    public bool TryRead(out T item)
    {
        item = default;

        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        if (readPos >= flushedPos)
            return false;

        int index = (int)(readPos % _capacity);
        item = _frontBuffer[index];

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _readPosition, readPos + 1);
        return true;
    }

    /// <summary>
    /// Reads multiple elements from the buffer.
    /// </summary>
    /// <param name="destination">Span to read into.</param>
    /// <returns>Number of elements actually read.</returns>
    public int Read(Span<T> destination)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos;
        int toRead = (int)Math.Min(destination.Length, available);

        if (toRead <= 0)
            return 0;

        int startIndex = (int)(readPos % _capacity);

        if (startIndex + toRead <= _capacity)
        {
            new Span<T>(_frontBuffer + startIndex, toRead).CopyTo(destination);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            new Span<T>(_frontBuffer + startIndex, firstPart).CopyTo(destination);
            new Span<T>(_frontBuffer, toRead - firstPart).CopyTo(destination.Slice(firstPart));
        }

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _readPosition, readPos + toRead);
        return toRead;
    }

    /// <summary>
    /// Reads elements into a native pointer.
    /// </summary>
    /// <param name="destination">Destination pointer.</param>
    /// <param name="count">Maximum elements to read.</param>
    /// <returns>Number of elements actually read.</returns>
    public int Read(T* destination, int count)
    {
        return Read(new Span<T>(destination, count));
    }

    /// <summary>
    /// Peeks at the next element without consuming it.
    /// </summary>
    /// <param name="item">The element to peek at.</param>
    /// <returns>True if an element was available; false if empty.</returns>
    public bool TryPeek(out T item)
    {
        item = default;

        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        if (readPos >= flushedPos)
            return false;

        int index = (int)(readPos % _capacity);
        item = _frontBuffer[index];
        return true;
    }

    /// <summary>
    /// Peeks at multiple elements without consuming them.
    /// </summary>
    /// <param name="destination">Span to peek into.</param>
    /// <param name="offset">Number of elements to skip before peeking.</param>
    /// <returns>Number of elements peeked.</returns>
    public int Peek(Span<T> destination, int offset = 0)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos - offset;
        int toPeek = (int)Math.Min(destination.Length, available);

        if (toPeek <= 0)
            return 0;

        int startIndex = (int)((readPos + offset) % _capacity);

        if (startIndex + toPeek <= _capacity)
        {
            new Span<T>(_frontBuffer + startIndex, toPeek).CopyTo(destination);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            new Span<T>(_frontBuffer + startIndex, firstPart).CopyTo(destination);
            new Span<T>(_frontBuffer, toPeek - firstPart).CopyTo(destination.Slice(firstPart));
        }

        return toPeek;
    }

    /// <summary>
    /// Advances the read position without reading data (skips elements).
    /// </summary>
    /// <param name="count">Number of elements to skip.</param>
    /// <returns>Number of elements actually skipped.</returns>
    public long Advance(long count)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos;
        long toAdvance = Math.Min(count, available);

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _readPosition, readPos + toAdvance);
        return toAdvance;
    }

    /// <summary>
    /// Advances the write position without writing data (reserves space).
    /// Caller must write to the returned span before the next read/flush.
    /// </summary>
    /// <param name="count">Number of elements to reserve.</param>
    /// <returns>Span pointing to the reserved region (may be partial on wrap).</returns>
    public Span<T> Reserve(int count)
    {
        long writePos = Volatile.Read(ref _writePosition);
        long readPos = Volatile.Read(ref _readPosition);

        if (writePos - readPos + count > _capacity)
            return Span<T>.Empty;

        int startIndex = (int)(writePos % _capacity);

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _writePosition, writePos + count);

        if (startIndex + count <= _capacity)
        {
            return new Span<T>(_frontBuffer + startIndex, count);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            return new Span<T>(_frontBuffer + startIndex, firstPart);
        }
    }

    /// <summary>
    /// Flushes the write buffer, making written data available for reading.
    /// Call this after writing via Reserve or direct pointer access.
    /// </summary>
    public void Flush()
    {
        long writePos = Volatile.Read(ref _writePosition);
        Interlocked.MemoryBarrier();
        Volatile.Write(ref _flushedPosition, writePos);
    }

    /// <summary>
    /// Resets the buffer positions to the beginning.
    /// </summary>
    public void Reset()
    {
        lock (_swapLock)
        {
            Volatile.Write(ref _writePosition, 0);
            Volatile.Write(ref _readPosition, 0);
            Volatile.Write(ref _flushedPosition, 0);
        }
    }

    /// <summary>
    /// Swaps front and back buffers for double-buffered GPU upload.
    /// The consumer should call this after consuming the current front buffer.
    /// </summary>
    public bool TrySwapBuffers()
    {
        if (Interlocked.CompareExchange(ref _isSwapping, 1, 0) == 1)
            return false;

        try
        {
            lock (_swapLock)
            {
                var tmp = _frontBuffer;
                _frontBuffer = _backBuffer;
                _backBuffer = tmp;
                _activeBuffer = _activeBuffer == 0 ? 1 : 0;

                // Reset write position for the new front buffer
                Volatile.Write(ref _writePosition, 0);
                Volatile.Write(ref _flushedPosition, 0);
            }
            return true;
        }
        finally
        {
            Volatile.Write(ref _isSwapping, 0);
        }
    }

    /// <summary>
    /// Gets a direct pointer to the front buffer for zero-copy GPU upload.
    /// </summary>
    /// <returns>Pointer to the beginning of the front buffer.</returns>
    public T* GetFrontBufferPointer()
    {
        return _frontBuffer;
    }

    /// <summary>
    /// Gets a direct pointer to the back buffer.
    /// </summary>
    public T* GetBackBufferPointer()
    {
        return _backBuffer;
    }

    /// <summary>
    /// Gets a span view of the front buffer.
    /// </summary>
    public Span<T> GetFrontBufferSpan()
    {
        return new Span<T>(_frontBuffer, _capacity);
    }

    /// <summary>
    /// Gets a span view of the back buffer.
    /// </summary>
    public Span<T> GetBackBufferSpan()
    {
        return new Span<T>(_backBuffer, _capacity);
    }

    /// <summary>
    /// Copies data from the front buffer to the back buffer.
    /// </summary>
    public void CopyFrontToBack()
    {
        int sizeInBytes = _capacity * _elementSize;
        Buffer.MemoryCopy(_frontBuffer, _backBuffer, sizeInBytes, sizeInBytes);
    }

    /// <summary>
    /// Copies data from the back buffer to the front buffer.
    /// </summary>
    public void CopyBackToFront()
    {
        int sizeInBytes = _capacity * _elementSize;
        Buffer.MemoryCopy(_backBuffer, _frontBuffer, sizeInBytes, sizeInBytes);
    }

    /// <summary>
    /// Copies a portion of the front buffer to a span.
    /// </summary>
    /// <param name="destination">Destination span.</param>
    /// <param name="offset">Offset in elements from the read position.</param>
    public void CopyTo(Span<T> destination, int offset = 0)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos - offset;
        int toCopy = (int)Math.Min(destination.Length, available);

        if (toCopy <= 0)
            return;

        int startIndex = (int)((readPos + offset) % _capacity);

        if (startIndex + toCopy <= _capacity)
        {
            new Span<T>(_frontBuffer + startIndex, toCopy).CopyTo(destination);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            new Span<T>(_frontBuffer + startIndex, firstPart).CopyTo(destination);
            new Span<T>(_frontBuffer, toCopy - firstPart).CopyTo(destination.Slice(firstPart));
        }
    }

    /// <summary>
    /// Gets the contiguous readable span (no wrap) starting from current read position.
    /// </summary>
    public Span<T> GetContiguousReadableSpan()
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos;
        if (available <= 0)
            return Span<T>.Empty;

        int startIndex = (int)(readPos % _capacity);
        int contiguous = (int)Math.Min(available, _capacity - startIndex);

        return new Span<T>(_frontBuffer + startIndex, contiguous);
    }

    /// <summary>
    /// Gets the contiguous writable span (no wrap) starting from current write position.
    /// </summary>
    public Span<T> GetContiguousWritableSpan()
    {
        long writePos = Volatile.Read(ref _writePosition);
        long readPos = Volatile.Read(ref _readPosition);

        long available = _capacity - (writePos - readPos);
        if (available <= 0)
            return Span<T>.Empty;

        int startIndex = (int)(writePos % _capacity);
        int contiguous = (int)Math.Min(available, _capacity - startIndex);

        return new Span<T>(_frontBuffer + startIndex, contiguous);
    }

    /// <summary>
    /// Fills the buffer with a default value.
    /// </summary>
    /// <param name="value">The value to fill with.</param>
    public void Fill(T value)
    {
        int sizeInBytes = _capacity * _elementSize;
        int elemCount = _capacity;

        Span<T> front = new Span<T>(_frontBuffer, elemCount);
        front.Fill(value);

        Span<T> back = new Span<T>(_backBuffer, elemCount);
        back.Fill(value);
    }

    /// <summary>
    /// Clears all data and resets positions.
    /// </summary>
    public void Clear()
    {
        NativeMemory.Clear(_bufferA, (nuint)(_capacity * _elementSize));
        NativeMemory.Clear(_bufferB, (nuint)(_capacity * _elementSize));
        Reset();
    }

    /// <summary>
    /// Attempts to write an element without locking (lock-free).
    /// Uses Interlocked operations for thread safety.
    /// </summary>
    /// <param name="item">The item to write.</param>
    /// <returns>True if written; false if buffer is full.</returns>
    public bool TryWriteLockFree(in T item)
    {
        long initialWritePos, initialReadPos;

        do
        {
            initialWritePos = Volatile.Read(ref _writePosition);
            initialReadPos = Volatile.Read(ref _readPosition);

            if (initialWritePos - initialReadPos >= _capacity)
                return false;
        }
        while (Interlocked.CompareExchange(ref _writePosition, initialWritePos + 1, initialWritePos) != initialWritePos);

        int index = (int)(initialWritePos % _capacity);
        _frontBuffer[index] = item;
        return true;
    }

    /// <summary>
    /// Attempts to read an element without locking (lock-free).
    /// Uses Interlocked operations for thread safety.
    /// </summary>
    /// <param name="item">The item read.</param>
    /// <returns>True if read; false if empty.</returns>
    public bool TryReadLockFree(out T item)
    {
        item = default;
        long initialReadPos, initialFlushedPos;

        do
        {
            initialReadPos = Volatile.Read(ref _readPosition);
            initialFlushedPos = Volatile.Read(ref _flushedPosition);

            if (initialReadPos >= initialFlushedPos)
                return false;
        }
        while (Interlocked.CompareExchange(ref _readPosition, initialReadPos + 1, initialReadPos) != initialReadPos);

        int index = (int)(initialReadPos % _capacity);
        item = _frontBuffer[index];
        return true;
    }

    /// <summary>
    /// Gets diagnostic information about the buffer state.
    /// </summary>
    public BufferDiagnostics GetDiagnostics()
    {
        return new BufferDiagnostics
        {
            Capacity = _capacity,
            WritePosition = Volatile.Read(ref _writePosition),
            ReadPosition = Volatile.Read(ref _readPosition),
            FlushedPosition = Volatile.Read(ref _flushedPosition),
            AvailableToRead = AvailableToRead,
            AvailableToWrite = AvailableToWrite,
            ActiveBuffer = _activeBuffer,
            IsSwapping = IsSwapping,
            ElementSize = _elementSize,
            TotalMemoryBytes = TotalMemoryUsage
        };
    }

    /// <summary>
    /// Creates a producer-consumer pair from this buffer.
    /// </summary>
    public (StreamingProducer<T> Producer, StreamingConsumer<T> Consumer) CreateProducerConsumerPair()
    {
        var producer = new StreamingProducer<T>(this);
        var consumer = new StreamingConsumer<T>(this);
        return (producer, consumer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;

            if (_usingMemoryMapped)
            {
                _mappedAccessor?.Dispose();
                _mappedFile?.Dispose();
            }
            else
            {
                NativeMemory.AlignedFree(_bufferA);
                NativeMemory.AlignedFree(_bufferB);
            }

            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// Diagnostic information about the buffer.
    /// </summary>
    public readonly struct BufferDiagnostics
    {
        public int Capacity { get; init; }
        public long WritePosition { get; init; }
        public long ReadPosition { get; init; }
        public long FlushedPosition { get; init; }
        public long AvailableToRead { get; init; }
        public long AvailableToWrite { get; init; }
        public int ActiveBuffer { get; init; }
        public bool IsSwapping { get; init; }
        public int ElementSize { get; init; }
        public long TotalMemoryBytes { get; init; }

        public override string ToString()
        {
            return $"Buffer[Cap={Capacity}, Read={ReadPosition}, Write={WritePosition}, " +
                   $"Flushed={FlushedPosition}, Avail={AvailableToRead}, Active={ActiveBuffer}]";
        }
    }
}

/// <summary>
/// Producer side of a streaming buffer pair. Provides a simplified write interface.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
public sealed class StreamingProducer<T> : IDisposable
    where T : unmanaged
{
    private readonly StreamingBuffer<T> _buffer;
    private bool _disposed;

    /// <summary>The underlying buffer.</summary>
    public StreamingBuffer<T> Buffer => _buffer;

    /// <summary>Number of elements available to write.</summary>
    public long AvailableToWrite => _buffer.AvailableToWrite;

    /// <summary>Whether the buffer is full.</summary>
    public bool IsFull => _buffer.IsFull;

    public StreamingProducer(StreamingBuffer<T> buffer)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>
    /// Writes a single element.
    /// </summary>
    public bool TryWrite(in T item) => _buffer.TryWrite(item);

    /// <summary>
    /// Writes a span of elements.
    /// </summary>
    public int Write(ReadOnlySpan<T> data) => _buffer.Write(data);

    /// <summary>
    /// Lock-free write attempt.
    /// </summary>
    public bool TryWriteLockFree(in T item) => _buffer.TryWriteLockFree(item);

    /// <summary>
    /// Reserves space for writing.
    /// </summary>
    public Span<T> Reserve(int count) => _buffer.Reserve(count);

    /// <summary>
    /// Flushes written data for the consumer.
    /// </summary>
    public void Flush() => _buffer.Flush();

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>
/// Consumer side of a streaming buffer pair. Provides a simplified read interface.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
public sealed class StreamingConsumer<T> : IDisposable
    where T : unmanaged
{
    private readonly StreamingBuffer<T> _buffer;
    private bool _disposed;

    /// <summary>The underlying buffer.</summary>
    public StreamingBuffer<T> Buffer => _buffer;

    /// <summary>Number of elements available to read.</summary>
    public long AvailableToRead => _buffer.AvailableToRead;

    /// <summary>Whether the buffer is empty.</summary>
    public bool IsEmpty => _buffer.IsEmpty;

    public StreamingConsumer(StreamingBuffer<T> buffer)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>
    /// Reads a single element.
    /// </summary>
    public bool TryRead(out T item) => _buffer.TryRead(out item);

    /// <summary>
    /// Reads multiple elements.
    /// </summary>
    public int Read(Span<T> destination) => _buffer.Read(destination);

    /// <summary>
    /// Lock-free read attempt.
    /// </summary>
    public bool TryReadLockFree(out T item) => _buffer.TryReadLockFree(out item);

    /// <summary>
    /// Peeks at the next element.
    /// </summary>
    public bool TryPeek(out T item) => _buffer.TryPeek(out item);

    /// <summary>
    /// Skips elements without reading.
    /// </summary>
    public long Advance(long count) => _buffer.Advance(count);

    /// <summary>
    /// Swaps to the next buffer.
    /// </summary>
    public bool TrySwapBuffers() => _buffer.TrySwapBuffers();

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>
/// Non-generic streaming buffer using byte arrays for type-erased streaming.
/// Useful for streaming heterogeneous data or raw byte streams.
/// </summary>
public sealed class StreamingByteBuffer : IDisposable
{
    private readonly byte[] _bufferA;
    private readonly byte[] _bufferB;
    private byte[] _frontBuffer;
    private byte[] _backBuffer;

    private readonly int _capacity;
    private long _writePosition;
    private long _readPosition;
    private long _flushedPosition;
    private int _activeBuffer;
    private readonly object _swapLock = new();
    private bool _disposed;

    /// <summary>Capacity in bytes.</summary>
    public int Capacity => _capacity;

    /// <summary>Available bytes for reading.</summary>
    public long AvailableToRead => Volatile.Read(ref _flushedPosition) - Volatile.Read(ref _readPosition);

    /// <summary>Available bytes for writing.</summary>
    public long AvailableToWrite => _capacity - (Volatile.Read(ref _writePosition) - Volatile.Read(ref _readPosition));

    /// <summary>Whether the buffer is empty.</summary>
    public bool IsEmpty => AvailableToRead <= 0;

    /// <summary>Whether the buffer is full.</summary>
    public bool IsFull => AvailableToWrite <= 0;

    /// <summary>
    /// Initializes a new streaming byte buffer.
    /// </summary>
    public StreamingByteBuffer(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(capacity, 0);

        _capacity = capacity;
        _bufferA = new byte[capacity];
        _bufferB = new byte[capacity];
        _frontBuffer = _bufferA;
        _backBuffer = _bufferB;
        _activeBuffer = 0;
    }

    /// <summary>
    /// Writes bytes to the buffer.
    /// </summary>
    public int Write(ReadOnlySpan<byte> data)
    {
        long writePos = Volatile.Read(ref _writePosition);
        long readPos = Volatile.Read(ref _readPosition);

        long available = _capacity - (writePos - readPos);
        int toWrite = (int)Math.Min(data.Length, available);

        if (toWrite <= 0)
            return 0;

        int startIndex = (int)(writePos % _capacity);

        if (startIndex + toWrite <= _capacity)
        {
            data.Slice(0, toWrite).CopyTo(_frontBuffer.AsSpan(startIndex));
        }
        else
        {
            int firstPart = _capacity - startIndex;
            data.Slice(0, firstPart).CopyTo(_frontBuffer.AsSpan(startIndex));
            data.Slice(firstPart, toWrite - firstPart).CopyTo(_frontBuffer.AsSpan(0));
        }

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _writePosition, writePos + toWrite);
        return toWrite;
    }

    /// <summary>
    /// Reads bytes from the buffer.
    /// </summary>
    public int Read(Span<byte> destination)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos;
        int toRead = (int)Math.Min(destination.Length, available);

        if (toRead <= 0)
            return 0;

        int startIndex = (int)(readPos % _capacity);

        if (startIndex + toRead <= _capacity)
        {
            _frontBuffer.AsSpan(startIndex, toRead).CopyTo(destination);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            _frontBuffer.AsSpan(startIndex, firstPart).CopyTo(destination);
            _frontBuffer.AsSpan(0, toRead - firstPart).CopyTo(destination.Slice(firstPart));
        }

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _readPosition, readPos + toRead);
        return toRead;
    }

    /// <summary>
    /// Peeks at bytes without consuming.
    /// </summary>
    public int Peek(Span<byte> destination, int offset = 0)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos - offset;
        int toPeek = (int)Math.Min(destination.Length, available);

        if (toPeek <= 0)
            return 0;

        int startIndex = (int)((readPos + offset) % _capacity);

        if (startIndex + toPeek <= _capacity)
        {
            _frontBuffer.AsSpan(startIndex, toPeek).CopyTo(destination);
        }
        else
        {
            int firstPart = _capacity - startIndex;
            _frontBuffer.AsSpan(startIndex, firstPart).CopyTo(destination);
            _frontBuffer.AsSpan(0, toPeek - firstPart).CopyTo(destination.Slice(firstPart));
        }

        return toPeek;
    }

    /// <summary>
    /// Advances the read position.
    /// </summary>
    public long Advance(long count)
    {
        long readPos = Volatile.Read(ref _readPosition);
        long flushedPos = Volatile.Read(ref _flushedPosition);

        long available = flushedPos - readPos;
        long toAdvance = Math.Min(count, available);

        Interlocked.MemoryBarrier();
        Volatile.Write(ref _readPosition, readPos + toAdvance);
        return toAdvance;
    }

    /// <summary>
    /// Flushes written data for the consumer.
    /// </summary>
    public void Flush()
    {
        long writePos = Volatile.Read(ref _writePosition);
        Interlocked.MemoryBarrier();
        Volatile.Write(ref _flushedPosition, writePos);
    }

    /// <summary>
    /// Swaps front and back buffers.
    /// </summary>
    public void SwapBuffers()
    {
        lock (_swapLock)
        {
            (_frontBuffer, _backBuffer) = (_backBuffer, _frontBuffer);
            _activeBuffer = _activeBuffer == 0 ? 1 : 0;
            Volatile.Write(ref _writePosition, 0);
            Volatile.Write(ref _flushedPosition, 0);
        }
    }

    /// <summary>
    /// Resets all positions.
    /// </summary>
    public void Reset()
    {
        Volatile.Write(ref _writePosition, 0);
        Volatile.Write(ref _readPosition, 0);
        Volatile.Write(ref _flushedPosition, 0);
    }

    /// <summary>
    /// Clears all data.
    /// </summary>
    public void Clear()
    {
        Array.Clear(_bufferA);
        Array.Clear(_bufferB);
        Reset();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
