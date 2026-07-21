using System;
// ============================================================
// FILE: ZeroCopyBuffer.cs
// PATH: Memory/ZeroCopyBuffer.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - Zero-Copy Buffer
// Zero-copy buffer for CPU-GPU data sharing with memory-mapped and circular variants.

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
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
using GDNN.Rendering.Compat;

namespace GDNN.Memory;

/// <summary>
/// Represents the current state of a <see cref="ZeroCopyBuffer"/>.
/// </summary>
[Flags]
public enum BufferState
{
    /// <summary>Buffer is available for writing.</summary>
    Available = 0,

    /// <summary>Buffer is being written to by the CPU.</summary>
    Writing = 1,

    /// <summary>Buffer contains valid data ready for reading.</summary>
    Ready = 2,

    /// <summary>Buffer is being read (e.g., by the GPU or consumer thread).</summary>
    Reading = 4,

    /// <summary>Buffer has been invalidated and must not be accessed.</summary>
    Invalid = 8
}

/// <summary>
/// A zero-copy buffer for sharing data between CPU and GPU without memory copies.
/// Backed by native aligned memory with atomic read/write operations and fence
/// synchronization for lock-free producer-consumer patterns.
/// </summary>
[DebuggerDisplay("Capacity={Capacity}, State={State}, ReadPos={ReadPosition}, WritePos={WritePosition}")]
public sealed unsafe class ZeroCopyBuffer : IDisposable
{
    /// <summary>Default alignment for zero-copy buffers (64-byte cache line).</summary>
    public const int DefaultAlignment = 64;

    /// <summary>Maximum number of fence signals before old signals are overwritten.</summary>
    public const int MaxFenceDepth = 16;

    private readonly byte* _pointer;
    private readonly byte* _rawPointer;
    private readonly int _capacity;
    private readonly int _alignment;
    private readonly MemoryMappedFile? _mmf;
    private readonly MemoryMappedViewAccessor? _accessor;
    private bool _viewHandleHeld;
    private volatile BufferState _state;
    private bool _disposed;

    // Fence synchronization.
    private int _fenceCount;
    private readonly int[] _fenceSignals;
    private int _fenceHead;
    private int _fenceTail;

    // Producer-consumer positions for circular usage.
    private volatile int _readPosition;
    private volatile int _writePosition;
    private long _totalBytesWritten;
    private long _totalBytesRead;

    /// <summary>
    /// Gets the base pointer to the buffer memory.
    /// Only valid while the buffer is not disposed.
    /// </summary>
    public byte* Pointer => _pointer;

    /// <summary>
    /// Gets the total capacity of the buffer in bytes.
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Gets the alignment of the buffer in bytes.
    /// </summary>
    public int Alignment => _alignment;

    /// <summary>
    /// Gets or sets the current state of the buffer.
    /// </summary>
    public BufferState State
    {
        get => _state;
        set => _state = value;
    }

    /// <summary>
    /// Gets the current read position for circular buffer usage.
    /// </summary>
    public int ReadPosition => _readPosition;

    /// <summary>
    /// Gets the current write position for circular buffer usage.
    /// </summary>
    public int WritePosition => _writePosition;

    /// <summary>
    /// Gets the number of bytes available to read (from read position to write position).
    /// </summary>
    public int AvailableToRead => (_writePosition - _readPosition + _capacity) % _capacity;

    /// <summary>
    /// Gets the number of bytes available to write before reaching the read position.
    /// </summary>
    public int AvailableToWrite => _capacity - AvailableToRead - 1;

    /// <summary>
    /// Gets the total bytes written since creation or last reset.
    /// </summary>
    public long TotalBytesWritten => Interlocked.Read(ref _totalBytesWritten);

    /// <summary>
    /// Gets the total bytes read since creation or last reset.
    /// </summary>
    public long TotalBytesRead => Interlocked.Read(ref _totalBytesRead);

    /// <summary>
    /// Gets whether the buffer has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Gets whether the buffer is full (no space available to write).
    /// </summary>
    public bool IsFull => AvailableToRead >= _capacity - 1;

    /// <summary>
    /// Gets whether the buffer is empty (no data available to read).
    /// </summary>
    public bool IsEmpty => _readPosition == _writePosition;

    /// <summary>
    /// Initializes a new zero-copy buffer with the specified capacity and alignment.
    /// </summary>
    /// <param name="capacity">Total capacity in bytes.</param>
    /// <param name="alignment">Memory alignment in bytes.</param>
    public ZeroCopyBuffer(int capacity, int alignment = DefaultAlignment)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");
        if (alignment <= 0 || !BitOperations.IsPow2((uint)alignment))
            throw new ArgumentOutOfRangeException(nameof(alignment), "Alignment must be a positive power of two.");

        _capacity = capacity;
        _alignment = alignment;
        _state = BufferState.Available;
        _mmf = null;
        _accessor = null;
        _viewHandleHeld = false;

        // Allocate aligned memory.
        int allocSize = capacity + alignment; // Extra space for alignment.
        _rawPointer = (byte*)NativeMemory.AlignedAlloc((nuint)allocSize, (nuint)alignment);

        // Align the pointer.
        nuint rawAddr = (nuint)_rawPointer;
        nuint alignedAddr = (rawAddr + (nuint)(alignment - 1)) & ~(nuint)(alignment - 1);
        _pointer = (byte*)alignedAddr;

        // Zero-initialize.
        NativeMemory.Clear(_pointer, (nuint)capacity);

        // Initialize fence signals.
        _fenceSignals = new int[MaxFenceDepth];
        _fenceCount = 0;
        _fenceHead = 0;
        _fenceTail = 0;
    }

    private ZeroCopyBuffer(
        byte* mappedPointer,
        int capacity,
        MemoryMappedFile mmf,
        MemoryMappedViewAccessor accessor,
        bool viewHandleHeld)
    {
        _capacity = capacity;
        _alignment = 1;
        _state = BufferState.Available;
        _pointer = mappedPointer;
        _rawPointer = null;
        _mmf = mmf;
        _accessor = accessor;
        _viewHandleHeld = viewHandleHeld;
        _fenceSignals = new int[MaxFenceDepth];
        _fenceCount = 0;
        _fenceHead = 0;
        _fenceTail = 0;
    }

    /// <summary>
    /// Creates a zero-copy buffer backed by a memory-mapped file.
    /// Enables inter-process data sharing without copying.
    /// </summary>
    /// <param name="filePath">Path to the memory-mapped file.</param>
    /// <param name="capacity">Size of the mapping in bytes.</param>
    /// <param name="offset">Offset within the file to map from.</param>
    /// <returns>A new ZeroCopyBuffer backed by the memory-mapped file.</returns>
    public static ZeroCopyBuffer CreateFromFile(string filePath, int capacity, long offset = 0)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset));

        long requiredLength = offset + capacity;
        if (!File.Exists(filePath))
        {
            using var fs = File.Create(filePath);
            fs.SetLength(requiredLength);
        }
        else if (new FileInfo(filePath).Length < requiredLength)
        {
            throw new InvalidDataException(
                $"File '{filePath}' is smaller than required mapping size ({requiredLength} bytes).");
        }

        var mmf = MemoryMappedFile.CreateFromFile(
            filePath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.ReadWrite);
        MemoryMappedViewAccessor? accessor = null;
        bool addRefSuccess = false;
        try
        {
            accessor = mmf.CreateViewAccessor(offset, capacity, MemoryMappedFileAccess.ReadWrite);
            accessor.SafeMemoryMappedViewHandle.DangerousAddRef(ref addRefSuccess);
            if (!addRefSuccess)
                throw new InvalidOperationException("Failed to pin memory-mapped view handle.");

            byte* ptr = (byte*)accessor.SafeMemoryMappedViewHandle.DangerousGetHandle().ToPointer();
            return new ZeroCopyBuffer(ptr, capacity, mmf, accessor, viewHandleHeld: true);
        }
        catch
        {
            if (addRefSuccess && accessor != null)
                accessor.SafeMemoryMappedViewHandle.DangerousRelease();
            accessor?.Dispose();
            mmf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Gets a Span&lt;T&gt; view over a region of the buffer.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="byteOffset">Byte offset from the buffer start.</param>
    /// <param name="count">Number of elements.</param>
    public Span<T> GetSpan<T>(int byteOffset, int count) where T : struct
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int requiredBytes = count * Unsafe.SizeOf<T>();
        if (byteOffset < 0 || byteOffset + requiredBytes > _capacity)
            throw new ArgumentOutOfRangeException(nameof(byteOffset));

        return new Span<T>(_pointer + byteOffset, count);
    }

    /// <summary>
    /// Gets a ReadOnlySpan&lt;T&gt; view over a region of the buffer.
    /// </summary>
    public ReadOnlySpan<T> GetReadOnlySpan<T>(int byteOffset, int count) where T : struct
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int requiredBytes = count * Unsafe.SizeOf<T>();
        if (byteOffset < 0 || byteOffset + requiredBytes > _capacity)
            throw new ArgumentOutOfRangeException(nameof(byteOffset));

        return new ReadOnlySpan<T>(_pointer + byteOffset, count);
    }

    /// <summary>
    /// Gets a raw byte Span over the entire buffer.
    /// </summary>
    public Span<byte> GetByteSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new Span<byte>(_pointer, _capacity);
    }

    /// <summary>
    /// Gets a raw byte ReadOnlySpan over the entire buffer.
    /// </summary>
    public ReadOnlySpan<byte> GetReadOnlyByteSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<byte>(_pointer, _capacity);
    }

    /// <summary>
    /// Atomically writes data to the buffer at the specified offset.
    /// Uses <see cref="Interlocked"/> operations for thread-safe writes.
    /// </summary>
    /// <typeparam name="T">Value type to write.</typeparam>
    /// <param name="offset">Byte offset to write at.</param>
    /// <param name="value">The value to write atomically.</param>
    public void AtomicWrite<T>(int offset, T value) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || offset + Unsafe.SizeOf<T>() > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        T* target = (T*)(_pointer + offset);

        if (sizeof(T) == sizeof(int))
        {
            int intVal = Unsafe.As<T, int>(ref value);
            Volatile.Write(ref Unsafe.AsRef<int>(target), intVal);
        }
        else if (sizeof(T) == sizeof(long))
        {
            long longVal = Unsafe.As<T, long>(ref value);
            Interlocked.Exchange(ref Unsafe.AsRef<long>(target), longVal);
        }
        else if (sizeof(T) == sizeof(float))
        {
            int bits = BitConverter.SingleToInt32Bits(Unsafe.As<T, float>(ref value));
            Volatile.Write(ref Unsafe.AsRef<int>((int*)target), bits);
        }
        else
        {
            // For other sizes, use a pinned copy.
            Unsafe.Copy(target, ref value);
        }
    }

    /// <summary>
    /// Atomically reads a value from the buffer at the specified offset.
    /// </summary>
    /// <typeparam name="T">Value type to read.</typeparam>
    /// <param name="offset">Byte offset to read from.</param>
    /// <returns>The atomically read value.</returns>
    public T AtomicRead<T>(int offset) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || offset + Unsafe.SizeOf<T>() > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        T* source = (T*)(_pointer + offset);

        if (sizeof(T) == sizeof(int))
        {
            int intVal = Volatile.Read(ref Unsafe.AsRef<int>(source));
            return Unsafe.As<int, T>(ref intVal);
        }
        else if (sizeof(T) == sizeof(long))
        {
            long longVal = Interlocked.Read(ref Unsafe.AsRef<long>(source));
            return Unsafe.As<long, T>(ref longVal);
        }
        else if (sizeof(T) == sizeof(float))
        {
            int bits = Volatile.Read(ref Unsafe.AsRef<int>((int*)source));
            return Unsafe.As<int, T>(ref bits);
        }
        else
        {
            return Unsafe.Read<T>(source);
        }
    }

    /// <summary>
    /// Performs a compare-and-swap operation at the specified offset.
    /// </summary>
    /// <typeparam name="T">Value type.</typeparam>
    /// <param name="offset">Byte offset.</param>
    /// <param name="comparand">The value to compare against.</param>
    /// <param name="replacement">The value to swap in if comparison succeeds.</param>
    /// <returns>The original value at the offset (whether or not the swap succeeded).</returns>
    public T AtomicCompareExchange<T>(int offset, T comparand, T replacement) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || offset + Unsafe.SizeOf<T>() > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        T* target = (T*)(_pointer + offset);

        if (sizeof(T) == sizeof(int))
        {
            int original = Interlocked.CompareExchange(
                ref Unsafe.AsRef<int>(target),
                Unsafe.As<T, int>(ref replacement),
                Unsafe.As<T, int>(ref comparand));
            return Unsafe.As<int, T>(ref original);
        }
        else if (sizeof(T) == sizeof(long))
        {
            long original = Interlocked.CompareExchange(
                ref Unsafe.AsRef<long>(target),
                Unsafe.As<T, long>(ref replacement),
                Unsafe.As<T, long>(ref comparand));
            return Unsafe.As<long, T>(ref original);
        }

        // Fallback: not truly atomic for non-interlocked sizes.
        T originalVal = Unsafe.Read<T>(target);
        if (EqualityComparer<T>.Default.Equals(originalVal, comparand))
            Unsafe.Write(target, replacement);
        return originalVal;
    }

    #region Circular Buffer Operations

    /// <summary>
    /// Writes data to the circular buffer at the current write position.
    /// Advances the write position atomically.
    /// </summary>
    /// <param name="data">Data to write.</param>
    /// <returns>Number of bytes actually written.</returns>
    public int Write(ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int available = AvailableToWrite;
        int toWrite = Math.Min(data.Length, available);
        if (toWrite == 0)
            return 0;

        int writePos = _writePosition;
        int firstChunk = Math.Min(toWrite, _capacity - writePos);

        fixed (byte* srcPtr = &MemoryMarshal.GetReference(data))
        {
            Buffer.MemoryCopy(srcPtr, _pointer + writePos, firstChunk, firstChunk);
            if (toWrite > firstChunk)
            {
                Buffer.MemoryCopy(srcPtr + firstChunk, _pointer, toWrite - firstChunk, toWrite - firstChunk);
            }
        }

        // Advance write position with wrap-around.
        Interlocked.Exchange(ref _writePosition, (writePos + toWrite) % _capacity);
        Interlocked.Add(ref _totalBytesWritten, toWrite);

        return toWrite;
    }

    /// <summary>
    /// Reads data from the circular buffer at the current read position.
    /// Advances the read position atomically.
    /// </summary>
    /// <param name="destination">Span to read data into.</param>
    /// <returns>Number of bytes actually read.</returns>
    public int Read(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int available = AvailableToRead;
        int toRead = Math.Min(destination.Length, available);
        if (toRead == 0)
            return 0;

        int readPos = _readPosition;
        int firstChunk = Math.Min(toRead, _capacity - readPos);

        fixed (byte* dstPtr = &MemoryMarshal.GetReference(destination))
        {
            Buffer.MemoryCopy(_pointer + readPos, dstPtr, firstChunk, firstChunk);
            if (toRead > firstChunk)
            {
                Buffer.MemoryCopy(_pointer, dstPtr + firstChunk, toRead - firstChunk, toRead - firstChunk);
            }
        }

        Interlocked.Exchange(ref _readPosition, (readPos + toRead) % _capacity);
        Interlocked.Add(ref _totalBytesRead, toRead);

        return toRead;
    }

    /// <summary>
    /// Peeks at data in the circular buffer without advancing the read position.
    /// </summary>
    /// <param name="destination">Span to peek data into.</param>
    /// <returns>Number of bytes peeked.</returns>
    public int Peek(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int available = AvailableToRead;
        int toPeek = Math.Min(destination.Length, available);
        if (toPeek == 0)
            return 0;

        int readPos = _readPosition;
        int firstChunk = Math.Min(toPeek, _capacity - readPos);

        fixed (byte* dstPtr = &MemoryMarshal.GetReference(destination))
        {
            Buffer.MemoryCopy(_pointer + readPos, dstPtr, firstChunk, firstChunk);
            if (toPeek > firstChunk)
            {
                Buffer.MemoryCopy(_pointer, dstPtr + firstChunk, toPeek - firstChunk, toPeek - firstChunk);
            }
        }

        return toPeek;
    }

    /// <summary>
    /// Skips the specified number of bytes in the read position.
    /// </summary>
    /// <param name="count">Number of bytes to skip.</param>
    /// <returns>Number of bytes actually skipped.</returns>
    public int Skip(int count)
    {
        int available = AvailableToRead;
        int toSkip = Math.Min(count, available);
        if (toSkip == 0)
            return 0;

        Interlocked.Exchange(ref _readPosition, (_readPosition + toSkip) % _capacity);
        Interlocked.Add(ref _totalBytesRead, toSkip);

        return toSkip;
    }

    /// <summary>
    /// Resets the circular buffer, clearing all data and resetting positions.
    /// </summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        NativeMemory.Clear(_pointer, (nuint)_capacity);
        Interlocked.Exchange(ref _readPosition, 0);
        Interlocked.Exchange(ref _writePosition, 0);
        _fenceCount = 0;
        _fenceHead = 0;
        _fenceTail = 0;
        _state = BufferState.Available;
    }

    #endregion

    #region Fence Synchronization

    /// <summary>
    /// Signals a fence, indicating that data up to the current write position is ready.
    /// Consumers can wait on this fence using <see cref="WaitForFence"/>.
    /// </summary>
    /// <returns>A fence ID that can be used to wait on this signal.</returns>
    public int SignalFence()
    {
        int fenceId = Interlocked.Increment(ref _fenceCount);
        int slot = Interlocked.Increment(ref _fenceHead) % MaxFenceDepth;

        Volatile.Write(ref _fenceSignals[slot], fenceId);
        return fenceId;
    }

    /// <summary>
    /// Waits for a specific fence to be signaled.
    /// Blocks the calling thread until the fence is reached.
    /// </summary>
    /// <param name="fenceId">The fence ID to wait for.</param>
    /// <param name="timeoutMs">Timeout in milliseconds. -1 for infinite wait.</param>
    /// <returns>True if the fence was reached; false on timeout.</returns>
    public bool WaitForFence(int fenceId, int timeoutMs = -1)
    {
        var sw = Stopwatch.StartNew();

        while (true)
        {
            // Check if the fence has been signaled.
            int head = _fenceHead;
            for (int i = _fenceTail; i <= head; i++)
            {
                int slot = i % MaxFenceDepth;
                if (Volatile.Read(ref _fenceSignals[slot]) >= fenceId)
                {
                    _fenceTail = i + 1;
                    return true;
                }
            }

            if (timeoutMs >= 0 && sw.ElapsedMilliseconds >= timeoutMs)
                return false;

            Thread.SpinWait(1);
        }
    }

    /// <summary>
    /// Resets all fence signals.
    /// </summary>
    public void ResetFences()
    {
        _fenceCount = 0;
        _fenceHead = 0;
        _fenceTail = 0;
        Array.Clear(_fenceSignals, 0, _fenceSignals.Length);
    }

    #endregion

    #region Bulk Operations

    /// <summary>
    /// Clears the entire buffer to zero.
    /// </summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMemory.Clear(_pointer, (nuint)_capacity);
    }

    /// <summary>
    /// Fills the entire buffer with the specified byte value.
    /// </summary>
    /// <param name="value">The byte value to fill with.</param>
    public void Fill(byte value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Unsafe.InitBlockUnaligned(_pointer, value, (uint)_capacity);
    }

    /// <summary>
    /// Copies data from a source span into the buffer at the specified offset.
    /// </summary>
    /// <param name="offset">Byte offset to start writing at.</param>
    /// <param name="source">Source data to copy.</param>
    public void CopyFrom(int offset, ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || offset + source.Length > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        fixed (byte* srcPtr = &MemoryMarshal.GetReference(source))
        {
            Buffer.MemoryCopy(srcPtr, _pointer + offset, _capacity - offset, source.Length);
        }
    }

    /// <summary>
    /// Copies data from the buffer into a destination span starting at the specified offset.
    /// </summary>
    /// <param name="offset">Byte offset to start reading from.</param>
    /// <param name="destination">Destination span to copy data into.</param>
    public void CopyTo(int offset, Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (offset < 0 || offset + destination.Length > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        fixed (byte* dstPtr = &MemoryMarshal.GetReference(destination))
        {
            Buffer.MemoryCopy(_pointer + offset, dstPtr, destination.Length, destination.Length);
        }
    }

    /// <summary>
    /// Returns a view of the buffer as a Memory&lt;T&gt; for interop with managed APIs.
    /// </summary>
    public Memory<T> GetMemory<T>(int offset, int count) where T : struct
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int requiredBytes = count * Unsafe.SizeOf<T>();
        if (offset < 0 || offset + requiredBytes > _capacity)
            throw new ArgumentOutOfRangeException(nameof(offset));

        // Pin the memory and create a Memory from it.
        // Note: For true zero-copy, the buffer must remain pinned during use.
        return MemoryMarshalCompat.CreateMemory(
            ref Unsafe.AsRef<T>(_pointer + offset),
            count);
    }

    #endregion

    /// <summary>
    /// Disposes the buffer and frees native or memory-mapped resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_viewHandleHeld && _accessor != null)
        {
            _accessor.SafeMemoryMappedViewHandle.DangerousRelease();
            _viewHandleHeld = false;
        }

        _accessor?.Dispose();
        _mmf?.Dispose();

        if (_rawPointer != null)
            NativeMemory.Free(_rawPointer);
    }
}

/// <summary>
/// A memory-mapped file backed buffer for inter-process data sharing.
/// Provides zero-copy access to shared memory regions between processes.
/// </summary>
[DebuggerDisplay("FilePath={FilePath}, Size={Size}, IsOpen={IsOpen}")]
public sealed unsafe class MappedBuffer : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer;
    private readonly int _size;
    private readonly string _filePath;
    private bool _disposed;
    private bool _isOpen;
    private bool _viewHandleHeld;

    /// <summary>
    /// Gets the file path of the memory-mapped file.
    /// </summary>
    public string FilePath => _filePath;

    /// <summary>
    /// Gets the size of the mapped region in bytes.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Gets whether the mapped file is currently open.
    /// </summary>
    public bool IsOpen => _isOpen;

    /// <summary>
    /// Gets the base pointer to the mapped memory.
    /// </summary>
    public unsafe byte* BasePointer => _basePointer;

    /// <summary>
    /// Creates a new mapped buffer backed by the specified file.
    /// </summary>
    /// <param name="filePath">Path to the memory-mapped file.</param>
    /// <param name="size">Size of the mapping in bytes.</param>
    /// <param name="createIfMissing">If true, creates the file if it doesn't exist.</param>
    public MappedBuffer(string filePath, int size, bool createIfMissing = true)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size));

        _filePath = filePath;
        _size = size;
        _isOpen = false;
    }

    /// <summary>
    /// Opens the memory-mapped file for read/write access.
    /// </summary>
    /// <exception cref="FileNotFoundException">If the file doesn't exist and createIfMissing was false.</exception>
    public unsafe void Open()
    {
        if (_isOpen)
            return;

        if (!File.Exists(_filePath))
        {
            using var fs = File.Create(_filePath);
            fs.SetLength(_size);
        }

        _mmf = MemoryMappedFile.CreateFromFile(
            _filePath, FileMode.Open, null, _size, MemoryMappedFileAccess.ReadWrite);

        _accessor = _mmf.CreateViewAccessor(0, _size, MemoryMappedFileAccess.ReadWrite);

        // Keep the view handle alive for the lifetime of the mapped pointer.
        bool addedRef = false;
        _accessor.SafeMemoryMappedViewHandle.DangerousAddRef(ref addedRef);
        if (!addedRef)
        {
            _accessor.Dispose();
            _accessor = null;
            _mmf.Dispose();
            _mmf = null;
            throw new InvalidOperationException("Failed to pin memory-mapped view handle.");
        }

        _viewHandleHeld = true;
        _basePointer = (byte*)_accessor.SafeMemoryMappedViewHandle.DangerousGetHandle().ToPointer();
        _isOpen = true;
    }

    /// <summary>
    /// Gets a Span&lt;T&gt; view over a region of the mapped memory.
    /// </summary>
    public Span<T> GetSpan<T>(int offset, int count) where T : struct
    {
        if (!_isOpen || _basePointer == null)
            throw new InvalidOperationException("Mapped buffer is not open.");

        int requiredBytes = count * Unsafe.SizeOf<T>();
        if (offset < 0 || offset + requiredBytes > _size)
            throw new ArgumentOutOfRangeException(nameof(offset));

        return new Span<T>(_basePointer + offset, count);
    }

    /// <summary>
    /// Gets a ReadOnlySpan&lt;T&gt; view over a region of the mapped memory.
    /// </summary>
    public ReadOnlySpan<T> GetReadOnlySpan<T>(int offset, int count) where T : struct
    {
        if (!_isOpen || _basePointer == null)
            throw new InvalidOperationException("Mapped buffer is not open.");

        int requiredBytes = count * Unsafe.SizeOf<T>();
        if (offset < 0 || offset + requiredBytes > _size)
            throw new ArgumentOutOfRangeException(nameof(offset));

        return new ReadOnlySpan<T>(_basePointer + offset, count);
    }

    /// <summary>
    /// Flushes all modified pages in the mapped region to the file on disk.
    /// </summary>
    public void Flush()
    {
        _accessor?.Flush();
    }

    /// <summary>
    /// Flushes a specific region of modified pages to disk.
    /// </summary>
    /// <param name="offset">Byte offset to start flushing from.</param>
    /// <param name="length">Number of bytes to flush.</param>
    public void FlushRegion(long offset, long length)
    {
        _accessor?.Flush();
    }

    /// <summary>
    /// Closes the mapped buffer.
    /// </summary>
    public void Close()
    {
        if (!_isOpen)
            return;
        _isOpen = false;

        if (_viewHandleHeld && _accessor != null)
        {
            _accessor.SafeMemoryMappedViewHandle.DangerousRelease();
            _viewHandleHeld = false;
        }

        _accessor?.Dispose();
        _accessor = null;
        _mmf?.Dispose();
        _mmf = null;
        _basePointer = null;
    }

    /// <summary>
    /// Disposes the mapped buffer and releases all resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        Close();
    }
}

/// <summary>
/// A lock-free single-producer single-consumer (SPSC) ring buffer for streaming data.
/// Designed for high-throughput data transfer between threads without allocation.
/// </summary>
/// <typeparam name="T">Element type (must be unmanaged for unsafe access).</typeparam>
[DebuggerDisplay("Count={Count}, Capacity={Capacity}")]
public sealed unsafe class SPSCRingBuffer<T> : IDisposable where T : unmanaged
{
    private readonly T* _buffer;
    private readonly T* _rawBuffer;
    private readonly int _capacity;
    private readonly int _alignment;
    private volatile int _head; // Write position (producer).
    private volatile int _tail; // Read position (consumer).
    private bool _disposed;

    /// <summary>
    /// Gets the maximum number of elements the ring buffer can hold.
    /// </summary>
    public int Capacity => _capacity - 1; // One slot wasted for full/empty disambiguation.

    /// <summary>
    /// Gets the number of elements currently in the buffer.
    /// </summary>
    public int Count => (_head - _tail + _capacity) % _capacity;

    /// <summary>
    /// Gets whether the buffer is empty.
    /// </summary>
    public bool IsEmpty => _head == _tail;

    /// <summary>
    /// Gets whether the buffer is full.
    /// </summary>
    public bool IsFull => ((_head + 1) % _capacity) == _tail;

    /// <summary>
    /// Initializes a new SPSC ring buffer with the specified capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of elements.</param>
    /// <param name="alignment">Memory alignment in bytes.</param>
    public SPSCRingBuffer(int capacity, int alignment = 64)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        if (alignment <= 0 || !BitOperations.IsPow2((uint)alignment))
            throw new ArgumentOutOfRangeException(nameof(alignment));

        _capacity = capacity + 1; // +1 for disambiguation.
        _alignment = alignment;

        int allocSize = _capacity * sizeof(T) + alignment;
        _rawBuffer = (T*)NativeMemory.AlignedAlloc((nuint)allocSize, (nuint)alignment);

        // Align.
        nuint rawAddr = (nuint)_rawBuffer;
        nuint alignedAddr = (rawAddr + (nuint)(alignment - 1)) & ~(nuint)(alignment - 1);
        _buffer = (T*)alignedAddr;

        NativeMemory.Clear(_buffer, (nuint)(_capacity * sizeof(T)));
    }

    /// <summary>
    /// Attempts to write a single element to the buffer (producer side).
    /// </summary>
    /// <param name="item">The item to write.</param>
    /// <returns>True if the item was written; false if the buffer is full.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryWrite(T item)
    {
        int currentHead = _head;
        int nextHead = (currentHead + 1) % _capacity;

        if (nextHead == _tail)
            return false; // Full.

        _buffer[currentHead] = item;
        Volatile.Write(ref _head, nextHead);
        return true;
    }

    /// <summary>
    /// Attempts to read a single element from the buffer (consumer side).
    /// </summary>
    /// <param name="item">The read item.</param>
    /// <returns>True if an item was read; false if the buffer is empty.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryRead(out T item)
    {
        int currentTail = _tail;

        if (currentTail == _head)
        {
            item = default!;
            return false; // Empty.
        }

        item = _buffer[currentTail];
        Volatile.Write(ref _tail, (currentTail + 1) % _capacity);
        return true;
    }

    /// <summary>
    /// Writes multiple elements to the buffer (producer side).
    /// </summary>
    /// <param name="data">Elements to write.</param>
    /// <returns>Number of elements actually written.</returns>
    public int Write(ReadOnlySpan<T> data)
    {
        int written = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (TryWrite(data[i]))
                written++;
            else
                break;
        }
        return written;
    }

    /// <summary>
    /// Reads multiple elements from the buffer (consumer side).
    /// </summary>
    /// <param name="destination">Span to read elements into.</param>
    /// <returns>Number of elements actually read.</returns>
    public int Read(Span<T> destination)
    {
        int read = 0;
        for (int i = 0; i < destination.Length; i++)
        {
            if (TryRead(out var item))
            {
                destination[i] = item;
                read++;
            }
            else
                break;
        }
        return read;
    }

    /// <summary>
    /// Peeks at the next element without consuming it.
    /// </summary>
    /// <param name="item">The peeked item.</param>
    /// <returns>True if an item was peeked; false if empty.</returns>
    public bool TryPeek(out T item)
    {
        int currentTail = _tail;
        if (currentTail == _head)
        {
            item = default!;
            return false;
        }
        item = _buffer[currentTail];
        return true;
    }

    /// <summary>
    /// Resets the buffer, discarding all data.
    /// </summary>
    public void Reset()
    {
        _head = 0;
        _tail = 0;
        NativeMemory.Clear(_buffer, (nuint)(_capacity * sizeof(T)));
    }

    /// <summary>
    /// Disposes the ring buffer and frees native memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_rawBuffer != null)
        {
            NativeMemory.Free(_rawBuffer);
        }
    }
}
