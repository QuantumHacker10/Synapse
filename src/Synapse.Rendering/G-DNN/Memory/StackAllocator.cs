using System;
// ============================================================
// FILE: StackAllocator.cs
// PATH: Memory/StackAllocator.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - Stack Allocator
// Fixed-size stack allocator for frame-based allocations with zero GC pressure.

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

namespace GDNN.Memory;

/// <summary>
/// A fixed-size stack allocator for frame-based allocations.
/// Provides fast bump allocations within a pre-allocated contiguous block,
/// with per-frame reset, nested scopes, overflow detection, and statistics.
/// All allocations are automatically freed when the owning frame ends or
/// the allocator is reset, eliminating GC pressure entirely.
/// </summary>
[DebuggerDisplay("UsedBytes={UsedBytes}/{Capacity}, Allocations={AllocationCount}, Scopes={ScopeDepth}")]
public sealed unsafe class StackAllocator : IDisposable
{
    /// <summary>
    /// Default capacity: 256 KB. Sufficient for most per-frame neural geometry
    /// scratch data (cluster evaluations, tile sorting, etc.).
    /// </summary>
    public const int DefaultCapacity = 256 * 1024;

    /// <summary>
    /// Default alignment for all allocations within the stack.
    /// 16-byte alignment satisfies SIMD vector requirements (Vector128).
    /// </summary>
    public const int DefaultAlignment = 16;

    /// <summary>
    /// Maximum alignment supported. Prevents unbounded alignment waste.
    /// </summary>
    public const int MaxAlignment = 256;

    private readonly byte* _basePointer;
    private readonly int _capacity;
    private int _usedBytes;
    private int _allocationCount;
    private int _scopeDepth;
    private int _frameIndex;
    private bool _disposed;

    // Per-scope bookkeeping for nested allocation scopes.
    private readonly Stack<int> _scopeStack;
    private readonly ScopeInfo[] _scopes;
    private const int MaxScopes = 64;

    // Statistics.
    private long _totalAllocationsEver;
    private long _totalBytesEver;
    private int _peakUsedBytes;
    private int _peakAllocationCount;

    // Thread-local storage for thread-specific allocators.
    [ThreadStatic]
    private static StackAllocator? t_current;

    // Overflow fallback: allocations exceeding remaining space fall back to the heap.
    private readonly bool _allowHeapFallback;

    /// <summary>
    /// Gets or sets the thread-local stack allocator for the current thread.
    /// </summary>
    public static StackAllocator? Current
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => t_current;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => t_current = value;
    }

    /// <summary>
    /// Gets the base pointer of the allocation block.
    /// </summary>
    public byte* BasePointer => _basePointer;

    /// <summary>
    /// Gets the total capacity of the stack in bytes.
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Gets the number of bytes currently used (allocated) from the stack.
    /// </summary>
    public int UsedBytes => Volatile.Read(ref _usedBytes);

    /// <summary>
    /// Gets the number of bytes remaining in the stack.
    /// </summary>
    public int RemainingBytes => _capacity - Volatile.Read(ref _usedBytes);

    /// <summary>
    /// Gets the total number of allocations made in the current frame.
    /// </summary>
    public int AllocationCount => Volatile.Read(ref _allocationCount);

    /// <summary>
    /// Gets the current nesting depth of allocation scopes.
    /// </summary>
    public int ScopeDepth => _scopeDepth;

    /// <summary>
    /// Gets the current frame index, incremented on each <see cref="Reset"/> call.
    /// </summary>
    public int FrameIndex => _frameIndex;

    /// <summary>
    /// Gets the peak number of bytes used since construction.
    /// </summary>
    public int PeakUsedBytes => _peakUsedBytes;

    /// <summary>
    /// Gets the peak allocation count in a single frame since construction.
    /// </summary>
    public int PeakAllocationCount => _peakAllocationCount;

    /// <summary>
    /// Gets the total number of allocations made across all frames.
    /// </summary>
    public long TotalAllocationsEver => Interlocked.Read(ref _totalAllocationsEver);

    /// <summary>
    /// Gets the total bytes allocated across all frames.
    /// </summary>
    public long TotalBytesEver => Interlocked.Read(ref _totalBytesEver);

    /// <summary>
    /// Gets the utilization ratio: UsedBytes / Capacity.
    /// </summary>
    public double Utilization => (double)UsedBytes / _capacity;

    /// <summary>
    /// Whether heap fallback is enabled for allocations that exceed remaining space.
    /// </summary>
    public bool AllowHeapFallback => _allowHeapFallback;

    /// <summary>
    /// Initializes a new stack allocator with the specified capacity.
    /// </summary>
    /// <param name="capacity">Total size of the allocation block in bytes.</param>
    /// <param name="alignment">Default alignment for allocations.</param>
    /// <param name="allowHeapFallback">
    /// If true, allocations that exceed remaining space fall back to the managed heap.
    /// If false, an <see cref="OutOfMemoryException"/> is thrown on overflow.
    /// </param>
    public StackAllocator(int capacity = DefaultCapacity, int alignment = DefaultAlignment, bool allowHeapFallback = false)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");
        if (alignment <= 0 || alignment > MaxAlignment)
            throw new ArgumentOutOfRangeException(nameof(alignment), $"Alignment must be between 1 and {MaxAlignment}.");
        if (!BitOperations.IsPow2((uint)alignment))
            throw new ArgumentException("Alignment must be a power of two.", nameof(alignment));

        _capacity = capacity;
        _allowHeapFallback = allowHeapFallback;
        _scopeStack = new Stack<int>(MaxScopes);
        _scopes = new ScopeInfo[MaxScopes];

        // Allocate the backing memory with maximum alignment to guarantee
        // any sub-alignment within the buffer is satisfiable.
        _basePointer = (byte*)NativeMemory.AlignedAlloc((nuint)capacity, (nuint)MaxAlignment);
        NativeMemory.Clear(_basePointer, (nuint)capacity);

        _usedBytes = 0;
        _allocationCount = 0;
        _scopeDepth = 0;
        _frameIndex = 0;
    }

    /// <summary>
    /// Allocates a block of the specified size with default alignment.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    /// <returns>A pointer to the allocated memory.</returns>
    /// <exception cref="ArgumentOutOfRangeException">If size is negative or zero.</exception>
    /// <exception cref="OutOfMemoryException">If allocation fails and heap fallback is disabled.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void* Allocate(int size)
    {
        return Allocate(size, DefaultAlignment);
    }

    /// <summary>
    /// Allocates a block of the specified size and alignment.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    /// <param name="alignment">Alignment requirement in bytes (must be a power of two).</param>
    /// <returns>A pointer to the allocated, properly aligned memory.</returns>
    /// <exception cref="ArgumentOutOfRangeException">If size is non-positive or alignment is invalid.</exception>
    /// <exception cref="OutOfMemoryException">If allocation fails and heap fallback is disabled.</exception>
    public void* Allocate(int size, int alignment)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Allocation size must be positive.");
        if (alignment <= 0 || alignment > MaxAlignment)
            throw new ArgumentOutOfRangeException(nameof(alignment), $"Alignment must be between 1 and {MaxAlignment}.");
        if (!BitOperations.IsPow2((uint)alignment))
            throw new ArgumentException("Alignment must be a power of two.", nameof(alignment));

        // Compute aligned offset.
        int currentUsed = _usedBytes;
        int alignedOffset = (currentUsed + alignment - 1) & ~(alignment - 1);
        int totalSize = alignedOffset + size;

        if (totalSize > _capacity)
        {
            if (_allowHeapFallback)
            {
                return AllocateHeapFallback(size, alignment);
            }

            ThrowOverflow(totalSize);
        }

        // Atomically reserve space. This allows safe concurrent reads of _usedBytes
        // for statistics while still providing thread-safe single-writer allocation.
        _usedBytes = totalSize;
        _allocationCount++;

        // Update statistics.
        UpdateStatistics(totalSize);

        // Return pointer into the contiguous block.
        return _basePointer + alignedOffset;
    }

    /// <summary>
    /// Allocates a typed buffer of the specified count with proper alignment for the element type.
    /// </summary>
    /// <typeparam name="T">Element type (must be an unmanaged struct).</typeparam>
    /// <param name="count">Number of elements to allocate.</param>
    /// <returns>A pointer to the allocated typed buffer.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T* Allocate<T>(int count) where T : unmanaged
    {
        int size = count * sizeof(T);
        int alignment = Math.Clamp(AlignmentOf<T>(), 1, MaxAlignment);
        void* ptr = Allocate(size, alignment);
        return (T*)ptr;
    }

    /// <summary>
    /// Allocates a typed buffer and returns it as a Span for safe indexed access.
    /// </summary>
    /// <typeparam name="T">Element type (must be an unmanaged struct).</typeparam>
    /// <param name="count">Number of elements to allocate.</param>
    /// <returns>A Span&lt;T&gt; over the allocated memory.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AllocateSpan<T>(int count) where T : unmanaged
    {
        T* ptr = Allocate<T>(count);
        return new Span<T>(ptr, count);
    }

    /// <summary>
    /// Allocates a ReadOnlySpan view over the specified count of typed elements.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="count">Number of elements.</param>
    /// <returns>A ReadOnlySpan&lt;T&gt; over the allocated memory.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AllocateReadOnlySpan<T>(int count) where T : unmanaged
    {
        T* ptr = Allocate<T>(count);
        return new ReadOnlySpan<T>(ptr, count);
    }

    /// <summary>
    /// Allocates memory for a single value and returns a pointer to it.
    /// Useful for frame-local temporaries that don't need to survive the frame.
    /// </summary>
    /// <typeparam name="T">Value type.</typeparam>
    /// <returns>A pointer to the allocated value (uninitialized).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T* AllocateOne<T>() where T : unmanaged
    {
        return Allocate<T>(1);
    }

    /// <summary>
    /// Allocates memory and initializes it with a specific value.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="count">Number of elements.</param>
    /// <param name="value">Value to fill each element with.</param>
    /// <returns>A Span&lt;T&gt; over the initialized memory.</returns>
    public Span<T> AllocateAndInitialize<T>(int count, T value) where T : unmanaged
    {
        Span<T> span = AllocateSpan<T>(count);
        span.Fill(value);
        return span;
    }

    /// <summary>
    /// Allocates a zeroed block of memory.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    /// <returns>A pointer to zeroed memory.</returns>
    public void* AllocateZeroed(int size)
    {
        void* ptr = Allocate(size);
        NativeMemory.Clear(ptr, (nuint)size);
        return ptr;
    }

    /// <summary>
    /// Allocates a zeroed typed buffer.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="count">Number of elements.</param>
    /// <returns>A Span&lt;T&gt; over the zeroed memory.</returns>
    public Span<T> AllocateZeroedSpan<T>(int count) where T : unmanaged
    {
        Span<T> span = AllocateSpan<T>(count);
        span.Clear();
        return span;
    }

    /// <summary>
    /// Returns the current allocation pointer (high-water mark) without allocating.
    /// Useful for manual rewind with <see cref="Rewind"/>.
    /// </summary>
    public int GetMarker()
    {
        return _usedBytes;
    }

    /// <summary>
    /// Rewinds the stack allocator to a previously saved marker.
    /// All allocations after the marker are invalidated.
    /// </summary>
    /// <param name="marker">A marker obtained from <see cref="GetMarker"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">If the marker is invalid.</exception>
    public void Rewind(int marker)
    {
        if (marker < 0 || marker > _usedBytes)
            throw new ArgumentOutOfRangeException(nameof(marker), "Marker is outside the valid allocation range.");

        // Clear the memory being rewound for deterministic zeroing.
        int rewindSize = _usedBytes - marker;
        NativeMemory.Clear(_basePointer + marker, (nuint)rewindSize);

        _usedBytes = marker;
    }

    /// <summary>
    /// Opens a new nested allocation scope. All allocations within this scope
    /// can be bulk-freed by calling <see cref="PopScope"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">If maximum scope depth is exceeded.</exception>
    public void PushScope()
    {
        if (_scopeDepth >= MaxScopes)
            throw new InvalidOperationException($"Maximum scope depth ({MaxScopes}) exceeded.");

        _scopes[_scopeDepth] = new ScopeInfo
        {
            Marker = _usedBytes,
            AllocationCount = _allocationCount
        };
        _scopeStack.Push(_scopeDepth);
        _scopeDepth++;
    }

    /// <summary>
    /// Closes the current allocation scope and frees all allocations made within it.
    /// This rewinds the stack to the state before the matching <see cref="PushScope"/> call.
    /// </summary>
    /// <exception cref="InvalidOperationException">If no scope is currently open.</exception>
    public void PopScope()
    {
        if (_scopeDepth == 0)
            throw new InvalidOperationException("No scope is currently open.");

        _scopeDepth--;
        int scopeIndex = _scopeStack.Pop();
        ScopeInfo scope = _scopes[scopeIndex];

        Rewind(scope.Marker);
        _allocationCount = scope.AllocationCount;
    }

    /// <summary>
    /// Executes an action within a new allocation scope, automatically popping the scope
    /// when the action completes (even if an exception is thrown).
    /// </summary>
    /// <param name="action">The action to execute within the scope.</param>
    public void WithScope(Action action)
    {
        PushScope();
        try
        {
            action();
        }
        finally
        {
            PopScope();
        }
    }

    /// <summary>
    /// Executes a function within a new allocation scope, returning its result.
    /// The scope is automatically popped when the function completes.
    /// </summary>
    /// <typeparam name="T">Return type.</typeparam>
    /// <param name="func">The function to execute within the scope.</param>
    /// <returns>The return value of the function.</returns>
    public T WithScope<T>(Func<T> func)
    {
        PushScope();
        try
        {
            return func();
        }
        finally
        {
            PopScope();
        }
    }

    /// <summary>
    /// Resets the allocator for a new frame.
    /// Frees all allocations, resets the scope depth, and increments the frame counter.
    /// This does not free the backing memory - only resets the allocation pointer.
    /// </summary>
    public void Reset()
    {
        // Zero the entire buffer for deterministic state.
        NativeMemory.Clear(_basePointer, (nuint)_capacity);

        _usedBytes = 0;
        _allocationCount = 0;
        _scopeDepth = 0;
        _scopeStack.Clear();
        _frameIndex++;

        // Update peak statistics.
        Interlocked.Exchange(ref _peakAllocationCount,
            Math.Max(_peakAllocationCount, _allocationCount));
    }

    /// <summary>
    /// Resets the allocator and sets it as the current thread-local allocator.
    /// </summary>
    public void ResetAndMakeCurrent()
    {
        Reset();
        t_current = this;
    }

    /// <summary>
    /// Gets a detailed snapshot of allocator statistics.
    /// </summary>
    public StackAllocatorStats GetStats()
    {
        return new StackAllocatorStats
        {
            Capacity = _capacity,
            UsedBytes = UsedBytes,
            RemainingBytes = RemainingBytes,
            Utilization = Utilization,
            AllocationCount = AllocationCount,
            ScopeDepth = ScopeDepth,
            FrameIndex = _frameIndex,
            PeakUsedBytes = _peakUsedBytes,
            PeakAllocationCount = _peakAllocationCount,
            TotalAllocationsEver = TotalAllocationsEver,
            TotalBytesEver = TotalBytesEver
        };
    }

    /// <summary>
    /// Attempts to allocate without throwing. Returns false if the allocation would overflow.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    /// <param name="pointer">The allocated pointer, or null on failure.</param>
    /// <returns>True if allocation succeeded; false on overflow.</returns>
    public bool TryAllocate(int size, out void* pointer)
    {
        return TryAllocate(size, DefaultAlignment, out pointer);
    }

    /// <summary>
    /// Attempts to allocate with alignment without throwing.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    /// <param name="alignment">Alignment requirement.</param>
    /// <param name="pointer">The allocated pointer, or null on failure.</param>
    /// <returns>True if allocation succeeded; false on overflow.</returns>
    public bool TryAllocate(int size, int alignment, out void* pointer)
    {
        int currentUsed = _usedBytes;
        int alignedOffset = (currentUsed + alignment - 1) & ~(alignment - 1);
        int totalSize = alignedOffset + size;

        if (totalSize > _capacity)
        {
            pointer = null;
            return false;
        }

        pointer = _basePointer + alignedOffset;
        _usedBytes = totalSize;
        _allocationCount++;
        UpdateStatistics(totalSize);
        return true;
    }

    /// <summary>
    /// Copies data from a source span into newly allocated stack memory.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="source">Source data to copy.</param>
    /// <returns>A Span&lt;T&gt; containing the copied data.</returns>
    public Span<T> AllocateCopy<T>(ReadOnlySpan<T> source) where T : unmanaged
    {
        Span<T> dest = AllocateSpan<T>(source.Length);
        source.CopyTo(dest);
        return dest;
    }

    /// <summary>
    /// Allocates a string on the stack and copies the specified string content into it.
    /// The returned pointer is null-terminated.
    /// </summary>
    /// <param name="text">The string to copy.</param>
    /// <returns>A pointer to the null-terminated UTF-8 string.</returns>
    public byte* AllocateString(string text)
    {
        int byteCount = System.Text.Encoding.UTF8.GetByteCount(text) + 1;
        byte* ptr = (byte*)Allocate(byteCount, 1);
        System.Text.Encoding.UTF8.GetBytes(text, new Span<byte>(ptr, byteCount - 1));
        ptr[byteCount - 1] = 0; // null terminator
        return ptr;
    }

    /// <summary>
    /// Gets a Span&lt;byte&gt; view over a region of the allocator's memory.
    /// </summary>
    /// <param name="offset">Byte offset from the base pointer.</param>
    /// <param name="length">Number of bytes.</param>
    public Span<byte> GetSpan(int offset, int length)
    {
        if (offset < 0 || offset + length > _usedBytes)
            throw new ArgumentOutOfRangeException(nameof(offset));

        return new Span<byte>(_basePointer + offset, length);
    }

    /// <summary>
    /// Gets a typed Span view over a region of the allocator's memory.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="offset">Byte offset from the base pointer.</param>
    /// <param name="count">Number of elements.</param>
    public Span<T> GetSpan<T>(int offset, int count) where T : unmanaged
    {
        int byteSize = count * sizeof(T);
        if (offset < 0 || offset + byteSize > _usedBytes)
            throw new ArgumentOutOfRangeException(nameof(offset));

        return new Span<T>(_basePointer + offset, count);
    }

    /// <summary>
    /// Gets a byte offset pointer relative to the base.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void* GetPointer(int offset)
    {
        return _basePointer + offset;
    }

    /// <summary>
    /// Determines the natural alignment for a given type.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int AlignmentOf<T>() where T : unmanaged
    {
        int size = sizeof(T);
        if (size >= 8)
            return 8;
        if (size >= 4)
            return 4;
        if (size >= 2)
            return 2;
        return 1;
    }

    /// <summary>
    /// Allocates from the managed heap as a fallback when the stack is full.
    /// </summary>
    private unsafe void* AllocateHeapFallback(int size, int alignment)
    {
        void* rawPtr = NativeMemory.AlignedAlloc((nuint)size, (nuint)alignment);
        _allocationCount++;
        Interlocked.Add(ref _totalBytesEver, size);
        Interlocked.Increment(ref _totalAllocationsEver);
        return rawPtr;
    }

    /// <summary>
    /// Updates internal peak statistics atomically.
    /// </summary>
    private void UpdateStatistics(int totalSize)
    {
        Interlocked.Add(ref _totalBytesEver, 0); // Touch for reads
        Interlocked.Increment(ref _totalAllocationsEver);

        int currentPeak = _peakUsedBytes;
        if (totalSize > currentPeak)
        {
            Interlocked.CompareExchange(ref _peakUsedBytes, totalSize, currentPeak);
        }

        int currentAllocPeak = _peakAllocationCount;
        if (_allocationCount > currentAllocPeak)
        {
            Interlocked.CompareExchange(ref _peakAllocationCount, _allocationCount, currentAllocPeak);
        }
    }

    /// <summary>
    /// Throws an <see cref="OutOfMemoryException"/> with details about the overflow.
    /// </summary>
    private static void ThrowOverflow(int requested)
    {
        throw new OutOfMemoryException(
            $"Stack allocator overflow: requested {requested} bytes. " +
            "Consider increasing capacity or enabling heap fallback.");
    }

    /// <summary>
    /// Disposes the allocator, freeing the backing native memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_basePointer != null)
        {
            NativeMemory.Free(_basePointer);
        }

        if (t_current == this)
        {
            t_current = null;
        }
    }

    /// <summary>
    /// Internal scope bookkeeping information.
    /// </summary>
    private struct ScopeInfo
    {
        /// <summary>Stack pointer (byte offset) when the scope was opened.</summary>
        public int Marker;

        /// <summary>Allocation count when the scope was opened.</summary>
        public int AllocationCount;
    }
}

/// <summary>
/// Thread-safe scoped guard for stack allocator scopes.
/// Automatically pops the scope when disposed, even on exceptions.
/// </summary>
public readonly struct StackScope : IDisposable
{
    private readonly StackAllocator _allocator;

    /// <summary>
    /// Creates a new scoped guard that will pop the scope on dispose.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public StackScope(StackAllocator allocator)
    {
        _allocator = allocator;
        _allocator.PushScope();
    }

    /// <summary>
    /// Pops the allocation scope when the guard is disposed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispose()
    {
        _allocator?.PopScope();
    }
}

/// <summary>
/// Comprehensive statistics snapshot for a <see cref="StackAllocator"/>.
/// </summary>
[DebuggerDisplay("Used={UsedBytes}/{Capacity} ({Utilization:P1}), Allocs={AllocationCount}, Frame={FrameIndex}")]
public sealed class StackAllocatorStats
{
    /// <summary>Total capacity in bytes.</summary>
    public int Capacity { get; init; }

    /// <summary>Bytes currently allocated.</summary>
    public int UsedBytes { get; init; }

    /// <summary>Bytes remaining.</summary>
    public int RemainingBytes { get; init; }

    /// <summary>Utilization ratio (0.0 - 1.0).</summary>
    public double Utilization { get; init; }

    /// <summary>Number of allocations in the current frame.</summary>
    public int AllocationCount { get; init; }

    /// <summary>Current scope nesting depth.</summary>
    public int ScopeDepth { get; init; }

    /// <summary>Current frame index (resets are frame boundaries).</summary>
    public int FrameIndex { get; init; }

    /// <summary>Peak bytes used in any single frame.</summary>
    public int PeakUsedBytes { get; init; }

    /// <summary>Peak allocation count in any single frame.</summary>
    public int PeakAllocationCount { get; init; }

    /// <summary>Total allocations across all frames.</summary>
    public long TotalAllocationsEver { get; init; }

    /// <summary>Total bytes allocated across all frames.</summary>
    public long TotalBytesEver { get; init; }

    /// <summary>Returns a formatted summary string.</summary>
    public override string ToString() =>
        $"StackAlloc: {UsedBytes / 1024.0:F1}KB / {Capacity / 1024.0:F1}KB " +
        $"({Utilization:P0}), {AllocationCount} allocs, Frame #{FrameIndex}";
}

/// <summary>
/// Factory and registry for managing thread-local stack allocators.
/// Provides centralized allocation management for the engine's frame pipeline.
/// </summary>
public sealed class StackAllocatorRegistry : IDisposable
{
    private readonly int _capacity;
    private readonly bool _allowHeapFallback;
    private readonly ConcurrentDictionary<int, StackAllocator> _allocators;
    private bool _disposed;

    /// <summary>
    /// Gets the number of registered thread-local allocators.
    /// </summary>
    public int Count => _allocators.Count;

    /// <summary>
    /// Creates a new registry with the specified default capacity for thread-local allocators.
    /// </summary>
    /// <param name="capacity">Capacity in bytes for each thread's allocator.</param>
    /// <param name="allowHeapFallback">Whether to allow heap fallback on overflow.</param>
    public StackAllocatorRegistry(int capacity = StackAllocator.DefaultCapacity, bool allowHeapFallback = false)
    {
        _capacity = capacity;
        _allowHeapFallback = allowHeapFallback;
        _allocators = new ConcurrentDictionary<int, StackAllocator>();
    }

    /// <summary>
    /// Gets or creates a stack allocator for the current thread.
    /// If one doesn't exist yet, it is created and registered.
    /// </summary>
    public StackAllocator GetCurrentThreadAllocator()
    {
        int threadId = Thread.CurrentThread.ManagedThreadId;
        return _allocators.GetOrAdd(threadId, _ => new StackAllocator(_capacity, StackAllocator.DefaultAlignment, _allowHeapFallback));
    }

    /// <summary>
    /// Gets the stack allocator for the specified thread, if it exists.
    /// </summary>
    /// <param name="threadId">The managed thread ID.</param>
    /// <returns>The allocator, or null if not registered.</returns>
    public StackAllocator? GetAllocator(int threadId)
    {
        _allocators.TryGetValue(threadId, out var allocator);
        return allocator;
    }

    /// <summary>
    /// Resets all registered allocators for a new frame.
    /// </summary>
    public void ResetAll()
    {
        foreach (var kvp in _allocators)
        {
            kvp.Value.Reset();
        }
    }

    /// <summary>
    /// Gets aggregate statistics across all registered allocators.
    /// </summary>
    public RegistryStats GetAggregateStats()
    {
        long totalUsed = 0;
        long totalCapacity = 0;
        int totalAllocations = 0;
        long totalPeakUsed = 0;

        foreach (var kvp in _allocators)
        {
            var stats = kvp.Value.GetStats();
            totalUsed += stats.UsedBytes;
            totalCapacity += stats.Capacity;
            totalAllocations += stats.AllocationCount;
            totalPeakUsed += stats.PeakUsedBytes;
        }

        return new RegistryStats
        {
            AllocatorCount = _allocators.Count,
            TotalUsedBytes = totalUsed,
            TotalCapacityBytes = totalCapacity,
            TotalAllocationCount = totalAllocations,
            TotalPeakUsedBytes = totalPeakUsed,
            AverageUtilization = totalCapacity > 0 ? (double)totalUsed / totalCapacity : 0.0
        };
    }

    /// <summary>
    /// Disposes all registered allocators.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        foreach (var kvp in _allocators)
        {
            kvp.Value.Dispose();
        }
        _allocators.Clear();
    }

    /// <summary>
    /// Aggregate statistics across all registered allocators.
    /// </summary>
    public sealed class RegistryStats
    {
        /// <summary>Number of registered allocators.</summary>
        public int AllocatorCount { get; init; }

        /// <summary>Total bytes used across all allocators.</summary>
        public long TotalUsedBytes { get; init; }

        /// <summary>Total capacity across all allocators.</summary>
        public long TotalCapacityBytes { get; init; }

        /// <summary>Total allocation count across all allocators.</summary>
        public int TotalAllocationCount { get; init; }

        /// <summary>Sum of peak usage across all allocators.</summary>
        public long TotalPeakUsedBytes { get; init; }

        /// <summary>Average utilization across all allocators.</summary>
        public double AverageUtilization { get; init; }

        /// <summary>Returns a formatted summary.</summary>
        public override string ToString() =>
            $"Registry: {AllocatorCount} allocators, " +
            $"{TotalUsedBytes / 1024.0:F1}KB / {TotalCapacityBytes / 1024.0:F1}KB " +
            $"({AverageUtilization:P0}), {TotalAllocationCount} allocs";
    }
}
