using System;
// ============================================================
// FILE: NativeBuffer.cs
// PATH: Memory/NativeBuffer.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - Native Buffer
// Native memory buffer wrapper with aligned allocation, typed access, and Span views.

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
/// Disposal state of a <see cref="NativeBuffer{T}"/>.
/// </summary>
public enum NativeBufferState
{
    Unallocated,
    Allocated,
    Disposed
}

/// <summary>
/// A native memory buffer wrapper using <see cref="NativeMemory"/> for aligned allocation.
/// Provides typed access via <see cref="Unsafe"/>, Span/Memory views, and safe disposal.
/// Designed for high-performance scenarios where GC-managed arrays are unacceptable.
/// </summary>
/// <typeparam name="T">Element type (must be an unmanaged struct).</typeparam>
[DebuggerDisplay("Length={Length}, ElementSize={sizeof(T)}, State={State}, Alignment={Alignment}")]
public sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>Default alignment: 16 bytes (Vector128 compatible).</summary>
    public const int DefaultAlignment = 16;

    /// <summary>SIMD-friendly alignment: 32 bytes (Vector256 compatible).</summary>
    public const int SimdAlignment = 32;

    /// <summary>AVX-512 alignment: 64 bytes.</summary>
    public const int Avx512Alignment = 64;

    private T* _pointer;
    private T* _rawPointer;
    private int _length;
    private int _alignment;
    private NativeBufferState _state;
    private bool _disposed;
    private int _version;

    /// <summary>
    /// Gets the pointer to the first element of the buffer.
    /// Valid only while the buffer is not disposed.
    /// </summary>
    public T* Pointer => _pointer;

    /// <summary>
    /// Gets the number of elements in the buffer.
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// Gets the element size in bytes.
    /// </summary>
    public int ElementSize => sizeof(T);

    /// <summary>
    /// Gets the total size of the buffer in bytes.
    /// </summary>
    public int ByteLength => _length * sizeof(T);

    /// <summary>
    /// Gets the alignment of the buffer in bytes.
    /// </summary>
    public int Alignment => _alignment;

    /// <summary>
    /// Gets the current state of the buffer.
    /// </summary>
    public NativeBufferState State => _state;

    /// <summary>
    /// Gets the version of the buffer (incremented on structural changes).
    /// </summary>
    public int Version => _version;

    /// <summary>
    /// Gets whether the buffer has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Gets whether the buffer is allocated and valid for use.
    /// </summary>
    public bool IsValid => _state == NativeBufferState.Allocated && !_disposed;

    /// <summary>
    /// Gets or sets the element at the specified index.
    /// Does not perform bounds checking for performance.
    /// </summary>
    public T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _pointer[index];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => _pointer[index] = value;
    }

    /// <summary>
    /// Initializes a new native buffer with the specified length and default alignment.
    /// </summary>
    /// <param name="length">Number of elements to allocate.</param>
    /// <param name="alignment">Memory alignment in bytes (must be a power of two).</param>
    public NativeBuffer(int length, int alignment = DefaultAlignment)
    {
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        if (alignment <= 0 || !BitOperations.IsPow2((uint)alignment))
            throw new ArgumentOutOfRangeException(nameof(alignment), "Alignment must be a positive power of two.");

        _length = length;
        _alignment = alignment;
        _state = NativeBufferState.Unallocated;

        if (length > 0)
        {
            AllocateInternal();
        }
    }

    /// <summary>
    /// Allocates the internal buffer.
    /// </summary>
    private void AllocateInternal()
    {
        int totalBytes = _length * sizeof(T) + _alignment;
        _rawPointer = (T*)NativeMemory.AlignedAlloc((nuint)totalBytes, (nuint)_alignment);

        // Align the pointer.
        nuint rawAddr = (nuint)_rawPointer;
        nuint alignedAddr = (rawAddr + (nuint)(_alignment - 1)) & ~(nuint)(_alignment - 1);
        _pointer = (T*)alignedAddr;

        // Zero-initialize for deterministic state.
        NativeMemory.Clear(_pointer, (nuint)(_length * sizeof(T)));

        _state = NativeBufferState.Allocated;
        _version++;
    }

    /// <summary>
    /// Creates a new native buffer from an existing pointer.
    /// The caller is responsible for the memory lifetime.
    /// </summary>
    /// <param name="pointer">Pointer to the pre-allocated memory.</param>
    /// <param name="length">Number of elements.</param>
    /// <returns>A new NativeBuffer wrapping the pointer.</returns>
    public static NativeBuffer<T> FromPointer(T* pointer, int length)
    {
        if (pointer == null)
            throw new ArgumentNullException(nameof(pointer));
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));

        var buffer = new NativeBuffer<T>(0);
        buffer._pointer = pointer;
        buffer._rawPointer = null;
        buffer._length = length;
        buffer._state = NativeBufferState.Allocated;
        return buffer;
    }

    /// <summary>
    /// Allocates a new native buffer from the specified <see cref="StackAllocator"/>.
    /// The buffer's lifetime is tied to the stack allocator's current scope.
    /// </summary>
    /// <param name="stackAllocator">The stack allocator to allocate from.</param>
    /// <param name="count">Number of elements.</param>
    public static NativeBuffer<T> FromStackAllocator(StackAllocator stackAllocator, int count)
    {
        if (stackAllocator == null)
            throw new ArgumentNullException(nameof(stackAllocator));

        T* ptr = stackAllocator.Allocate<T>(count);
        var buffer = new NativeBuffer<T>(0);
        buffer._pointer = ptr;
        buffer._rawPointer = null;
        buffer._length = count;
        buffer._state = NativeBufferState.Allocated;
        return buffer;
    }

    /// <summary>
    /// Gets a Span<T> view over the entire buffer.
    /// </summary>
    public Span<T> AsSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new Span<T>(_pointer, _length);
    }

    /// <summary>
    /// Gets a Span<T> view over a portion of the buffer.
    /// </summary>
    /// <param name="start">Starting element index.</param>
    /// <param name="length">Number of elements.</param>
    public Span<T> AsSpan(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (start < 0 || start + length > _length)
            throw new ArgumentOutOfRangeException(nameof(start));
        return new Span<T>(_pointer + start, length);
    }

    /// <summary>
    /// Gets a ReadOnlySpan<T> view over the entire buffer.
    /// </summary>
    public ReadOnlySpan<T> AsReadOnlySpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<T>(_pointer, _length);
    }

    /// <summary>
    /// Gets a ReadOnlySpan<T> view over a portion of the buffer.
    /// </summary>
    public ReadOnlySpan<T> AsReadOnlySpan(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (start < 0 || start + length > _length)
            throw new ArgumentOutOfRangeException(nameof(start));
        return new ReadOnlySpan<T>(_pointer + start, length);
    }

    /// <summary>
    /// Gets a Span<byte> view over the raw bytes of the buffer.
    /// </summary>
    public Span<byte> AsByteSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new Span<byte>(_pointer, _length * sizeof(T));
    }

    /// <summary>
    /// Gets a Memory<T> view over the buffer for interop with managed APIs.
    /// </summary>
    public Memory<T> AsMemory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var array = new T[_length];
        if (_length > 0)
        {
            unsafe
            {
                for (int i = 0; i < _length; i++)
                    array[i] = _pointer[i];
            }
        }
        return array.AsMemory();
    }

    /// <summary>
    /// Gets a Memory<T> view over a portion of the buffer.
    /// </summary>
    public Memory<T> AsMemory(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (start < 0 || start + length > _length)
            throw new ArgumentOutOfRangeException(nameof(start));
        var array = new T[length];
        if (length > 0)
        {
            unsafe
            {
                for (int i = 0; i < length; i++)
                    array[i] = _pointer[start + i];
            }
        }
        return array.AsMemory();
    }

    /// <summary>
    /// Resizes the buffer to the specified new length.
    /// Existing data is preserved up to the minimum of old and new lengths.
    /// New elements are zero-initialized.
    /// </summary>
    /// <param name="newLength">The new number of elements.</param>
    public void Resize(int newLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (newLength < 0)
            throw new ArgumentOutOfRangeException(nameof(newLength));
        if (newLength == _length)
            return;

        if (newLength == 0)
        {
            Dispose();
            return;
        }

        int copyCount = Math.Min(_length, newLength);
        T* newRawPointer = (T*)NativeMemory.AlignedAlloc(
            (nuint)(newLength * sizeof(T) + _alignment), (nuint)_alignment);

        nuint rawAddr = (nuint)newRawPointer;
        nuint alignedAddr = (rawAddr + (nuint)(_alignment - 1)) & ~(nuint)(_alignment - 1);
        T* newPointer = (T*)alignedAddr;

        // Copy existing data.
        if (copyCount > 0 && _pointer != null)
        {
            var source = new ReadOnlySpan<T>(_pointer, copyCount);
            var destination = new Span<T>(newPointer, copyCount);
            source.CopyTo(destination);
        }

        // Zero new elements.
        if (newLength > copyCount)
        {
            NativeMemory.Clear(newPointer + copyCount, (nuint)((newLength - copyCount) * sizeof(T)));
        }

        // Free old allocation.
        if (_rawPointer != null)
        {
            NativeMemory.Free(_rawPointer);
        }

        _rawPointer = newRawPointer;
        _pointer = newPointer;
        _length = newLength;
        _version++;
    }

    /// <summary>
    /// Grows the buffer to at least the specified capacity.
    /// If the buffer is already large enough, this is a no-op.
    /// Growth strategy: double the current capacity, or use the requested capacity
    /// if it is larger.
    /// </summary>
    /// <param name="capacity">Minimum capacity required.</param>
    public void EnsureCapacity(int capacity)
    {
        if (_length >= capacity)
            return;
        int newCapacity = Math.Max(capacity, _length * 2);
        Resize(newCapacity);
    }

    /// <summary>
    /// Clears all elements to their default value.
    /// </summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_length == 0)
            return;
        NativeMemory.Clear(_pointer, (nuint)(_length * sizeof(T)));
        _version++;
    }

    /// <summary>
    /// Fills all elements with the specified value.
    /// </summary>
    /// <param name="value">The value to fill with.</param>
    public void Fill(T value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        AsSpan().Fill(value);
        _version++;
    }

    /// <summary>
    /// Copies data from a ReadOnlySpan into the buffer.
    /// </summary>
    /// <param name="source">Source data to copy.</param>
    public void CopyFrom(ReadOnlySpan<T> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length > _length)
            throw new ArgumentException("Source is larger than the buffer.");

        source.CopyTo(AsSpan());
        _version++;
    }

    /// <summary>
    /// Copies data from the buffer into a destination Span.
    /// </summary>
    /// <param name="destination">Destination span to copy into.</param>
    public void CopyTo(Span<T> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        AsReadOnlySpan().CopyTo(destination);
    }

    /// <summary>
    /// Returns a pointer to the element at the specified index.
    /// </summary>
    /// <param name="index">Element index.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T* GetPointer(int index)
    {
        return _pointer + index;
    }

    /// <summary>
    /// Reads a value at the specified index using Unsafe.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref T GetRef(int index)
    {
        return ref Unsafe.AsRef<T>(_pointer + index);
    }

    /// <summary>
    /// Reads a value of a different type from the specified byte offset.
    /// Useful for reinterpret casting within the buffer.
    /// </summary>
    /// <typeparam name="U">Target type to read.</typeparam>
    /// <param name="byteOffset">Byte offset from the start of the buffer.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public U ReadAs<U>(int byteOffset) where U : unmanaged
    {
        return Unsafe.Read<U>((byte*)_pointer + byteOffset);
    }

    /// <summary>
    /// Writes a value of a different type at the specified byte offset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteAs<U>(int byteOffset, U value) where U : unmanaged
    {
        Unsafe.Write((byte*)_pointer + byteOffset, value);
    }

    /// <summary>
    /// Reinterprets the buffer as a span of a different type with the same byte size per element.
    /// </summary>
    /// <typeparam name="U">Target element type.</typeparam>
    /// <returns>A Span<U> over the same memory.</returns>
    public Span<U> ReinterpretAs<U>() where U : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (sizeof(T) != sizeof(U))
            throw new ArgumentException(
                $"Cannot reinterpret between types of different sizes: {sizeof(T)} vs {sizeof(U)} bytes.");

        return MemoryMarshal.Cast<T, U>(AsSpan());
    }

    /// <summary>
    /// Swaps the contents of this buffer with another buffer of the same type.
    /// </summary>
    /// <param name="other">The other buffer to swap with.</param>
    public void Swap(NativeBuffer<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        var tempPointer = _pointer;
        _pointer = other._pointer;
        other._pointer = tempPointer;

        var tempRawPointer = _rawPointer;
        _rawPointer = other._rawPointer;
        other._rawPointer = tempRawPointer;

        (_length, other._length) = (other._length, _length);
        (_alignment, other._alignment) = (other._alignment, _alignment);
        (_state, other._state) = (other._state, _state);
        _version++;
        other._version++;
    }

    /// <summary>
    /// Creates a deep copy of this buffer.
    /// </summary>
    /// <returns>A new NativeBuffer containing a copy of the data.</returns>
    public NativeBuffer<T> Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var clone = new NativeBuffer<T>(_length, _alignment);
        if (_length > 0)
        {
            AsReadOnlySpan().CopyTo(clone.AsSpan());
        }
        return clone;
    }

    /// <summary>
    /// Slices the buffer to create a view over a sub-range.
    /// The returned buffer shares the same memory (no copy).
    /// The returned buffer does not own the memory and must not be disposed independently.
    /// </summary>
    /// <param name="start">Starting element index.</param>
    /// <param name="length">Number of elements in the slice.</param>
    public NativeBuffer<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (start < 0 || start + length > _length)
            throw new ArgumentOutOfRangeException(nameof(start));

        return FromPointer(_pointer + start, length);
    }

    /// <summary>
    /// Checks if the buffer's starting address is aligned to the specified boundary.
    /// </summary>
    public bool IsAligned(int alignment)
    {
        return ((nuint)_pointer % (nuint)alignment) == 0;
    }

    /// <summary>
    /// Disposes the buffer and frees native memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_rawPointer != null)
        {
            NativeMemory.Free(_rawPointer);
            _rawPointer = null;
        }

        _pointer = null;
        _state = NativeBufferState.Disposed;
    }
}

/// <summary>
/// Non-generic native buffer for raw byte-level operations.
/// Useful when the element type is determined at runtime.
/// </summary>
[DebuggerDisplay("ByteLength={ByteLength}, State={State}, Alignment={Alignment}")]
public sealed unsafe class NativeByteBuffer : IDisposable
{
    private byte* _pointer;
    private byte* _rawPointer;
    private int _byteLength;
    private int _alignment;
    private bool _disposed;

    /// <summary>Gets the pointer to the buffer data.</summary>
    public byte* Pointer => _pointer;

    /// <summary>Gets the buffer length in bytes.</summary>
    public int ByteLength => _byteLength;

    /// <summary>Gets the alignment in bytes.</summary>
    public int Alignment => _alignment;

    /// <summary>Gets whether the buffer is disposed.</summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Initializes a new native byte buffer.
    /// </summary>
    /// <param name="byteLength">Size in bytes.</param>
    /// <param name="alignment">Alignment in bytes.</param>
    public NativeByteBuffer(int byteLength, int alignment = 16)
    {
        if (byteLength < 0)
            throw new ArgumentOutOfRangeException(nameof(byteLength));
        if (alignment <= 0 || !BitOperations.IsPow2((uint)alignment))
            throw new ArgumentOutOfRangeException(nameof(alignment));

        _byteLength = byteLength;
        _alignment = alignment;

        if (byteLength > 0)
        {
            int totalBytes = byteLength + alignment;
            _rawPointer = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, (nuint)alignment);
            nuint rawAddr = (nuint)_rawPointer;
            nuint alignedAddr = (rawAddr + (nuint)(alignment - 1)) & ~(nuint)(alignment - 1);
            _pointer = (byte*)alignedAddr;
            NativeMemory.Clear(_pointer, (nuint)byteLength);
        }
    }

    /// <summary>
    /// Gets a Span<byte> over the entire buffer.
    /// </summary>
    public Span<byte> AsSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new Span<byte>(_pointer, _byteLength);
    }

    /// <summary>
    /// Gets a Span<byte> over a portion of the buffer.
    /// </summary>
    public Span<byte> AsSpan(int start, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (start < 0 || start + length > _byteLength)
            throw new ArgumentOutOfRangeException(nameof(start));
        return new Span<byte>(_pointer + start, length);
    }

    /// <summary>
    /// Gets a ReadOnlySpan<byte> over the entire buffer.
    /// </summary>
    public ReadOnlySpan<byte> AsReadOnlySpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<byte>(_pointer, _byteLength);
    }

    /// <summary>
    /// Gets a typed Span<T> view over the buffer.
    /// </summary>
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int count = _byteLength / sizeof(T);
        return new Span<T>(_pointer, count);
    }

    /// <summary>
    /// Gets a typed Span<T> view starting at the specified byte offset.
    /// </summary>
    public Span<T> AsSpan<T>(int byteOffset, int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int requiredBytes = count * sizeof(T);
        if (byteOffset < 0 || byteOffset + requiredBytes > _byteLength)
            throw new ArgumentOutOfRangeException(nameof(byteOffset));
        return new Span<T>(_pointer + byteOffset, count);
    }

    /// <summary>
    /// Copies data from a source span into the buffer.
    /// </summary>
    public void CopyFrom(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length > _byteLength)
            throw new ArgumentException("Source is larger than the buffer.");
        source.CopyTo(AsSpan());
    }

    /// <summary>
    /// Copies data from the buffer into a destination span.
    /// </summary>
    public void CopyTo(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        AsReadOnlySpan().CopyTo(destination);
    }

    /// <summary>
    /// Clears the entire buffer to zero.
    /// </summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMemory.Clear(_pointer, (nuint)_byteLength);
    }

    /// <summary>
    /// Fills the buffer with the specified byte value.
    /// </summary>
    public void Fill(byte value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Unsafe.InitBlockUnaligned(_pointer, value, (uint)_byteLength);
    }

    /// <summary>
    /// Resizes the buffer, preserving existing data.
    /// </summary>
    public void Resize(int newByteLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (newByteLength < 0)
            throw new ArgumentOutOfRangeException(nameof(newByteLength));
        if (newByteLength == _byteLength)
            return;

        int copyLength = Math.Min(_byteLength, newByteLength);
        int totalBytes = newByteLength + _alignment;
        byte* newRaw = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, (nuint)_alignment);
        nuint rawAddr = (nuint)newRaw;
        nuint alignedAddr = (rawAddr + (nuint)(_alignment - 1)) & ~(nuint)(_alignment - 1);
        byte* newPtr = (byte*)alignedAddr;

        if (copyLength > 0 && _pointer != null)
        {
            Buffer.MemoryCopy(_pointer, newPtr, newByteLength, copyLength);
        }

        if (newByteLength > copyLength)
        {
            NativeMemory.Clear(newPtr + copyLength, (nuint)(newByteLength - copyLength));
        }

        if (_rawPointer != null)
            NativeMemory.Free(_rawPointer);

        _rawPointer = newRaw;
        _pointer = newPtr;
        _byteLength = newByteLength;
    }

    /// <summary>
    /// Disposes the buffer and frees native memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_rawPointer != null)
        {
            NativeMemory.Free(_rawPointer);
            _rawPointer = null;
        }
        _pointer = null;
    }
}


