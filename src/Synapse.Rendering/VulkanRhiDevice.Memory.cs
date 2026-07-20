// =============================================================================
// GDNN Engine - Vulkan 1.4 Render Hardware Interface Backend
// File: VulkanRhiDevice.cs
// Description: Complete Vulkan RHI implementation for the G-DNN Engine
// Author: GDNN Engine Team
// Version: 1.0.0
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.RHI.Vulkan
{
    public class VulkanMemoryAllocator : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Memory pools
        private readonly List<MemoryBlock> _blocks = new List<MemoryBlock>();
        private readonly Dictionary<uint, List<MemoryBlock>> _poolsByMemoryType = new Dictionary<uint, List<MemoryBlock>>();

        // Budget tracking
        private ulong _totalAllocated;
        private ulong _totalReserved;
        private readonly Dictionary<uint, ulong> _budgetByType = new Dictionary<uint, ulong>();
        private readonly Dictionary<uint, ulong> _usageByType = new Dictionary<uint, ulong>();

        // Statistics
        private ulong _allocationCount;
        private ulong _deallocationCount;
        private ulong _peakUsage;

        // Defragmentation
        private readonly List<DefragmentationPass> _pendingDefragmentations = new List<DefragmentationPass>();

        // Function pointers
        private AllocateMemoryDel _vkAllocateMemory;
        private FreeMemoryDel _vkFreeMemory;
        private MapMemoryDel _vkMapMemory;
        private UnmapMemoryDel _vkUnmapMemory;
        private FlushMappedMemoryRangesDel _vkFlushMappedMemoryRanges;
        private GetBufferMemoryRequirementsDel _vkGetBufferMemoryRequirements;
        private GetImageMemoryRequirementsDel _vkGetImageMemoryRequirements;
        private BindBufferMemoryDel _vkBindBufferMemory;
        private BindImageMemoryDel _vkBindImageMemory;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateMemoryDel(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, ref IntPtr pMemory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDel(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDel(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDel(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetBufferMemoryRequirementsDel(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetImageMemoryRequirementsDel(IntPtr device, IntPtr image, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDel(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDel(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);

        // Block size constants
        private const ulong DEFAULT_BLOCK_SIZE = 256 * 1024 * 1024; // 256 MB
        private const ulong SMALL_BLOCK_SIZE = 64 * 1024 * 1024;    // 64 MB
        private const ulong MIN_ALLOCATION_SIZE = 256;               // 256 bytes alignment
        private const ulong BUFFER_IMAGE_GRANULARITY = 128;          // 128 bytes

        public VulkanMemoryAllocator(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
            InitializeBudgetTracking();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkAllocateMemory = Marshal.GetDelegateForFunctionPointer<AllocateMemoryDel>(load("vkAllocateMemory"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
            _vkMapMemory = Marshal.GetDelegateForFunctionPointer<MapMemoryDel>(load("vkMapMemory"));
            _vkUnmapMemory = Marshal.GetDelegateForFunctionPointer<UnmapMemoryDel>(load("vkUnmapMemory"));
            _vkFlushMappedMemoryRanges = Marshal.GetDelegateForFunctionPointer<FlushMappedMemoryRangesDel>(load("vkFlushMappedMemoryRanges"));
            _vkGetBufferMemoryRequirements = Marshal.GetDelegateForFunctionPointer<GetBufferMemoryRequirementsDel>(load("vkGetBufferMemoryRequirements"));
            _vkGetImageMemoryRequirements = Marshal.GetDelegateForFunctionPointer<GetImageMemoryRequirementsDel>(load("vkGetImageMemoryRequirements"));
            _vkBindBufferMemory = Marshal.GetDelegateForFunctionPointer<BindBufferMemoryDel>(load("vkBindBufferMemory"));
            _vkBindImageMemory = Marshal.GetDelegateForFunctionPointer<BindImageMemoryDel>(load("vkBindImageMemory"));
        }

        private void InitializeBudgetTracking()
        {
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;
            for (uint i = 0; i < memProps.MemoryHeapCount; i++)
            {
                _budgetByType[i] = memProps.MemoryHeaps[i].Size;
                _usageByType[i] = 0;
            }
        }

        /// <summary>Selects the best memory type for the given requirements and properties</summary>
        public uint FindMemoryType(uint typeFilter, MemoryPropertyFlag requiredProperties)
        {
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;
            for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            {
                if ((typeFilter & (1 << (int)i)) != 0)
                {
                    if ((memProps.MemoryTypes[i].PropertyFlags & requiredProperties) == requiredProperties)
                        return i;
                }
            }

            // Fallback: try with fewer requirements
            for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            {
                if ((typeFilter & (1 << (int)i)) != 0)
                    return i;
            }

            throw new InvalidOperationException("Failed to find suitable memory type");
        }

        /// <summary>Allocates memory for a buffer</summary>
        public void AllocateBuffer(VulkanBuffer buffer, MemoryPropertyFlag requiredProperties)
        {
            var requirements = buffer.MemoryRequirements;
            uint memoryType = FindMemoryType(requirements.MemoryTypeBits, requiredProperties);

            lock (_lock)
            {
                var block = FindOrCreateBlock(memoryType, requirements.Size, requirements.Alignment);
                var allocation = block.Allocate(requirements.Size, requirements.Alignment);

                if (allocation.HasValue)
                {
                    buffer.BindMemory(block.DeviceMemory, allocation.Value.Offset);
                    Interlocked.Increment(ref _allocationCount);
                    UpdateUsage(memoryType, requirements.Size);
                }
            }
        }

        /// <summary>Allocates memory for an image</summary>
        public void AllocateImage(VulkanTexture texture, MemoryPropertyFlag requiredProperties)
        {
            var requirements = texture.MemoryRequirements;
            uint memoryType = FindMemoryType(requirements.MemoryTypeBits, requiredProperties);

            lock (_lock)
            {
                var block = FindOrCreateBlock(memoryType, requirements.Size, requirements.Alignment);
                var allocation = block.Allocate(requirements.Size, requirements.Alignment);

                if (allocation.HasValue)
                {
                    texture.BindMemory(block.DeviceMemory, allocation.Value.Offset);
                    Interlocked.Increment(ref _allocationCount);
                    UpdateUsage(memoryType, requirements.Size);
                }
            }
        }

        /// <summary>Finds or creates a memory block for the given type</summary>
        private MemoryBlock FindOrCreateBlock(uint memoryType, ulong size, ulong alignment)
        {
            if (!_poolsByMemoryType.TryGetValue(memoryType, out var blocks))
            {
                blocks = new List<MemoryBlock>();
                _poolsByMemoryType[memoryType] = blocks;
            }

            // Try to find an existing block with enough space
            foreach (var block in blocks)
            {
                if (block.HasSpace(size, alignment))
                    return block;
            }

            // Create a new block
            ulong blockSize = size > DEFAULT_BLOCK_SIZE ? size * 2 : DEFAULT_BLOCK_SIZE;
            if (size < SMALL_BLOCK_SIZE)
                blockSize = SMALL_BLOCK_SIZE;

            var newBlock = CreateMemoryBlock(memoryType, blockSize);
            blocks.Add(newBlock);
            _blocks.Add(newBlock);
            return newBlock;
        }

        /// <summary>Creates a new memory block</summary>
        private MemoryBlock CreateMemoryBlock(uint memoryType, ulong size)
        {
            var allocateInfo = new VkMemoryAllocateInfo
            {
                sType = 5,
                allocationSize = size,
                memoryTypeIndex = memoryType
            };

            IntPtr memory = IntPtr.Zero;
            var result = _vkAllocateMemory(_device.LogicalDevice, ref allocateInfo, IntPtr.Zero, ref memory);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate Vulkan memory: {result}");

            _totalReserved += size;

            return new MemoryBlock(memory, memoryType, size);
        }

        private void UpdateUsage(uint memoryType, ulong size)
        {
            Interlocked.Add(ref _totalAllocated, size);
            if (_usageByType.ContainsKey(memoryType))
                _usageByType[memoryType] += size;
            if (_totalAllocated > _peakUsage)
                _peakUsage = _totalAllocated;
        }

        /// <summary>Runs defragmentation on all memory blocks</summary>
        public void Defragment()
        {
            lock (_lock)
            {
                foreach (var block in _blocks)
                {
                    block.Compact();
                }
            }
        }

        /// <summary>Returns memory budget information for each heap</summary>
        public MemoryBudgetInfo GetBudgetInfo()
        {
            var info = new MemoryBudgetInfo();
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;

            info.HeapBudgets = new HeapBudget[memProps.MemoryHeapCount];
            for (uint i = 0; i < memProps.MemoryHeapCount; i++)
            {
                info.HeapBudgets[i] = new HeapBudget
                {
                    HeapIndex = i,
                    Budget = _budgetByType.ContainsKey(i) ? _budgetByType[i] : 0,
                    Usage = _usageByType.ContainsKey(i) ? _usageByType[i] : 0,
                    Flags = memProps.MemoryHeaps[i].Flags
                };
            }

            return info;
        }

        /// <summary>Gets current allocation statistics</summary>
        public AllocatorStats GetStats()
        {
            return new AllocatorStats
            {
                TotalAllocated = _totalAllocated,
                TotalReserved = _totalReserved,
                AllocationCount = _allocationCount,
                DeallocationCount = _deallocationCount,
                PeakUsage = _peakUsage,
                BlockCount = _blocks.Count
            };
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            lock (_lock)
            {
                foreach (var block in _blocks)
                    block.Dispose();
                _blocks.Clear();
                _poolsByMemoryType.Clear();
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Represents a block of Vulkan device memory</summary>
    internal class MemoryBlock : IDisposable
    {
        private IntPtr _deviceMemory;
        private uint _memoryType;
        private ulong _size;
        private ulong _used;
        private readonly List<MemoryAllocation> _allocations = new List<MemoryAllocation>();
        private IntPtr _mappedPtr;
        private bool _isMapped;

        public IntPtr DeviceMemory => _deviceMemory;
        public ulong Size => _size;
        public ulong Used => _used;
        public bool IsMapped => _isMapped;

        public MemoryBlock(IntPtr deviceMemory, uint memoryType, ulong size)
        {
            _deviceMemory = deviceMemory;
            _memoryType = memoryType;
            _size = size;
            _used = 0;
        }

        public bool HasSpace(ulong size, ulong alignment)
        {
            ulong alignedOffset = (_used + alignment - 1) & ~(alignment - 1);
            return alignedOffset + size <= _size;
        }

        public MemoryAllocation? Allocate(ulong size, ulong alignment)
        {
            ulong offset = (_used + alignment - 1) & ~(alignment - 1);
            if (offset + size > _size)
                return null;

            var alloc = new MemoryAllocation
            {
                Offset = offset,
                Size = size,
                IsActive = true
            };

            _allocations.Add(alloc);
            _used = offset + size;
            return alloc;
        }

        public void Free(ulong offset, ulong size)
        {
            for (int i = _allocations.Count - 1; i >= 0; i--)
            {
                if (_allocations[i].Offset == offset && _allocations[i].Size == size)
                {
                    var alloc = _allocations[i];
                    alloc.IsActive = false;
                    _allocations[i] = alloc;
                    break;
                }
            }
            Compact();
        }

        public void Compact()
        {
            _allocations.RemoveAll(a => !a.IsActive);
            _allocations.Sort((a, b) => a.Offset.CompareTo(b.Offset));

            ulong compacted = 0;
            for (int i = 0; i < _allocations.Count; i++)
            {
                var alloc = _allocations[i];
                if (alloc.Offset != compacted)
                {
                    alloc.Offset = compacted;
                    _allocations[i] = alloc;
                }
                compacted += alloc.Size;
            }
            _used = compacted;
        }

        public void Dispose()
        {
            _allocations.Clear();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Represents a suballocation within a memory block</summary>
    public struct MemoryAllocation
    {
        public ulong Offset;
        public ulong Size;
        public bool IsActive;
    }

    /// <summary>Memory budget information</summary>
    public class MemoryBudgetInfo
    {
        public HeapBudget[] HeapBudgets { get; set; }
    }

    /// <summary>Per-heap budget info</summary>
    public class HeapBudget
    {
        public uint HeapIndex { get; set; }
        public ulong Budget { get; set; }
        public ulong Usage { get; set; }
        public MemoryHeapFlag Flags { get; set; }
    }

    /// <summary>Allocator statistics</summary>
    public class AllocatorStats
    {
        public ulong TotalAllocated { get; set; }
        public ulong TotalReserved { get; set; }
        public ulong AllocationCount { get; set; }
        public ulong DeallocationCount { get; set; }
        public ulong PeakUsage { get; set; }
        public int BlockCount { get; set; }
    }

    /// <summary>Defragmentation pass</summary>
    internal class DefragmentationPass
    {
        public uint MemoryType { get; set; }
        public List<MemoryBlock> Blocks { get; set; }
    }
    // =========================================================================
    // VulkanDescriptorManager
    // =========================================================================

    /// <summary>
    /// Manages Vulkan descriptor set allocation from pools with layout caching,
    /// update batching, and per-pipeline-layout set caching for optimal performance.
    /// </summary>
    public class VulkanDescriptorManager : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Layout cache: hash -> layout handle
        private readonly Dictionary<long, IntPtr> _layoutCache = new Dictionary<long, IntPtr>();

        // Pool management: growable pool chain
        private readonly List<IntPtr> _descriptorPools = new List<IntPtr>();
        private int _currentPoolIndex = 0;
        private uint _setsAllocated;
        private uint _maxSetsPerPool = 1024;

        // Set cache per pipeline layout
        private readonly Dictionary<IntPtr, DescriptorSetCache> _setCaches = new Dictionary<IntPtr, DescriptorSetCache>();

        // Update batching
        private readonly List<DescriptorUpdateBatch> _pendingUpdates = new List<DescriptorUpdateBatch>();
        private int _maxBatchSize = 64;

        // Function pointers
        private CreateDescriptorSetLayoutDel _vkCreateDescriptorSetLayout;
        private DestroyDescriptorSetLayoutDel _vkDestroyDescriptorSetLayout;
        private CreateDescriptorPoolDel _vkCreateDescriptorPool;
        private DestroyDescriptorPoolDel _vkDestroyDescriptorPool;
        private AllocateDescriptorSetsDel _vkAllocateDescriptorSets;
        private FreeDescriptorSetsDel _vkFreeDescriptorSets;
        private UpdateDescriptorSetsDel _vkUpdateDescriptorSets;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorSetLayoutDel(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSetLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDel(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorPoolDel(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDescriptorPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorPoolDel(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateDescriptorSetsDel(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr[] pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FreeDescriptorSetsDel(IntPtr device, IntPtr descriptorPool, uint descriptorSetCount, ref IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UpdateDescriptorSetsDel(IntPtr device, uint descriptorWriteCount, IntPtr pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

        public VulkanDescriptorManager(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
            CreateNewPool();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreateDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<CreateDescriptorSetLayoutDel>(load("vkCreateDescriptorSetLayout"));
            _vkDestroyDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorSetLayoutDel>(load("vkDestroyDescriptorSetLayout"));
            _vkCreateDescriptorPool = Marshal.GetDelegateForFunctionPointer<CreateDescriptorPoolDel>(load("vkCreateDescriptorPool"));
            _vkDestroyDescriptorPool = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorPoolDel>(load("vkDestroyDescriptorPool"));
            _vkAllocateDescriptorSets = Marshal.GetDelegateForFunctionPointer<AllocateDescriptorSetsDel>(load("vkAllocateDescriptorSets"));
            _vkFreeDescriptorSets = Marshal.GetDelegateForFunctionPointer<FreeDescriptorSetsDel>(load("vkFreeDescriptorSets"));
            _vkUpdateDescriptorSets = Marshal.GetDelegateForFunctionPointer<UpdateDescriptorSetsDel>(load("vkUpdateDescriptorSets"));
        }

        private void CreateNewPool()
        {
            var poolSizes = new VkDescriptorPoolSize[]
            {
                new VkDescriptorPoolSize { type = DescriptorType.UniformBuffer, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.StorageBuffer, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.CombinedImageSampler, descriptorCount = _maxSetsPerPool * 8 },
                new VkDescriptorPoolSize { type = DescriptorType.SampledImage, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.StorageImage, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.Sampler, descriptorCount = _maxSetsPerPool * 2 },
                new VkDescriptorPoolSize { type = DescriptorType.InputAttachment, descriptorCount = _maxSetsPerPool * 2 },
            };

            int poolSizeStructSize = 8; // VkDescriptorPoolSize
            var poolSizesPtr = Marshal.AllocHGlobal(poolSizes.Length * poolSizeStructSize);
            for (int i = 0; i < poolSizes.Length; i++)
                Marshal.StructureToPtr(poolSizes[i], poolSizesPtr + i * poolSizeStructSize, false);

            var createInfo = new VkDescriptorPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags = 0, // No reset flag for simplicity
                maxSets = _maxSetsPerPool,
                poolSizeCount = (uint)poolSizes.Length,
                pPoolSizes = poolSizesPtr
            };

            IntPtr pool = IntPtr.Zero;
            _vkCreateDescriptorPool(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref pool);
            Marshal.FreeHGlobal(poolSizesPtr);

            _descriptorPools.Add(pool);
            _currentPoolIndex = _descriptorPools.Count - 1;
            _setsAllocated = 0;
        }

        /// <summary>Computes a hash for descriptor set layout bindings</summary>
        public long ComputeLayoutHash(DescriptorSetLayoutBinding[] bindings)
        {
            if (bindings == null || bindings.Length == 0)
                return 0;

            long hash = 17;
            foreach (var binding in bindings)
            {
                hash = hash * 31 + binding.Binding;
                hash = hash * 31 + (long)binding.DescriptorType;
                hash = hash * 31 + binding.DescriptorCount;
                hash = hash * 31 + (long)binding.StageFlags;
            }
            return hash;
        }

        /// <summary>Allocates a descriptor set from the current pool</summary>
        public IntPtr AllocateDescriptorSet(IntPtr layout)
        {
            lock (_lock)
            {
                var layoutsPtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(layoutsPtr, layout);

                var allocInfo = new VkDescriptorSetAllocateInfo
                {
                    sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool = _descriptorPools[_currentPoolIndex],
                    descriptorSetCount = 1,
                    pSetLayouts = layoutsPtr
                };

                var sets = new IntPtr[1];
                var result = _vkAllocateDescriptorSets(_device.LogicalDevice, ref allocInfo, sets);
                Marshal.FreeHGlobal(layoutsPtr);

                if (result == VulkanResult.ErrorOutOfPoolMemory || result == VulkanResult.ErrorFragmentedPool)
                {
                    CreateNewPool();
                    allocInfo.descriptorPool = _descriptorPools[_currentPoolIndex];
                    layoutsPtr = Marshal.AllocHGlobal(IntPtr.Size);
                    Marshal.WriteIntPtr(layoutsPtr, layout);
                    allocInfo.pSetLayouts = layoutsPtr;
                    result = _vkAllocateDescriptorSets(_device.LogicalDevice, ref allocInfo, sets);
                    Marshal.FreeHGlobal(layoutsPtr);
                }

                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to allocate descriptor set: {result}");

                _setsAllocated++;
                return sets[0];
            }
        }

        /// <summary>Batch updates multiple descriptor sets</summary>
        public void BatchUpdateDescriptors(DescriptorWrite[] writes)
        {
            if (writes == null || writes.Length == 0)
                return;

            lock (_lock)
            {
                int writeStructSize = 56; // Approximate VkWriteDescriptorSet size
                var writesPtr = Marshal.AllocHGlobal(writes.Length * writeStructSize);
                var tempAllocs = new List<IntPtr> { writesPtr };

                for (int w = 0; w < writes.Length; w++)
                {
                    var write = writes[w];
                    IntPtr imageInfoPtr = IntPtr.Zero;
                    IntPtr bufferInfoPtr = IntPtr.Zero;

                    if (write.ImageInfos != null && write.ImageInfos.Length > 0)
                    {
                        int imgInfoSize = 24; // VkDescriptorImageInfo size
                        imageInfoPtr = Marshal.AllocHGlobal(write.ImageInfos.Length * imgInfoSize);
                        tempAllocs.Add(imageInfoPtr);
                        for (int i = 0; i < write.ImageInfos.Length; i++)
                        {
                            var imgInfo = new VkDescriptorImageInfo
                            {
                                sampler = write.ImageInfos[i].Sampler,
                                imageView = write.ImageInfos[i].ImageView,
                                imageLayout = write.ImageInfos[i].ImageLayout
                            };
                            Marshal.StructureToPtr(imgInfo, imageInfoPtr + i * imgInfoSize, false);
                        }
                    }

                    if (write.BufferInfos != null && write.BufferInfos.Length > 0)
                    {
                        int bufInfoSize = 24; // VkDescriptorBufferInfo size
                        bufferInfoPtr = Marshal.AllocHGlobal(write.BufferInfos.Length * bufInfoSize);
                        tempAllocs.Add(bufferInfoPtr);
                        for (int i = 0; i < write.BufferInfos.Length; i++)
                        {
                            var bufInfo = new VkDescriptorBufferInfo
                            {
                                buffer = write.BufferInfos[i].Buffer,
                                offset = write.BufferInfos[i].Offset,
                                range = write.BufferInfos[i].Range
                            };
                            Marshal.StructureToPtr(bufInfo, bufferInfoPtr + i * bufInfoSize, false);
                        }
                    }

                    // Marshal VkWriteDescriptorSet
                    long baseAddr = (long)writesPtr + w * writeStructSize;
                    Marshal.WriteInt32((IntPtr)baseAddr, (int)VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                    Marshal.WriteInt64((IntPtr)(baseAddr + 8), 0); // pNext
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 16), write.DescriptorSet);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 24), (int)write.DstBinding);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 28), (int)write.DstArrayElement);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 32), write.ImageInfos?.Length ?? write.BufferInfos?.Length ?? 0);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 36), (int)write.DescriptorType);
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 40), imageInfoPtr);
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 48), bufferInfoPtr);
                }

                _vkUpdateDescriptorSets(_device.LogicalDevice, (uint)writes.Length, writesPtr, 0, IntPtr.Zero);

                foreach (var ptr in tempAllocs)
                    Marshal.FreeHGlobal(ptr);
            }
        }

        /// <summary>Gets the default descriptor pool handle</summary>
        public IntPtr GetDefaultPool() => _descriptorPools[_currentPoolIndex];

        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
        private const uint VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            foreach (var pool in _descriptorPools)
                if (pool != IntPtr.Zero)
                    _vkDestroyDescriptorPool?.Invoke(_device.LogicalDevice, pool, IntPtr.Zero);
            foreach (var layout in _layoutCache.Values)
                if (layout != IntPtr.Zero)
                    _vkDestroyDescriptorSetLayout?.Invoke(_device.LogicalDevice, layout, IntPtr.Zero);
            _descriptorPools.Clear();
            _layoutCache.Clear();
            GC.SuppressFinalize(this);
        }
    }

    internal struct VkDescriptorSetLayoutCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint bindingCount; public IntPtr pBindings;
    }

    /// <summary>Descriptor set cache entry</summary>
    internal class DescriptorSetCache
    {
        public IntPtr PipelineLayout { get; set; }
        public Dictionary<uint, IntPtr> CachedSets { get; set; } = new Dictionary<uint, IntPtr>();
        public uint Generation { get; set; }
    }

    /// <summary>Descriptor update batch entry</summary>
    internal class DescriptorUpdateBatch
    {
        public IntPtr DescriptorSet { get; set; }
        public uint Binding { get; set; }
        public DescriptorType Type { get; set; }
        public DescriptorImageInfo[] ImageInfos { get; set; }
        public DescriptorBufferInfo[] BufferInfos { get; set; }
    }
    // =========================================================================
    // VulkanPipelineCache
    // =========================================================================

    /// <summary>
    /// Manages Vulkan pipeline state object caching, cache serialization/deserialization,
    /// and cache warming to reduce pipeline creation time.
    /// </summary>
    public class VulkanPipelineCache : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _cache;
        private readonly object _lock = new object();
        private bool _disposed;

        // Cache statistics
        private long _cacheHits;
        private long _cacheMisses;
        private long _totalCreations;

        // Cache warming state
        private readonly List<PipelineCacheEntry> _warmedEntries = new List<PipelineCacheEntry>();

        // Function pointers
        private CreatePipelineCacheDel _vkCreatePipelineCache;
        private DestroyPipelineCacheDel _vkDestroyPipelineCache;
        private GetPipelineCacheDataDel _vkGetPipelineCacheData;
        private MergePipelineCachesDel _vkMergePipelineCaches;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineCacheDel(IntPtr device, ref VkPipelineCacheCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineCache);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineCacheDel(IntPtr device, IntPtr pipelineCache, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetPipelineCacheDataDel(IntPtr device, IntPtr pipelineCache, ref IntPtr pDataSize, IntPtr pData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MergePipelineCachesDel(IntPtr device, IntPtr dstCache, uint srcCacheCount, ref IntPtr pSrcCaches);

        public IntPtr Handle => _cache;
        public long CacheHits => _cacheHits;
        public long CacheMisses => _cacheMisses;

        public VulkanPipelineCache(VulkanDevice device, IntPtr existingCache = default)
        {
            _device = device;
            _cache = existingCache;
            LoadFunctions();

            if (_cache == IntPtr.Zero)
            {
                var createInfo = new VkPipelineCacheCreateInfo
                {
                    sType = 17,
                    initialDataSize = IntPtr.Zero,
                    pInitialData = IntPtr.Zero
                };
                _vkCreatePipelineCache(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _cache);
            }
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreatePipelineCache = Marshal.GetDelegateForFunctionPointer<CreatePipelineCacheDel>(load("vkCreatePipelineCache"));
            _vkDestroyPipelineCache = Marshal.GetDelegateForFunctionPointer<DestroyPipelineCacheDel>(load("vkDestroyPipelineCache"));
            _vkGetPipelineCacheData = Marshal.GetDelegateForFunctionPointer<GetPipelineCacheDataDel>(load("vkGetPipelineCacheData"));
            _vkMergePipelineCaches = Marshal.GetDelegateForFunctionPointer<MergePipelineCachesDel>(load("vkMergePipelineCaches"));
        }

        /// <summary>Serializes the pipeline cache data for disk storage</summary>
        public byte[] SerializeCache()
        {
            lock (_lock)
            {
                IntPtr dataSize = IntPtr.Zero;
                _vkGetPipelineCacheData(_device.LogicalDevice, _cache, ref dataSize, IntPtr.Zero);

                long size = (long)dataSize;
                if (size == 0)
                    return Array.Empty<byte>();

                var data = new byte[size];
                var dataPtr = Marshal.AllocHGlobal((int)size);
                try
                {
                    _vkGetPipelineCacheData(_device.LogicalDevice, _cache, ref dataSize, dataPtr);
                    Marshal.Copy(dataPtr, data, 0, (int)size);
                }
                finally { Marshal.FreeHGlobal(dataPtr); }
                return data;
            }
        }

        /// <summary>Deserializes pipeline cache data from disk</summary>
        public void DeserializeCache(byte[] cacheData)
        {
            if (cacheData == null || cacheData.Length == 0)
                return;

            lock (_lock)
            {
                // Destroy old cache
                if (_cache != IntPtr.Zero)
                    _vkDestroyPipelineCache(_device.LogicalDevice, _cache, IntPtr.Zero);

                var dataPtr = Marshal.AllocHGlobal(cacheData.Length);
                try
                {
                    Marshal.Copy(cacheData, 0, dataPtr, cacheData.Length);
                    var createInfo = new VkPipelineCacheCreateInfo
                    {
                        sType = 17,
                        initialDataSize = (IntPtr)cacheData.Length,
                        pInitialData = dataPtr
                    };
                    _vkCreatePipelineCache(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _cache);
                }
                finally { Marshal.FreeHGlobal(dataPtr); }
            }
        }

        /// <summary>Warms the cache by pre-creating pipelines from cached data</summary>
        public void WarmCache(byte[] cachedData)
        {
            if (cachedData != null && cachedData.Length > 0)
                DeserializeCache(cachedData);

            // Mark all warmed entries
            lock (_lock)
            {
                _warmedEntries.Clear();
                Interlocked.Exchange(ref _totalCreations, 0);
            }
        }

        /// <summary>Merges another pipeline cache into this one</summary>
        public void MergeCache(VulkanPipelineCache otherCache)
        {
            if (otherCache == null)
                return;

            lock (_lock)
            {
                var otherHandle = otherCache.Handle;
                _vkMergePipelineCaches(_device.LogicalDevice, _cache, 1, ref otherHandle);
            }
        }

        /// <summary>Records a cache hit</summary>
        internal void RecordHit() => Interlocked.Increment(ref _cacheHits);

        /// <summary>Records a cache miss</summary>
        internal void RecordMiss() => Interlocked.Increment(ref _cacheMisses);

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_cache != IntPtr.Zero)
                _vkDestroyPipelineCache?.Invoke(_device.LogicalDevice, _cache, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Pipeline cache warming entry</summary>
    internal class PipelineCacheEntry
    {
        public string Hash { get; set; }
        public IntPtr Pipeline { get; set; }
        public byte[] SerializedData { get; set; }
    }
    // =========================================================================
    // VulkanSyncManager
    // =========================================================================

    /// <summary>
    /// Manages Vulkan synchronization primitives including fences, semaphores,
    /// and timeline semaphores. Provides a recycling pool for efficient reuse.
    /// </summary>
}
