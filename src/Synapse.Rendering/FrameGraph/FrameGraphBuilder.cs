using System.Collections.Generic;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>Collects pass resource declarations for one compiled frame.</summary>
    public sealed class FrameGraphBuilder
    {
        private readonly List<FrameGraphResourceDesc> _resources = new();
        private readonly Dictionary<string, FrameGraphTextureHandle> _handles = new();
        private int _nextId;

        public IReadOnlyList<FrameGraphResourceDesc> Resources => _resources;

        public FrameGraphTextureHandle CreateTexture(string name, int width, int height, FrameGraphResourceUsage usage = FrameGraphResourceUsage.Write)
        {
            if (_handles.TryGetValue(name, out var existing))
                return existing;

            var handle = new FrameGraphTextureHandle(_nextId++, name);
            _handles[name] = handle;
            _resources.Add(new FrameGraphResourceDesc
            {
                Name = name,
                Usage = usage,
                Width = width,
                Height = height
            });
            return handle;
        }

        public FrameGraphTextureHandle ImportTexture(string name)
        {
            if (_handles.TryGetValue(name, out var existing))
                return existing;

            var handle = new FrameGraphTextureHandle(_nextId++, name);
            _handles[name] = handle;
            _resources.Add(new FrameGraphResourceDesc
            {
                Name = name,
                Usage = FrameGraphResourceUsage.Import,
                Width = 0,
                Height = 0
            });
            return handle;
        }

        public FrameGraphTextureHandle Get(string name)
            => _handles.TryGetValue(name, out var h) ? h : FrameGraphTextureHandle.Invalid;

        public void Reset()
        {
            _resources.Clear();
            _handles.Clear();
            _nextId = 0;
        }
    }
}
