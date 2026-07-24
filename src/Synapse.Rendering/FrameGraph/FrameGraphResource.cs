namespace GDNN.Rendering.FrameGraph
{
    /// <summary>Logical handle for a transient or imported frame-graph texture.</summary>
    public readonly struct FrameGraphTextureHandle
    {
        public FrameGraphTextureHandle(int id, string name)
        {
            Id = id;
            Name = name;
        }

        public int Id { get; }
        public string Name { get; }

        public bool IsValid => Id >= 0;

        public static FrameGraphTextureHandle Invalid => new(-1, "");
    }

    public enum FrameGraphResourceUsage
    {
        Read,
        Write,
        ReadWrite,
        Import
    }

    public sealed class FrameGraphResourceDesc
    {
        public required string Name { get; init; }
        public FrameGraphResourceUsage Usage { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
    }
}
