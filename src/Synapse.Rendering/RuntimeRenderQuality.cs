namespace GDNN.Rendering.Quality
{
    /// <summary>
    /// Rendering-side quality snapshot consumed by the native FrameGraph.
    /// Mapped from <c>Synapse.Infrastructure.RuntimeQualityManager.QualityLevel</c>
    /// so Physics / Simulation / Studio can drive LOD, shadows, GI and post without
    /// reaching into Vulkan internals.
    /// </summary>
    public sealed class RuntimeRenderQuality
    {
        public string PresetName { get; init; } = "High";
        public float ResolutionScale { get; init; } = 1f;
        public int MaxLodLevel { get; init; } = 1;
        public int ShadowCascades { get; init; } = 3;
        public int ShadowQuality { get; init; } = 2;
        public bool EnableGlobalIllumination { get; init; } = true;
        public bool EnableScreenSpaceGi { get; init; } = true;
        public bool EnableSsao { get; init; } = true;
        public bool EnableBloom { get; init; } = true;
        public bool EnableTaa { get; init; } = true;
        public bool EnableVolumetricLighting { get; init; } = true;
        public int ParticleQuality { get; init; } = 2;
        public int GiMaxBounces { get; init; } = 2;
    }
}
