using System;
using System.IO;
using FluentAssertions;
using GDNN.Rendering;
using GDNN.Rendering.Bridge;
using System.Numerics;
using Xunit;

namespace Synapse.Tests.Rendering;

public sealed class GiGoldenSnapshotTests
{
    private const int Size = 32;

    /// <summary>Deterministic procedural GI fingerprint (update via SYNAPSE_UPDATE_GI_GOLDEN=1).</summary>
    private const string ExpectedHash = "PLACEHOLDER";

    [Fact]
    public void ProceduralPreview_MatchesGoldenHash()
    {
        var gi = RenderProceduralGi();
        string hash = GiGoldenSnapshot.ComputeHash(gi);

        if (string.Equals(Environment.GetEnvironmentVariable("SYNAPSE_UPDATE_GI_GOLDEN"), "1", StringComparison.Ordinal))
        {
            var sourceGolden = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Rendering", "Golden", "gi-procedural-32x32.sha256"));
            Directory.CreateDirectory(Path.GetDirectoryName(sourceGolden)!);
            File.WriteAllText(sourceGolden, hash);
        }

        string expected = LoadExpectedHash();
        hash.Should().Be(expected, "procedural L-DNN GI output changed — run with SYNAPSE_UPDATE_GI_GOLDEN=1 to refresh");
    }

    private static Vector3[,] RenderProceduralGi()
    {
        using var bridge = new LDNNBridge(Size, Size);
        bridge.Initialize();
        bridge.UpdateCamera(
            Matrix4x4.Identity,
            Matrix4x4.Identity,
            new Vector3(0, 1.5f, 3f),
            Vector3.Normalize(new Vector3(0, -0.2f, -1f)),
            Vector3.UnitX,
            Vector3.UnitY,
            60f,
            1f,
            0.1f,
            100f);
        bridge.FillGBufferProceduralPreview();
        return bridge.RenderGI();
    }

    private static string LoadExpectedHash()
    {
        var outputGolden = Path.Combine(AppContext.BaseDirectory, "Golden", "gi-procedural-32x32.sha256");
        if (File.Exists(outputGolden))
            return File.ReadAllText(outputGolden).Trim();

        var sourceGolden = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Rendering", "Golden", "gi-procedural-32x32.sha256"));
        if (File.Exists(sourceGolden))
            return File.ReadAllText(sourceGolden).Trim();

        return ExpectedHash;
    }
}
