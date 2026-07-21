using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.Streaming;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class UsdPixarRuntimeSurfaceTests
{
    [Fact]
    public void Udim_ExpandsExistingTiles()
    {
        var dir = Path.GetDirectoryName(Resolve("samples/meshes/textures/albedo.1001.png"))!;
        var tiles = UsdUdim.ExpandTiles("./albedo.<UDIM>.png", dir);
        tiles.Should().ContainKey(1001);
        tiles.Should().ContainKey(1002);
        tiles[1001].Should().EndWith("albedo.1001.png");
        UsdUdim.TileFromUv(new Vector2(1.2f, 0.1f)).Should().Be(1002);
    }

    [Fact]
    public void Usd_ParsesUdimMdlAndBlendShapes()
    {
        var path = Resolve("samples/meshes/udim_mdl_blend.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        var mat = result.Asset!.Materials.Should().ContainSingle(m => m.Name == "UdimMdlMat").Subject;
        mat.MdlAssetPath.Should().EndWith(Path.Combine("materials", "demo.mdl"));
        mat.MdlMaterialName.Should().Contain("Demo");
        mat.UdimTiles.Should().ContainKey(1001);
        mat.UdimTiles.Should().ContainKey(1002);
        result.Asset.BlendShapes.Should().HaveCount(2);
        result.Asset.BlendShapes[0].DeltaPositions.Should().HaveCount(3);

        var positions = result.Asset.Primitives[0].Vertices.ConvertAll(v => v.Position);
        result.Asset.BlendShapes[0].Apply(positions, 1f);
        positions[0].Y.Should().BeApproximately(0.1f, 0.001f);
    }

    [Fact]
    public async Task GpuTextureStreamer_PagesDiskAndCaches()
    {
        await using var streamer = new GpuTextureStreamer(maxResidentBytes: 1024 * 1024);
        var path = Resolve("samples/meshes/textures/albedo.1001.png");
        var page1 = await streamer.RequestAsync("albedo", path);
        page1.Bytes.Should().NotBeEmpty();
        var page2 = await streamer.RequestAsync("albedo", path);
        streamer.CacheHits.Should().BeGreaterThan(0);
        page2.Bytes.Length.Should().Be(page1.Bytes.Length);

        var tiles = new Dictionary<int, string> { [1001] = path };
        await streamer.PrefetchUdimAsync("mat", tiles);
        streamer.ResidentCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task RemoteMarketplace_InstallsFromLoopbackCatalog()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var pluginDir = Path.Combine(Path.GetTempPath(), $"rmkt-{Guid.NewGuid():N}");
        var serveDir = Path.Combine(Path.GetTempPath(), $"rmkt-serve-{Guid.NewGuid():N}");
        Directory.CreateDirectory(pluginDir);
        Directory.CreateDirectory(serveDir);
        try
        {
            var dllPath = Path.Combine(serveDir, "demo.dll");
            await File.WriteAllBytesAsync(dllPath, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            var hash = PluginMarketplace.ComputeFileSha256(dllPath);

            using var listener = new HttpListener();
            listener.Prefixes.Add("http://127.0.0.1:18765/");
            listener.Start();
            var serveTask = Task.Run(async () =>
            {
                for (int i = 0; i < 2; i++)
                {
                    var ctx = await listener.GetContextAsync();
                    if (ctx.Request.Url!.AbsolutePath.Contains("catalog", StringComparison.Ordinal))
                    {
                        var json = $$"""
                            [{"Id":"demo","Name":"Demo","FileName":"demo.dll","Sha256":"{{hash}}","DownloadUrl":"http://127.0.0.1:18765/demo.dll"}]
                            """;
                        var buf = System.Text.Encoding.UTF8.GetBytes(json);
                        ctx.Response.ContentType = "application/json";
                        await ctx.Response.OutputStream.WriteAsync(buf);
                    }
                    else
                    {
                        var bytes = await File.ReadAllBytesAsync(dllPath);
                        await ctx.Response.OutputStream.WriteAsync(bytes);
                    }

                    ctx.Response.Close();
                }
            });

            var remote = new RemotePluginMarketplace(logger);
            var n = await remote.SyncAsync("http://127.0.0.1:18765/catalog.json", pluginDir);
            n.Should().Be(1);
            File.Exists(Path.Combine(pluginDir, "demo.dll")).Should().BeTrue();
            File.Exists(Path.Combine(pluginDir, "marketplace.json")).Should().BeTrue();

            listener.Stop();
            await serveTask;
        }
        finally
        {
            try
            { Directory.Delete(pluginDir, true); }
            catch { }
            try
            { Directory.Delete(serveDir, true); }
            catch { }
        }
    }

    private static string Resolve(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new FileNotFoundException(relative);
    }
}
