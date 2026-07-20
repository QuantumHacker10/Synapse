using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using GDNN.Rendering.LOD;
using Xunit;

namespace Synapse.Tests.Rendering;

public class IndustrialRenderingTests
{
    [Fact]
    public void ComputeAO_WithDepthVariation_ShouldOccludeConcavePixel()
    {
        var bridge = new LDNNBridge(32, 32);
        bridge.Initialize();

        var depth = new float[32 * 32];
        var normals = new Vector3[32 * 32];
        var albedo = new Vector3[32 * 32];
        var emissive = new Vector3[32 * 32];

        for (int y = 0; y < 32; y++)
        {
            for (int x = 0; x < 32; x++)
            {
                int i = y * 32 + x;
                // Center is deeper (farther) — neighbors closer create occlusion pressure on edges.
                float dx = x - 16;
                float dy = y - 16;
                float r = MathF.Sqrt(dx * dx + dy * dy);
                depth[i] = 2.0f + (r < 6 ? 0.0f : 0.4f);
                normals[i] = Vector3.UnitZ;
                albedo[i] = Vector3.One;
            }
        }

        bridge.UpdateCamera(
            Matrix4x4.CreateLookAt(new Vector3(0, 0, 5), Vector3.Zero, Vector3.UnitY),
            Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 1f, 0.1f, 100f),
            new Vector3(0, 0, 5), -Vector3.UnitZ, Vector3.UnitX, Vector3.UnitY,
            MathF.PI / 3f, 1f, 0.1f, 100f);

        bridge.FillGBuffer(depth, normals, albedo, emissive);

        float edgeAo = bridge.ComputeAO(10, 16);
        float farAo = bridge.ComputeAO(30, 30);

        edgeAo.Should().BeInRange(0f, 1f);
        farAo.Should().BeInRange(0f, 1f);
        // Flat far region should be less occluded than the cavity rim.
        farAo.Should().BeGreaterThanOrEqualTo(edgeAo - 0.05f);
    }

    [Fact]
    public void DispatchComputeShaders_SSAO_ShouldFillAoBuffer()
    {
        var renderer = new LDNNRenderer();
        renderer.Initialize(new LDNNConfig(), 16, 16);

        var gbuffer = new GBuffer
        {
            Width = 16,
            Height = 16,
            Depth = new float[16 * 16],
            Normals = new Vector3[16 * 16],
            Albedo = new Vector3[16 * 16],
            Velocity = new Vector2[16 * 16],
            MaterialProps = new Vector4[16 * 16],
            Specular = new Vector3[16 * 16],
            Emissive = new Vector3[16 * 16]
        };

        for (int i = 0; i < gbuffer.Depth.Length; i++)
        {
            gbuffer.Depth[i] = 1.0f + (i % 16) * 0.02f;
            gbuffer.Normals[i] = Vector3.UnitY;
        }

        var camera = new CameraState
        {
            Position = new Vector3(0, 2, 5),
            Forward = -Vector3.UnitZ,
            Right = Vector3.UnitX,
            Up = Vector3.UnitY,
            FieldOfView = MathF.PI / 3f,
            AspectRatio = 1f,
            NearPlane = 0.1f,
            FarPlane = 100f
        };
        camera.ViewMatrix = Matrix4x4.CreateLookAt(camera.Position, Vector3.Zero, Vector3.UnitY);
        camera.ProjectionMatrix = Matrix4x4.CreatePerspectiveFieldOfView(
            camera.FieldOfView, camera.AspectRatio, camera.NearPlane, camera.FarPlane);
        camera.Recompute();

        renderer.DispatchComputeShaders("ssao", 2, 2, 1, new Dictionary<string, object>
        {
            ["gbuffer"] = gbuffer,
            ["camera"] = camera,
            ["kernelSize"] = 16,
            ["radius"] = 0.5f
        });

        renderer.AmbientOcclusion.AOBuffer.Should().NotBeNull();
        renderer.AmbientOcclusion.AOBuffer.Length.Should().Be(16 * 16);
        renderer.AmbientOcclusion.AOBuffer.Should().Contain(v => v > 0f && v <= 1f);
    }

    [Fact]
    public void QuadricMeshSimplifier_ShouldReduceTriangleCount()
    {
        // Icosahedron-like fan: enough triangles for meaningful QEM collapses.
        var vertices = new List<Vector3>
        {
            Vector3.Zero,
            new(1, 0, 0), new(0.5f, 0.87f, 0), new(-0.5f, 0.87f, 0),
            new(-1, 0, 0), new(-0.5f, -0.87f, 0), new(0.5f, -0.87f, 0),
            new(0, 0, 1), new(0, 0, -1)
        };

        var indices = new List<uint>();
        for (uint i = 1; i <= 6; i++)
        {
            uint next = i == 6 ? 1u : i + 1;
            indices.Add(0);
            indices.Add(i);
            indices.Add(next);
            indices.Add(7);
            indices.Add(i);
            indices.Add(next);
            indices.Add(8);
            indices.Add(next);
            indices.Add(i);
        }

        int before = indices.Count / 3;
        var simplified = QuadricMeshSimplifier.Simplify(vertices, indices, targetTriangles: Math.Max(2, before / 2));
        (simplified.Count / 3).Should().BeLessThan(before);
        (simplified.Count / 3).Should().BeGreaterThan(0);
    }

    [Fact]
    public void LodGenerator_ShouldStoreSimplifiedMeshesInLevels()
    {
        var generator = new LodGenerator();
        var vertices = new List<Vector3>
        {
            new(0, 0, 0), new(1, 0, 0), new(0.5f, 1, 0),
            new(0, 0, 1), new(1, 0, 1), new(0.5f, 1, 1),
            new(0.25f, 0.25f, 0.5f), new(0.75f, 0.25f, 0.5f)
        };
        var indices = new List<uint>
        {
            0, 1, 2, 3, 4, 5, 0, 3, 2, 1, 4, 2,
            0, 1, 6, 1, 7, 6, 3, 5, 6, 4, 5, 7
        };

        var levels = generator.GenerateLevels(vertices, indices, targetLevelCount: 3);
        levels.Should().HaveCountGreaterOrEqualTo(3);
        levels[0].MeshData.Should().BeOfType<List<uint>>();
        levels[^1].TriangleCount.Should().BeLessThanOrEqualTo(levels[0].TriangleCount);
    }
}
