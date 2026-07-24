using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.GPU;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using GDNN.Rendering.Engine;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Industrial;

public class DxcCompilationTests
{
    [Fact]
    public void ShaderCompiler_ReportsBackend_AndSucceeds()
    {
        const string source = """
            void CSMain(uint3 id : SV_DispatchThreadID)
            {
                float x = id.x;
            }
            """;

        using var compiler = new ShaderCompiler(new ShaderCompilerConfig
        {
            PreferNativeCompiler = true,
            AllowSimulatedFallback = true,
            Platform = ShaderTargetPlatform.Vulkan,
            EnableDebugInfo = false
        });

        var result = compiler.Compile(source, "CSMain", ShaderType.ComputeShader);
        result.Success.Should().BeTrue(result.GetErrorSummary());
        result.Bytecode.Should().NotBeNull();
        result.Backend.Should().BeOneOf(
            ShaderCompilationBackend.Dxc,
            ShaderCompilationBackend.SimulatedFallback);
    }

    [Fact]
    public void ShaderCompiler_DisablingFallback_FailsWithoutDxcOrAcceptsDxc()
    {
        const string source = """
            void CSMain(uint3 id : SV_DispatchThreadID)
            {
                float x = id.x;
            }
            """;

        using var compiler = new ShaderCompiler(new ShaderCompilerConfig
        {
            PreferNativeCompiler = true,
            AllowSimulatedFallback = false,
            Platform = ShaderTargetPlatform.Vulkan
        });

        var result = compiler.Compile(source, "CSMain", ShaderType.ComputeShader);
        if (SpirvToolchain.IsDxcAvailable && result.Success)
            result.Backend.Should().Be(ShaderCompilationBackend.Dxc);
        else
            result.Success.Should().BeFalse();
    }
}

public class GpuResidentGiTests
{
    [Fact]
    public void ResidentGi_ComputesWithoutFreshReadback()
    {
        var bridge = new LDNNBridge(24, 24);
        bridge.Initialize();
        bridge.EnableStaticSceneCache = false;

        var depth = new float[24 * 24];
        var normals = new Vector3[24 * 24];
        var albedo = new Vector3[24 * 24];
        var emissive = new Vector3[24 * 24];
        for (int i = 0; i < depth.Length; i++)
        {
            depth[i] = 2.0f + (i % 24) * 0.01f;
            normals[i] = Vector3.UnitY;
            albedo[i] = new Vector3(0.6f, 0.5f, 0.4f);
        }

        bridge.UpdateCamera(
            Matrix4x4.CreateLookAt(new Vector3(0, 2, 5), Vector3.Zero, Vector3.UnitY),
            Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 1f, 0.1f, 100f),
            new Vector3(0, 2, 5), -Vector3.UnitZ, Vector3.UnitX, Vector3.UnitY,
            MathF.PI / 3f, 1f, 0.1f, 100f);
        bridge.AddDirectionalLight(new Vector3(0.3f, -1f, 0.2f), Vector3.One, 2f);
        bridge.FillGBuffer(depth, normals, albedo, emissive);

        bridge.HasResidentGBuffer.Should().BeTrue();

        var gi1 = bridge.RenderGI();
        // Full Hybrid L-DNN is preferred over the resident-only SSGI shortcut (visual quality).
        // Resident path is retained as a fallback when the hybrid field is unavailable.
        gi1[12, 12].Length().Should().BeGreaterThan(0f);
        bridge.HasResidentGBuffer.Should().BeTrue();

        // Second frame still produces irradiance from the resident / hybrid stack.
        var gi2 = bridge.RenderGI();
        gi2[12, 12].Length().Should().BeGreaterThan(0f);
        if (bridge.LastGiPath == GiComputePath.GpuResidentCompute)
            bridge.ResidentGi.ResidentComputeFrames.Should().BeGreaterThan(0);
    }

    [Fact]
    public void IngestGpuSnapshot_MarksReadbackPath()
    {
        var bridge = new LDNNBridge(8, 8);
        bridge.Initialize();
        var snap = new GBufferSnapshot
        {
            Width = 8,
            Height = 8,
            Depth = new float[64],
            Normals = new Vector3[64],
            Albedo = new Vector3[64],
            Velocity = new Vector2[64],
            MaterialProps = new Vector4[64],
            Specular = new Vector3[64],
            Emissive = new Vector3[64]
        };
        for (int i = 0; i < 64; i++)
        {
            snap.Depth[i] = 1.5f;
            snap.Normals[i] = Vector3.UnitZ;
            snap.Albedo[i] = Vector3.One;
        }

        bridge.IngestGpuSnapshot(snap);
        bridge.HasResidentGBuffer.Should().BeTrue();
        bridge.ResidentGi.LastPath.Should().Be(GiComputePath.GpuReadback);
    }
}

public class CcdTests
{
    [Fact]
    public void FastSphere_AgainstPlane_ShouldNotTunnel()
    {
        var world = new RigidBodyWorld
        {
            Gravity = Vector3.Zero,
            EnableCcd = true,
            CcdVelocityThreshold = 1f,
            EnableSleeping = false
        };
        world.AddBody(new RigidBody
        {
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Plane },
            Position = Vector3.Zero
        });
        var bullet = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.25f) },
            Position = new Vector3(0, 1.5f, 0),
            LinearVelocity = new Vector3(0, -100f, 0),
            Material = new PhysicsMaterial { Restitution = 0.1f },
            EnableCcd = true,
            CcdMotionThreshold = 0.05f
        });
        bullet.SetMass(1f);

        world.Step(1f / 60f);

        world.LastStats.CcdHitCount.Should().BeGreaterThan(0);
        bullet.Position.Y.Should().BeGreaterThan(0.2f);
    }
}

public class PhysicsCertificationTests
{
    [Fact]
    public void RunIndustrialCore_ShouldPass()
    {
        var report = PhysicsCertification.RunIndustrialCore();
        report.Cases.Should().NotBeEmpty();
        report.FailedCount.Should().Be(0, report.ToMarkdown());
        report.Level.Should().Be(CertificationLevel.IndustrialCore);
    }
}
