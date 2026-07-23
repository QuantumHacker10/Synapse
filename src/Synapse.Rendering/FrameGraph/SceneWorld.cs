using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Materials.SubstrateOmega;
using GDNN.Rendering.LOD;

namespace GDNN.Rendering.FrameGraph
{
    public sealed class SceneMesh
    {
        public required float[] VertexData { get; init; }
        public int VertexStride { get; init; }
        public required uint[] IndexData { get; init; }
        public int MaterialIndex { get; set; }
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;
    }

    public sealed class SceneLight
    {
        public Vector3 Position { get; set; }
        public Vector3 Direction { get; set; }
        public Vector3 Color { get; set; } = Vector3.One;
        public float Intensity { get; set; } = 1f;
        public float Range { get; set; } = 100f;
    }

    public sealed class SceneDraw
    {
        public int MeshIndex { get; set; }
        public uint FirstIndex { get; set; }
        public uint IndexCount { get; set; }
        public int MaterialIndex { get; set; }
        public Matrix4x4 WorldMatrix { get; set; }
    }

    /// <summary>
    /// Scene snapshot consumed by FrameGraph passes (G-DNN cull, shadows, G-buffer).
    /// Populated by SceneRenderer / RenderEngine each frame.
    /// </summary>
    public sealed class SceneWorld
    {
        public List<SceneMesh> Meshes { get; } = new();
        public List<SceneLight> Lights { get; } = new();
        public List<SubstrateMaterial> Materials { get; } = new();
        public List<SceneDraw> Draws { get; } = new();

        public Vector3 CameraPosition { get; set; }
        public Vector3 CameraForward { get; set; } = -Vector3.UnitZ;
        public Vector3 CameraRight { get; set; } = Vector3.UnitX;
        public Vector3 CameraUp { get; set; } = Vector3.UnitY;
        public float FieldOfViewRadians { get; set; } = MathF.PI / 3f;
        public float ViewportHeight { get; set; } = 1080f;

        public LodManager? Lod { get; set; }

        /// <summary>G-DNN LOD update before draw submission.</summary>
        public void UpdateLod()
        {
            Lod?.UpdateAll(CameraPosition, FieldOfViewRadians, ViewportHeight);
        }

        public void ClearTransientDraws()
        {
            Draws.Clear();
        }
    }
}
