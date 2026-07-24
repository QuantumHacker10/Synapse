using System;
using System.Numerics;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// Embedded production smoke checks for the Synapse OpenUSD MeshIO runtime
/// (no filesystem sample dependency — used by <c>--health</c> and unit tests).
/// </summary>
public static class UsdProductionSmoke
{
    public const string MinimalStage = """
        #usda 1.0
        (
            defaultPrim = "Root"
            timeCodesPerSecond = 30
            startTimeCode = 0
            endTimeCode = 30
        )

        def Xform "Root"
        {
            def Mesh "Hero"
            {
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                uniform bool doubleSided = 1
                token purpose = "render"
                point3f[] points = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
                normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3, 3]
                int[] faceVertexIndices = [0, 1, 2, 0, 1, 3]
                rel material:binding = </Looks/Mat>
            }

            def Mesh "GuideOnly"
            {
                token purpose = "guide"
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
            }

            def Mesh "Hidden"
            {
                token visibility = "invisible"
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
            }
        }

        def Scope "Looks"
        {
            def Material "Mat"
            {
                token outputs:surface.connect = </Looks/Mat/PBR.outputs:surface>

                def Shader "PBR"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0.8, 0.2, 0.1)
                    float inputs:roughness = 0.4
                    float inputs:metallic = 0.1
                    float inputs:opacity = 0.95
                    float inputs:opacityThreshold = 0.5
                    color3f inputs:emissiveColor = (0.1, 0.0, 0.0)
                    float inputs:emissiveIntensity = 2
                }
            }
        }
        """;

    /// <summary>Returns true when MeshIO parses the embedded production stage correctly.</summary>
    public static bool TryVerify(out string detail)
    {
        try
        {
            var loader = new UsdAsciiLoader();
            var result = loader.LoadLeafFromText(MinimalStage, "smoke", new MeshLoadConfig(), applyLocalXform: false);
            if (!result.Success || result.Asset == null)
            {
                detail = result.ErrorMessage;
                return false;
            }

            var asset = result.Asset;
            if (asset.Primitives.Count != 1)
            {
                detail = $"expected 1 production mesh (guide/invisible skipped), got {asset.Primitives.Count}";
                return false;
            }

            var prim = asset.Primitives[0];
            if (prim.Indices.Count != 6)
            {
                detail = $"faceVertexCounts triangulation expected 6 indices, got {prim.Indices.Count}";
                return false;
            }

            if (Math.Abs(prim.Vertices[0].Normal.Y - 1f) > 0.01f)
            {
                detail = "authored normals not applied";
                return false;
            }

            if (asset.Materials.Count == 0 || !asset.Materials[0].DoubleSided)
            {
                detail = "doubleSided not applied to material";
                return false;
            }

            if (asset.Materials[0].AlphaCutoff < 0.4f || asset.Materials[0].EmissiveIntensity < 1.5f)
            {
                detail = "opacityThreshold / emissiveIntensity missing";
                return false;
            }

            if (asset.Bounds.Min.X > -0.9f || asset.Bounds.Max.X < 0.9f)
            {
                detail = "extent bounds missing";
                return false;
            }

            detail = $"ok prims={asset.Primitives.Count} tris={asset.TotalTriangleCount} mats={asset.Materials.Count}";
            return true;
        }
        catch (Exception ex)
        {
            detail = ex.Message;
            return false;
        }
    }
}
