using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GDNN.Rendering.Compat;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: DebugUtils.cs
// PATH: Utilities/DebugUtils.cs
// ============================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace GDNN.Utilities;

/// <summary>
/// Debug visualization data generation for bounding volumes, rays,
/// point clouds, gradient fields, and neural network heatmaps.
/// </summary>
public static class DebugUtils
{
    /// <summary>
    /// Generates line segment vertices for an axis-aligned bounding box.
    /// </summary>
    /// <param name="min">Minimum corner of the box.</param>
    /// <param name="max">Maximum corner of the box.</param>
    /// <returns>24 vertices forming 12 line segments (2 per edge).</returns>
    public static Vector3[] GenerateBoundingBoxLines(Vector3 min, Vector3 max)
    {
        Vector3[] v = new Vector3[24];
        v[0] = new Vector3(min.X, min.Y, min.Z);
        v[1] = new Vector3(max.X, min.Y, min.Z);
        v[2] = new Vector3(min.X, min.Y, min.Z);
        v[3] = new Vector3(min.X, max.Y, min.Z);
        v[4] = new Vector3(min.X, min.Y, min.Z);
        v[5] = new Vector3(min.X, min.Y, max.Z);

        v[6] = new Vector3(max.X, max.Y, max.Z);
        v[7] = new Vector3(min.X, max.Y, max.Z);
        v[8] = new Vector3(max.X, max.Y, max.Z);
        v[9] = new Vector3(max.X, min.Y, max.Z);
        v[10] = new Vector3(max.X, max.Y, max.Z);
        v[11] = new Vector3(max.X, max.Y, min.Z);

        v[12] = new Vector3(min.X, max.Y, min.Z);
        v[13] = new Vector3(max.X, max.Y, min.Z);
        v[14] = new Vector3(min.X, max.Y, min.Z);
        v[15] = new Vector3(min.X, max.Y, max.Z);

        v[16] = new Vector3(min.X, min.Y, max.Z);
        v[17] = new Vector3(max.X, min.Y, max.Z);
        v[18] = new Vector3(min.X, min.Y, max.Z);
        v[19] = new Vector3(min.X, max.Y, max.Z);

        v[20] = new Vector3(max.X, min.Y, min.Z);
        v[21] = new Vector3(max.X, max.Y, min.Z);
        v[22] = new Vector3(max.X, min.Y, min.Z);
        v[23] = new Vector3(max.X, min.Y, max.Z);

        return v;
    }

    /// <summary>
    /// Generates line segments for a bounding sphere rendered as wireframe circles.
    /// </summary>
    /// <param name="center">Sphere center.</param>
    /// <param name="radius">Sphere radius.</param>
    /// <param name="segments">Number of segments per circle (default 32).</param>
    /// <returns>Vertices forming three orthogonal circles.</returns>
    public static Vector3[] GenerateBoundingSphereLines(Vector3 center, float radius, int segments = 32)
    {
        var vertices = new Vector3[segments * 6];
        float step = MathF.Tau / segments;

        for (int i = 0; i < segments; i++)
        {
            float angle0 = i * step;
            float angle1 = (i + 1) * step;
            float cos0 = MathF.Cos(angle0), sin0 = MathF.Sin(angle0);
            float cos1 = MathF.Cos(angle1), sin1 = MathF.Sin(angle1);

            // XY plane circle
            vertices[i * 2] = center + new Vector3(cos0 * radius, sin0 * radius, 0);
            vertices[i * 2 + 1] = center + new Vector3(cos1 * radius, sin1 * radius, 0);

            // XZ plane circle
            int off = segments * 2;
            vertices[off + i * 2] = center + new Vector3(cos0 * radius, 0, sin0 * radius);
            vertices[off + i * 2 + 1] = center + new Vector3(cos1 * radius, 0, sin1 * radius);

            // YZ plane circle
            off = segments * 4;
            vertices[off + i * 2] = center + new Vector3(0, cos0 * radius, sin0 * radius);
            vertices[off + i * 2 + 1] = center + new Vector3(0, cos1 * radius, sin1 * radius);
        }

        return vertices;
    }

    /// <summary>
    /// Generates line segments for an oriented bounding box.
    /// </summary>
    /// <param name="center">Box center.</param>
    /// <param name="halfExtents">Half-extents along each axis.</param>
    /// <param name="orientation">Rotation quaternion.</param>
    /// <returns>24 vertices forming 12 line segments.</returns>
    public static Vector3[] GenerateOrientedBoxLines(Vector3 center, Vector3 halfExtents, Quaternion orientation)
    {
        Matrix4x4 transform = Matrix4x4.CreateFromQuaternion(orientation);
        transform.Translation = center;

        Vector3[] corners = new Vector3[8];
        int idx = 0;
        for (int z = -1; z <= 1; z += 2)
        {
            for (int y = -1; y <= 1; y += 2)
            {
                for (int x = -1; x <= 1; x += 2)
                {
                    corners[idx++] = Vector3.Transform(
                        new Vector3(x * halfExtents.X, y * halfExtents.Y, z * halfExtents.Z),
                        transform);
                }
            }
        }

        return GenerateBoxFromCorners(corners);
    }

    /// <summary>
    /// Generates line segments from 8 pre-computed box corners.
    /// </summary>
    public static Vector3[] GenerateBoxFromCorners(Span<Vector3> corners)
    {
        var vertices = new Vector3[24];
        int[][] edges = {
            new[]{0,1}, new[]{0,2}, new[]{0,4},
            new[]{7,5}, new[]{7,6}, new[]{7,3},
            new[]{2,6}, new[]{2,3},
            new[]{5,1}, new[]{5,4},
            new[]{1,3}, new[]{6,4}
        };

        for (int i = 0; i < edges.Length; i++)
        {
            vertices[i * 2] = corners[edges[i][0]];
            vertices[i * 2 + 1] = corners[edges[i][1]];
        }

        return vertices;
    }

    /// <summary>
    /// Generates line segments for octree cell boundaries.
    /// </summary>
    /// <param name="center">Cell center.</param>
    /// <param name="halfSize">Half-size of the cell.</param>
    /// <returns>24 vertices forming the cell wireframe.</returns>
    public static Vector3[] GenerateOctreeCellLines(Vector3 center, float halfSize)
    {
        return GenerateBoundingBoxLines(center - Vector3.One * halfSize, center + Vector3.One * halfSize);
    }

    /// <summary>
    /// Generates an octree wireframe recursively for visualization.
    /// </summary>
    /// <param name="center">Root cell center.</param>
    /// <param name="halfSize">Root cell half-size.</param>
    /// <param name="depth">Current recursion depth.</param>
    /// <param name="maxDepth">Maximum recursion depth.</param>
    /// <param name="occupiedCells">Set of occupied cell centers to subdivide.</param>
    /// <returns>All vertices for the octree wireframe.</returns>
    public static List<Vector3> GenerateOctreeWireframe(
        Vector3 center, float halfSize, int depth, int maxDepth,
        HashSet<Vector3>? occupiedCells = null)
    {
        var allVertices = new List<Vector3>();
        GenerateOctreeWireframeRecursive(center, halfSize, depth, maxDepth, occupiedCells, allVertices);
        return allVertices;
    }

    private static void GenerateOctreeWireframeRecursive(
        Vector3 center, float halfSize, int depth, int maxDepth,
        HashSet<Vector3>? occupiedCells, List<Vector3> vertices)
    {
        if (depth > maxDepth) return;

        bool shouldSubdivide = occupiedCells == null || occupiedCells.Contains(center);
        if (!shouldSubdivide) return;

        vertices.AddRange(GenerateOctreeCellLines(center, halfSize));

        if (depth < maxDepth)
        {
            float childHalf = halfSize * 0.5f;
            for (int z = -1; z <= 1; z += 2)
            {
                for (int y = -1; y <= 1; y += 2)
                {
                    for (int x = -1; x <= 1; x += 2)
                    {
                        Vector3 childCenter = center + new Vector3(x * childHalf, y * childHalf, z * childHalf);
                        GenerateOctreeWireframeRecursive(childCenter, childHalf, depth + 1, maxDepth, occupiedCells, vertices);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates line segments representing a ray for visualization.
    /// </summary>
    /// <param name="origin">Ray origin.</param>
    /// <param name="direction">Ray direction (normalized).</param>
    /// <param name="length">Visual length of the ray.</param>
    /// <param name="arrowSize">Size of the arrow head.</param>
    /// <returns>Vertices for the ray and arrowhead.</returns>
    public static Vector3[] GenerateRayLines(Vector3 origin, Vector3 direction, float length, float arrowSize = 0.1f)
    {
        Vector3 tip = origin + direction * length;
        Vector3 up = MathHelpers.GetPerpendicular(direction);
        Vector3 right = Vector3.Cross(direction, up);

        return new[]
        {
            origin, tip,
            tip, tip - direction * arrowSize + right * arrowSize * 0.5f,
            tip, tip - direction * arrowSize - right * arrowSize * 0.5f,
            tip, tip - direction * arrowSize + up * arrowSize * 0.5f,
            tip, tip - direction * arrowSize - up * arrowSize * 0.5f
        };
    }

    /// <summary>
    /// Generates line segments for a coordinate axis tripod.
    /// </summary>
    /// <param name="origin">Tripod origin.</param>
    /// <param name="length">Length of each axis.</param>
    /// <returns>6 vertices (3 lines for X, Y, Z axes).</returns>
    public static Vector3[] GenerateAxisTripod(Vector3 origin, float length = 1.0f)
    {
        return new[]
        {
            origin, origin + Vector3.UnitX * length,
            origin, origin + Vector3.UnitY * length,
            origin, origin + Vector3.UnitZ * length
        };
    }

    /// <summary>
    /// Generates a colored point cloud from positions and a scalar field.
    /// </summary>
    /// <param name="positions">Point positions.</param>
    /// <param name="scalarField">Scalar values per point (used for coloring).</param>
    /// <param name="minColor">Color at minimum scalar value.</param>
    /// <param name="maxColor">Color at maximum scalar value.</param>
    /// <returns>Tuples of position and color for each point.</returns>
    public static (Vector3 Position, Vector4 Color)[] GeneratePointCloudVisualization(
        ReadOnlySpan<Vector3> positions,
        ReadOnlySpan<float> scalarField,
        Vector4 minColor,
        Vector4 maxColor)
    {
        if (positions.Length != scalarField.Length)
            throw new ArgumentException("Positions and scalar field must have the same length.");

        var result = new (Vector3, Vector4)[positions.Length];
        float minVal = float.MaxValue, maxVal = float.MinValue;

        for (int i = 0; i < scalarField.Length; i++)
        {
            if (scalarField[i] < minVal) minVal = scalarField[i];
            if (scalarField[i] > maxVal) maxVal = scalarField[i];
        }

        float range = maxVal - minVal;
        if (range < 1e-10f) range = 1;

        for (int i = 0; i < positions.Length; i++)
        {
            float t = (scalarField[i] - minVal) / range;
            result[i] = (positions[i], Vector4.Lerp(minColor, maxColor, t));
        }

        return result;
    }

    /// <summary>
    /// Generates a gradient field visualization as arrows from a 3D scalar field.
    /// </summary>
    /// <param name="field">Scalar field values (flat 3D grid).</param>
    /// <param name="gridSize">Grid dimensions (width, height, depth).</param>
    /// <param name="cellSize">Size of each grid cell.</param>
    /// <param name="origin">Origin of the grid in world space.</param>
    /// <param name="arrowScale">Scale factor for gradient arrows.</param>
    /// <returns>Line segments representing gradient arrows.</returns>
    public static List<(Vector3 Start, Vector3 End, Vector4 Color)> GenerateGradientFieldVisualization(
        ReadOnlySpan<float> field,
        Vector3Int gridSize,
        float cellSize,
        Vector3 origin,
        float arrowScale = 1.0f)
    {
        var arrows = new List<(Vector3, Vector3, Vector4)>();

        for (int z = 1; z < gridSize.Z - 1; z++)
        {
            for (int y = 1; y < gridSize.Y - 1; y++)
            {
                for (int x = 1; x < gridSize.X - 1; x++)
                {
                    int idx = x + y * gridSize.X + z * gridSize.X * gridSize.Y;
                    float gx = (field[idx + 1] - field[idx - 1]) * 0.5f;
                    float gy = (field[idx + gridSize.X] - field[idx - gridSize.X]) * 0.5f;
                    float gz = (field[idx + gridSize.X * gridSize.Y] - field[idx - gridSize.X * gridSize.Y]) * 0.5f;

                    Vector3 grad = new Vector3(gx, gy, gz);
                    float magnitude = grad.Length();
                    if (magnitude < 1e-6f) continue;

                    Vector3 pos = origin + new Vector3(x, y, z) * cellSize;
                    Vector3 tip = pos + grad * arrowScale / magnitude;

                    float t = Math.Clamp(magnitude * arrowScale, 0, 1);
                    Vector4 color = Vector4.Lerp(new Vector4(0, 0, 1, 1), new Vector4(1, 0, 0, 1), t);

                    arrows.Add((pos, tip, color));
                }
            }
        }

        return arrows;
    }

    /// <summary>
    /// Generates a neural network evaluation heatmap as colored quads.
    /// </summary>
    /// <param name="evaluations">2D grid of evaluation values.</param>
    /// <param name="gridWidth">Width of the evaluation grid.</param>
    /// <param name="gridHeight">Height of the evaluation grid.</param>
    /// <param name="worldMin">World-space minimum corner.</param>
    /// <param name="worldMax">World-space maximum corner.</param>
    /// <param name="colorMap">Color map function mapping [0,1] to RGBA color.</param>
    /// <returns>Quad vertices with colors for the heatmap.</returns>
    public static List<DebugQuad> GenerateNeuralHeatmap(
        ReadOnlySpan<float> evaluations,
        int gridWidth, int gridHeight,
        Vector3 worldMin, Vector3 worldMax,
        Func<float, Vector4>? colorMap = null)
    {
        colorMap ??= DefaultHeatmapColor;
        var quads = new List<DebugQuad>();

        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < evaluations.Length; i++)
        {
            if (evaluations[i] < minVal) minVal = evaluations[i];
            if (evaluations[i] > maxVal) maxVal = evaluations[i];
        }

        float range = maxVal - minVal;
        if (range < 1e-10f) range = 1;

        Vector3 size = worldMax - worldMin;
        float cellW = size.X / gridWidth;
        float cellH = size.Y / gridHeight;

        for (int y = 0; y < gridHeight; y++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                int idx = x + y * gridWidth;
                float normalized = (evaluations[idx] - minVal) / range;
                Vector4 color = colorMap(normalized);

                float cx = worldMin.X + x * cellW;
                float cy = worldMin.Y + y * cellH;
                float z = worldMin.Z;

                quads.Add(new DebugQuad(
                    new Vector3(cx, cy, z),
                    new Vector3(cx + cellW, cy, z),
                    new Vector3(cx + cellW, cy + cellH, z),
                    new Vector3(cx, cy + cellH, z),
                    color
                ));
            }
        }

        return quads;
    }

    /// <summary>
    /// Generates a wireframe sphere for debug rendering.
    /// </summary>
    /// <param name="center">Sphere center.</param>
    /// <param name="radius">Sphere radius.</param>
    /// <param name="latitudeLines">Number of latitude lines.</param>
    /// <param name="longitudeLines">Number of longitude lines.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateWireframeSphere(Vector3 center, float radius,
        int latitudeLines = 8, int longitudeLines = 16)
    {
        int totalVerts = (latitudeLines * longitudeLines + longitudeLines) * 2;
        var vertices = new Vector3[totalVerts];
        int idx = 0;

        float latStep = MathF.PI / latitudeLines;
        float lonStep = MathF.Tau / longitudeLines;

        for (int lat = 1; lat < latitudeLines; lat++)
        {
            float theta = lat * latStep;
            float sinLat = MathF.Sin(theta);
            float cosLat = MathF.Cos(theta);

            for (int lon = 0; lon < longitudeLines; lon++)
            {
                float phi0 = lon * lonStep;
                float phi1 = (lon + 1) * lonStep;

                vertices[idx++] = center + new Vector3(
                    sinLat * MathF.Cos(phi0),
                    cosLat,
                    sinLat * MathF.Sin(phi0)
                ) * radius;

                vertices[idx++] = center + new Vector3(
                    sinLat * MathF.Cos(phi1),
                    cosLat,
                    sinLat * MathF.Sin(phi1)
                ) * radius;
            }
        }

        for (int lon = 0; lon < longitudeLines; lon++)
        {
            float phi = lon * lonStep;
            float cosPhi = MathF.Cos(phi);
            float sinPhi = MathF.Sin(phi);

            vertices[idx++] = center + new Vector3(sinTheta(0) * cosPhi, cosTheta(0), sinTheta(0) * sinPhi) * radius;
            vertices[idx++] = center + new Vector3(sinTheta(latStep) * cosPhi, cosTheta(latStep), sinTheta(latStep) * sinPhi) * radius;
        }

        return vertices;

        static float sinTheta(float angle) => MathF.Sin(angle);
        static float cosTheta(float angle) => MathF.Cos(angle);
    }

    /// <summary>
    /// Generates line segments for a capsule shape.
    /// </summary>
    /// <param name="start">Capsule start point.</param>
    /// <param name="end">Capsule end point.</param>
    /// <param name="radius">Capsule radius.</param>
    /// <param name="segments">Number of segments for the circular parts.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateCapsuleLines(Vector3 start, Vector3 end, float radius, int segments = 16)
    {
        var vertices = new List<Vector3>();
        Vector3 axis = end - start;
        float length = axis.Length();
        if (length < 1e-6f) return GenerateBoundingSphereLines(start, radius, segments);

        Vector3 direction = axis / length;
        Vector3 up = MathHelpers.GetPerpendicular(direction);
        Vector3 right = Vector3.Cross(direction, up);

        float halfLen = length * 0.5f;
        Vector3 center = (start + end) * 0.5f;

        float step = MathF.Tau / segments;
        for (int i = 0; i < segments; i++)
        {
            float a0 = i * step;
            float a1 = (i + 1) * step;
            float c0 = MathF.Cos(a0), s0 = MathF.Sin(a0);
            float c1 = MathF.Cos(a1), s1 = MathF.Sin(a1);

            // Cylinder body
            Vector3 p0 = center + right * c0 * radius + up * s0 * radius;
            Vector3 p1 = center + right * c1 * radius + up * s1 * radius;
            vertices.Add(p0 - direction * halfLen);
            vertices.Add(p1 - direction * halfLen);
            vertices.Add(p0 + direction * halfLen);
            vertices.Add(p1 + direction * halfLen);

            // Connecting rings
            vertices.Add(p0 - direction * halfLen);
            vertices.Add(p0 + direction * halfLen);
        }

        // Hemisphere caps
        for (int i = 0; i < segments; i++)
        {
            float a0 = i * step;
            float a1 = (i + 1) * step;
            float c0 = MathF.Cos(a0), s0 = MathF.Sin(a0);
            float c1 = MathF.Cos(a1), s1 = MathF.Sin(a1);

            for (int j = 0; j < segments / 2; j++)
            {
                float phi0 = j * (MathF.PI / (segments / 2));
                float phi1 = (j + 1) * (MathF.PI / (segments / 2));
                float sp0 = MathF.Sin(phi0), cp0 = MathF.Cos(phi0);
                float sp1 = MathF.Sin(phi1), cp1 = MathF.Cos(phi1);

                // Start cap
                vertices.Add(start + (direction * cp0 + right * c0 * sp0 + up * s0 * sp0) * radius);
                vertices.Add(start + (direction * cp0 + right * c1 * sp0 + up * s1 * sp0) * radius);

                // End cap
                vertices.Add(end + (-direction * cp0 + right * c0 * sp0 + up * s0 * sp0) * radius);
                vertices.Add(end + (-direction * cp0 + right * c1 * sp0 + up * s1 * sp0) * radius);
            }
        }

        return vertices.ToArray();
    }

    /// <summary>
    /// Generates a debug grid on the XZ plane.
    /// </summary>
    /// <param name="center">Grid center.</param>
    /// <param name="size">Grid total size.</param>
    /// <param name="divisions">Number of divisions per axis.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateDebugGrid(Vector3 center, float size, int divisions = 10)
    {
        var vertices = new List<Vector3>();
        float halfSize = size * 0.5f;
        float step = size / divisions;

        for (int i = 0; i <= divisions; i++)
        {
            float offset = -halfSize + i * step;

            // Lines along X
            vertices.Add(center + new Vector3(-halfSize, 0, offset));
            vertices.Add(center + new Vector3(halfSize, 0, offset));

            // Lines along Z
            vertices.Add(center + new Vector3(offset, 0, -halfSize));
            vertices.Add(center + new Vector3(offset, 0, halfSize));
        }

        return vertices.ToArray();
    }

    /// <summary>
    /// Generates line segments for a frustum.
    /// </summary>
    /// <param name="viewProjection">View-projection matrix.</param>
    /// <returns>24 vertices forming frustum wireframe.</returns>
    public static Vector3[] GenerateFrustumLines(Matrix4x4 viewProjection)
    {
        Matrix4x4 invVP = Matrix4x4.Identity;
        bool inverted = Matrix4x4.Invert(viewProjection, out invVP);
        if (!inverted) return Array.Empty<Vector3>();

        ReadOnlySpan<Vector4> ndcCorners = stackalloc Vector4[]
        {
            new(-1, -1, 0, 1), new(1, -1, 0, 1),
            new(1, 1, 0, 1), new(-1, 1, 0, 1),
            new(-1, -1, 1, 1), new(1, -1, 1, 1),
            new(1, 1, 1, 1), new(-1, 1, 1, 1)
        };

        Vector3[] worldCorners = new Vector3[8];
        for (int i = 0; i < 8; i++)
        {
            Vector4 transformed = Vector4.Transform(ndcCorners[i], invVP);
            worldCorners[i] = new Vector3(transformed.X, transformed.Y, transformed.Z) / transformed.W;
        }

        return GenerateBoxFromCorners(worldCorners);
    }

    /// <summary>
    /// Generates a text label representation for debug overlay.
    /// </summary>
    /// <param name="position">World-space position.</param>
    /// <param name="text">Label text.</param>
    /// <param name="color">Label color.</param>
    /// <returns>Debug label data.</returns>
    public static DebugLabel CreateDebugLabel(Vector3 position, string text, Vector4 color)
    {
        return new DebugLabel(position, text, color);
    }

    /// <summary>
    /// Generates line segments for a cone shape.
    /// </summary>
    /// <param name="apex">Cone apex.</param>
    /// <param name="direction">Cone direction (normalized).</param>
    /// <param name="height">Cone height.</param>
    /// <param name="angle">Cone half-angle in radians.</param>
    /// <param name="segments">Number of segments for the base circle.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateConeLines(Vector3 apex, Vector3 direction, float height,
        float angle, int segments = 16)
    {
        Vector3 baseCenter = apex + direction * height;
        float baseRadius = height * MathF.Tan(angle);
        Vector3 up = MathHelpers.GetPerpendicular(direction);
        Vector3 right = Vector3.Cross(direction, up);

        var vertices = new Vector3[segments * 4];
        float step = MathF.Tau / segments;

        for (int i = 0; i < segments; i++)
        {
            float a0 = i * step;
            float a1 = (i + 1) * step;
            Vector3 p0 = baseCenter + (right * MathF.Cos(a0) + up * MathF.Sin(a0)) * baseRadius;
            Vector3 p1 = baseCenter + (right * MathF.Cos(a1) + up * MathF.Sin(a1)) * baseRadius;

            // Base circle
            vertices[i * 4] = p0;
            vertices[i * 4 + 1] = p1;

            // Lines from apex to base
            vertices[i * 4 + 2] = apex;
            vertices[i * 4 + 3] = p0;
        }

        return vertices;
    }

    /// <summary>
    /// Generates a spiral path for debug visualization.
    /// </summary>
    /// <param name="center">Spiral center.</param>
    /// <param name="axis">Spiral axis direction.</param>
    /// <param name="radius">Spiral radius.</param>
    /// <param name="height">Total height of the spiral.</param>
    /// <param name="turns">Number of turns.</param>
    /// <param name="points">Number of points along the spiral.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateSpiralLines(Vector3 center, Vector3 axis, float radius,
        float height, int turns, int points = 256)
    {
        Vector3 up = MathHelpers.GetPerpendicular(axis);
        Vector3 right = Vector3.Cross(axis, up);

        var vertices = new Vector3[points * 2];
        float step = (turns * MathF.Tau) / points;

        for (int i = 0; i < points; i++)
        {
            float t0 = (float)i / points;
            float t1 = (float)(i + 1) / points;
            float angle0 = i * step;
            float angle1 = (i + 1) * step;

            Vector3 p0 = center + axis * (t0 * height - height * 0.5f) +
                (right * MathF.Cos(angle0) + up * MathF.Sin(angle0)) * radius;
            Vector3 p1 = center + axis * (t1 * height - height * 0.5f) +
                (right * MathF.Cos(angle1) + up * MathF.Sin(angle1)) * radius;

            vertices[i * 2] = p0;
            vertices[i * 2 + 1] = p1;
        }

        return vertices;
    }

    /// <summary>
    /// Generates colored line segments based on a scalar field along a path.
    /// </summary>
    /// <param name="path">Path points.</param>
    /// <param name="values">Scalar values at each path point.</param>
    /// <param name="colorLow">Color for low values.</param>
    /// <param name="colorHigh">Color for high values.</param>
    /// <returns>Line segments with per-vertex colors.</returns>
    public static (Vector3 Start, Vector3 End, Vector4 ColorStart, Vector4 ColorEnd)[] GenerateColorPath(
        ReadOnlySpan<Vector3> path,
        ReadOnlySpan<float> values,
        Vector4 colorLow,
        Vector4 colorHigh)
    {
        if (path.Length < 2 || path.Length != values.Length)
            throw new ArgumentException("Path and values must have the same length >= 2.");

        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] < minVal) minVal = values[i];
            if (values[i] > maxVal) maxVal = values[i];
        }
        float range = maxVal - minVal;
        if (range < 1e-10f) range = 1;

        var segments = new (Vector3, Vector3, Vector4, Vector4)[path.Length - 1];
        for (int i = 0; i < segments.Length; i++)
        {
            float t0 = (values[i] - minVal) / range;
            float t1 = (values[i + 1] - minVal) / range;
            segments[i] = (path[i], path[i + 1],
                Vector4.Lerp(colorLow, colorHigh, t0),
                Vector4.Lerp(colorLow, colorHigh, t1));
        }

        return segments;
    }

    /// <summary>
    /// Generates a heatmap color from a normalized value using a classic blue-cyan-green-yellow-red ramp.
    /// </summary>
    public static Vector4 DefaultHeatmapColor(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        if (t < 0.25f)
            return Vector4.Lerp(new Vector4(0, 0, 1, 1), new Vector4(0, 1, 1, 1), t * 4);
        if (t < 0.5f)
            return Vector4.Lerp(new Vector4(0, 1, 1, 1), new Vector4(0, 1, 0, 1), (t - 0.25f) * 4);
        if (t < 0.75f)
            return Vector4.Lerp(new Vector4(0, 1, 0, 1), new Vector4(1, 1, 0, 1), (t - 0.5f) * 4);
        return Vector4.Lerp(new Vector4(1, 1, 0, 1), new Vector4(1, 0, 0, 1), (t - 0.75f) * 4);
    }

    /// <summary>
    /// Generates a viridis-inspired heatmap color.
    /// </summary>
    public static Vector4 ViridisColorMap(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        float r = Math.Clamp(0.267f + t * (0.329f + t * (-1.45f + t * 1.78f)), 0, 1);
        float g = Math.Clamp(0.004f + t * (1.42f + t * (-1.69f + t * 0.90f)), 0, 1);
        float b = Math.Clamp(0.329f + t * (1.44f + t * (-3.32f + t * 2.16f)), 0, 1);
        return new Vector4(r, g, b, 1);
    }

    /// <summary>
    /// Generates a hot-to-cold diverging colormap.
    /// </summary>
    public static Vector4 HotColdColorMap(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        Vector4 cold = new(0.2f, 0.4f, 0.9f, 1);
        Vector4 neutral = new(0.9f, 0.9f, 0.9f, 1);
        Vector4 hot = new(0.9f, 0.2f, 0.1f, 1);

        if (t < 0.5f)
            return Vector4.Lerp(cold, neutral, t * 2);
        return Vector4.Lerp(neutral, hot, (t - 0.5f) * 2);
    }

    /// <summary>
    /// Generates line segments for a torus shape.
    /// </summary>
    /// <param name="center">Torus center.</param>
    /// <param name="majorRadius">Major radius.</param>
    /// <param name="minorRadius">Minor radius.</param>
    /// <param name="majorSegments">Major segments.</param>
    /// <param name="minorSegments">Minor segments.</param>
    /// <returns>Line segment vertices.</returns>
    public static Vector3[] GenerateTorusLines(Vector3 center, float majorRadius, float minorRadius,
        int majorSegments = 24, int minorSegments = 12)
    {
        var vertices = new List<Vector3>();
        float majorStep = MathF.Tau / majorSegments;
        float minorStep = MathF.Tau / minorSegments;

        for (int i = 0; i < majorSegments; i++)
        {
            float theta0 = i * majorStep;
            float theta1 = (i + 1) * majorStep;
            Vector3 center0 = center + new Vector3(MathF.Cos(theta0), 0, MathF.Sin(theta0)) * majorRadius;
            Vector3 center1 = center + new Vector3(MathF.Cos(theta1), 0, MathF.Sin(theta1)) * majorRadius;

            for (int j = 0; j < minorSegments; j++)
            {
                float phi0 = j * minorStep;
                float phi1 = (j + 1) * minorStep;

                Vector3 p0 = GetTorusPoint(center0, theta0, phi0, minorRadius);
                Vector3 p1 = GetTorusPoint(center0, theta0, phi1, minorRadius);
                Vector3 p2 = GetTorusPoint(center1, theta1, phi0, minorRadius);

                vertices.Add(p0);
                vertices.Add(p1);
                vertices.Add(p0);
                vertices.Add(p2);
            }
        }

        return vertices.ToArray();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector3 GetTorusPoint(Vector3 ringCenter, float theta, float phi, float minorRadius)
    {
        Vector3 normal = new Vector3(MathF.Cos(theta), 0, MathF.Sin(theta));
        Vector3 bitangent = new Vector3(0, 1, 0);
        return ringCenter + (bitangent * MathF.Sin(phi) + normal * MathF.Cos(phi)) * minorRadius;
    }

    /// <summary>
    /// Generates arrow visualization for a vector field.
    /// </summary>
    /// <param name="field">Vector field values.</param>
    /// <param name="gridSize">Grid dimensions.</param>
    /// <param name="cellSize">Cell size.</param>
    /// <param name="origin">Grid origin.</param>
    /// <param name="scale">Arrow scale factor.</param>
    /// <returns>Line segments for field arrows.</returns>
    public static Vector3[] GenerateVectorFieldArrows(
        ReadOnlySpan<Vector3> field,
        Vector3Int gridSize,
        float cellSize,
        Vector3 origin,
        float scale = 1.0f)
    {
        var vertices = new List<Vector3>();

        for (int z = 0; z < gridSize.Z; z++)
        {
            for (int y = 0; y < gridSize.Y; y++)
            {
                for (int x = 0; x < gridSize.X; x++)
                {
                    int idx = x + y * gridSize.X + z * gridSize.X * gridSize.Y;
                    Vector3 value = field[idx];
                    float magnitude = value.Length();
                    if (magnitude < 1e-6f) continue;

                    Vector3 pos = origin + new Vector3(x, y, z) * cellSize;
                    vertices.AddRange(GenerateRayLines(pos, value / magnitude, magnitude * scale, magnitude * scale * 0.2f));
                }
            }
        }

        return vertices.ToArray();
    }

    /// <summary>
    /// Generates debug visualization for a collision hit result.
    /// </summary>
    /// <param name="rayOrigin">Original ray origin.</param>
    /// <param name="hitPoint">Hit point in world space.</param>
    /// <param name="hitNormal">Surface normal at hit point.</param>
    /// <param name="hitDistance">Distance to the hit.</param>
    /// <param name="normalLength">Length of the normal visualization line.</param>
    /// <returns>Line segments for the hit visualization.</returns>
    public static Vector3[] GenerateHitVisualization(
        Vector3 rayOrigin, Vector3 hitPoint, Vector3 hitNormal,
        float hitDistance, float normalLength = 0.5f)
    {
        var vertices = new List<Vector3>();

        // Ray to hit point
        vertices.Add(rayOrigin);
        vertices.Add(hitPoint);

        // Hit normal
        vertices.Add(hitPoint);
        vertices.Add(hitPoint + hitNormal * normalLength);

        // Small cross at hit point
        float crossSize = 0.05f;
        Vector3 up = MathHelpers.GetPerpendicular(hitNormal);
        Vector3 right = Vector3.Cross(hitNormal, up);

        vertices.Add(hitPoint - right * crossSize);
        vertices.Add(hitPoint + right * crossSize);
        vertices.Add(hitPoint - up * crossSize);
        vertices.Add(hitPoint + up * crossSize);

        return vertices.ToArray();
    }

    /// <summary>
    /// Generates a text mesh representation for debug overlay.
    /// </summary>
    /// <param name="text">Text to render.</param>
    /// <param name="position">World-space position.</param>
    /// <param name="scale">Text scale.</param>
    /// <param name="color">Text color.</param>
    /// <returns>Debug text data.</returns>
    public static DebugTextMesh GenerateDebugText(string text, Vector3 position, float scale, Vector4 color)
    {
        return new DebugTextMesh(text, position, scale, color);
    }

    /// <summary>
    /// Generates bounding volume hierarchy (BVH) wireframe for debugging.
    /// </summary>
    /// <param name="nodes">BVH nodes with bounds and child indices.</param>
    /// <param name="rootIndex">Index of the root node.</param>
    /// <returns>All bounding box line segments for the BVH.</returns>
    public static List<Vector3> GenerateBvhWireframe(ReadOnlySpan<BvhDebugNode> nodes, int rootIndex)
    {
        var allVertices = new List<Vector3>();
        GenerateBvhWireframeRecursive(nodes, rootIndex, allVertices);
        return allVertices;
    }

    private static void GenerateBvhWireframeRecursive(
        ReadOnlySpan<BvhDebugNode> nodes, int index, List<Vector3> vertices)
    {
        if (index < 0 || index >= nodes.Length) return;

        ref readonly BvhDebugNode node = ref nodes[index];
        vertices.AddRange(GenerateBoundingBoxLines(node.BoundsMin, node.BoundsMax));

        if (node.LeftChild >= 0)
            GenerateBvhWireframeRecursive(nodes, node.LeftChild, vertices);
        if (node.RightChild >= 0)
            GenerateBvhWireframeRecursive(nodes, node.RightChild, vertices);
    }
}

/// <summary>
/// Represents a 3D integer vector for grid dimensions.
/// </summary>
public readonly struct Vector3Int : IEquatable<Vector3Int>
{
    /// <summary>X component.</summary>
    public int X { get; }

    /// <summary>Y component.</summary>
    public int Y { get; }

    /// <summary>Z component.</summary>
    public int Z { get; }

    public Vector3Int(int x, int y, int z) { X = x; Y = y; Z = z; }

    public static Vector3Int Zero => new(0, 0, 0);
    public static Vector3Int One => new(1, 1, 1);

    public bool Equals(Vector3Int other) => X == other.X && Y == other.Y && Z == other.Z;
    public override bool Equals(object? obj) => obj is Vector3Int other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(X, Y, Z);
    public override string ToString() => $"({X}, {Y}, {Z})";

    public static bool operator ==(Vector3Int left, Vector3Int right) => left.Equals(right);
    public static bool operator !=(Vector3Int left, Vector3Int right) => !left.Equals(right);
}

/// <summary>
/// Debug quad for heatmap visualization.
/// </summary>
public readonly struct DebugQuad
{
    /// <summary>First corner position.</summary>
    public Vector3 V0 { get; }

    /// <summary>Second corner position.</summary>
    public Vector3 V1 { get; }

    /// <summary>Third corner position.</summary>
    public Vector3 V2 { get; }

    /// <summary>Fourth corner position.</summary>
    public Vector3 V3 { get; }

    /// <summary>Quad color.</summary>
    public Vector4 Color { get; }

    public DebugQuad(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3, Vector4 color)
    {
        V0 = v0; V1 = v1; V2 = v2; V3 = v3; Color = color;
    }
}

/// <summary>
/// Debug label for world-space text overlay.
/// </summary>
public sealed class DebugLabel
{
    /// <summary>World-space position.</summary>
    public Vector3 Position { get; }

    /// <summary>Label text.</summary>
    public string Text { get; }

    /// <summary>Label color.</summary>
    public Vector4 Color { get; }

    public DebugLabel(Vector3 position, string text, Vector4 color)
    {
        Position = position;
        Text = text;
        Color = color;
    }
}

/// <summary>
/// Debug text mesh for rendering text in 3D space.
/// </summary>
public sealed class DebugTextMesh
{
    /// <summary>Text content.</summary>
    public string Text { get; }

    /// <summary>World-space position.</summary>
    public Vector3 Position { get; }

    /// <summary>Text scale.</summary>
    public float Scale { get; }

    /// <summary>Text color.</summary>
    public Vector4 Color { get; }

    public DebugTextMesh(string text, Vector3 position, float scale, Vector4 color)
    {
        Text = text;
        Position = position;
        Scale = scale;
        Color = color;
    }
}

/// <summary>
/// BVH debug node for wireframe generation.
/// </summary>
public readonly struct BvhDebugNode
{
    /// <summary>Minimum bounds of this node.</summary>
    public Vector3 BoundsMin { get; }

    /// <summary>Maximum bounds of this node.</summary>
    public Vector3 BoundsMax { get; }

    /// <summary>Left child index, or -1 if leaf.</summary>
    public int LeftChild { get; }

    /// <summary>Right child index, or -1 if leaf.</summary>
    public int RightChild { get; }

    public BvhDebugNode(Vector3 boundsMin, Vector3 boundsMax, int leftChild, int rightChild)
    {
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        LeftChild = leftChild;
        RightChild = rightChild;
    }
}

/// <summary>
/// Assert utilities with detailed failure messages and conditional compilation support.
/// </summary>
public static class AssertUtils
{
    /// <summary>
    /// Asserts that a condition is true, throwing with a detailed message on failure.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertTrue(bool condition, string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        if (!condition)
        {
            throw new DebugAssertionException(
                $"Assertion failed: {message}\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a condition is false.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertFalse(bool condition, string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        AssertTrue(!condition, $"Expected false but got true: {message}", memberName, filePath, lineNumber);
    }

    /// <summary>
    /// Asserts that two values are approximately equal within a tolerance.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertApproximately(float expected, float actual, float tolerance = 1e-6f,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        if (MathF.Abs(expected - actual) > tolerance)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Expected approximately {expected} but got {actual} (tolerance {tolerance})\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that two vectors are approximately equal.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertApproximately(Vector3 expected, Vector3 actual, float tolerance = 1e-6f,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        float dist = Vector3.Distance(expected, actual);
        if (dist > tolerance)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Vectors differ by {dist} (tolerance {tolerance})\n" +
                $"  Expected: {expected}\n  Actual:   {actual}\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a value is within a range.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertInRange(float value, float min, float max,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        if (value < min || value > max)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Value {value} is not in range [{min}, {max}]\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a value is not NaN or infinity.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertFinite(float value, string name = "value",
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
        {
            throw new DebugAssertionException(
                $"Assertion failed: {name} is {value}\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a vector is normalized.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertNormalized(Vector3 vector, float tolerance = 1e-4f,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        float length = vector.Length();
        if (MathF.Abs(length - 1.0f) > tolerance)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Vector is not normalized (length={length})\n" +
                $"  Vector: {vector}\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that an index is within valid bounds.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertBounds(int index, int count,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        if (index < 0 || index >= count)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Index {index} is out of bounds [0, {count})\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a reference is not null.
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertNotNull<T>(T? value, string name = "value",
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0) where T : class
    {
        if (value is null)
        {
            throw new DebugAssertionException(
                $"Assertion failed: {name} is null\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a quaternion is a valid rotation (unit length).
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertValidRotation(Quaternion rotation, float tolerance = 1e-4f,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        float length = rotation.Length();
        if (MathF.Abs(length - 1.0f) > tolerance)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Quaternion is not a valid rotation (length={length})\n" +
                $"  Quaternion: {rotation}\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Asserts that a matrix is invertible (non-zero determinant).
    /// </summary>
    [Conditional("DEBUG")]
    public static void AssertInvertible(Matrix4x4 matrix,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        float det = Matrix4x4Compat.Determinant(matrix);
        if (MathF.Abs(det) < 1e-6f)
        {
            throw new DebugAssertionException(
                $"Assertion failed: Matrix is not invertible (determinant={det})\n" +
                $"  at {memberName} in {filePath}:{lineNumber}");
        }
    }

    /// <summary>
    /// Logs a debug warning without throwing.
    /// </summary>
    [Conditional("DEBUG")]
    public static void DebugWarning(string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        System.Diagnostics.Debug.WriteLine(
            $"[GDNN Warning] {message}\n  at {memberName} in {filePath}:{lineNumber}");
    }

    /// <summary>
    /// Logs a debug info message.
    /// </summary>
    [Conditional("DEBUG")]
    public static void DebugLog(string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        System.Diagnostics.Debug.WriteLine(
            $"[GDNN] {message}\n  at {memberName} in {filePath}:{lineNumber}");
    }
}

/// <summary>
/// Exception thrown when a debug assertion fails.
/// </summary>
public sealed class DebugAssertionException : Exception
{
    public DebugAssertionException(string message) : base(message) { }
    public DebugAssertionException(string message, Exception inner) : base(message, inner) { }
}
