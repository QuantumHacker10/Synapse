using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Polygonization;

/// <summary>
/// Cible de rasterisation : visibility buffer 64 bits par pixel, façon Nanite.
/// Bits 63..32 : clé de profondeur (plus grand = plus proche) ;
/// bits 31..0 : payload (meshlet, triangle) + 1, 0 = pixel vide.
/// La résolution du matériau/éclairage se fait en différé à partir du payload.
/// </summary>
public sealed class RasterTarget
{
    private readonly ulong[] _buffer;

    public int Width { get; }
    public int Height { get; }

    public RasterTarget(int width, int height)
    {
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width), "Target must be at least 1x1.");
        Width = width;
        Height = height;
        _buffer = new ulong[width * height];
    }

    internal ulong[] Buffer => _buffer;

    public void Clear() => Array.Clear(_buffer);

    /// <summary>Décode (meshlet, triangle) du pixel ; false si le pixel est vide.</summary>
    public bool TryDecode(int x, int y, out int meshletIndex, out int triangleIndex)
    {
        uint payload = unchecked((uint)_buffer[y * Width + x]);
        if (payload == 0)
        {
            meshletIndex = 0;
            triangleIndex = 0;
            return false;
        }
        payload--;
        meshletIndex = (int)(payload >> 10);
        triangleIndex = (int)(payload & 0x3FF);
        return true;
    }

    /// <summary>Proximité normalisée du pixel (0 = vide/loin, 1 = plan proche).</summary>
    public float ClosenessAt(int x, int y)
        => (_buffer[y * Width + x] >> 32) / (float)uint.MaxValue;

    public bool IsCovered(int x, int y) => unchecked((uint)_buffer[y * Width + x]) != 0;

    public int CountCoveredPixels()
    {
        int count = 0;
        foreach (ulong v in _buffer)
        {
            if (unchecked((uint)v) != 0)
                count++;
        }
        return count;
    }
}

/// <summary>Statistiques mesurées d'une passe de rasterisation.</summary>
public sealed class RasterStats
{
    public int MeshletsSubmitted { get; init; }
    public int TrianglesSubmitted { get; init; }
    public int TrianglesBackfaceCulled { get; init; }
    public int TrianglesNearClipped { get; init; }
    public int TrianglesRasterized { get; init; }
    public double ElapsedMs { get; init; }

    public override string ToString() =>
        $"{MeshletsSubmitted} meshlets, tris {TrianglesRasterized}/{TrianglesSubmitted} " +
        $"(backface −{TrianglesBackfaceCulled}, near −{TrianglesNearClipped}), {ElapsedMs:F2}ms";
}

/// <summary>
/// Rasteriseur software de meshlets — implémentation CPU de référence de
/// l'algorithme compute façon Nanite (un groupe par meshlet, fonctions d'arête,
/// règle top-left, atomicMax 64 bits sur le visibility buffer).
/// <see cref="GDNN.GPU.MeshletRasterizerShaderGenerator"/> émet le même
/// algorithme en GLSL compute ; ce chemin CPU sert de référence de conformité
/// et de repli sans GPU, parallélisé sur les meshlets.
/// </summary>
public sealed class SoftwareRasterizer
{
    /// <summary>Epsilon de rejet près du plan proche (w de clip minimal).</summary>
    public float NearClipEpsilon { get; set; } = 1e-4f;

    public RasterStats LastStats { get; private set; } = new();

    /// <summary>
    /// Rasterise les meshlets sélectionnés dans la cible. Les positions viennent
    /// du maillage du niveau de LOD dont sont issus les meshlets.
    /// </summary>
    public RasterStats Rasterize(
        RasterTarget target,
        NeuralPolygonMesh mesh,
        IReadOnlyList<NeuralMeshlet> meshlets,
        in CameraView camera)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(meshlets);

        var sw = Stopwatch.StartNew();
        Matrix4x4 viewProjection = camera.ViewProjection;

        int submitted = 0, backface = 0, nearClipped = 0, rasterized = 0;

        Parallel.For(0, meshlets.Count, meshletIndex =>
        {
            var (s, b, n, r) = RasterizeMeshlet(
                target, mesh, meshlets[meshletIndex], meshletIndex, viewProjection);
            Interlocked.Add(ref submitted, s);
            Interlocked.Add(ref backface, b);
            Interlocked.Add(ref nearClipped, n);
            Interlocked.Add(ref rasterized, r);
        });

        sw.Stop();
        LastStats = new RasterStats
        {
            MeshletsSubmitted = meshlets.Count,
            TrianglesSubmitted = submitted,
            TrianglesBackfaceCulled = backface,
            TrianglesNearClipped = nearClipped,
            TrianglesRasterized = rasterized,
            ElapsedMs = sw.Elapsed.TotalMilliseconds
        };
        return LastStats;
    }

    private (int Submitted, int Backface, int NearClipped, int Rasterized) RasterizeMeshlet(
        RasterTarget target, NeuralPolygonMesh mesh, NeuralMeshlet meshlet,
        int meshletIndex, in Matrix4x4 viewProjection)
    {
        // Équivalent de la mémoire partagée du workgroup : transformer une fois
        // les ≤ 64 sommets du meshlet.
        Span<Vector4> clip = stackalloc Vector4[meshlet.VertexCount];
        for (int v = 0; v < meshlet.VertexCount; v++)
        {
            clip[v] = Vector4.Transform(
                new Vector4(mesh.Positions[meshlet.VertexIndices[v]], 1f), viewProjection);
        }

        int backface = 0, nearClipped = 0, rasterized = 0;
        int triangleCount = meshlet.TriangleCount;

        for (int t = 0; t < triangleCount; t++)
        {
            Vector4 c0 = clip[meshlet.LocalIndices[t * 3]];
            Vector4 c1 = clip[meshlet.LocalIndices[t * 3 + 1]];
            Vector4 c2 = clip[meshlet.LocalIndices[t * 3 + 2]];

            // Rejet conservatif près du plan proche (pas de clipping polygonal :
            // pour un visibility buffer, perdre un triangle traversant la caméra
            // est acceptable et compté).
            if (c0.W <= NearClipEpsilon || c1.W <= NearClipEpsilon || c2.W <= NearClipEpsilon)
            {
                nearClipped++;
                continue;
            }

            Vector3 s0 = ToScreen(c0, target.Width, target.Height);
            Vector3 s1 = ToScreen(c1, target.Width, target.Height);
            Vector3 s2 = ToScreen(c2, target.Width, target.Height);

            // Virgule fixe 24.8 : l'arithmétique d'arête entière est exacte, donc
            // deux triangles partageant une arête se partagent les pixels sans
            // trou ni recouvrement (impossible à garantir en flottant).
            var f0 = Snap(s0);
            var f1 = Snap(s1);
            var f2 = Snap(s2);

            // Aire signée en écran (y vers le bas) : le winding CCW vu de
            // l'extérieur devient négatif ; on réordonne en aire positive.
            long area2 = (f1.X - f0.X) * (f2.Y - f0.Y) - (f2.X - f0.X) * (f1.Y - f0.Y);
            if (area2 == 0)
                continue; // dégénéré
            if (area2 > 0)
            {
                backface++;
                continue;
            }
            (f1, f2) = (f2, f1);
            (s1, s2) = (s2, s1);
            area2 = -area2;

            if (RasterizeTriangle(target, f0, f1, f2, s0.Z, s1.Z, s2.Z, area2,
                    EncodePayload(meshletIndex, t)))
                rasterized++;
        }

        return (triangleCount, backface, nearClipped, rasterized);
    }

    private static Vector3 ToScreen(Vector4 clip, int width, int height)
    {
        float invW = 1f / clip.W;
        return new Vector3(
            (clip.X * invW * 0.5f + 0.5f) * width,
            (0.5f - clip.Y * invW * 0.5f) * height,
            clip.Z * invW); // z NDC ∈ [0,1]
    }

    private const int SubPixelBits = 8;
    private const int SubPixelScale = 1 << SubPixelBits;
    private const int PixelCenter = SubPixelScale / 2;

    private static uint EncodePayload(int meshletIndex, int triangleIndex)
        => (uint)((meshletIndex << 10) | triangleIndex) + 1u;

    private static (long X, long Y) Snap(Vector3 screen)
        => ((long)MathF.Round(screen.X * SubPixelScale), (long)MathF.Round(screen.Y * SubPixelScale));

    private static bool RasterizeTriangle(
        RasterTarget target,
        (long X, long Y) f0, (long X, long Y) f1, (long X, long Y) f2,
        float z0, float z1, float z2, long area2, uint payload)
    {
        long fMinX = Math.Min(f0.X, Math.Min(f1.X, f2.X));
        long fMaxX = Math.Max(f0.X, Math.Max(f1.X, f2.X));
        long fMinY = Math.Min(f0.Y, Math.Min(f1.Y, f2.Y));
        long fMaxY = Math.Max(f0.Y, Math.Max(f1.Y, f2.Y));

        int minX = Math.Max(0, (int)(fMinX >> SubPixelBits));
        int maxX = Math.Min(target.Width - 1, (int)((fMaxX + SubPixelScale - 1) >> SubPixelBits));
        int minY = Math.Max(0, (int)(fMinY >> SubPixelBits));
        int maxY = Math.Min(target.Height - 1, (int)((fMaxY + SubPixelScale - 1) >> SubPixelBits));
        if (minX > maxX || minY > maxY)
            return false;

        // Fonctions d'arête entières (positives à l'intérieur, aire positive).
        // w(p) = (b−a) × (p−a), incréments exacts de ±(b−a) par pixel.
        long a0 = f1.Y - f2.Y, b0 = f2.X - f1.X;
        long a1 = f2.Y - f0.Y, b1 = f0.X - f2.X;
        long a2 = f0.Y - f1.Y, b2 = f1.X - f0.X;

        // Règle top-left : arêtes top/left inclusives (bias 0), autres exclusives
        // (bias −1 en arithmétique entière) — chaque pixel d'une arête partagée
        // appartient à exactement un des deux triangles.
        long bias0 = IsTopLeftEdge(f1, f2) ? 0 : -1;
        long bias1 = IsTopLeftEdge(f2, f0) ? 0 : -1;
        long bias2 = IsTopLeftEdge(f0, f1) ? 0 : -1;

        long px = ((long)minX << SubPixelBits) + PixelCenter;
        long py = ((long)minY << SubPixelBits) + PixelCenter;
        long rowW0 = (f2.X - f1.X) * (py - f1.Y) - (f2.Y - f1.Y) * (px - f1.X) + bias0;
        long rowW1 = (f0.X - f2.X) * (py - f2.Y) - (f0.Y - f2.Y) * (px - f2.X) + bias1;
        long rowW2 = (f1.X - f0.X) * (py - f0.Y) - (f1.Y - f0.Y) * (px - f0.X) + bias2;

        long stepX0 = a0 << SubPixelBits, stepY0 = b0 << SubPixelBits;
        long stepX1 = a1 << SubPixelBits, stepY1 = b1 << SubPixelBits;
        long stepX2 = a2 << SubPixelBits, stepY2 = b2 << SubPixelBits;

        float invArea = 1f / area2;
        var buffer = target.Buffer;
        bool any = false;

        for (int y = minY; y <= maxY; y++)
        {
            long w0 = rowW0, w1 = rowW1, w2 = rowW2;
            int rowBase = y * target.Width;

            for (int x = minX; x <= maxX; x++)
            {
                if ((w0 | w1 | w2) >= 0)
                {
                    // Interpolation barycentrique du z NDC (affine en écran) ;
                    // on retire le bias pour ne pas fausser les poids.
                    float z = ((w0 - bias0) * z0 + (w1 - bias1) * z1 + (w2 - bias2) * z2) * invArea;
                    float closeness = 1f - Math.Clamp(z, 0f, 1f);
                    ulong key = ((ulong)(uint)(closeness * uint.MaxValue) << 32) | payload;

                    AtomicMax(ref buffer[rowBase + x], key);
                    any = true;
                }
                w0 += stepX0; w1 += stepX1; w2 += stepX2;
            }
            rowW0 += stepY0; rowW1 += stepY1; rowW2 += stepY2;
        }

        return any;
    }

    /// <summary>
    /// Arête (p→q) top ou left, repère y-bas, aire positive (winding horaire à
    /// l'écran) : top = horizontale vers la droite ; left = montante (dy &lt; 0).
    /// </summary>
    private static bool IsTopLeftEdge((long X, long Y) p, (long X, long Y) q)
    {
        long dx = q.X - p.X, dy = q.Y - p.Y;
        return dy < 0 || (dy == 0 && dx > 0);
    }

    /// <summary>Équivalent CPU de l'atomicMax 64 bits du shader compute.</summary>
    private static void AtomicMax(ref ulong location, ulong value)
    {
        ulong current = Volatile.Read(ref location);
        while (value > current)
        {
            ulong previous = Interlocked.CompareExchange(ref location, value, current);
            if (previous == current)
                return;
            current = previous;
        }
    }
}
