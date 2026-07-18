using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Polygonization;

/// <summary>
/// Maillage polygonal extrait d'un SDF neuronal.
/// Les normales proviennent du gradient du réseau (pas du maillage),
/// ce qui préserve le détail neuronal même à basse résolution polygonale.
/// </summary>
public sealed class NeuralPolygonMesh
{
    public required Vector3[] Positions { get; init; }
    public required Vector3[] Normals { get; init; }
    public required int[] Indices { get; init; }

    public int VertexCount => Positions.Length;
    public int TriangleCount => Indices.Length / 3;

    /// <summary>Bornes englobantes du maillage.</summary>
    public AABB ComputeBounds()
    {
        if (Positions.Length == 0)
            return new AABB(Vector3.Zero, Vector3.Zero);

        Vector3 min = Positions[0], max = Positions[0];
        foreach (var p in Positions)
        {
            min = Vector3.Min(min, p);
            max = Vector3.Max(max, p);
        }
        return new AABB(min, max, fromCorners: true);
    }
}

/// <summary>Stratégie de placement des sommets dans les cellules actives.</summary>
public enum VertexPlacementMode
{
    /// <summary>Barycentre des croisements d'arêtes (lisse, rapide).</summary>
    SurfaceNets,

    /// <summary>
    /// Dual Contouring : minimisation d'une fonction d'erreur quadratique (QEF)
    /// sur les données de Hermite (croisements + gradients neuronaux).
    /// Reconstruit les arêtes vives et les coins.
    /// </summary>
    DualContouringQef
}

/// <summary>
/// Rapport mesurable d'une extraction — conforme à l'esprit de GDNNValidationProtocol :
/// aucune affirmation de gain sans métrique.
/// </summary>
public sealed class PolygonizationReport
{
    public int GridResolution { get; init; }

    /// <summary>Évaluations SDF réellement effectuées (grille + centres d'élagage).</summary>
    public int SdfEvaluations { get; init; }

    /// <summary>Évaluations qu'aurait coûté la grille dense — pour mesurer le gain épars.</summary>
    public int TheoreticalDenseEvaluations { get; init; }

    public int VertexCount { get; init; }
    public int TriangleCount { get; init; }
    public double ElapsedMs { get; init; }

    /// <summary>Déviation max |sdf(v)| aux sommets — borne d'erreur de surface.</summary>
    public float MaxSurfaceDeviation { get; init; }

    /// <summary>Erreur géométrique nominale du niveau (diagonale d'un voxel).</summary>
    public float GeometricError { get; init; }

    public override string ToString() =>
        $"Grid={GridResolution}³, Evals={SdfEvaluations}/{TheoreticalDenseEvaluations}, " +
        $"Verts={VertexCount}, Tris={TriangleCount}, " +
        $"MaxDev={MaxSurfaceDeviation:F5}, GeomErr={GeometricError:F5}, {ElapsedMs:F1}ms";
}

/// <summary>
/// Polygonisation neuronale : extraction de l'iso-surface d'un <see cref="ISdfNetwork"/>.
///
/// Deux stratégies d'échantillonnage :
/// - dense : grille complète (référence) ;
/// - éparse : élagage récursif par la borne de Lipschitz du SDF
///   (|sdf(centre)| &gt; demi-diagonale ⇒ aucune surface dans le nœud),
///   qui ne visite que les blocs traversés par la surface. Topologie identique
///   au dense, une fraction des évaluations réseau.
///
/// Deux placements de sommets : Surface Nets (barycentre) ou Dual Contouring QEF
/// (arêtes vives). La topologie (un sommet par cellule active, un quad par arête
/// traversée) est commune et garantit un maillage étanche.
/// </summary>
public sealed class NeuralPolygonizer
{
    private const int BlockCells = 2;

    /// <summary>Itérations de relaxation vers l'iso-surface (mode SurfaceNets uniquement :
    /// en mode QEF, la projection le long du gradient éroderait les arêtes vives).</summary>
    public int SurfaceProjectionIterations { get; set; } = 2;

    /// <summary>Taille de lot pour l'échantillonnage du réseau.</summary>
    public int BatchSize { get; set; } = 4096;

    /// <summary>Active l'élagage épars par borne de Lipschitz.</summary>
    public bool UseSparseTraversal { get; set; } = true;

    /// <summary>
    /// Marge de sécurité sur la borne de Lipschitz : un SDF neuronal n'est
    /// qu'approximativement 1-Lipschitz, on élargit donc les nœuds conservés.
    /// </summary>
    public float PruneSafetyFactor { get; set; } = 1.5f;

    public VertexPlacementMode VertexPlacement { get; set; } = VertexPlacementMode.SurfaceNets;

    /// <summary>Régularisation QEF (tire la solution vers le barycentre des croisements).</summary>
    public float QefRegularization { get; set; } = 0.05f;

    /// <summary>
    /// Parallélise les passes par sommet (projection, normales) et
    /// l'échantillonnage dense. L'évaluation des réseaux est pure (état local
    /// en pile), donc sûre en concurrence.
    /// </summary>
    public bool EnableParallelPasses { get; set; } = true;

    public PolygonizationReport LastReport { get; private set; } = new();

    /// <summary>
    /// Extrait l'iso-surface (sdf = 0) sur une grille <paramref name="resolution"/>³
    /// dans <paramref name="bounds"/>.
    /// </summary>
    public NeuralPolygonMesh Extract(ISdfNetwork network, AABB bounds, int resolution)
    {
        ArgumentNullException.ThrowIfNull(network);
        if (resolution < 2)
            throw new ArgumentOutOfRangeException(nameof(resolution), "Resolution must be at least 2.");

        var sw = Stopwatch.StartNew();

        Vector3 min = bounds.Min;
        Vector3 cellSize = bounds.Size / resolution;
        int corners = resolution + 1;

        // 1) Cellules actives (traversées par la surface) + leurs 8 distances de coin.
        var cells = new List<(int X, int Y, int Z)>();
        var cellDistances = new List<float>(); // 8 par cellule, aplaties
        int evalCount = UseSparseTraversal
            ? CollectActiveCellsSparse(network, min, cellSize, resolution, cells, cellDistances)
            : CollectActiveCellsDense(network, min, cellSize, resolution, cells, cellDistances);

        // 2) Un sommet par cellule active.
        var cellVertex = new Dictionary<long, int>(cells.Count);
        var positions = new List<Vector3>(cells.Count);
        for (int i = 0; i < cells.Count; i++)
        {
            var (cx, cy, cz) = cells[i];
            Vector3 cellMin = min + new Vector3(cx * cellSize.X, cy * cellSize.Y, cz * cellSize.Z);
            Vector3 vertex = PlaceVertex(
                network, cellMin, cellSize,
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(cellDistances).Slice(i * 8, 8));

            cellVertex[CellKey(cx, cy, cz, resolution)] = positions.Count;
            positions.Add(vertex);
        }

        // 3) Relaxation vers l'iso-surface (SurfaceNets uniquement). Chaque sommet
        //    est indépendant : parallèle sur les sommets (le réseau est immuable
        //    en évaluation).
        if (VertexPlacement == VertexPlacementMode.SurfaceNets)
        {
            for (int iter = 0; iter < SurfaceProjectionIterations; iter++)
            {
                ParallelForVertices(positions.Count, i =>
                {
                    float d = network.EvaluateWithGradient(positions[i], out Vector3 g);
                    if (g.LengthSquared() > 1e-12f)
                        positions[i] -= d * Vector3.Normalize(g);
                });
            }
        }

        // 4) Quads : chaque arête min-corner traversée d'une cellule active connecte
        //    les 4 cellules qui la partagent (dont la cellule courante est le max).
        var indices = new List<int>();
        EmitQuads(cells, cellDistances, cellVertex, resolution, indices);

        // 5) Normales depuis le gradient neuronal + déviation mesurée (parallèle).
        var normals = new Vector3[positions.Count];
        var deviations = new float[positions.Count];
        ParallelForVertices(positions.Count, i =>
        {
            float d = network.EvaluateWithGradient(positions[i], out Vector3 g);
            normals[i] = g.LengthSquared() > 1e-12f ? Vector3.Normalize(g) : Vector3.UnitY;
            deviations[i] = MathF.Abs(d);
        });
        float maxDeviation = 0f;
        foreach (float d in deviations)
            maxDeviation = MathF.Max(maxDeviation, d);

        sw.Stop();
        LastReport = new PolygonizationReport
        {
            GridResolution = resolution,
            SdfEvaluations = evalCount,
            TheoreticalDenseEvaluations = corners * corners * corners,
            VertexCount = positions.Count,
            TriangleCount = indices.Count / 3,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            MaxSurfaceDeviation = maxDeviation,
            GeometricError = cellSize.Length()
        };

        return new NeuralPolygonMesh
        {
            Positions = positions.ToArray(),
            Normals = normals,
            Indices = indices.ToArray()
        };
    }

    // ------------------------------------------------------------------
    // Échantillonnage dense (référence)
    // ------------------------------------------------------------------

    private int CollectActiveCellsDense(
        ISdfNetwork network, Vector3 min, Vector3 cellSize, int resolution,
        List<(int, int, int)> cells, List<float> cellDistances)
    {
        int corners = resolution + 1;
        int total = corners * corners * corners;
        var samples = new float[total];

        // Lots disjoints → parallélisables sans synchronisation.
        int batchSize = Math.Min(BatchSize, total);
        int batchCount = (total + batchSize - 1) / batchSize;
        ParallelFor(batchCount, EnableParallelPasses, batch =>
        {
            int start = batch * batchSize;
            int count = Math.Min(batchSize, total - start);
            var batchPoints = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                int linear = start + i;
                int x = linear % corners;
                int y = linear / corners % corners;
                int z = linear / (corners * corners);
                batchPoints[i] = min + new Vector3(x * cellSize.X, y * cellSize.Y, z * cellSize.Z);
            }
            network.EvaluateBatch(batchPoints, samples.AsSpan(start, count));
        });

        Span<float> d = stackalloc float[8];
        for (int z = 0; z < resolution; z++)
        for (int y = 0; y < resolution; y++)
        for (int x = 0; x < resolution; x++)
        {
            bool anyNeg = false, anyPos = false;
            for (int i = 0; i < 8; i++)
            {
                d[i] = samples[
                    (x + (i & 1)) + corners * ((y + ((i >> 1) & 1)) + corners * (z + ((i >> 2) & 1)))];
                if (d[i] < 0) anyNeg = true; else anyPos = true;
            }
            if (!anyNeg || !anyPos)
                continue;

            cells.Add((x, y, z));
            for (int i = 0; i < 8; i++)
                cellDistances.Add(d[i]);
        }

        return total;
    }

    // ------------------------------------------------------------------
    // Échantillonnage épars (élagage Lipschitz)
    // ------------------------------------------------------------------

    private int CollectActiveCellsSparse(
        ISdfNetwork network, Vector3 min, Vector3 cellSize, int resolution,
        List<(int, int, int)> cells, List<float> cellDistances)
    {
        var cornerCache = new Dictionary<long, float>();
        var scratchPoints = new List<Vector3>();
        var scratchKeys = new List<long>();
        int evalCount = 0;

        RecurseNode(0, 0, 0, resolution, resolution, resolution);
        return evalCount;

        void RecurseNode(int x0, int y0, int z0, int sx, int sy, int sz)
        {
            Vector3 nodeMin = min + new Vector3(x0 * cellSize.X, y0 * cellSize.Y, z0 * cellSize.Z);
            Vector3 nodeHalf = new Vector3(sx * cellSize.X, sy * cellSize.Y, sz * cellSize.Z) * 0.5f;
            Vector3 center = nodeMin + nodeHalf;

            float d = network.Evaluate(center);
            evalCount++;

            // Borne de Lipschitz : si |sdf| dépasse la demi-diagonale (avec marge),
            // la surface ne peut pas traverser ce nœud.
            if (MathF.Abs(d) > nodeHalf.Length() * PruneSafetyFactor + 1e-6f)
                return;

            if (Math.Max(sx, Math.Max(sy, sz)) <= BlockCells)
            {
                ProcessBlock(x0, y0, z0, sx, sy, sz);
                return;
            }

            int hx = Math.Max(1, sx / 2), hy = Math.Max(1, sy / 2), hz = Math.Max(1, sz / 2);
            for (int oz = 0; oz < 2; oz++)
            for (int oy = 0; oy < 2; oy++)
            for (int ox = 0; ox < 2; ox++)
            {
                int cx0 = x0 + ox * hx, cy0 = y0 + oy * hy, cz0 = z0 + oz * hz;
                int csx = ox == 0 ? hx : sx - hx;
                int csy = oy == 0 ? hy : sy - hy;
                int csz = oz == 0 ? hz : sz - hz;
                if (csx > 0 && csy > 0 && csz > 0)
                    RecurseNode(cx0, cy0, cz0, csx, csy, csz);
            }
        }

        void ProcessBlock(int x0, int y0, int z0, int sx, int sy, int sz)
        {
            // Échantillonner les coins manquants du treillis (sx+1)(sy+1)(sz+1) en lot.
            scratchPoints.Clear();
            scratchKeys.Clear();
            for (int z = z0; z <= z0 + sz; z++)
            for (int y = y0; y <= y0 + sy; y++)
            for (int x = x0; x <= x0 + sx; x++)
            {
                long key = CornerKey(x, y, z, resolution);
                if (cornerCache.ContainsKey(key))
                    continue;
                cornerCache[key] = float.NaN; // réservé, évite les doublons dans ce lot
                scratchKeys.Add(key);
                scratchPoints.Add(min + new Vector3(x * cellSize.X, y * cellSize.Y, z * cellSize.Z));
            }

            if (scratchPoints.Count > 0)
            {
                var pts = scratchPoints.ToArray();
                var dist = new float[pts.Length];
                network.EvaluateBatch(pts, dist);
                evalCount += pts.Length;
                for (int i = 0; i < scratchKeys.Count; i++)
                    cornerCache[scratchKeys[i]] = dist[i];
            }

            Span<float> d = stackalloc float[8];
            for (int z = z0; z < z0 + sz; z++)
            for (int y = y0; y < y0 + sy; y++)
            for (int x = x0; x < x0 + sx; x++)
            {
                bool anyNeg = false, anyPos = false;
                for (int i = 0; i < 8; i++)
                {
                    d[i] = cornerCache[CornerKey(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1), resolution)];
                    if (d[i] < 0) anyNeg = true; else anyPos = true;
                }
                if (!anyNeg || !anyPos)
                    continue;

                cells.Add((x, y, z));
                for (int i = 0; i < 8; i++)
                    cellDistances.Add(d[i]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Placement des sommets
    // ------------------------------------------------------------------

    private Vector3 PlaceVertex(
        ISdfNetwork network, Vector3 cellMin, Vector3 cellSize, ReadOnlySpan<float> d)
    {
        // Paires de coins formant les 12 arêtes du cube (indexation bit 0=x, 1=y, 2=z).
        ReadOnlySpan<byte> edges = [
            0, 1, 2, 3, 4, 5, 6, 7,   // arêtes X
            0, 2, 1, 3, 4, 6, 5, 7,   // arêtes Y
            0, 4, 1, 5, 2, 6, 3, 7];  // arêtes Z

        Span<Vector3> crossings = stackalloc Vector3[12];
        int crossingCount = 0;
        Vector3 sum = Vector3.Zero;

        for (int e = 0; e < edges.Length; e += 2)
        {
            float a = d[edges[e]];
            float b = d[edges[e + 1]];
            if (a < 0 == b < 0)
                continue;

            float t = a / (a - b);
            Vector3 local = Vector3.Lerp(CornerOffset(edges[e]), CornerOffset(edges[e + 1]), t);
            Vector3 world = cellMin + local * cellSize;
            crossings[crossingCount++] = world;
            sum += world;
        }

        if (crossingCount == 0)
            return cellMin + cellSize * 0.5f; // ne devrait pas arriver (cellule active)

        Vector3 massPoint = sum / crossingCount;
        if (VertexPlacement == VertexPlacementMode.SurfaceNets)
            return massPoint;

        return SolveQef(network, crossings[..crossingCount], massPoint, cellMin, cellMin + cellSize);
    }

    /// <summary>
    /// Minimise Σ (nᵢ · (x − pᵢ))² sur les données de Hermite, régularisée vers le
    /// barycentre, solution bornée à la cellule (Dual Contouring classique).
    /// </summary>
    private Vector3 SolveQef(
        ISdfNetwork network, ReadOnlySpan<Vector3> crossings, Vector3 massPoint,
        Vector3 cellMin, Vector3 cellMax)
    {
        float a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
        Vector3 atb = Vector3.Zero;

        foreach (var p in crossings)
        {
            Vector3 n = network.ComputeGradient(p);
            if (n.LengthSquared() < 1e-12f)
                continue;
            n = Vector3.Normalize(n);

            a00 += n.X * n.X; a01 += n.X * n.Y; a02 += n.X * n.Z;
            a11 += n.Y * n.Y; a12 += n.Y * n.Z; a22 += n.Z * n.Z;
            atb += n * Vector3.Dot(n, p);
        }

        float reg = QefRegularization;
        a00 += reg; a11 += reg; a22 += reg;
        atb += massPoint * reg;

        // Inversion 3x3 symétrique par cofacteurs.
        float c00 = a11 * a22 - a12 * a12;
        float c01 = a02 * a12 - a01 * a22;
        float c02 = a01 * a12 - a02 * a11;
        float det = a00 * c00 + a01 * c01 + a02 * c02;
        if (MathF.Abs(det) < 1e-12f)
            return massPoint;

        float inv = 1f / det;
        float c11 = a00 * a22 - a02 * a02;
        float c12 = a01 * a02 - a00 * a12;
        float c22 = a00 * a11 - a01 * a01;

        var solution = new Vector3(
            (c00 * atb.X + c01 * atb.Y + c02 * atb.Z) * inv,
            (c01 * atb.X + c11 * atb.Y + c12 * atb.Z) * inv,
            (c02 * atb.X + c12 * atb.Y + c22 * atb.Z) * inv);

        return Vector3.Clamp(solution, cellMin, cellMax);
    }

    // ------------------------------------------------------------------
    // Topologie (commune aux deux stratégies)
    // ------------------------------------------------------------------

    /// <summary>
    /// Pour chaque cellule active, examine les 3 arêtes partant de son coin minimal.
    /// Une arête traversée est partagée par 4 cellules dont la cellule courante est
    /// le maximum sur les deux tangentes — chaque arête n'est donc émise qu'une fois,
    /// et les 4 cellules sont nécessairement actives (elles contiennent l'arête).
    /// </summary>
    private static void EmitQuads(
        List<(int X, int Y, int Z)> cells, List<float> cellDistances,
        Dictionary<long, int> cellVertex, int resolution, List<int> indices)
    {
        // Triplets droitiers (axe d'arête e, tangentes u et v) : e = u × v.
        // Le coin opposé sur l'arête correspond au bit de l'axe (X=1, Y=2, Z=4).
        ReadOnlySpan<(int bit, int ux, int uy, int uz, int vx, int vy, int vz)> axes = [
            (1, 0, 1, 0, 0, 0, 1),
            (2, 0, 0, 1, 1, 0, 0),
            (4, 1, 0, 0, 0, 1, 0)];

        for (int i = 0; i < cells.Count; i++)
        {
            var (cx, cy, cz) = cells[i];
            float d0 = cellDistances[i * 8];

            foreach (var (bit, ux, uy, uz, vx, vy, vz) in axes)
            {
                float dEnd = cellDistances[i * 8 + bit];
                if (d0 < 0 == dEnd < 0)
                    continue;

                // Bord de grille : pas de cellule de l'autre côté.
                if (cx - ux - vx < 0 || cy - uy - vy < 0 || cz - uz - vz < 0)
                    continue;

                if (!cellVertex.TryGetValue(CellKey(cx - ux - vx, cy - uy - vy, cz - uz - vz, resolution), out int q0) ||
                    !cellVertex.TryGetValue(CellKey(cx - vx, cy - vy, cz - vz, resolution), out int q1) ||
                    !cellVertex.TryGetValue(CellKey(cx, cy, cz, resolution), out int q2) ||
                    !cellVertex.TryGetValue(CellKey(cx - ux, cy - uy, cz - uz, resolution), out int q3))
                    continue;

                // Winding par défaut → normale vers +e ; l'extérieur (sdf > 0)
                // est du côté +e quand d0 < 0.
                if (d0 < 0)
                {
                    indices.AddRange([q0, q1, q2, q0, q2, q3]);
                }
                else
                {
                    indices.AddRange([q0, q2, q1, q0, q3, q2]);
                }
            }
        }
    }

    private void ParallelForVertices(int count, Action<int> body)
        => ParallelFor(count, EnableParallelPasses, body);

    private static void ParallelFor(int count, bool parallel, Action<int> body)
    {
        if (parallel && count > 64)
        {
            System.Threading.Tasks.Parallel.For(0, count, body);
        }
        else
        {
            for (int i = 0; i < count; i++)
                body(i);
        }
    }

    private static long CornerKey(int x, int y, int z, int resolution)
    {
        long corners = resolution + 1;
        return x + corners * (y + corners * z);
    }

    private static long CellKey(int x, int y, int z, int resolution)
        => x + (long)resolution * (y + (long)resolution * z);

    private static Vector3 CornerOffset(int corner)
        => new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
}
