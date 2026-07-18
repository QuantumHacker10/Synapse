using System;
using System.Diagnostics;
using System.Numerics;
using GDNN.Evaluation;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Campagne d'entraînement offline d'un HashEncodedDeepMLP contre un maillage
/// de référence haute résolution (distances signées).
/// </summary>
public sealed class OfflineHashMeshTrainer
{
    public sealed class TrainingReport
    {
        public int Epochs { get; init; }
        public int SampleCount { get; init; }
        public int MeshTriangles { get; init; }
        public int MeshVertices { get; init; }
        public float LossBefore { get; init; }
        public float LossAfter { get; init; }
        public float HausdorffError { get; init; }
        public float RmsError { get; init; }
        public int MemoryBytes { get; init; }
        public double ElapsedMs { get; init; }

        public bool Improved => LossAfter < LossBefore;

        public override string ToString() =>
            $"Loss {LossBefore:F4}→{LossAfter:F4}, Hausdorff={HausdorffError:F4}, " +
            $"RMS={RmsError:F4}, Tris={MeshTriangles}, Mem={MemoryBytes}B, {ElapsedMs:F0}ms";
    }

    public float LearningRate { get; set; } = 5e-3f;
    public float HashLearningRate { get; set; } = 5e-2f;
    public float SurfaceBand { get; set; } = 0.15f;
    public float InteriorFraction { get; set; } = 0.25f;

    /// <summary>
    /// Entraîne un réseau sur une sphère maillée subdivisée (asset procédural HR).
    /// </summary>
    public TrainingReport TrainOnSubdividedSphere(
        HashEncodedDeepMLP network,
        int subdivisions = 3,
        float radius = 0.5f,
        int sampleCount = 4096,
        int epochs = 40,
        Random? random = null)
    {
        var mesh = ReferenceMeshSdf.CreateUnitSphereIcosahedron(radius, subdivisions);
        return TrainOnMesh(network, mesh, sampleCount, epochs, random,
            // Pour une sphère, la GT analytique est plus précise que le raycast maillage
            p => ReferenceMeshSdf.AnalyticSphereSdf(p, radius));
    }

    /// <summary>
    /// Entraîne contre les distances signées d'un maillage arbitraire.
    /// </summary>
    public TrainingReport TrainOnMesh(
        HashEncodedDeepMLP network,
        ReferenceMeshSdf mesh,
        int sampleCount = 4096,
        int epochs = 40,
        Random? random = null,
        Func<Vector3, float>? overrideTarget = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(mesh);
        random ??= new Random(42);

        var sw = Stopwatch.StartNew();
        var (points, targets) = SampleTrainingSet(mesh, sampleCount, random, overrideTarget);

        float lossBefore = MeanSquaredError(network, points, targets);

        var trainer = new HashEncodedDeepMLPTrainer(network)
        {
            LearningRate = LearningRate,
            HashLearningRate = HashLearningRate
        };
        trainer.Fit(points, targets, epochs, random);

        float lossAfter = MeanSquaredError(network, points, targets);

        // Validation géométrique sur un jeu hold-out
        var valPoints = GDNNValidationProtocol.SamplePointCloud(Math.Min(512, sampleCount / 4), new Random(99));
        float hausdorff = 0f, rms = 0f;
        if (overrideTarget != null)
        {
            for (int i = 0; i < valPoints.Length; i++)
            {
                float err = MathF.Abs(network.Evaluate(valPoints[i]) - overrideTarget(valPoints[i]));
                hausdorff = MathF.Max(hausdorff, err);
                rms += err * err;
            }
            rms = MathF.Sqrt(rms / valPoints.Length);
        }
        else
        {
            hausdorff = mesh.ComputeHausdorffError(network, valPoints);
            // Compare signed network to |mesh| is imperfect — use signed mesh distances
            float sumSq = 0f;
            for (int i = 0; i < valPoints.Length; i++)
            {
                float err = MathF.Abs(network.Evaluate(valPoints[i]) - mesh.SignedDistance(valPoints[i]));
                hausdorff = MathF.Max(hausdorff, err);
                sumSq += err * err;
            }
            rms = MathF.Sqrt(sumSq / valPoints.Length);
        }

        sw.Stop();
        return new TrainingReport
        {
            Epochs = epochs,
            SampleCount = sampleCount,
            MeshTriangles = mesh.TriangleCount,
            MeshVertices = mesh.VertexCount,
            LossBefore = lossBefore,
            LossAfter = lossAfter,
            HausdorffError = hausdorff,
            RmsError = rms,
            MemoryBytes = network.GetMemoryFootprintBytes(),
            ElapsedMs = sw.Elapsed.TotalMilliseconds
        };
    }

    /// <summary>
    /// Échantillonne points de surface (bande) + volume pour un SDF bien conditionné.
    /// </summary>
    public (Vector3[] Points, float[] Targets) SampleTrainingSet(
        ReferenceMeshSdf mesh,
        int sampleCount,
        Random random,
        Func<Vector3, float>? overrideTarget = null)
    {
        var points = new Vector3[sampleCount];
        var targets = new float[sampleCount];
        int interiorCount = (int)(sampleCount * InteriorFraction);

        for (int i = 0; i < sampleCount; i++)
        {
            Vector3 p;
            if (i < interiorCount)
            {
                // Volume uniforme dans [-1,1]^3
                p = new Vector3(
                    (float)(random.NextDouble() * 2 - 1),
                    (float)(random.NextDouble() * 2 - 1),
                    (float)(random.NextDouble() * 2 - 1));
            }
            else
            {
                // Bande autour de la surface : point aléatoire + offset normal approximatif
                p = new Vector3(
                    (float)(random.NextDouble() * 2 - 1),
                    (float)(random.NextDouble() * 2 - 1),
                    (float)(random.NextDouble() * 2 - 1));
                float ud = mesh.UnsignedDistance(p);
                // Rejeter trop loin ; sinon garder avec petite perturbation
                if (ud > SurfaceBand)
                {
                    p = Vector3.Normalize(p) * (0.5f + (float)(random.NextDouble() * 2 - 1) * SurfaceBand);
                }
            }

            points[i] = p;
            targets[i] = overrideTarget?.Invoke(p) ?? mesh.SignedDistance(p);
        }

        return (points, targets);
    }

    private static float MeanSquaredError(HashEncodedDeepMLP network, Vector3[] points, float[] targets)
    {
        float sum = 0f;
        for (int i = 0; i < points.Length; i++)
        {
            float d = network.Evaluate(points[i]) - targets[i];
            sum += d * d;
        }
        return sum / points.Length;
    }
}
