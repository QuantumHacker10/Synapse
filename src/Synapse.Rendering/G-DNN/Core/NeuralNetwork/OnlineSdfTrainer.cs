using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using GDNN.Core.DataStructures;

namespace GDNN.Core.NeuralNetwork;

/// <summary>Opération CSG appliquée par un édit de sculpt.</summary>
public enum SdfEditOperation
{
    /// <summary>Ajout de matière : min(sdf, brosse).</summary>
    Union,

    /// <summary>Retrait de matière : max(sdf, −brosse).</summary>
    Subtract
}

/// <summary>Résultat d'une tranche d'entraînement budgétée.</summary>
public sealed class TrainingSliceReport
{
    public int StepsExecuted { get; init; }
    public float MeanLoss { get; init; }
    public double ElapsedMs { get; init; }
    public int PendingRemaining { get; init; }
    public bool ConsumedEditSamples { get; init; }
    public long GeometryVersion { get; init; }

    public override string ToString() =>
        $"Steps={StepsExecuted}, Loss={MeanLoss:F6}, {ElapsedMs:F2}ms, " +
        $"Pending={PendingRemaining}, v{GeometryVersion}";
}

/// <summary>
/// Entraînement en temps réel d'un SDF neuronal : micro-tranches de SGD
/// budgétées en millisecondes, à appeler chaque frame.
///
/// Deux mécanismes contre l'oubli catastrophique pendant l'édition live :
/// - un replay buffer circulaire qui ré-entraîne sur les échantillons passés ;
/// - des ancres d'auto-distillation : à chaque édit, des points hors de la zone
///   éditée sont ré-étiquetés avec la sortie actuelle du réseau, figeant la
///   géométrie que l'édit ne doit pas toucher.
///
/// La version de géométrie s'incrémente quand des échantillons d'édit sont
/// consommés ; les bornes sales accumulées permettent une invalidation ciblée.
/// </summary>
public sealed class OnlineSdfTrainer
{
    private readonly HashEncodedDeepMLP _network;
    private readonly HashEncodedDeepMLPTrainer _trainer;
    private readonly Queue<(Vector3 Point, float Target)> _pending = new();
    private readonly (Vector3 Point, float Target)[] _replay;
    private readonly Random _random;
    private int _replayCount;
    private int _replayWrite;
    private AABB? _dirtyBounds;

    /// <summary>Domaine spatial du SDF (échantillonnage des ancres).</summary>
    public AABB Domain { get; }

    /// <summary>S'incrémente à chaque tranche ayant consommé des échantillons d'édit.</summary>
    public long GeometryVersion { get; private set; }

    public int PendingCount => _pending.Count;
    public int ReplayCount => _replayCount;

    /// <summary>Fraction des pas consacrée au replay quand des édits sont en attente.</summary>
    public float ReplayFraction { get; set; } = 0.35f;

    public float LearningRate
    {
        get => _trainer.LearningRate;
        set => _trainer.LearningRate = value;
    }

    public float HashLearningRate
    {
        get => _trainer.HashLearningRate;
        set => _trainer.HashLearningRate = value;
    }

    public OnlineSdfTrainer(
        HashEncodedDeepMLP network, AABB domain, int replayCapacity = 8192, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(network);
        if (replayCapacity < 1)
            throw new ArgumentOutOfRangeException(nameof(replayCapacity));

        _network = network;
        _trainer = new HashEncodedDeepMLPTrainer(network)
        {
            LearningRate = 1e-2f,
            HashLearningRate = 1e-1f
        };
        Domain = domain;
        _replay = new (Vector3, float)[replayCapacity];
        _random = new Random(seed);
    }

    /// <summary>Empile des échantillons (point, distance cible) à apprendre.</summary>
    public void EnqueueSamples(
        ReadOnlySpan<Vector3> points, ReadOnlySpan<float> targets, AABB dirtyBounds)
    {
        if (points.Length != targets.Length)
            throw new ArgumentException("Points and targets must have the same length.");

        for (int i = 0; i < points.Length; i++)
            _pending.Enqueue((points[i], targets[i]));

        _dirtyBounds = _dirtyBounds is { } existing ? AABB.Merge(existing, dirtyBounds) : dirtyBounds;
    }

    /// <summary>
    /// Édit de sculpt sphérique : les cibles sont la combinaison CSG de la sortie
    /// actuelle du réseau et de la brosse, plus des ancres d'auto-distillation
    /// hors de la zone pour préserver le reste de la géométrie.
    /// </summary>
    public void ApplySphereEdit(
        Vector3 center, float radius, SdfEditOperation operation,
        int editSampleCount = 512, int anchorSampleCount = 256)
    {
        if (radius <= 0f)
            throw new ArgumentOutOfRangeException(nameof(radius));

        var brushBounds = new AABB(center, new Vector3(radius * 1.5f));
        int total = editSampleCount + anchorSampleCount;
        var points = new Vector3[total];
        var targets = new float[total];

        for (int i = 0; i < editSampleCount; i++)
        {
            // Échantillonnage en importance : 70 % dans la boule de la brosse
            // (là où le champ change), 30 % dans la coquille de transition —
            // un tirage uniforme dans la boîte diluerait le signal d'édition.
            float shellFactor = i % 10 < 7 ? 1f : 1.5f;
            float u = _random.NextSingle();
            Vector3 p = center + RandomUnitVector() * (radius * shellFactor * MathF.Cbrt(u));
            p = Vector3.Clamp(p, Domain.Min, Domain.Max);

            float current = _network.Evaluate(p);
            float brush = (p - center).Length() - radius;
            targets[i] = operation == SdfEditOperation.Union
                ? MathF.Min(current, brush)
                : MathF.Max(current, -brush);
            points[i] = p;
        }

        // Ancres : la sortie actuelle devient la cible — fige la géométrie distante.
        // Impératif : hors de la zone éditée, sinon elles y figeraient l'ancienne
        // géométrie et combattraient l'édit.
        for (int i = 0; i < anchorSampleCount; i++)
        {
            Vector3 p;
            int attempts = 0;
            do
            {
                p = RandomPointIn(Domain);
            } while (Contains(brushBounds, p) && ++attempts < 16);

            points[editSampleCount + i] = p;
            targets[editSampleCount + i] = _network.Evaluate(p);
        }

        EnqueueSamples(points, targets, brushBounds);
    }

    /// <summary>
    /// Exécute des pas de SGD jusqu'à épuisement du budget temps (ou des données).
    /// Mélange échantillons frais et replay selon <see cref="ReplayFraction"/>.
    /// </summary>
    public TrainingSliceReport TrainSlice(double budgetMs, int maxSteps = int.MaxValue)
    {
        var sw = Stopwatch.StartNew();
        int steps = 0;
        float lossSum = 0f;
        bool consumedEdits = false;

        while (steps < maxSteps && sw.Elapsed.TotalMilliseconds < budgetMs)
        {
            (Vector3 Point, float Target) sample;
            bool takeFresh = _pending.Count > 0 &&
                (_replayCount == 0 || _random.NextSingle() >= ReplayFraction);

            if (takeFresh)
            {
                sample = _pending.Dequeue();
                AddToReplay(sample);
                consumedEdits = true;
            }
            else if (_replayCount > 0)
            {
                sample = _replay[_random.Next(_replayCount)];
            }
            else
            {
                break; // rien à apprendre
            }

            lossSum += _trainer.TrainStep(sample.Point, sample.Target);
            steps++;
        }

        if (consumedEdits)
            GeometryVersion++;

        sw.Stop();
        return new TrainingSliceReport
        {
            StepsExecuted = steps,
            MeanLoss = steps > 0 ? lossSum / steps : 0f,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            PendingRemaining = _pending.Count,
            ConsumedEditSamples = consumedEdits,
            GeometryVersion = GeometryVersion
        };
    }

    /// <summary>Récupère et efface les bornes sales accumulées depuis le dernier appel.</summary>
    public bool TryConsumeDirtyBounds(out AABB dirty)
    {
        if (_dirtyBounds is { } bounds)
        {
            dirty = bounds;
            _dirtyBounds = null;
            return true;
        }
        dirty = default;
        return false;
    }

    private void AddToReplay((Vector3, float) sample)
    {
        _replay[_replayWrite] = sample;
        _replayWrite = (_replayWrite + 1) % _replay.Length;
        _replayCount = Math.Min(_replayCount + 1, _replay.Length);
    }

    private Vector3 RandomUnitVector()
    {
        Vector3 v;
        do
        {
            v = new Vector3(
                _random.NextSingle() * 2f - 1f,
                _random.NextSingle() * 2f - 1f,
                _random.NextSingle() * 2f - 1f);
        } while (v.LengthSquared() is < 1e-4f or > 1f);
        return Vector3.Normalize(v);
    }

    private static bool Contains(AABB bounds, Vector3 p)
        => p.X >= bounds.Min.X && p.X <= bounds.Max.X &&
           p.Y >= bounds.Min.Y && p.Y <= bounds.Max.Y &&
           p.Z >= bounds.Min.Z && p.Z <= bounds.Max.Z;

    private Vector3 RandomPointIn(AABB bounds)
    {
        Vector3 min = bounds.Min, size = bounds.Size;
        return min + new Vector3(
            _random.NextSingle() * size.X,
            _random.NextSingle() * size.Y,
            _random.NextSingle() * size.Z);
    }
}
