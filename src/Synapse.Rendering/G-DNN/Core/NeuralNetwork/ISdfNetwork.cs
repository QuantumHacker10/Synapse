using System;
using System.Numerics;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Abstraction commune pour l'évaluation d'un SDF neuronal.
/// Permet à SurfaceEvaluator / RayMarcher d'accepter MicroMLP, DeepMicroMLP,
/// HashEncodedDeepMLP et QuantizedDeepMLP sans couplage fort.
/// </summary>
public interface ISdfNetwork
{
    /// <summary>Évalue la distance signée en un point.</summary>
    float Evaluate(Vector3 point);

    /// <summary>Évalue un lot de points.</summary>
    void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances);

    /// <summary>Gradient normalisé (normale de surface) via différences finies ou autodiff.</summary>
    Vector3 ComputeGradient(Vector3 point);

    /// <summary>Évalue la distance et le gradient en un seul appel.</summary>
    float EvaluateWithGradient(Vector3 point, out Vector3 gradient);
}
