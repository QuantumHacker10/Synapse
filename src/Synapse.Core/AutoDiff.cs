// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

// SECTION 5: DIFFERENTIATION AUTOMATIQUE FORWARD-MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Scalaire differentiable : valeur + derivee (forward-mode AD).
/// Chaque operation arithmetique propage automatiquement les derivees.
/// Utilise pour l'optimisation de parametres physiques et le calcul de gradient.
///
/// Exemple : si x = DiffScalar(2.0, 1.0) (valeur=2, dx/dx=1)
/// alors x*x = DiffScalar(4.0, 4.0) (valeur=4, d(x^2)/dx=2x=4).
///
/// MEMORY LAYOUT: 16 bytes (2 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
[DebuggerDisplay("d={Value:F6} (d/dx={Derivative:F6})")]
public struct DiffScalar : IEquatable<DiffScalar>
{
    /// <summary>Valeur de la fonction en ce point.</summary>
    [FieldOffset(0)] public double Value;
    /// <summary>Derivee par rapport a la variable independante.</summary>
    [FieldOffset(8)] public double Derivative;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public DiffScalar(double value, double derivative) { Value = value; Derivative = derivative; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public DiffScalar(double value) { Value = value; Derivative = 0; }

    public static readonly DiffScalar Zero = new(0, 0);
    public static readonly DiffScalar One = new(1, 0);
    public static DiffScalar Variable(double value) => new(value, 1.0);

    // Addition : d(a+b)/dx = da/dx + db/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(DiffScalar a, DiffScalar b) => new(a.Value + b.Value, a.Derivative + b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(DiffScalar a, double b) => new(a.Value + b, a.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(double a, DiffScalar b) => new(a + b.Value, b.Derivative);
    // Soustraction
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a, DiffScalar b) => new(a.Value - b.Value, a.Derivative - b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a, double b) => new(a.Value - b, a.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(double a, DiffScalar b) => new(a - b.Value, -b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a) => new(-a.Value, -a.Derivative);
    // Multiplication : d(a*b)/dx = a'*b + a*b'
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(DiffScalar a, DiffScalar b) => new(a.Value * b.Value, a.Derivative * b.Value + a.Value * b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(DiffScalar a, double b) => new(a.Value * b, a.Derivative * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(double a, DiffScalar b) => new(a * b.Value, a * b.Derivative);
    // Division : d(a/b)/dx = (a'*b - a*b') / b^2
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(DiffScalar a, DiffScalar b) { double q = a.Value / b.Value; return new(q, (a.Derivative * b.Value - a.Value * b.Derivative) / (b.Value * b.Value)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(DiffScalar a, double b) => new(a.Value / b, a.Derivative / b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(double a, DiffScalar b) { double q = a / b.Value; return new(q, -a * b.Derivative / (b.Value * b.Value)); }

    // Puissance : d(x^n)/dx = n*x^(n-1)*dx/dx
    public static DiffScalar Pow(DiffScalar x, double n) => new(Math.Pow(x.Value, n), n * Math.Pow(x.Value, n - 1) * x.Derivative);
    // Racine carree : d(sqrt(x))/dx = 1/(2*sqrt(x)) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sqrt(DiffScalar x) { double s = Math.Sqrt(x.Value); return new(s, x.Derivative / (2 * s)); }
    // Racine nieme : d(x^(1/n))/dx = (1/n)*x^(1/n-1)*dx/dx
    public static DiffScalar NthRoot(DiffScalar x, double n) { double r = Math.Pow(x.Value, 1.0 / n); return new(r, (1.0 / n) * Math.Pow(x.Value, 1.0 / n - 1) * x.Derivative); }
    // Exponentielle : d(e^x)/dx = e^x * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Exp(DiffScalar x) { double e = Math.Exp(x.Value); return new(e, e * x.Derivative); }
    // Logarithme : d(ln(x))/dx = (1/x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log(DiffScalar x) => new(Math.Log(x.Value), x.Derivative / x.Value);
    // Logarithme base 10 : d(log10(x))/dx = 1/(x*ln(10)) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log10(DiffScalar x) => new(Math.Log10(x.Value), x.Derivative / (x.Value * Math.Log(10)));
    // Logarithme base 2
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log2(DiffScalar x) => new(Math.Log2(x.Value), x.Derivative / (x.Value * Math.Log(2)));

    // Trigonometrie : d(sin(x))/dx = cos(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sin(DiffScalar x) => new(Math.Sin(x.Value), Math.Cos(x.Value) * x.Derivative);
    // d(cos(x))/dx = -sin(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Cos(DiffScalar x) => new(Math.Cos(x.Value), -Math.Sin(x.Value) * x.Derivative);
    // d(tan(x))/dx = sec^2(x) * dx/dx = 1/cos^2(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Tan(DiffScalar x) { double c = Math.Cos(x.Value); return new(Math.Tan(x.Value), x.Derivative / (c * c)); }
    // Inverse trigonometrie : d(asin(x))/dx = 1/sqrt(1-x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Asin(DiffScalar x) => new(Math.Asin(x.Value), x.Derivative / Math.Sqrt(1 - x.Value * x.Value));
    // d(acos(x))/dx = -1/sqrt(1-x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Acos(DiffScalar x) => new(Math.Acos(x.Value), -x.Derivative / Math.Sqrt(1 - x.Value * x.Value));
    // d(atan(x))/dx = 1/(1+x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Atan(DiffScalar x) => new(Math.Atan(x.Value), x.Derivative / (1 + x.Value * x.Value));
    // d(atan2(y,x))/dx = (x*dy/dx - y*dx/dx) / (x^2+y^2)
    public static DiffScalar Atan2(DiffScalar y, DiffScalar x) => new(Math.Atan2(y.Value, x.Value), (x.Value * y.Derivative - y.Value * x.Derivative) / (x.Value * x.Value + y.Value * y.Value));

    // Hyperbolique
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sinh(DiffScalar x) { double h = Math.Sinh(x.Value); return new(h, Math.Cosh(x.Value) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Cosh(DiffScalar x) { double h = Math.Cosh(x.Value); return new(h, Math.Sinh(x.Value) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Tanh(DiffScalar x) { double t = Math.Tanh(x.Value); return new(t, (1 - t * t) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Asinh(DiffScalar x) => new(Math.Asinh(x.Value), x.Derivative / Math.Sqrt(x.Value * x.Value + 1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Acosh(DiffScalar x) => new(Math.Acosh(x.Value), x.Derivative / Math.Sqrt(x.Value * x.Value - 1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Atanh(DiffScalar x) => new(Math.Atanh(x.Value), x.Derivative / (1 - x.Value * x.Value));

    // Valeur absolue : d|x|/dx = sign(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Abs(DiffScalar x) => new(Math.Abs(x.Value), Math.Sign(x.Value) * x.Derivative);
    // Clamp : clamp(x, a, b)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Clamp(DiffScalar x, double a, double b) => x.Value < a ? new(a, 0) : x.Value > b ? new(b, 0) : x;
    // Min/Max
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Max(DiffScalar a, DiffScalar b) => a.Value >= b.Value ? a : b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Min(DiffScalar a, DiffScalar b) => a.Value <= b.Value ? a : b;
    // Sigmoid : sigma(x) = 1/(1+e^-x), d/dx = sigma(x)*(1-sigma(x))
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sigmoid(DiffScalar x) { double s = 1.0 / (1.0 + Math.Exp(-x.Value)); return new(s, s * (1 - s) * x.Derivative); }
    // Softplus : ln(1+e^x), d/dx = sigmoid(x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Softplus(DiffScalar x) => new(Math.Log(1 + Math.Exp(x.Value)), Sigmoid(x).Derivative);
    // GELU approx : 0.5*x*(1+erf(x/sqrt(2)))
    // ReLU : max(0, x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar ReLU(DiffScalar x) => x.Value > 0 ? x : new(0, 0);
    // LeakyReLU : max(alpha*x, x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar LeakyReLU(DiffScalar x, double alpha = 0.01) => x.Value > 0 ? x : new(alpha * x.Value, alpha * x.Derivative);

    /// <summary>Chain rule : compose this with f(x) : d/dx f(g(x)) = f'(g(x)) * g'(x).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public DiffScalar ChainRule(DiffScalar outerFuncDerivative) => new(Value, Value * outerFuncDerivative.Derivative);

    /// <summary>Cosinue la differentiation avec un nouveau point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar Chain(DiffScalar inner, DiffScalar outerAtInner) => new(outerAtInner.Value, outerAtInner.Derivative * inner.Derivative);

    /// <summary>Promote to a second-order jet (value, first derivative, second derivative).</summary>
    public readonly DiffScalar2 WithSecondDerivative(double secondDerivative) => new(Value, Derivative, secondDerivative);

    // Equality
    public readonly bool Equals(DiffScalar o) => Math.Abs(Value - o.Value) < 1e-15 && Math.Abs(Derivative - o.Derivative) < 1e-15;
    public override readonly bool Equals(object? obj) => obj is DiffScalar o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Value, Derivative);
    public static bool operator ==(DiffScalar a, DiffScalar b) => a.Equals(b);
    public static bool operator !=(DiffScalar a, DiffScalar b) => !a.Equals(b);
    public override readonly string ToString() => $"d={Value:F6} (d/dx={Derivative:F6})";
}

/// <summary>
/// Second-order dual number (jet): tracks f, f', f'' for Hessian-vector products
/// and curvature-aware physics optimization.
/// MEMORY LAYOUT: 24 bytes (3 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 24, Pack = 8)]
[DebuggerDisplay("d={Value:F6} (d/dx={Derivative:F6}, d²/dx²={SecondDerivative:F6})")]
public struct DiffScalar2 : IEquatable<DiffScalar2>
{
    [FieldOffset(0)] public double Value;
    [FieldOffset(8)] public double Derivative;
    [FieldOffset(16)] public double SecondDerivative;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public DiffScalar2(double value, double derivative = 0, double secondDerivative = 0)
    {
        Value = value;
        Derivative = derivative;
        SecondDerivative = secondDerivative;
    }

    public static DiffScalar2 Variable(double value) => new(value, 1.0, 0.0);
    public DiffScalar ToFirstOrder() => new(Value, Derivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator +(DiffScalar2 a, DiffScalar2 b) =>
        new(a.Value + b.Value, a.Derivative + b.Derivative, a.SecondDerivative + b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator -(DiffScalar2 a, DiffScalar2 b) =>
        new(a.Value - b.Value, a.Derivative - b.Derivative, a.SecondDerivative - b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator *(DiffScalar2 a, DiffScalar2 b) =>
        new(
            a.Value * b.Value,
            a.Derivative * b.Value + a.Value * b.Derivative,
            a.SecondDerivative * b.Value + 2 * a.Derivative * b.Derivative + a.Value * b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Exp(DiffScalar2 x)
    {
        double e = Math.Exp(x.Value);
        return new(e, e * x.Derivative, e * (x.SecondDerivative + x.Derivative * x.Derivative));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Sin(DiffScalar2 x)
    {
        double s = Math.Sin(x.Value), c = Math.Cos(x.Value);
        return new(s, c * x.Derivative, -s * x.Derivative * x.Derivative + c * x.SecondDerivative);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Cos(DiffScalar2 x)
    {
        double s = Math.Sin(x.Value), c = Math.Cos(x.Value);
        return new(c, -s * x.Derivative, -c * x.Derivative * x.Derivative - s * x.SecondDerivative);
    }

    public readonly bool Equals(DiffScalar2 o) =>
        Math.Abs(Value - o.Value) < 1e-15 &&
        Math.Abs(Derivative - o.Derivative) < 1e-15 &&
        Math.Abs(SecondDerivative - o.SecondDerivative) < 1e-15;

    public override readonly bool Equals(object? obj) => obj is DiffScalar2 o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Value, Derivative, SecondDerivative);
    public override readonly string ToString() =>
        $"d={Value:F6} (d/dx={Derivative:F6}, d²/dx²={SecondDerivative:F6})";
}

/// <summary>
/// Expression differentiable — contient un graphe de computation pour le calcul
/// de gradients. Supporte le forward-mode AD via DiffScalar et le backward-mode
/// via un accumulateur de gradient. Utilise pour l'optimisation de parametres
/// physiques dans le PINN (Physics-Informed Neural Network).
///
/// L'expression est construite par composition d'operateurs differentiables.
/// Chaque evaluation propage les valeurs et les derivees simultanement.
/// </summary>
public sealed class DiffExpression
{
    private readonly List<(Func<DiffScalar[], DiffScalar> fn, int[] inputIndices)> _operations = new();
    private readonly List<DiffScalar> _tapes = new();
    private int _nextIndex = 0;

    /// <summary>Enregistre une variable comme entree differentiable.</summary>
    public int AddVariable(double value)
    {
        _tapes.Add(DiffScalar.Variable(value));
        return _nextIndex++;
    }

    /// <summary>Enregistre une constante (pas de derivee).</summary>
    public int AddConstant(double value)
    {
        _tapes.Add(new DiffScalar(value, 0));
        return _nextIndex++;
    }

    /// <summary>Ajoute une operation : result = fn(inputs).</summary>
    public int AddOperation(Func<DiffScalar[], DiffScalar> fn, params int[] inputIndices)
    {
        var inputs = new DiffScalar[inputIndices.Length];
        for (int i = 0; i < inputIndices.Length; i++)
            inputs[i] = _tapes[inputIndices[i]];
        _tapes.Add(fn(inputs));
        _operations.Add((fn, inputIndices));
        return _nextIndex++;
    }

    /// <summary>Evalue l'expression et retourne le resultat differentiable.</summary>
    public DiffScalar Evaluate() => _tapes.Count > 0 ? _tapes[^1] : DiffScalar.Zero;

    /// <summary>Calcule le gradient par rapport a toutes les variables d'entree.</summary>
    public double[] ComputeGradients()
    {
        var result = new double[_tapes.Count];
        for (int i = 0; i < _tapes.Count; i++)
            result[i] = _tapes[i].Derivative;
        return result;
    }

    /// <summary>Met a jour une variable et re-evalue.</summary>
    public void UpdateVariable(int index, double newValue)
    {
        _tapes[index] = DiffScalar.Variable(newValue);
        // Re-evaluer toutes les operations dependantes
        for (int i = 0; i < _operations.Count; i++)
        {
            var (fn, inputIndices) = _operations[i];
            var inputs = new DiffScalar[inputIndices.Length];
            for (int j = 0; j < inputIndices.Length; j++)
                inputs[j] = _tapes[inputIndices[j]];
            _tapes[_tapes.Count - _operations.Count + i] = fn(inputs);
        }
    }

    /// <summary>Nombre total de noeuds dans le graphe de computation.</summary>
    public int TapeSize => _tapes.Count;

    /// <summary>Reinitialise l'expression pour un nouvel evaluation.</summary>
    public void Reset() { _tapes.Clear(); _operations.Clear(); _nextIndex = 0; }

    // Factory methods pour les expressions courantes
    public static DiffExpression Quadratic(double a, double b, double c)
    {
        var expr = new DiffExpression();
        int x = expr.AddVariable(0);
        int aI = expr.AddConstant(a);
        int bI = expr.AddConstant(b);
        int cI = expr.AddConstant(c);
        int x2 = expr.AddOperation(d => d[0] * d[0], x);
        int ax2 = expr.AddOperation(d => d[0] * d[1], aI, x2);
        int bx = expr.AddOperation(d => d[0] * d[1], bI, x);
        expr.AddOperation(d => d[0] + d[1] + d[2], ax2, bx, cI);
        return expr;
    }

    public static DiffExpression Polynomial(double[] coefficients)
    {
        var expr = new DiffExpression();
        int x = expr.AddVariable(0);
        int[] powerIndices = new int[coefficients.Length];
        powerIndices[0] = expr.AddConstant(1); // x^0 = 1
        for (int i = 1; i < coefficients.Length; i++)
        {
            int pow = i;
            powerIndices[i] = expr.AddOperation(d => DiffScalar.Pow(d[0], pow), x);
        }
        for (int i = 0; i < coefficients.Length; i++)
        {
            int cI = expr.AddConstant(coefficients[i]);
            // Multiply coefficient by power
        }
        return expr;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
