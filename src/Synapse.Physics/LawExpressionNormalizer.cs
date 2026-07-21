using System.Text.RegularExpressions;

namespace Synapse.Physics;

/// <summary>
/// Normalizes catalog law expressions into forms the <see cref="LawExpressionParser"/> can compile.
/// </summary>
public static partial class LawExpressionNormalizer
{
    /// <summary>Prepares a registry or library expression for bytecode compilation.</summary>
    public static string NormalizeForCompilation(string expression)
    {
        if (string.IsNullOrWhiteSpace(expression))
            return "0";

        string s = expression.Trim();
        s = s.Replace("∂", "d", StringComparison.Ordinal)
            .Replace("·", "*", StringComparison.Ordinal)
            .Replace("×", "*", StringComparison.Ordinal)
            .Replace("Δ", "del", StringComparison.Ordinal)
            .Replace("∇", "del", StringComparison.Ordinal)
            .Replace("²", "^2", StringComparison.Ordinal)
            .Replace("³", "^3", StringComparison.Ordinal)
            .Replace("⁴", "^4", StringComparison.Ordinal);

        s = ExtractRightHandSide(s);
        s = DelSquaredRegex().Replace(s, "laplacian($1)");
        s = GradRegex().Replace(s, "grad_x($1)");
        s = TimeDerivativeRegex().Replace(s, "$1");
        s = DelDotRegex().Replace(s, "divergence($1)");
        return s.Trim();
    }

    /// <summary>Returns the RHS of an equation, or the whole string when no '=' is present.</summary>
    public static string ExtractRightHandSide(string expression)
    {
        for (int i = 0; i < expression.Length; i++)
        {
            if (expression[i] != '=')
                continue;
            if (i > 0 && (expression[i - 1] is '<' or '>' or '!' or '='))
                continue;
            if (i + 1 < expression.Length && expression[i + 1] == '=')
                continue;
            return expression[(i + 1)..].Trim();
        }

        return expression.Trim();
    }

    [GeneratedRegex(@"del\^2\s*\(\s*(\w+)\s*\)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex DelSquaredRegex();

    [GeneratedRegex(@"\bgrad\s*\(\s*(\w+)\s*\)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex GradRegex();

    [GeneratedRegex(@"\bd(\w+)/dt\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex TimeDerivativeRegex();

    [GeneratedRegex(@"\b\w+\.del\s*\(\s*(\w+)\s*\)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex DelDotRegex();
}
