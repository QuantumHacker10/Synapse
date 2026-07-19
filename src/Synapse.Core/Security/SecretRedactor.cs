using System.Text.RegularExpressions;

namespace Synapse.Core.Security
{
    public static class SecretRedactor
    {
        private static readonly (Regex Pattern, string Replacement)[] Patterns =
        {
            (new(@"(?i)(api[_-]?key|authorization|bearer|token|secret|password)\s*[=:]\s*([^\s,;]+)", RegexOptions.Compiled),
                "$1=***"),
            (new(@"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*", RegexOptions.Compiled),
                "Bearer ***"),
            (new(@"(?i)key=[A-Za-z0-9_\-]{8,}", RegexOptions.Compiled),
                "key=***"),
            (new(@"(?i)sk-[A-Za-z0-9]{10,}", RegexOptions.Compiled),
                "sk-***"),
        };

        public static string Redact(string? text)
        {
            if (string.IsNullOrEmpty(text))
                return text ?? "";
            var result = text;
            foreach (var (pattern, replacement) in Patterns)
                result = pattern.Replace(result, replacement);
            return result;
        }
    }
}
