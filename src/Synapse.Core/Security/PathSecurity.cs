using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Synapse.Core.Security
{
    /// <summary>
    /// Path canonicalization and jail helpers to prevent traversal / symlink escapes.
    /// </summary>
    public static class PathSecurity
    {
        private static readonly Regex SafeAssetId = new(
            @"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,200}$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex SafeIdentifier = new(
            @"^[A-Za-z_][A-Za-z0-9_]{0,127}$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        public static string RequireSafeAssetId(string assetId)
        {
            if (string.IsNullOrWhiteSpace(assetId) || !SafeAssetId.IsMatch(assetId) || assetId.Contains("..", StringComparison.Ordinal))
                throw new ArgumentException("Invalid asset identifier.", nameof(assetId));
            return assetId;
        }

        public static string RequireSafeIdentifier(string name, string paramName = "name")
        {
            if (string.IsNullOrWhiteSpace(name) || !SafeIdentifier.IsMatch(name))
                throw new ArgumentException("Invalid identifier.", paramName);
            return name;
        }

        public static string SanitizeFileName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Empty file name.", nameof(name));

            var invalid = Path.GetInvalidFileNameChars();
            var cleaned = new string(name.Select(c => invalid.Contains(c) || c == '/' || c == '\\' ? '_' : c).ToArray());
            cleaned = cleaned.Trim('.', ' ');
            if (cleaned.Length == 0 || cleaned.Contains("..", StringComparison.Ordinal) || !SafeAssetId.IsMatch(cleaned))
                throw new ArgumentException("Unsafe file name.", nameof(name));
            return cleaned;
        }

        public static string GetFullPathChecked(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path is empty.", nameof(path));
            return Path.GetFullPath(path);
        }

        /// <summary>
        /// Ensures <paramref name="path"/> resolves inside <paramref name="rootDirectory"/>.
        /// </summary>
        public static string EnsureUnderRoot(string rootDirectory, string path)
        {
            var root = AppendDirectorySeparator(GetFullPathChecked(rootDirectory));
            var full = GetFullPathChecked(path);
            if (!full.StartsWith(root, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
                throw new UnauthorizedAccessException($"Path escapes allowed root: {rootDirectory}");
            return full;
        }

        /// <summary>
        /// Combines root + relative segments and ensures the result stays under root.
        /// </summary>
        public static string CombineUnderRoot(string rootDirectory, params string[] relativeParts)
        {
            var root = GetFullPathChecked(rootDirectory);
            foreach (var part in relativeParts)
            {
                if (string.IsNullOrWhiteSpace(part))
                    throw new ArgumentException("Empty path segment.");
                if (Path.IsPathRooted(part) || part.Contains("..", StringComparison.Ordinal) ||
                    part.Contains('/') || part.Contains('\\'))
                {
                    throw new ArgumentException($"Unsafe path segment: {part}");
                }
            }

            var combined = Path.Combine(new[] { root }.Concat(relativeParts).ToArray());
            return EnsureUnderRoot(root, combined);
        }

        public static bool IsUnderRoot(string rootDirectory, string path)
        {
            try
            {
                EnsureUnderRoot(rootDirectory, path);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public static void EnsureParentDirectoryUnderRoot(string rootDirectory, string filePath)
        {
            var full = EnsureUnderRoot(rootDirectory, filePath);
            var parent = Path.GetDirectoryName(full);
            if (parent != null)
                Directory.CreateDirectory(EnsureUnderRoot(rootDirectory, parent));
        }

        private static string AppendDirectorySeparator(string path)
        {
            if (path.EndsWith(Path.DirectorySeparatorChar) || path.EndsWith(Path.AltDirectorySeparatorChar))
                return path;
            return path + Path.DirectorySeparatorChar;
        }
    }
}
