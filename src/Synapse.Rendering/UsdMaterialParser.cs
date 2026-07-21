using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// Parses UsdPreviewSurface materials, UsdUVTexture file inputs, and material:binding relations.
/// Maps connected texture shaders to <see cref="MeshMaterial"/> PBR texture slots.
/// </summary>
public static class UsdMaterialParser
{
    private static readonly Regex MaterialDef = new(
        @"def\s+Material\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex ShaderDef = new(
        @"def\s+Shader\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex Binding = new(
        @"rel\s+material:binding(?:\s*:\s*\w+)?\s*=\s*<([^>]+)>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex ConnectAttr = new(
        @"inputs:(\w+)\.connect\s*=\s*<([^>]+)>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex AssetFile = new(
        @"(?:asset\s+)?inputs:file\s*=\s*@([^@]+)@",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex InfoId = new(
        @"uniform\s+token\s+info:id\s*=\s*""([^""]+)""",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Input name on UsdPreviewSurface → mesh material texture slot.</summary>
    public static readonly Dictionary<string, string> PreviewSurfaceTextureInputs = new(StringComparer.OrdinalIgnoreCase)
    {
        ["diffuseColor"] = nameof(MeshMaterial.AlbedoTexturePath),
        ["albedo"] = nameof(MeshMaterial.AlbedoTexturePath),
        ["baseColor"] = nameof(MeshMaterial.AlbedoTexturePath),
        ["normal"] = nameof(MeshMaterial.NormalTexturePath),
        ["roughness"] = nameof(MeshMaterial.MetallicRoughnessTexturePath),
        ["metallic"] = nameof(MeshMaterial.MetallicRoughnessTexturePath),
        ["metallicRoughness"] = nameof(MeshMaterial.MetallicRoughnessTexturePath),
        ["occlusion"] = nameof(MeshMaterial.AOTexturePath),
        ["occlusionRoughnessMetallic"] = nameof(MeshMaterial.MetallicRoughnessTexturePath),
        ["emissiveColor"] = nameof(MeshMaterial.EmissiveTexturePath),
        ["displacement"] = nameof(MeshMaterial.HeightTexturePath),
        ["height"] = nameof(MeshMaterial.HeightTexturePath),
        ["clearcoat"] = nameof(MeshMaterial.ClearcoatTexturePath),
        ["specularColor"] = nameof(MeshMaterial.SpecularTexturePath),
        ["opacity"] = nameof(MeshMaterial.OpacityTexturePath),
    };

    public static List<MeshMaterial> ParseMaterials(string usdaText, string? baseDirectory = null)
    {
        var materials = new List<MeshMaterial>();
        if (string.IsNullOrEmpty(usdaText))
            return materials;

        // Index all shaders by name for connection resolution.
        var shaders = IndexShaders(usdaText);

        foreach (Match matMatch in MaterialDef.Matches(usdaText))
        {
            string matName = matMatch.Groups[1].Value;
            int bodyStart = matMatch.Index + matMatch.Length;
            // Material body starts at '{' just before bodyStart-1 from regex end... regex ends after '{'
            string matBody = ExtractBalancedBlock(usdaText, matMatch.Index + matMatch.Length - 1);

            var mat = new MeshMaterial { Name = matName };

            // Prefer PreviewSurface inside this material; fall back to any in file that shares name prefix.
            var preview = FindPreviewSurface(matBody, shaders) ?? FindPreviewSurface(usdaText, shaders);
            if (preview != null)
            {
                ApplyPreviewSurfaceScalars(mat, preview.Value.Body);
                ApplyTextureConnections(mat, preview.Value.Body, shaders, baseDirectory);
            }

            // Also scan material body for direct UsdUVTexture children named conventionally.
            ApplyDirectUvTexturesInScope(mat, matBody, shaders, baseDirectory);
            ApplyMdlReference(mat, matBody, usdaText, baseDirectory);
            FinalizeUdimMaps(mat, baseDirectory);

            materials.Add(mat);
        }

        if (materials.Count == 0)
        {
            // Flat PreviewSurface-only stages (no Material prim).
            foreach (var (name, body, id) in EnumerateShaders(usdaText))
            {
                if (!IsPreviewSurface(id, body))
                    continue;
                var mat = new MeshMaterial { Name = name };
                ApplyPreviewSurfaceScalars(mat, body);
                ApplyTextureConnections(mat, body, shaders, baseDirectory);
                FinalizeUdimMaps(mat, baseDirectory);
                materials.Add(mat);
            }
        }

        return materials;
    }

    public static void ApplyMdlReference(MeshMaterial mat, string matBody, string fullText, string? baseDirectory)
    {
        // sourceAsset = @./materials/foo.mdl@ or info:id = "mdl:..."
        var source = Regex.Match(matBody, @"(?:asset\s+)?(?:info:)?sourceAsset\s*=\s*@([^@]+)@",
            RegexOptions.IgnoreCase);
        if (!source.Success)
            source = Regex.Match(fullText, @"(?:asset\s+)?(?:info:)?sourceAsset\s*=\s*@([^@]+\.mdl)@",
                RegexOptions.IgnoreCase);
        if (source.Success)
            mat.MdlAssetPath = ResolveTexturePath(source.Groups[1].Value.Trim(), baseDirectory);

        var mdlName = Regex.Match(matBody, @"(?:uniform\s+)?token\s+(?:info:)?mdl:sourceMaterial\s*=\s*""([^""]+)""",
            RegexOptions.IgnoreCase);
        if (!mdlName.Success)
            mdlName = Regex.Match(matBody, @"string\s+inputs:mdlMaterial\s*=\s*""([^""]+)""", RegexOptions.IgnoreCase);
        if (mdlName.Success)
            mat.MdlMaterialName = mdlName.Groups[1].Value;

        // Heuristic: shader info:id containing .mdl
        var id = Regex.Match(matBody, @"info:id\s*=\s*""([^""]*\.mdl[^""]*)""", RegexOptions.IgnoreCase);
        if (id.Success && string.IsNullOrEmpty(mat.MdlAssetPath))
            mat.MdlAssetPath = ResolveTexturePath(id.Groups[1].Value, baseDirectory);
    }

    public static void FinalizeUdimMaps(MeshMaterial mat, string? baseDirectory)
    {
        ExpandSlotUdim(mat, nameof(MeshMaterial.AlbedoTexturePath), mat.AlbedoTexturePath, baseDirectory, primary: true);
        ExpandSlotUdim(mat, nameof(MeshMaterial.NormalTexturePath), mat.NormalTexturePath, baseDirectory, primary: false);
        ExpandSlotUdim(mat, nameof(MeshMaterial.MetallicRoughnessTexturePath), mat.MetallicRoughnessTexturePath, baseDirectory, primary: false);
        ExpandSlotUdim(mat, nameof(MeshMaterial.EmissiveTexturePath), mat.EmissiveTexturePath, baseDirectory, primary: false);
        ExpandSlotUdim(mat, nameof(MeshMaterial.AOTexturePath), mat.AOTexturePath, baseDirectory, primary: false);
        ExpandSlotUdim(mat, nameof(MeshMaterial.OpacityTexturePath), mat.OpacityTexturePath, baseDirectory, primary: false);
    }

    private static void ExpandSlotUdim(
        MeshMaterial mat,
        string slotName,
        string path,
        string? baseDirectory,
        bool primary)
    {
        if (string.IsNullOrEmpty(path))
            return;

        bool hasToken = path.Contains("<UDIM>", StringComparison.OrdinalIgnoreCase) ||
                        path.Contains("%(UDIM)", StringComparison.OrdinalIgnoreCase);
        if (!hasToken)
        {
            if (primary)
            {
                var m = Regex.Match(path, @"(\d{4})(?=\.|$)");
                if (m.Success && int.TryParse(m.Groups[1].Value, out var tile) && tile >= 1001)
                    mat.UdimTiles[tile] = path;
            }

            return;
        }

        var tiles = UsdUdim.ExpandTiles(path, baseDirectory);
        mat.UdimMapsBySlot[slotName] = tiles;
        if (tiles.TryGetValue(UsdUdim.BaseTile, out var t1001))
            AssignPathBySlot(mat, slotName, t1001);
        if (primary)
            mat.UdimTiles = tiles;
    }

    private static void AssignPathBySlot(MeshMaterial mat, string slotName, string path)
    {
        switch (slotName)
        {
            case nameof(MeshMaterial.AlbedoTexturePath):
                mat.AlbedoTexturePath = path;
                break;
            case nameof(MeshMaterial.NormalTexturePath):
                mat.NormalTexturePath = path;
                break;
            case nameof(MeshMaterial.MetallicRoughnessTexturePath):
                mat.MetallicRoughnessTexturePath = path;
                break;
            case nameof(MeshMaterial.EmissiveTexturePath):
                mat.EmissiveTexturePath = path;
                break;
            case nameof(MeshMaterial.AOTexturePath):
                mat.AOTexturePath = path;
                break;
            case nameof(MeshMaterial.OpacityTexturePath):
                mat.OpacityTexturePath = path;
                break;
        }
    }

    public static string? ParseBindingPath(string usdaText)
    {
        var m = Binding.Match(usdaText);
        return m.Success ? m.Groups[1].Value.Trim() : null;
    }

    public static int ResolveMaterialIndex(IReadOnlyList<MeshMaterial> materials, string? bindingPath)
    {
        if (materials.Count == 0)
            return 0;
        if (string.IsNullOrWhiteSpace(bindingPath))
            return 0;

        var last = bindingPath.Trim().Trim('/');
        if (last.Contains('/'))
            last = last[(last.LastIndexOf('/') + 1)..];

        for (int i = 0; i < materials.Count; i++)
        {
            if (string.Equals(materials[i].Name, last, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return 0;
    }

    /// <summary>Resolves relative @asset@ paths against a USDA directory.</summary>
    public static string ResolveTexturePath(string assetPath, string? baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(assetPath))
            return "";
        var cleaned = assetPath.Trim().Replace('\\', '/');
        while (cleaned.StartsWith("./", StringComparison.Ordinal))
            cleaned = cleaned[2..];
        if (Path.IsPathRooted(cleaned))
            return Path.GetFullPath(cleaned);
        if (string.IsNullOrWhiteSpace(baseDirectory))
            return cleaned;
        return Path.GetFullPath(Path.Combine(baseDirectory, cleaned));
    }

    public static void ResolveMaterialTexturePaths(IEnumerable<MeshMaterial> materials, string? baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(baseDirectory))
            return;
        foreach (var mat in materials)
        {
            mat.AlbedoTexturePath = ResolveIfRelative(mat.AlbedoTexturePath, baseDirectory);
            mat.NormalTexturePath = ResolveIfRelative(mat.NormalTexturePath, baseDirectory);
            mat.MetallicRoughnessTexturePath = ResolveIfRelative(mat.MetallicRoughnessTexturePath, baseDirectory);
            mat.EmissiveTexturePath = ResolveIfRelative(mat.EmissiveTexturePath, baseDirectory);
            mat.AOTexturePath = ResolveIfRelative(mat.AOTexturePath, baseDirectory);
            mat.HeightTexturePath = ResolveIfRelative(mat.HeightTexturePath, baseDirectory);
            mat.ClearcoatTexturePath = ResolveIfRelative(mat.ClearcoatTexturePath, baseDirectory);
            mat.SpecularTexturePath = ResolveIfRelative(mat.SpecularTexturePath, baseDirectory);
            mat.OpacityTexturePath = ResolveIfRelative(mat.OpacityTexturePath, baseDirectory);
        }
    }

    public static Vector3 ParseColor3(string text, string key, Vector3 fallback)
    {
        int idx = IndexOfInputAssignment(text, key);
        if (idx < 0)
            return fallback;
        // Skip .connect forms
        int connect = text.IndexOf(".connect", idx, StringComparison.OrdinalIgnoreCase);
        int eq = text.IndexOf('=', idx);
        if (eq < 0 || (connect >= 0 && connect < eq))
            return fallback;
        int open = text.IndexOf('(', eq);
        int close = open >= 0 ? text.IndexOf(')', open) : -1;
        if (open < 0 || close < open)
            return fallback;
        var parts = text.Substring(open + 1, close - open - 1)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 3)
            return fallback;
        if (float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var r) &&
            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var g) &&
            float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
            return new Vector3(r, g, b);
        return fallback;
    }

    public static float ParseFloat(string text, string key, float fallback)
    {
        int idx = IndexOfInputAssignment(text, key);
        if (idx < 0)
            return fallback;
        int connect = text.IndexOf(".connect", idx, StringComparison.OrdinalIgnoreCase);
        int eq = text.IndexOf('=', idx);
        if (eq < 0 || (connect >= 0 && connect < eq + 8 && connect < eq + 20))
        {
            // If this is a .connect line, skip scalar parse
            if (connect >= 0 && connect < idx + key.Length + 16)
                return fallback;
        }

        eq = text.IndexOf('=', idx);
        if (eq < 0)
            return fallback;
        if (text.IndexOf(".connect", idx, StringComparison.OrdinalIgnoreCase) is int c and >= 0 && c < eq)
            return fallback;

        int end = eq + 1;
        while (end < text.Length && char.IsWhiteSpace(text[end]))
            end++;
        int start = end;
        while (end < text.Length && (char.IsDigit(text[end]) || text[end] is '.' or '-' or '+' or 'e' or 'E'))
            end++;
        if (start == end)
            return fallback;
        return float.TryParse(text[start..end], NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v
            : fallback;
    }

    public static string? ExtractUvTextureFile(string shaderBody)
    {
        var m = AssetFile.Match(shaderBody);
        return m.Success ? m.Groups[1].Value.Trim() : null;
    }

    private static void ApplyPreviewSurfaceScalars(MeshMaterial mat, string body)
    {
        mat.BaseColor = ParseColor3(body, "inputs:diffuseColor", mat.BaseColor);
        mat.BaseColor = ParseColor3(body, "inputs:baseColor", mat.BaseColor);
        mat.Roughness = ParseFloat(body, "inputs:roughness", mat.Roughness);
        mat.Metallic = ParseFloat(body, "inputs:metallic", mat.Metallic);
        mat.Opacity = ParseFloat(body, "inputs:opacity", mat.Opacity);
        mat.AlphaCutoff = ParseFloat(body, "inputs:opacityThreshold", mat.AlphaCutoff);
        mat.EmissiveColor = ParseColor3(body, "inputs:emissiveColor", mat.EmissiveColor);
        mat.EmissiveIntensity = ParseFloat(body, "inputs:emissiveIntensity", mat.EmissiveIntensity);
        if (mat.EmissiveIntensity <= 0f && mat.EmissiveColor.LengthSquared() > 1e-8f)
            mat.EmissiveIntensity = 1f;
        mat.Clearcoat = ParseFloat(body, "inputs:clearcoat", mat.Clearcoat);
        mat.ClearcoatRoughness = ParseFloat(body, "inputs:clearcoatRoughness", mat.ClearcoatRoughness);
        mat.Ior = ParseFloat(body, "inputs:ior", mat.Ior);
        mat.Occlusion = ParseFloat(body, "inputs:occlusion", mat.Occlusion);
    }

    private static void ApplyTextureConnections(
        MeshMaterial mat,
        string previewBody,
        Dictionary<string, ShaderInfo> shaders,
        string? baseDirectory)
    {
        foreach (Match m in ConnectAttr.Matches(previewBody))
        {
            string input = m.Groups[1].Value;
            string target = m.Groups[2].Value.Trim();
            // </Looks/Mat/DiffuseTex.outputs:rgb> → DiffuseTex
            string shaderName = PrimLeafName(target);
            if (shaderName.Contains('.', StringComparison.Ordinal))
                shaderName = shaderName[..shaderName.IndexOf('.')];

            if (!shaders.TryGetValue(shaderName, out var shader))
            {
                foreach (var kv in shaders)
                {
                    if (kv.Key.Equals(shaderName, StringComparison.OrdinalIgnoreCase) ||
                        target.Contains("/" + kv.Key + ".", StringComparison.OrdinalIgnoreCase) ||
                        target.Contains("/" + kv.Key + ">", StringComparison.OrdinalIgnoreCase))
                    {
                        shader = kv.Value;
                        break;
                    }
                }
            }

            if (string.IsNullOrEmpty(shader.Body))
                continue;

            if (!IsUvTexture(shader.Id, shader.Body))
                continue;

            string? file = ExtractUvTextureFile(shader.Body);
            if (string.IsNullOrEmpty(file))
                continue;

            string resolved = ResolveTexturePath(file, baseDirectory);
            AssignTextureSlot(mat, input, resolved);
            ApplyUvTextureColorSpace(mat, shader.Body);
        }
    }

    private static void ApplyUvTextureColorSpace(MeshMaterial mat, string shaderBody)
    {
        var cs = Regex.Match(shaderBody,
            @"(?:token|string)\s+inputs:sourceColorSpace\s*=\s*""([^""]+)""",
            RegexOptions.IgnoreCase);
        if (cs.Success && string.IsNullOrEmpty(mat.ColorSpace))
            mat.ColorSpace = cs.Groups[1].Value;
    }

    private static void ApplyDirectUvTexturesInScope(
        MeshMaterial mat,
        string scopeBody,
        Dictionary<string, ShaderInfo> shaders,
        string? baseDirectory)
    {
        foreach (Match m in ShaderDef.Matches(scopeBody))
        {
            string name = m.Groups[1].Value;
            if (!shaders.TryGetValue(name, out var info))
                continue;
            if (!IsUvTexture(info.Id, info.Body))
                continue;
            string? file = ExtractUvTextureFile(info.Body);
            if (string.IsNullOrEmpty(file))
                continue;
            string resolved = ResolveTexturePath(file, baseDirectory);
            // Heuristic from shader name when not connected
            AssignTextureSlotByNameHint(mat, name, resolved);
        }
    }

    private static void AssignTextureSlot(MeshMaterial mat, string previewInput, string path)
    {
        if (!PreviewSurfaceTextureInputs.TryGetValue(previewInput, out var slot))
            AssignTextureSlotByNameHint(mat, previewInput, path);
        else
            SetSlot(mat, slot, path);
    }

    private static void AssignTextureSlotByNameHint(MeshMaterial mat, string hint, string path)
    {
        var h = hint.ToLowerInvariant();
        if (h.Contains("normal", StringComparison.Ordinal))
            mat.NormalTexturePath = Prefer(mat.NormalTexturePath, path);
        else if (h.Contains("emissive", StringComparison.Ordinal) || h.Contains("emit", StringComparison.Ordinal))
            mat.EmissiveTexturePath = Prefer(mat.EmissiveTexturePath, path);
        else if (h.Contains("occlusion", StringComparison.Ordinal) || h.Contains("ao", StringComparison.Ordinal))
            mat.AOTexturePath = Prefer(mat.AOTexturePath, path);
        else if (h.Contains("height", StringComparison.Ordinal) || h.Contains("disp", StringComparison.Ordinal))
            mat.HeightTexturePath = Prefer(mat.HeightTexturePath, path);
        else if (h.Contains("metal", StringComparison.Ordinal) || h.Contains("rough", StringComparison.Ordinal) || h.Contains("orm", StringComparison.Ordinal))
            mat.MetallicRoughnessTexturePath = Prefer(mat.MetallicRoughnessTexturePath, path);
        else if (h.Contains("clearcoat", StringComparison.Ordinal))
            mat.ClearcoatTexturePath = Prefer(mat.ClearcoatTexturePath, path);
        else if (h.Contains("specular", StringComparison.Ordinal))
            mat.SpecularTexturePath = Prefer(mat.SpecularTexturePath, path);
        else if (h.Contains("opacity", StringComparison.Ordinal) || h.Contains("alpha", StringComparison.Ordinal))
            mat.OpacityTexturePath = Prefer(mat.OpacityTexturePath, path);
        else if (h.Contains("diff", StringComparison.Ordinal) || h.Contains("albedo", StringComparison.Ordinal) || h.Contains("base", StringComparison.Ordinal) || h.Contains("color", StringComparison.Ordinal))
            mat.AlbedoTexturePath = Prefer(mat.AlbedoTexturePath, path);
    }

    private static void SetSlot(MeshMaterial mat, string slot, string path)
    {
        switch (slot)
        {
            case nameof(MeshMaterial.AlbedoTexturePath):
                mat.AlbedoTexturePath = Prefer(mat.AlbedoTexturePath, path);
                break;
            case nameof(MeshMaterial.NormalTexturePath):
                mat.NormalTexturePath = Prefer(mat.NormalTexturePath, path);
                break;
            case nameof(MeshMaterial.MetallicRoughnessTexturePath):
                mat.MetallicRoughnessTexturePath = Prefer(mat.MetallicRoughnessTexturePath, path);
                break;
            case nameof(MeshMaterial.EmissiveTexturePath):
                mat.EmissiveTexturePath = Prefer(mat.EmissiveTexturePath, path);
                break;
            case nameof(MeshMaterial.AOTexturePath):
                mat.AOTexturePath = Prefer(mat.AOTexturePath, path);
                break;
            case nameof(MeshMaterial.HeightTexturePath):
                mat.HeightTexturePath = Prefer(mat.HeightTexturePath, path);
                break;
            case nameof(MeshMaterial.ClearcoatTexturePath):
                mat.ClearcoatTexturePath = Prefer(mat.ClearcoatTexturePath, path);
                break;
            case nameof(MeshMaterial.SpecularTexturePath):
                mat.SpecularTexturePath = Prefer(mat.SpecularTexturePath, path);
                break;
            case nameof(MeshMaterial.OpacityTexturePath):
                mat.OpacityTexturePath = Prefer(mat.OpacityTexturePath, path);
                break;
        }
    }

    private static string Prefer(string existing, string next) =>
        string.IsNullOrEmpty(existing) ? next : existing;

    private static string ResolveIfRelative(string path, string baseDirectory) =>
        string.IsNullOrEmpty(path) ? path : ResolveTexturePath(path, baseDirectory);

    private static Dictionary<string, ShaderInfo> IndexShaders(string text)
    {
        var map = new Dictionary<string, ShaderInfo>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, body, id) in EnumerateShaders(text))
            map[name] = new ShaderInfo(name, body, id);
        return map;
    }

    private static IEnumerable<(string Name, string Body, string Id)> EnumerateShaders(string text)
    {
        foreach (Match m in ShaderDef.Matches(text))
        {
            string name = m.Groups[1].Value;
            string body = ExtractBalancedBlock(text, m.Index + m.Length - 1);
            string id = "";
            var idMatch = InfoId.Match(body);
            if (idMatch.Success)
                id = idMatch.Groups[1].Value;
            yield return (name, body, id);
        }
    }

    private static (string Name, string Body)? FindPreviewSurface(string scope, Dictionary<string, ShaderInfo> shaders)
    {
        foreach (Match m in ShaderDef.Matches(scope))
        {
            string name = m.Groups[1].Value;
            if (shaders.TryGetValue(name, out var info) && IsPreviewSurface(info.Id, info.Body))
                return (name, info.Body);
        }

        return null;
    }

    private static bool IsPreviewSurface(string id, string body) =>
        id.Contains("UsdPreviewSurface", StringComparison.OrdinalIgnoreCase) ||
        body.Contains("UsdPreviewSurface", StringComparison.OrdinalIgnoreCase) ||
        body.Contains("inputs:diffuseColor", StringComparison.OrdinalIgnoreCase);

    private static bool IsUvTexture(string id, string body) =>
        id.Contains("UsdUVTexture", StringComparison.OrdinalIgnoreCase) ||
        body.Contains("UsdUVTexture", StringComparison.OrdinalIgnoreCase) ||
        AssetFile.IsMatch(body);

    private static string PrimLeafName(string path)
    {
        var p = path.Trim().Trim('<', '>').Trim('/');
        int slash = p.LastIndexOf('/');
        return slash >= 0 ? p[(slash + 1)..] : p;
    }

    private static int IndexOfInputAssignment(string text, string key)
    {
        // Accept "inputs:foo" or bare "foo" if key already has inputs:
        int idx = text.IndexOf(key, StringComparison.OrdinalIgnoreCase);
        return idx;
    }

    private static string ExtractBalancedBlock(string text, int openBraceIndex)
    {
        if (openBraceIndex < 0 || openBraceIndex >= text.Length || text[openBraceIndex] != '{')
            return "";
        int depth = 0;
        for (int i = openBraceIndex; i < text.Length; i++)
        {
            if (text[i] == '{')
                depth++;
            else if (text[i] == '}')
            {
                depth--;
                if (depth == 0)
                    return text.Substring(openBraceIndex + 1, i - openBraceIndex - 1);
            }
        }

        return text[(openBraceIndex + 1)..];
    }

    private readonly record struct ShaderInfo(string Name, string Body, string Id);
}
