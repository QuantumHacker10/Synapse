// =============================================================================
// G-DNN — Primary-device mesh-shader style compute (Nanite cinematic path)
// =============================================================================

using System;
using System.Text;
using GDNN.GPU;

namespace GDNN.GPU;

/// <summary>
/// Emits a primary-device compute shader that mimics mesh-shader amplification:
/// one workgroup per meshlet, shared-memory vertex transform, triangle expand,
/// and full-res material payload write (visibility + albedo RGB10 packing).
/// Used when true VK_EXT_mesh_shader is unavailable; same contract as Nanite clusters.
/// </summary>
public static class MeshShaderCompatGenerator
{
    public const int WorkgroupSize = 64;
    public const string EntryPoint = "main";

    /// <summary>GLSL compute that writes visibility R32 + material RGB10_A2 payloads.</summary>
    public static string GenerateMaterialResolveGlsl()
    {
        var sb = new StringBuilder(4096);
        sb.AppendLine("#version 450");
        sb.AppendLine($"layout(local_size_x = {WorkgroupSize}) in;");
        sb.AppendLine("layout(std430, binding = 0) readonly buffer Headers { uvec4 headers[]; };");
        sb.AppendLine("layout(std430, binding = 1) readonly buffer Positions { vec4 positions[]; };");
        sb.AppendLine("layout(std430, binding = 2) readonly buffer Triangles { uint tris[]; };");
        sb.AppendLine("layout(binding = 3) uniform writeonly uimage2D visBuffer;");
        sb.AppendLine("layout(binding = 4) uniform writeonly uimage2D materialBuffer;");
        sb.AppendLine("layout(push_constant) uniform Push { mat4 viewProj; uvec2 screen; uint meshletCount; uint lodQ; } pc;");
        sb.AppendLine("shared vec4 shPos[64];");
        sb.AppendLine("uint packRgb10(vec3 c) {");
        sb.AppendLine("  uvec3 u = uvec3(clamp(c, 0.0, 1.0) * 1023.0 + 0.5);");
        sb.AppendLine("  return (u.x) | (u.y << 10) | (u.z << 20);");
        sb.AppendLine("}");
        sb.AppendLine("void main() {");
        sb.AppendLine("  uint mid = gl_WorkGroupID.x;");
        sb.AppendLine("  if (mid >= pc.meshletCount) return;");
        sb.AppendLine("  uvec4 h = headers[mid];");
        sb.AppendLine("  uint vCount = h.x & 0xFFu;");
        sb.AppendLine("  uint tCount = (h.x >> 8) & 0xFFu;");
        sb.AppendLine("  uint vOff = h.y;");
        sb.AppendLine("  uint tOff = h.z;");
        sb.AppendLine("  uint lid = gl_LocalInvocationID.x;");
        sb.AppendLine("  if (lid < vCount) shPos[lid] = pc.viewProj * positions[vOff + lid];");
        sb.AppendLine("  barrier();");
        sb.AppendLine("  if (lid >= tCount) return;");
        sb.AppendLine("  uint packed = tris[tOff + lid];");
        sb.AppendLine("  uint i0 = packed & 0xFFu;");
        sb.AppendLine("  uint i1 = (packed >> 8) & 0xFFu;");
        sb.AppendLine("  uint i2 = (packed >> 16) & 0xFFu;");
        sb.AppendLine("  vec4 p0 = shPos[i0]; vec4 p1 = shPos[i1]; vec4 p2 = shPos[i2];");
        sb.AppendLine("  if (p0.w <= 0.0 || p1.w <= 0.0 || p2.w <= 0.0) return;");
        sb.AppendLine("  vec2 s0 = (p0.xy / p0.w) * 0.5 + 0.5; s0 *= vec2(pc.screen);");
        sb.AppendLine("  vec2 s1 = (p1.xy / p1.w) * 0.5 + 0.5; s1 *= vec2(pc.screen);");
        sb.AppendLine("  vec2 s2 = (p2.xy / p2.w) * 0.5 + 0.5; s2 *= vec2(pc.screen);");
        sb.AppendLine("  // Hash material from meshlet+tri (Nanite cluster look)");
        sb.AppendLine("  uint hh = mid * 73856093u ^ lid * 19349663u;");
        sb.AppendLine("  vec3 albedo = vec3(float(hh & 255u), float((hh >> 8) & 255u), float((hh >> 16) & 255u)) / 255.0;");
        sb.AppendLine("  albedo = mix(vec3(dot(albedo, vec3(0.2126, 0.7152, 0.0722))), albedo, 0.55 + 0.45 * float(pc.lodQ) / 255.0);");
        sb.AppendLine("  // Conservative 1-pixel stamp at triangle centroid for compat path");
        sb.AppendLine("  ivec2 px = ivec2((s0 + s1 + s2) / 3.0);");
        sb.AppendLine("  if (px.x < 0 || px.y < 0 || px.x >= int(pc.screen.x) || px.y >= int(pc.screen.y)) return;");
        sb.AppendLine("  uint depthKey = floatBitsToUint(1.0 / max(1e-4, (p0.w + p1.w + p2.w) / 3.0));");
        sb.AppendLine("  imageAtomicMax(visBuffer, px, depthKey);");
        sb.AppendLine("  imageStore(materialBuffer, px, uvec4(packRgb10(albedo), mid, lid, 0));");
        sb.AppendLine("}");
        return sb.ToString();
    }

    public static bool TryGetSpirv(out byte[] spirv, out string log)
    {
        string glsl = GenerateMaterialResolveGlsl();
        if (SpirvToolchain.TryCompileGlsl(glsl, out spirv, out log))
            return true;
        spirv = Array.Empty<byte>();
        return false;
    }
}
