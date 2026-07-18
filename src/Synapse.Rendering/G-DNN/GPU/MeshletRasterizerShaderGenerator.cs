using System;
using System.Text;

namespace GDNN.GPU;

/// <summary>
/// Émet le shader compute GLSL de rasterisation software des meshlets —
/// le même algorithme que <see cref="GDNN.Polygonization.SoftwareRasterizer"/>
/// (fonctions d'arête en virgule fixe 24.8, règle top-left, atomicMax 64 bits
/// sur le visibility buffer), un workgroup par meshlet comme Nanite :
/// les sommets sont transformés en mémoire partagée puis les threads
/// rasterisent les triangles.
///
/// Le chemin CPU sert de référence de conformité ; ce shader est le chemin
/// rapide à brancher sur VulkanRhiDevice (VK_KHR_shader_atomic_int64).
/// </summary>
public static class MeshletRasterizerShaderGenerator
{
    public const int WorkgroupSize = 128;
    public const int MaxVerticesPerMeshlet = 64;
    public const int MaxTrianglesPerMeshlet = 124;

    public static string GenerateGlsl()
    {
        var sb = new StringBuilder();
        sb.AppendLine("#version 450");
        sb.AppendLine("#extension GL_EXT_shader_atomic_int64 : require");
        sb.AppendLine("#extension GL_ARB_gpu_shader_int64 : require");
        sb.AppendLine();
        sb.AppendLine($"layout(local_size_x = {WorkgroupSize}) in;");
        sb.AppendLine();
        sb.AppendLine("layout(push_constant) uniform PushConstants {");
        sb.AppendLine("    mat4 viewProjection;");
        sb.AppendLine("    uvec2 viewport;");
        sb.AppendLine("    uint meshletCount;");
        sb.AppendLine("} pc;");
        sb.AppendLine();
        sb.AppendLine("struct MeshletHeader { uint vertexOffset; uint vertexCount; uint triangleOffset; uint triangleCount; };");
        sb.AppendLine();
        sb.AppendLine("layout(std430, binding = 0) readonly buffer Headers   { MeshletHeader headers[]; };");
        sb.AppendLine("layout(std430, binding = 1) readonly buffer Positions { vec4 positions[]; };");
        sb.AppendLine("layout(std430, binding = 2) readonly buffer Triangles { uint packedTriangles[]; }; // 3 x 8 bits");
        sb.AppendLine("layout(std430, binding = 3) buffer Visibility { uint64_t visibility[]; };");
        sb.AppendLine();
        sb.AppendLine($"shared vec4 s_clip[{MaxVerticesPerMeshlet}];");
        sb.AppendLine();
        sb.AppendLine("const int SUBPIXEL_BITS = 8;");
        sb.AppendLine("const int SUBPIXEL_SCALE = 1 << SUBPIXEL_BITS;");
        sb.AppendLine("const int PIXEL_CENTER = SUBPIXEL_SCALE / 2;");
        sb.AppendLine("const float NEAR_CLIP_EPSILON = 1e-4;");
        sb.AppendLine();
        sb.AppendLine("vec3 toScreen(vec4 clip) {");
        sb.AppendLine("    float invW = 1.0 / clip.w;");
        sb.AppendLine("    return vec3(");
        sb.AppendLine("        (clip.x * invW * 0.5 + 0.5) * float(pc.viewport.x),");
        sb.AppendLine("        (0.5 - clip.y * invW * 0.5) * float(pc.viewport.y),");
        sb.AppendLine("        clip.z * invW);");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("// Arête top/left : repère y-bas, aire positive (voir SoftwareRasterizer).");
        sb.AppendLine("bool isTopLeftEdge(ivec2 p, ivec2 q) {");
        sb.AppendLine("    int dx = q.x - p.x, dy = q.y - p.y;");
        sb.AppendLine("    return dy < 0 || (dy == 0 && dx > 0);");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("void rasterizeTriangle(ivec2 f0, ivec2 f1, ivec2 f2,");
        sb.AppendLine("                       float z0, float z1, float z2, uint payload) {");
        sb.AppendLine("    // Aire signée fixe : front = négatif (CCW monde -> y écran inversé).");
        sb.AppendLine("    int64_t area2 = int64_t(f1.x - f0.x) * int64_t(f2.y - f0.y)");
        sb.AppendLine("               - int64_t(f2.x - f0.x) * int64_t(f1.y - f0.y);");
        sb.AppendLine("    if (area2 >= 0) return; // dos ou dégénéré");
        sb.AppendLine("    ivec2 t = f1; f1 = f2; f2 = t;");
        sb.AppendLine("    float tz = z1; z1 = z2; z2 = tz;");
        sb.AppendLine("    area2 = -area2;");
        sb.AppendLine();
        sb.AppendLine("    ivec2 minF = min(f0, min(f1, f2));");
        sb.AppendLine("    ivec2 maxF = max(f0, max(f1, f2));");
        sb.AppendLine("    int minX = max(0, minF.x >> SUBPIXEL_BITS);");
        sb.AppendLine("    int maxX = min(int(pc.viewport.x) - 1, (maxF.x + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
        sb.AppendLine("    int minY = max(0, minF.y >> SUBPIXEL_BITS);");
        sb.AppendLine("    int maxY = min(int(pc.viewport.y) - 1, (maxF.y + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
        sb.AppendLine("    if (minX > maxX || minY > maxY) return;");
        sb.AppendLine();
        sb.AppendLine("    int64_t bias0 = isTopLeftEdge(f1, f2) ? int64_t(0) : int64_t(-1);");
        sb.AppendLine("    int64_t bias1 = isTopLeftEdge(f2, f0) ? int64_t(0) : int64_t(-1);");
        sb.AppendLine("    int64_t bias2 = isTopLeftEdge(f0, f1) ? int64_t(0) : int64_t(-1);");
        sb.AppendLine();
        sb.AppendLine("    int64_t px = (int64_t(minX) << SUBPIXEL_BITS) + PIXEL_CENTER;");
        sb.AppendLine("    int64_t py = (int64_t(minY) << SUBPIXEL_BITS) + PIXEL_CENTER;");
        sb.AppendLine("    int64_t rowW0 = int64_t(f2.x - f1.x) * (py - f1.y) - int64_t(f2.y - f1.y) * (px - f1.x) + bias0;");
        sb.AppendLine("    int64_t rowW1 = int64_t(f0.x - f2.x) * (py - f2.y) - int64_t(f0.y - f2.y) * (px - f2.x) + bias1;");
        sb.AppendLine("    int64_t rowW2 = int64_t(f1.x - f0.x) * (py - f0.y) - int64_t(f1.y - f0.y) * (px - f0.x) + bias2;");
        sb.AppendLine("    int64_t stepX0 = int64_t(f1.y - f2.y) << SUBPIXEL_BITS, stepY0 = int64_t(f2.x - f1.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    int64_t stepX1 = int64_t(f2.y - f0.y) << SUBPIXEL_BITS, stepY1 = int64_t(f0.x - f2.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    int64_t stepX2 = int64_t(f0.y - f1.y) << SUBPIXEL_BITS, stepY2 = int64_t(f1.x - f0.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    float invArea = 1.0 / float(area2);");
        sb.AppendLine();
        sb.AppendLine("    for (int y = minY; y <= maxY; y++) {");
        sb.AppendLine("        int64_t w0 = rowW0, w1 = rowW1, w2 = rowW2;");
        sb.AppendLine("        for (int x = minX; x <= maxX; x++) {");
        sb.AppendLine("            if ((w0 | w1 | w2) >= 0) {");
        sb.AppendLine("                float z = (float(w0 - bias0) * z0 + float(w1 - bias1) * z1");
        sb.AppendLine("                         + float(w2 - bias2) * z2) * invArea;");
        sb.AppendLine("                float closeness = 1.0 - clamp(z, 0.0, 1.0);");
        sb.AppendLine("                uint64_t key = (uint64_t(uint(closeness * 4294967295.0)) << 32) | uint64_t(payload);");
        sb.AppendLine("                atomicMax(visibility[y * int(pc.viewport.x) + x], key);");
        sb.AppendLine("            }");
        sb.AppendLine("            w0 += stepX0; w1 += stepX1; w2 += stepX2;");
        sb.AppendLine("        }");
        sb.AppendLine("        rowW0 += stepY0; rowW1 += stepY1; rowW2 += stepY2;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("void main() {");
        sb.AppendLine("    uint meshletIndex = gl_WorkGroupID.x;");
        sb.AppendLine("    if (meshletIndex >= pc.meshletCount) return;");
        sb.AppendLine("    MeshletHeader h = headers[meshletIndex];");
        sb.AppendLine();
        sb.AppendLine("    // Phase 1 : transformer les sommets du meshlet en mémoire partagée.");
        sb.AppendLine("    for (uint v = gl_LocalInvocationID.x; v < h.vertexCount; v += gl_WorkGroupSize.x)");
        sb.AppendLine("        s_clip[v] = pc.viewProjection * vec4(positions[h.vertexOffset + v].xyz, 1.0);");
        sb.AppendLine("    barrier();");
        sb.AppendLine();
        sb.AppendLine("    // Phase 2 : un thread par triangle.");
        sb.AppendLine("    for (uint t = gl_LocalInvocationID.x; t < h.triangleCount; t += gl_WorkGroupSize.x) {");
        sb.AppendLine("        uint packed = packedTriangles[h.triangleOffset + t];");
        sb.AppendLine("        vec4 c0 = s_clip[packed & 0xFFu];");
        sb.AppendLine("        vec4 c1 = s_clip[(packed >> 8) & 0xFFu];");
        sb.AppendLine("        vec4 c2 = s_clip[(packed >> 16) & 0xFFu];");
        sb.AppendLine("        if (c0.w <= NEAR_CLIP_EPSILON || c1.w <= NEAR_CLIP_EPSILON || c2.w <= NEAR_CLIP_EPSILON)");
        sb.AppendLine("            continue; // rejet conservatif près du plan proche");
        sb.AppendLine();
        sb.AppendLine("        vec3 s0 = toScreen(c0), s1 = toScreen(c1), s2 = toScreen(c2);");
        sb.AppendLine("        ivec2 f0 = ivec2(round(s0.xy * float(SUBPIXEL_SCALE)));");
        sb.AppendLine("        ivec2 f1 = ivec2(round(s1.xy * float(SUBPIXEL_SCALE)));");
        sb.AppendLine("        ivec2 f2 = ivec2(round(s2.xy * float(SUBPIXEL_SCALE)));");
        sb.AppendLine("        uint payload = ((meshletIndex << 10) | t) + 1u;");
        sb.AppendLine("        rasterizeTriangle(f0, f1, f2, s0.z, s1.z, s2.z, payload);");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }
}
