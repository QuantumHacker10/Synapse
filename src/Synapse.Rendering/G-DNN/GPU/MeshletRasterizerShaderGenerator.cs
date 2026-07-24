using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using GDNN.Polygonization;

namespace GDNN.GPU;

/// <summary>
/// Émet le shader compute GLSL de rasterisation software des meshlets —
/// le même algorithme que <see cref="GDNN.Polygonization.SoftwareRasterizer"/>
/// (fonctions d'arête en virgule fixe 24.8, règle top-left, atomicMax
/// sur le visibility buffer), un workgroup par meshlet comme Nanite :
/// les sommets sont transformés en mémoire partagée puis les threads
/// rasterisent les triangles.
///
/// <see cref="GenerateGlsl"/> = référence int64 (VK_KHR_shader_atomic_int64).
/// <see cref="GenerateGlslR32"/> = chemin Vulkan déployé (atomicMax uint, portable).
/// </summary>
public static class MeshletRasterizerShaderGenerator
{
    public const int WorkgroupSize = 128;
    public const int MaxVerticesPerMeshlet = 64;
    public const int MaxTrianglesPerMeshlet = 124;
    public const int PushConstantBytes = 80; // mat4(64) + uvec2(8) + uint(4) + pad(4)

    public static string GenerateGlsl() => GenerateGlslCore(useInt64Visibility: true);

    /// <summary>
    /// Variante portable : depthKeys + payloads en R32UI (pas d'atomic int64).
    /// C'est le SPIR-V compilé et dispatché par <see cref="VulkanMeshletRasterizerDispatcher"/>.
    /// </summary>
    public static string GenerateGlslR32() => GenerateGlslCore(useInt64Visibility: false);

    public static bool TryGetSpirv(out byte[] spirv, out string entryPoint, out string log)
    {
        entryPoint = "main";
        string glsl = GenerateGlslR32();
        if (SpirvToolchain.TryCompileGlsl(glsl, out spirv, out log))
            return true;

        string hlsl = GenerateHlslR32();
        if (SpirvToolchain.TryCompileHlsl(hlsl, "CSMain", out spirv, out string hlslLog))
        {
            entryPoint = "CSMain";
            log = hlslLog;
            return true;
        }

        log = $"No SPIR-V compiler for meshlets. glslang: {log}; dxc: {hlslLog}";
        spirv = Array.Empty<byte>();
        return false;
    }

    /// <summary>Pack meshlets into GPU buffer layouts matching the compute shader.</summary>
    public static void PackMeshlets(
        NeuralPolygonMesh mesh,
        IReadOnlyList<NeuralMeshlet> meshlets,
        out uint[] headers,
        out float[] positions,
        out uint[] packedTriangles)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(meshlets);

        var posList = new List<float>(meshlets.Count * 16 * 4);
        var triList = new List<uint>(meshlets.Count * 32);
        headers = new uint[meshlets.Count * 4];

        for (int m = 0; m < meshlets.Count; m++)
        {
            var ml = meshlets[m];
            uint vertexOffset = (uint)(posList.Count / 4);
            for (int v = 0; v < ml.VertexIndices.Length; v++)
            {
                var p = mesh.Positions[ml.VertexIndices[v]];
                posList.Add(p.X);
                posList.Add(p.Y);
                posList.Add(p.Z);
                posList.Add(1f);
            }

            uint triangleOffset = (uint)triList.Count;
            int triCount = ml.LocalIndices.Length / 3;
            for (int t = 0; t < triCount; t++)
            {
                int o = t * 3;
                uint packed = ml.LocalIndices[o]
                             | ((uint)ml.LocalIndices[o + 1] << 8)
                             | ((uint)ml.LocalIndices[o + 2] << 16);
                triList.Add(packed);
            }

            int h = m * 4;
            headers[h] = vertexOffset;
            headers[h + 1] = (uint)ml.VertexIndices.Length;
            headers[h + 2] = triangleOffset;
            headers[h + 3] = (uint)triCount;
        }

        positions = posList.ToArray();
        packedTriangles = triList.Count > 0 ? triList.ToArray() : new uint[] { 0 };
        if (headers.Length == 0)
            headers = new uint[] { 0, 0, 0, 0 };
        if (positions.Length == 0)
            positions = new float[] { 0, 0, 0, 1 };
    }

    /// <summary>Push-constant blob: mat4 + uvec2 viewport + meshletCount (+ pad).</summary>
    public static byte[] PackPushConstants(Matrix4x4 viewProjection, uint width, uint height, uint meshletCount)
    {
        var bytes = new byte[PushConstantBytes];
        // System.Numerics row-major blit → GLSL mat4 column-major interprets as transpose,
        // which matches row-vector CPU transforms used elsewhere (v * M ≡ M^T * v).
        WriteFloat(bytes, 0, viewProjection.M11);
        WriteFloat(bytes, 4, viewProjection.M12);
        WriteFloat(bytes, 8, viewProjection.M13);
        WriteFloat(bytes, 12, viewProjection.M14);
        WriteFloat(bytes, 16, viewProjection.M21);
        WriteFloat(bytes, 20, viewProjection.M22);
        WriteFloat(bytes, 24, viewProjection.M23);
        WriteFloat(bytes, 28, viewProjection.M24);
        WriteFloat(bytes, 32, viewProjection.M31);
        WriteFloat(bytes, 36, viewProjection.M32);
        WriteFloat(bytes, 40, viewProjection.M33);
        WriteFloat(bytes, 44, viewProjection.M34);
        WriteFloat(bytes, 48, viewProjection.M41);
        WriteFloat(bytes, 52, viewProjection.M42);
        WriteFloat(bytes, 56, viewProjection.M43);
        WriteFloat(bytes, 60, viewProjection.M44);
        BitConverter.TryWriteBytes(bytes.AsSpan(64), width);
        BitConverter.TryWriteBytes(bytes.AsSpan(68), height);
        BitConverter.TryWriteBytes(bytes.AsSpan(72), meshletCount);
        return bytes;
    }

    private static void WriteFloat(byte[] dest, int offset, float value)
        => BitConverter.TryWriteBytes(dest.AsSpan(offset), value);

    private static string GenerateGlslCore(bool useInt64Visibility)
    {
        var sb = new StringBuilder();
        sb.AppendLine("#version 450");
        if (useInt64Visibility)
        {
            sb.AppendLine("#extension GL_EXT_shader_atomic_int64 : require");
            sb.AppendLine("#extension GL_ARB_gpu_shader_int64 : require");
        }
        sb.AppendLine();
        sb.AppendLine($"layout(local_size_x = {WorkgroupSize}) in;");
        sb.AppendLine();
        sb.AppendLine("layout(push_constant) uniform PushConstants {");
        sb.AppendLine("    mat4 viewProjection;");
        sb.AppendLine("    uvec2 viewport;");
        sb.AppendLine("    uint meshletCount;");
        sb.AppendLine("    uint _pad;");
        sb.AppendLine("} pc;");
        sb.AppendLine();
        sb.AppendLine("struct MeshletHeader { uint vertexOffset; uint vertexCount; uint triangleOffset; uint triangleCount; };");
        sb.AppendLine();
        sb.AppendLine("layout(std430, binding = 0) readonly buffer Headers   { MeshletHeader headers[]; };");
        sb.AppendLine("layout(std430, binding = 1) readonly buffer Positions { vec4 positions[]; };");
        sb.AppendLine("layout(std430, binding = 2) readonly buffer Triangles { uint packedTriangles[]; };");
        if (useInt64Visibility)
            sb.AppendLine("layout(std430, binding = 3) buffer Visibility { uint64_t visibility[]; };");
        else
        {
            sb.AppendLine("layout(std430, binding = 3) buffer DepthKeys { uint depthKeys[]; };");
            sb.AppendLine("layout(std430, binding = 4) buffer Payloads  { uint payloads[]; };");
        }
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
        sb.AppendLine("bool isTopLeftEdge(ivec2 p, ivec2 q) {");
        sb.AppendLine("    int dx = q.x - p.x, dy = q.y - p.y;");
        sb.AppendLine("    return dy < 0 || (dy == 0 && dx > 0);");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("void rasterizeTriangle(ivec2 f0, ivec2 f1, ivec2 f2,");
        sb.AppendLine("                       float z0, float z1, float z2, uint payload) {");
        if (useInt64Visibility)
        {
            sb.AppendLine("    int64_t area2 = int64_t(f1.x - f0.x) * int64_t(f2.y - f0.y)");
            sb.AppendLine("               - int64_t(f2.x - f0.x) * int64_t(f1.y - f0.y);");
            sb.AppendLine("    if (area2 >= 0) return;");
            sb.AppendLine("    ivec2 t = f1; f1 = f2; f2 = t;");
            sb.AppendLine("    float tz = z1; z1 = z2; z2 = tz;");
            sb.AppendLine("    area2 = -area2;");
            sb.AppendLine("    ivec2 minF = min(f0, min(f1, f2));");
            sb.AppendLine("    ivec2 maxF = max(f0, max(f1, f2));");
            sb.AppendLine("    int minX = max(0, minF.x >> SUBPIXEL_BITS);");
            sb.AppendLine("    int maxX = min(int(pc.viewport.x) - 1, (maxF.x + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
            sb.AppendLine("    int minY = max(0, minF.y >> SUBPIXEL_BITS);");
            sb.AppendLine("    int maxY = min(int(pc.viewport.y) - 1, (maxF.y + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
            sb.AppendLine("    if (minX > maxX || minY > maxY) return;");
            sb.AppendLine("    int64_t bias0 = isTopLeftEdge(f1, f2) ? int64_t(0) : int64_t(-1);");
            sb.AppendLine("    int64_t bias1 = isTopLeftEdge(f2, f0) ? int64_t(0) : int64_t(-1);");
            sb.AppendLine("    int64_t bias2 = isTopLeftEdge(f0, f1) ? int64_t(0) : int64_t(-1);");
            sb.AppendLine("    int64_t px = (int64_t(minX) << SUBPIXEL_BITS) + PIXEL_CENTER;");
            sb.AppendLine("    int64_t py = (int64_t(minY) << SUBPIXEL_BITS) + PIXEL_CENTER;");
            sb.AppendLine("    int64_t rowW0 = int64_t(f2.x - f1.x) * (py - f1.y) - int64_t(f2.y - f1.y) * (px - f1.x) + bias0;");
            sb.AppendLine("    int64_t rowW1 = int64_t(f0.x - f2.x) * (py - f2.y) - int64_t(f0.y - f2.y) * (px - f2.x) + bias1;");
            sb.AppendLine("    int64_t rowW2 = int64_t(f1.x - f0.x) * (py - f0.y) - int64_t(f1.y - f0.y) * (px - f0.x) + bias2;");
            sb.AppendLine("    int64_t stepX0 = int64_t(f1.y - f2.y) << SUBPIXEL_BITS, stepY0 = int64_t(f2.x - f1.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    int64_t stepX1 = int64_t(f2.y - f0.y) << SUBPIXEL_BITS, stepY1 = int64_t(f0.x - f2.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    int64_t stepX2 = int64_t(f0.y - f1.y) << SUBPIXEL_BITS, stepY2 = int64_t(f1.x - f0.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    float invArea = 1.0 / float(area2);");
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
        }
        else
        {
            // Portable R32 path: 32-bit edge math + atomicMax on closeness key.
            sb.AppendLine("    int area2 = (f1.x - f0.x) * (f2.y - f0.y) - (f2.x - f0.x) * (f1.y - f0.y);");
            sb.AppendLine("    if (area2 >= 0) return;");
            sb.AppendLine("    ivec2 t = f1; f1 = f2; f2 = t;");
            sb.AppendLine("    float tz = z1; z1 = z2; z2 = tz;");
            sb.AppendLine("    area2 = -area2;");
            sb.AppendLine("    ivec2 minF = min(f0, min(f1, f2));");
            sb.AppendLine("    ivec2 maxF = max(f0, max(f1, f2));");
            sb.AppendLine("    int minX = max(0, minF.x >> SUBPIXEL_BITS);");
            sb.AppendLine("    int maxX = min(int(pc.viewport.x) - 1, (maxF.x + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
            sb.AppendLine("    int minY = max(0, minF.y >> SUBPIXEL_BITS);");
            sb.AppendLine("    int maxY = min(int(pc.viewport.y) - 1, (maxF.y + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
            sb.AppendLine("    if (minX > maxX || minY > maxY) return;");
            sb.AppendLine("    int bias0 = isTopLeftEdge(f1, f2) ? 0 : -1;");
            sb.AppendLine("    int bias1 = isTopLeftEdge(f2, f0) ? 0 : -1;");
            sb.AppendLine("    int bias2 = isTopLeftEdge(f0, f1) ? 0 : -1;");
            sb.AppendLine("    int px = (minX << SUBPIXEL_BITS) + PIXEL_CENTER;");
            sb.AppendLine("    int py = (minY << SUBPIXEL_BITS) + PIXEL_CENTER;");
            sb.AppendLine("    int rowW0 = (f2.x - f1.x) * (py - f1.y) - (f2.y - f1.y) * (px - f1.x) + bias0;");
            sb.AppendLine("    int rowW1 = (f0.x - f2.x) * (py - f2.y) - (f0.y - f2.y) * (px - f2.x) + bias1;");
            sb.AppendLine("    int rowW2 = (f1.x - f0.x) * (py - f0.y) - (f1.y - f0.y) * (px - f0.x) + bias2;");
            sb.AppendLine("    int stepX0 = (f1.y - f2.y) << SUBPIXEL_BITS, stepY0 = (f2.x - f1.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    int stepX1 = (f2.y - f0.y) << SUBPIXEL_BITS, stepY1 = (f0.x - f2.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    int stepX2 = (f0.y - f1.y) << SUBPIXEL_BITS, stepY2 = (f1.x - f0.x) << SUBPIXEL_BITS;");
            sb.AppendLine("    float invArea = 1.0 / float(area2);");
            sb.AppendLine("    for (int y = minY; y <= maxY; y++) {");
            sb.AppendLine("        int w0 = rowW0, w1 = rowW1, w2 = rowW2;");
            sb.AppendLine("        for (int x = minX; x <= maxX; x++) {");
            sb.AppendLine("            if ((w0 | w1 | w2) >= 0) {");
            sb.AppendLine("                float z = (float(w0 - bias0) * z0 + float(w1 - bias1) * z1");
            sb.AppendLine("                         + float(w2 - bias2) * z2) * invArea;");
            sb.AppendLine("                float closeness = 1.0 - clamp(z, 0.0, 1.0);");
            sb.AppendLine("                uint key = uint(closeness * 4294967295.0);");
            sb.AppendLine("                int idx = y * int(pc.viewport.x) + x;");
            sb.AppendLine("                uint old = atomicMax(depthKeys[idx], key);");
            sb.AppendLine("                if (key >= old) payloads[idx] = payload;");
            sb.AppendLine("            }");
            sb.AppendLine("            w0 += stepX0; w1 += stepX1; w2 += stepX2;");
            sb.AppendLine("        }");
            sb.AppendLine("        rowW0 += stepY0; rowW1 += stepY1; rowW2 += stepY2;");
            sb.AppendLine("    }");
        }
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("void main() {");
        sb.AppendLine("    uint meshletIndex = gl_WorkGroupID.x;");
        sb.AppendLine("    if (meshletIndex >= pc.meshletCount) return;");
        sb.AppendLine("    MeshletHeader h = headers[meshletIndex];");
        sb.AppendLine("    for (uint v = gl_LocalInvocationID.x; v < h.vertexCount; v += gl_WorkGroupSize.x)");
        sb.AppendLine("        s_clip[v] = pc.viewProjection * vec4(positions[h.vertexOffset + v].xyz, 1.0);");
        sb.AppendLine("    barrier();");
        sb.AppendLine("    for (uint t = gl_LocalInvocationID.x; t < h.triangleCount; t += gl_WorkGroupSize.x) {");
        sb.AppendLine("        uint packed = packedTriangles[h.triangleOffset + t];");
        sb.AppendLine("        vec4 c0 = s_clip[packed & 0xFFu];");
        sb.AppendLine("        vec4 c1 = s_clip[(packed >> 8) & 0xFFu];");
        sb.AppendLine("        vec4 c2 = s_clip[(packed >> 16) & 0xFFu];");
        sb.AppendLine("        if (c0.w <= NEAR_CLIP_EPSILON || c1.w <= NEAR_CLIP_EPSILON || c2.w <= NEAR_CLIP_EPSILON)");
        sb.AppendLine("            continue;");
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

    /// <summary>HLSL R32 fallback for DXC → SPIR-V when glslang is absent.</summary>
    public static string GenerateHlslR32()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"#define WORKGROUP_SIZE {WorkgroupSize}");
        sb.AppendLine($"#define MAX_VERTS {MaxVerticesPerMeshlet}");
        sb.AppendLine("struct MeshletHeader { uint vertexOffset; uint vertexCount; uint triangleOffset; uint triangleCount; };");
        sb.AppendLine("struct PushConstants { float4x4 viewProjection; uint2 viewport; uint meshletCount; uint _pad; };");
        sb.AppendLine("[[vk::push_constant]] ConstantBuffer<PushConstants> pc;");
        sb.AppendLine("[[vk::binding(0, 0)]] StructuredBuffer<MeshletHeader> headers;");
        sb.AppendLine("[[vk::binding(1, 0)]] StructuredBuffer<float4> positions;");
        sb.AppendLine("[[vk::binding(2, 0)]] StructuredBuffer<uint> packedTriangles;");
        sb.AppendLine("[[vk::binding(3, 0)]] RWStructuredBuffer<uint> depthKeys;");
        sb.AppendLine("[[vk::binding(4, 0)]] RWStructuredBuffer<uint> payloads;");
        sb.AppendLine("groupshared float4 s_clip[MAX_VERTS];");
        sb.AppendLine("static const int SUBPIXEL_BITS = 8;");
        sb.AppendLine("static const int SUBPIXEL_SCALE = 1 << SUBPIXEL_BITS;");
        sb.AppendLine("static const int PIXEL_CENTER = SUBPIXEL_SCALE / 2;");
        sb.AppendLine("bool IsTopLeft(int2 p, int2 q) {");
        sb.AppendLine("    int dx = q.x - p.x, dy = q.y - p.y;");
        sb.AppendLine("    return dy < 0 || (dy == 0 && dx > 0);");
        sb.AppendLine("}");
        sb.AppendLine("float3 ToScreen(float4 clip) {");
        sb.AppendLine("    float invW = 1.0 / clip.w;");
        sb.AppendLine("    return float3((clip.x * invW * 0.5 + 0.5) * pc.viewport.x,");
        sb.AppendLine("                  (0.5 - clip.y * invW * 0.5) * pc.viewport.y,");
        sb.AppendLine("                  clip.z * invW);");
        sb.AppendLine("}");
        sb.AppendLine("void RasterizeTriangle(int2 f0, int2 f1, int2 f2, float z0, float z1, float z2, uint payload) {");
        sb.AppendLine("    int area2 = (f1.x - f0.x) * (f2.y - f0.y) - (f2.x - f0.x) * (f1.y - f0.y);");
        sb.AppendLine("    if (area2 >= 0) return;");
        sb.AppendLine("    int2 t = f1; f1 = f2; f2 = t; float tz = z1; z1 = z2; z2 = tz; area2 = -area2;");
        sb.AppendLine("    int2 minF = min(f0, min(f1, f2)); int2 maxF = max(f0, max(f1, f2));");
        sb.AppendLine("    int minX = max(0, minF.x >> SUBPIXEL_BITS);");
        sb.AppendLine("    int maxX = min((int)pc.viewport.x - 1, (maxF.x + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
        sb.AppendLine("    int minY = max(0, minF.y >> SUBPIXEL_BITS);");
        sb.AppendLine("    int maxY = min((int)pc.viewport.y - 1, (maxF.y + SUBPIXEL_SCALE - 1) >> SUBPIXEL_BITS);");
        sb.AppendLine("    if (minX > maxX || minY > maxY) return;");
        sb.AppendLine("    int bias0 = IsTopLeft(f1, f2) ? 0 : -1;");
        sb.AppendLine("    int bias1 = IsTopLeft(f2, f0) ? 0 : -1;");
        sb.AppendLine("    int bias2 = IsTopLeft(f0, f1) ? 0 : -1;");
        sb.AppendLine("    int px = (minX << SUBPIXEL_BITS) + PIXEL_CENTER;");
        sb.AppendLine("    int py = (minY << SUBPIXEL_BITS) + PIXEL_CENTER;");
        sb.AppendLine("    int rowW0 = (f2.x - f1.x) * (py - f1.y) - (f2.y - f1.y) * (px - f1.x) + bias0;");
        sb.AppendLine("    int rowW1 = (f0.x - f2.x) * (py - f2.y) - (f0.y - f2.y) * (px - f2.x) + bias1;");
        sb.AppendLine("    int rowW2 = (f1.x - f0.x) * (py - f0.y) - (f1.y - f0.y) * (px - f0.x) + bias2;");
        sb.AppendLine("    int stepX0 = (f1.y - f2.y) << SUBPIXEL_BITS, stepY0 = (f2.x - f1.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    int stepX1 = (f2.y - f0.y) << SUBPIXEL_BITS, stepY1 = (f0.x - f2.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    int stepX2 = (f0.y - f1.y) << SUBPIXEL_BITS, stepY2 = (f1.x - f0.x) << SUBPIXEL_BITS;");
        sb.AppendLine("    float invArea = 1.0 / (float)area2;");
        sb.AppendLine("    for (int y = minY; y <= maxY; y++) {");
        sb.AppendLine("        int w0 = rowW0, w1 = rowW1, w2 = rowW2;");
        sb.AppendLine("        for (int x = minX; x <= maxX; x++) {");
        sb.AppendLine("            if ((w0 | w1 | w2) >= 0) {");
        sb.AppendLine("                float z = ((float)(w0 - bias0) * z0 + (float)(w1 - bias1) * z1 + (float)(w2 - bias2) * z2) * invArea;");
        sb.AppendLine("                float closeness = 1.0 - saturate(z);");
        sb.AppendLine("                uint key = (uint)(closeness * 4294967295.0);");
        sb.AppendLine("                uint idx = (uint)(y * (int)pc.viewport.x + x);");
        sb.AppendLine("                uint old; InterlockedMax(depthKeys[idx], key, old);");
        sb.AppendLine("                if (key >= old) payloads[idx] = payload;");
        sb.AppendLine("            }");
        sb.AppendLine("            w0 += stepX0; w1 += stepX1; w2 += stepX2;");
        sb.AppendLine("        }");
        sb.AppendLine("        rowW0 += stepY0; rowW1 += stepY1; rowW2 += stepY2;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine("[numthreads(WORKGROUP_SIZE, 1, 1)]");
        sb.AppendLine("void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {");
        sb.AppendLine("    uint meshletIndex = gid.x;");
        sb.AppendLine("    if (meshletIndex >= pc.meshletCount) return;");
        sb.AppendLine("    MeshletHeader h = headers[meshletIndex];");
        sb.AppendLine("    for (uint v = lid.x; v < h.vertexCount; v += WORKGROUP_SIZE)");
        sb.AppendLine("        s_clip[v] = mul(pc.viewProjection, float4(positions[h.vertexOffset + v].xyz, 1.0));");
        sb.AppendLine("    GroupMemoryBarrierWithGroupSync();");
        sb.AppendLine("    for (uint t = lid.x; t < h.triangleCount; t += WORKGROUP_SIZE) {");
        sb.AppendLine("        uint packed = packedTriangles[h.triangleOffset + t];");
        sb.AppendLine("        float4 c0 = s_clip[packed & 0xFFu];");
        sb.AppendLine("        float4 c1 = s_clip[(packed >> 8) & 0xFFu];");
        sb.AppendLine("        float4 c2 = s_clip[(packed >> 16) & 0xFFu];");
        sb.AppendLine("        if (c0.w <= 1e-4 || c1.w <= 1e-4 || c2.w <= 1e-4) continue;");
        sb.AppendLine("        float3 s0 = ToScreen(c0), s1 = ToScreen(c1), s2 = ToScreen(c2);");
        sb.AppendLine("        int2 f0 = (int2)round(s0.xy * (float)SUBPIXEL_SCALE);");
        sb.AppendLine("        int2 f1 = (int2)round(s1.xy * (float)SUBPIXEL_SCALE);");
        sb.AppendLine("        int2 f2 = (int2)round(s2.xy * (float)SUBPIXEL_SCALE);");
        sb.AppendLine("        uint payload = ((meshletIndex << 10) | t) + 1u;");
        sb.AppendLine("        RasterizeTriangle(f0, f1, f2, s0.z, s1.z, s2.z, payload);");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }
}
