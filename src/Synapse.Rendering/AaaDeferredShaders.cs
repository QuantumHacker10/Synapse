// =============================================================================
// AaaDeferredShaders.cs — deferred lighting + G-buffer + HDR post SPIR-V
// Cook-Torrance GGX, hemisphere ambient, PCF shadows; tonemap/bloom is a separate pass.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Text;

namespace GDNN.Rendering.Shaders
{
    /// <summary>Deferred shaders for the Vulkan present path (HDR lighting + LDR post).</summary>
    public static class AaaDeferredShaders
    {
        /// <summary>
        /// G-Buffer VS: world pos/normal/UV/color + screen-space velocity from prev/current VP.
        /// Binding 0 = CameraUBO { mat4 vp; mat4 prevVp }, binding 1 = model.
        /// </summary>
        public static byte[] CompileGBufferVertex()
        {
            var b = new Spirv();
            b.CapShader();
            b.ImportGlsl();
            b.MemoryModel();

            uint tVoid = b.TVoid();
            uint tF = b.TFloat();
            uint tU = b.TIntU();
            uint tV2 = b.TVec(tF, 2);
            uint tV3 = b.TVec(tF, 3);
            uint tV4 = b.TVec(tF, 4);
            uint tM4 = b.TMat(tV4, 4);

            uint tCam = b.TStruct(tM4, tM4);
            b.DecBlock(tCam);
            b.DecMemberOffset(tCam, 0, 0); b.DecMemberMatrix(tCam, 0);
            b.DecMemberOffset(tCam, 1, 64); b.DecMemberMatrix(tCam, 1);

            uint pInV3 = b.TPtr(Spirv.Input, tV3);
            uint pInV2 = b.TPtr(Spirv.Input, tV2);
            uint pOutV3 = b.TPtr(Spirv.Output, tV3);
            uint pOutV2 = b.TPtr(Spirv.Output, tV2);
            uint pOutV4 = b.TPtr(Spirv.Output, tV4);
            uint pUCam = b.TPtr(Spirv.Uniform, tCam);
            uint pUM4 = b.TPtr(Spirv.Uniform, tM4);
            uint pUV4 = b.TPtr(Spirv.Uniform, tV4);

            b.BeginMain(tVoid, fragment: false);

            uint inPos = b.Var(pInV3, Spirv.Input); b.DecLocation(inPos, 0);
            uint inNrm = b.Var(pInV3, Spirv.Input); b.DecLocation(inNrm, 1);
            uint inUV = b.Var(pInV2, Spirv.Input); b.DecLocation(inUV, 2);
            uint inCol = b.Var(pInV3, Spirv.Input); b.DecLocation(inCol, 3);

            uint oPos = b.Var(pOutV3, Spirv.Output); b.DecLocation(oPos, 0);
            uint oNrm = b.Var(pOutV3, Spirv.Output); b.DecLocation(oNrm, 1);
            uint oUV = b.Var(pOutV2, Spirv.Output); b.DecLocation(oUV, 2);
            uint oCol = b.Var(pOutV3, Spirv.Output); b.DecLocation(oCol, 3);
            uint oVel = b.Var(pOutV2, Spirv.Output); b.DecLocation(oVel, 4);
            uint glPos = b.Var(pOutV4, Spirv.Output); b.DecBuiltIn(glPos, 0); // Position

            uint cam = b.Var(pUCam, Spirv.Uniform); b.DecSetBinding(cam, 0, 0);
            uint modelU = b.Var(pUM4, Spirv.Uniform); b.DecSetBinding(modelU, 0, 1);

            uint glsl = b.GlslId;
            b.Label();
            uint one = b.F(tF, 1f);
            uint zero = b.F(tF, 0f);
            uint half = b.F(tF, 0.5f);
            uint eps = b.F(tF, 1e-4f);

            uint pos = b.Load(tV3, inPos);
            uint pos4 = b.Comp(tV4, b.Extract(tF, pos, 0), b.Extract(tF, pos, 1), b.Extract(tF, pos, 2), one);
            uint model = b.Load(tM4, modelU);
            uint world4 = b.MulVM(tV4, pos4, model);
            uint wp = b.Comp(tV3, b.Extract(tF, world4, 0), b.Extract(tF, world4, 1), b.Extract(tF, world4, 2));
            b.Store(oPos, wp);

            uint vp = b.Load(tM4, b.Access(pUM4, cam, b.U(tU, 0)));
            uint prevVp = b.Load(tM4, b.Access(pUM4, cam, b.U(tU, 1)));
            // Correct clip: world * ViewProjection (fixes model transform on-screen).
            uint clip = b.MulVM(tV4, world4, vp);
            uint prevClip = b.MulVM(tV4, world4, prevVp);
            b.Store(glPos, clip);

            uint nrm = b.Load(tV3, inNrm);
            uint n4 = b.MulVM(tV4, b.Comp(tV4,
                b.Extract(tF, nrm, 0), b.Extract(tF, nrm, 1), b.Extract(tF, nrm, 2), zero), model);
            b.Store(oNrm, b.Comp(tV3, b.Extract(tF, n4, 0), b.Extract(tF, n4, 1), b.Extract(tF, n4, 2)));
            b.Store(oUV, b.Load(tV2, inUV));
            b.Store(oCol, b.Load(tV3, inCol));

            // NDC velocity: curr.xy/w - prev.xy/w.
            uint cw = b.Extract(tF, clip, 3);
            uint pw = b.Extract(tF, prevClip, 3);
            uint invCw = b.FDiv(tF, one, b.FMax(glsl, tF, b.FAbs(glsl, tF, cw), eps));
            uint invPw = b.FDiv(tF, one, b.FMax(glsl, tF, b.FAbs(glsl, tF, pw), eps));
            uint cNdcX = b.FMul(tF, b.Extract(tF, clip, 0), invCw);
            uint cNdcY = b.FMul(tF, b.Extract(tF, clip, 1), invCw);
            uint pNdcX = b.FMul(tF, b.Extract(tF, prevClip, 0), invPw);
            uint pNdcY = b.FMul(tF, b.Extract(tF, prevClip, 1), invPw);
            b.Store(oVel, b.Comp(tV2, b.FSub(tF, cNdcX, pNdcX), b.FSub(tF, cNdcY, pNdcY)));

            _ = (pUV4, half);
            b.Return();
            b.EndMain();
            return b.Build();
        }

        /// <summary>
        /// G-Buffer FS: samples albedo/normal/ORM (bindings 3–5) × MaterialUBO,
        /// writes velocity from VS. Albedo.a = emissive strength.
        /// </summary>
        public static byte[] CompileGBufferFragment()
        {
            var b = new Spirv();
            b.CapShader();
            b.ImportGlsl();
            b.MemoryModel();

            uint tVoid = b.TVoid();
            uint tF = b.TFloat();
            uint tU = b.TIntU();
            uint tV2 = b.TVec(tF, 2);
            uint tV3 = b.TVec(tF, 3);
            uint tV4 = b.TVec(tF, 4);
            uint tImg = b.TImage(tF, depth: false);
            uint tSImg = b.TSampled(tImg);

            uint tMat = b.TStruct(tV4, tV4, tF, tF, tF, tF, tF, tF, tF, tF, tF, tF);
            b.DecBlock(tMat);
            for (uint i = 0; i < 12; i++)
            {
                uint off = i switch
                {
                    0 => 0u,
                    1 => 16u,
                    2 => 32u,
                    3 => 36u,
                    4 => 40u,
                    5 => 44u,
                    6 => 48u,
                    7 => 52u,
                    8 => 56u,
                    9 => 60u,
                    10 => 64u,
                    _ => 68u
                };
                b.DecMemberOffset(tMat, i, off);
            }

            uint pInV3 = b.TPtr(Spirv.Input, tV3);
            uint pInV2 = b.TPtr(Spirv.Input, tV2);
            uint pOutV4 = b.TPtr(Spirv.Output, tV4);
            uint pUMat = b.TPtr(Spirv.Uniform, tMat);
            uint pUF = b.TPtr(Spirv.Uniform, tF);
            uint pUV4 = b.TPtr(Spirv.Uniform, tV4);
            uint pSI = b.TPtr(Spirv.UniformConstant, tSImg);

            b.BeginMain(tVoid, fragment: true);

            uint inPos = b.Var(pInV3, Spirv.Input); b.DecLocation(inPos, 0);
            uint inNrm = b.Var(pInV3, Spirv.Input); b.DecLocation(inNrm, 1);
            uint inUV = b.Var(pInV2, Spirv.Input); b.DecLocation(inUV, 2);
            uint inCol = b.Var(pInV3, Spirv.Input); b.DecLocation(inCol, 3);
            uint inVel = b.Var(pInV2, Spirv.Input); b.DecLocation(inVel, 4);

            uint oAlb = b.Var(pOutV4, Spirv.Output); b.DecLocation(oAlb, 0);
            uint oNrm = b.Var(pOutV4, Spirv.Output); b.DecLocation(oNrm, 1);
            uint oPos = b.Var(pOutV4, Spirv.Output); b.DecLocation(oPos, 2);
            uint oMat = b.Var(pOutV4, Spirv.Output); b.DecLocation(oMat, 3);
            uint oVel = b.Var(pOutV4, Spirv.Output); b.DecLocation(oVel, 4);

            uint ubo = b.Var(pUMat, Spirv.Uniform); b.DecSetBinding(ubo, 0, 2);
            uint texAlb = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texAlb, 0, 3);
            uint texNrm = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texNrm, 0, 4);
            uint texOrm = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texOrm, 0, 5);

            uint glsl = b.GlslId;
            b.Label();
            uint one = b.F(tF, 1f);
            uint zero = b.F(tF, 0f);
            uint uv = b.Load(tV2, inUV);

            uint base4 = b.Load(tV4, b.Access(pUV4, ubo, b.U(tU, 0)));
            uint baseRgb = b.Rgb(tF, tV3, base4);
            uint texA = b.Rgb(tF, tV3, b.Sample(tV4, b.Load(tSImg, texAlb), uv));
            uint albedo = b.FMul(tV3, baseRgb, texA);
            uint vertCol = b.Load(tV3, inCol);
            albedo = b.Mix(glsl, tV3, albedo, vertCol, b.Splat(tV3, b.F(tF, 0.25f)));

            uint emissive = b.Load(tV4, b.Access(pUV4, ubo, b.U(tU, 1)));
            uint eStr = b.Extract(tF, emissive, 3);
            b.Store(oAlb, b.Comp(tV4,
                b.Extract(tF, albedo, 0), b.Extract(tF, albedo, 1), b.Extract(tF, albedo, 2), eStr));

            // Tangent-ish normal map: remap [0,1] → [-1,1] and blend with geometric normal.
            uint nGeo = b.Normalize(glsl, tV3, b.Load(tV3, inNrm));
            uint nMap = b.Rgb(tF, tV3, b.Sample(tV4, b.Load(tSImg, texNrm), uv));
            uint nPert = b.Comp(tV3,
                b.FSub(tF, b.FMul(tF, b.Extract(tF, nMap, 0), b.F(tF, 2f)), one),
                b.FSub(tF, b.FMul(tF, b.Extract(tF, nMap, 1), b.F(tF, 2f)), one),
                b.Extract(tF, nMap, 2));
            uint nScale = b.Load(tF, b.Access(pUF, ubo, b.U(tU, 5)));
            uint nBlend = b.Normalize(glsl, tV3, b.FAdd(tV3, nGeo,
                b.Scale(tV3, nPert, b.FMul(tF, nScale, b.F(tF, 0.35f)))));
            b.Store(oNrm, b.Comp(tV4,
                b.Extract(tF, nBlend, 0), b.Extract(tF, nBlend, 1), b.Extract(tF, nBlend, 2), one));

            uint wp = b.Load(tV3, inPos);
            b.Store(oPos, b.Comp(tV4,
                b.Extract(tF, wp, 0), b.Extract(tF, wp, 1), b.Extract(tF, wp, 2), one));

            uint orm = b.Sample(tV4, b.Load(tSImg, texOrm), uv);
            uint rough = b.FMul(tF, b.Load(tF, b.Access(pUF, ubo, b.U(tU, 2))), b.Extract(tF, orm, 1));
            uint metal = b.FMul(tF, b.Load(tF, b.Access(pUF, ubo, b.U(tU, 3))), b.Extract(tF, orm, 2));
            uint ao = b.FMul(tF, b.Load(tF, b.Access(pUF, ubo, b.U(tU, 4))), b.Extract(tF, orm, 0));
            uint clearCoat = b.Load(tF, b.Access(pUF, ubo, b.U(tU, 9)));
            b.Store(oMat, b.Comp(tV4, rough, metal, ao, clearCoat));

            uint vel = b.Load(tV2, inVel);
            b.Store(oVel, b.Comp(tV4, b.Extract(tF, vel, 0), b.Extract(tF, vel, 1), zero, one));

            b.Return();
            b.EndMain();
            return b.Build();
        }

        /// <summary>
        /// Deferred lighting FS: GGX PBR + ClearCoat lobe, hemisphere + GI + fog, 4-tap PCF,
        /// up to 4 directional/point lights (baked constants). Writes linear HDR (no tonemap).
        /// Bindings 0–4 G-buffer+GI, 5 shadow, 6 lightVP, 7 SSAO, 8 fog, 9–10 cascades.
        /// </summary>
        public static byte[] CompileLightingFragment(
            float lx, float ly, float lz,
            float ambient, float giBoost,
            float camX, float camY, float camZ,
            float lightIntensity = 3.2f,
            float bloomStrength = 0.3f,
            ReadOnlySpan<float> extraLights = default)
        {
            // Pack extra lights as [dx,dy,dz, r,g,b, intensity, range] × N (max 3 extras → 4 total).
            Span<float> lights = stackalloc float[32];
            lights[0] = lx; lights[1] = ly; lights[2] = lz;
            lights[3] = 1f; lights[4] = 0.97f; lights[5] = 0.92f;
            lights[6] = lightIntensity; lights[7] = 0f; // range 0 = directional
            int lightCount = 1;
            if (!extraLights.IsEmpty)
            {
                int n = Math.Min(3, extraLights.Length / 8);
                for (int i = 0; i < n; i++)
                {
                    int s = i * 8;
                    int d = lightCount * 8;
                    for (int c = 0; c < 8; c++)
                        lights[d + c] = extraLights[s + c];
                    lightCount++;
                }
            }

            var b = new Spirv();
            b.CapShader();
            b.ImportGlsl();
            b.MemoryModel();

            uint tVoid = b.TVoid();
            uint tF = b.TFloat();
            uint tU = b.TIntU();
            uint tBool = b.TBool();
            uint tV2 = b.TVec(tF, 2);
            uint tV3 = b.TVec(tF, 3);
            uint tV4 = b.TVec(tF, 4);
            uint tM4 = b.TMat(tV4, 4);

            uint tImg = b.TImage(tF, depth: false);
            uint tSImg = b.TSampled(tImg);
            uint tDImg = b.TImage(tF, depth: true);
            uint tSDImg = b.TSampled(tDImg);

            // Cascade0/1/2 VP + vec4 splits (x=near, y=mid, z=cascadeCount, w=unused).
            uint tLight = b.TStruct(tM4, tM4, tM4, tV4);
            b.DecBlock(tLight);
            b.DecMemberOffset(tLight, 0, 0); b.DecMemberMatrix(tLight, 0);
            b.DecMemberOffset(tLight, 1, 64); b.DecMemberMatrix(tLight, 1);
            b.DecMemberOffset(tLight, 2, 128); b.DecMemberMatrix(tLight, 2);
            b.DecMemberOffset(tLight, 3, 192);

            uint pInV2 = b.TPtr(Spirv.Input, tV2);
            uint pOutV4 = b.TPtr(Spirv.Output, tV4);
            uint pSI = b.TPtr(Spirv.UniformConstant, tSImg);
            uint pSD = b.TPtr(Spirv.UniformConstant, tSDImg);
            uint pULight = b.TPtr(Spirv.Uniform, tLight);
            uint pUM4 = b.TPtr(Spirv.Uniform, tM4);
            uint pUV4 = b.TPtr(Spirv.Uniform, tV4);

            b.BeginMain(tVoid, fragment: true);

            uint inUV = b.Var(pInV2, Spirv.Input); b.DecLocation(inUV, 0);
            uint tex0 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex0, 0, 0);
            uint tex1 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex1, 0, 1);
            uint tex2 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex2, 0, 2);
            uint tex3 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex3, 0, 3);
            uint tex4 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex4, 0, 4);
            uint tex5 = b.Var(pSD, Spirv.UniformConstant); b.DecSetBinding(tex5, 0, 5);
            uint lightUbo = b.Var(pULight, Spirv.Uniform); b.DecSetBinding(lightUbo, 0, 6);
            uint texAo = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texAo, 0, 7);
            uint texFog = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texFog, 0, 8);
            uint texC1 = b.Var(pSD, Spirv.UniformConstant); b.DecSetBinding(texC1, 0, 9);
            uint texC2 = b.Var(pSD, Spirv.UniformConstant); b.DecSetBinding(texC2, 0, 10);
            uint outC = b.Var(pOutV4, Spirv.Output); b.DecLocation(outC, 0);

            uint glsl = b.GlslId;
            b.Label();

            uint uv = b.Load(tV2, inUV);
            uint one = b.F(tF, 1f);
            uint zero = b.F(tF, 0f);
            uint half = b.F(tF, 0.5f);
            uint eps = b.F(tF, 1e-4f);
            uint pi = b.F(tF, 3.14159265f);

            uint albedo4 = b.Sample(tV4, b.Load(tSImg, tex0), uv);
            uint normal4 = b.Sample(tV4, b.Load(tSImg, tex1), uv);
            uint world4 = b.Sample(tV4, b.Load(tSImg, tex2), uv);
            uint mat4v = b.Sample(tV4, b.Load(tSImg, tex3), uv);
            uint gi4 = b.Sample(tV4, b.Load(tSImg, tex4), uv);
            uint ao4 = b.Sample(tV4, b.Load(tSImg, texAo), uv);
            uint fog4 = b.Sample(tV4, b.Load(tSImg, texFog), uv);

            uint albedo = b.Rgb(tF, tV3, albedo4);
            uint N = b.Normalize(glsl, tV3, b.Rgb(tF, tV3, normal4));
            uint P = b.Rgb(tF, tV3, world4);
            uint roughness = b.Clamp(glsl, tF, b.Extract(tF, mat4v, 0), b.F(tF, 0.045f), one);
            uint metallic = b.Clamp(glsl, tF, b.Extract(tF, mat4v, 1), zero, one);
            uint matAo = b.Clamp(glsl, tF, b.Extract(tF, mat4v, 2), zero, one);
            uint clearCoat = b.Clamp(glsl, tF, b.Extract(tF, mat4v, 3), zero, one);
            uint emissiveI = b.Extract(tF, albedo4, 3);
            uint ssao = b.Clamp(glsl, tF, b.Extract(tF, ao4, 0), zero, one);
            uint ao = b.FMul(tF, matAo, ssao);

            uint cam = b.Comp(tV3, b.F(tF, camX), b.F(tF, camY), b.F(tF, camZ));
            uint V = b.Normalize(glsl, tV3, b.FSub(tV3, cam, P));

            uint lightVP0 = b.Load(tM4, b.Access(pUM4, lightUbo, b.U(tU, 0)));
            uint lightVP1 = b.Load(tM4, b.Access(pUM4, lightUbo, b.U(tU, 1)));
            uint lightVP2 = b.Load(tM4, b.Access(pUM4, lightUbo, b.U(tU, 2)));
            uint splits = b.Load(tV4, b.Access(pUV4, lightUbo, b.U(tU, 3)));
            uint split0 = b.Extract(tF, splits, 0);
            uint split1 = b.Extract(tF, splits, 1);

            uint dist = b.Length(glsl, tF, b.FSub(tV3, P, cam));
            uint sh0 = CascadeShadow(b, glsl, tF, tV2, tV3, tV4, tBool, tU, tSDImg,
                b.Load(tSDImg, tex5), lightVP0, P, one, half, zero, eps);
            uint sh1 = CascadeShadow(b, glsl, tF, tV2, tV3, tV4, tBool, tU, tSDImg,
                b.Load(tSDImg, texC1), lightVP1, P, one, half, zero, eps);
            uint sh2 = CascadeShadow(b, glsl, tF, tV2, tV3, tV4, tBool, tU, tSDImg,
                b.Load(tSDImg, texC2), lightVP2, P, one, half, zero, eps);

            uint useNear = b.FOrdLt(tBool, dist, split0);
            uint useMid = b.And(tBool, b.FOrdGe(tBool, dist, split0), b.FOrdLt(tBool, dist, split1));
            uint sh = b.Select(tF, useNear, sh0, b.Select(tF, useMid, sh1, sh2));

            uint ones = b.Splat(tV3, one);
            uint f0d = b.Comp(tV3, b.F(tF, 0.04f), b.F(tF, 0.04f), b.F(tF, 0.04f));
            uint F0 = b.Mix(glsl, tV3, f0d, albedo, b.Splat(tV3, metallic));
            uint a = b.FMul(tF, roughness, roughness);
            uint a2 = b.FMul(tF, a, a);
            uint NdotV = b.FMax(glsl, tF, b.Dot(tF, N, V), eps);
            uint rp1 = b.FAdd(tF, roughness, one);
            uint k = b.FDiv(tF, b.FMul(tF, rp1, rp1), b.F(tF, 8f));

            uint direct = b.Splat(tV3, zero);

            for (int li = 0; li < lightCount; li++)
            {
                int o = li * 8;
                uint L = b.Normalize(glsl, tV3, b.Comp(tV3, b.F(tF, lights[o]), b.F(tF, lights[o + 1]), b.F(tF, lights[o + 2])));
                uint lightCol = b.Scale(tV3,
                    b.Comp(tV3, b.F(tF, lights[o + 3]), b.F(tF, lights[o + 4]), b.F(tF, lights[o + 5])),
                    b.F(tF, lights[o + 6]));
                float range = lights[o + 7];
                uint atten = one;
                if (range > 0.01f)
                {
                    // Point light: xyz = position relative to camera, w-slot = range.
                    uint toLight = b.Comp(tV3,
                        b.F(tF, lights[o]),
                        b.F(tF, lights[o + 1]),
                        b.F(tF, lights[o + 2]));
                    uint lightPos = b.FAdd(tV3, cam, toLight);
                    uint delta = b.FSub(tV3, lightPos, P);
                    uint d = b.Length(glsl, tF, delta);
                    L = b.Normalize(glsl, tV3, delta);
                    uint r = b.F(tF, range);
                    uint nd = b.FDiv(tF, d, r);
                    atten = b.FMax(glsl, tF, b.FSub(tF, one, b.FMul(tF, nd, nd)), zero);
                    atten = b.FMul(tF, atten, atten);
                }

                uint H = b.Normalize(glsl, tV3, b.FAdd(tV3, L, V));
                uint NdotL = b.FMax(glsl, tF, b.Dot(tF, N, L), zero);
                uint NdotH = b.FMax(glsl, tF, b.Dot(tF, N, H), zero);
                uint VdotH = b.FMax(glsl, tF, b.Dot(tF, V, H), zero);

                uint nh2 = b.FMul(tF, NdotH, NdotH);
                uint denom = b.FAdd(tF, b.FMul(tF, nh2, b.FSub(tF, a2, one)), one);
                uint D = b.FDiv(tF, a2, b.FMul(tF, pi, b.FMul(tF, denom, denom)));

                uint omvh = b.FSub(tF, one, VdotH);
                uint vh2 = b.FMul(tF, omvh, omvh);
                uint vh5 = b.FMul(tF, b.FMul(tF, vh2, vh2), omvh);
                uint F = b.FAdd(tV3, F0, b.Scale(tV3, b.FSub(tV3, ones, F0), vh5));

                uint gv = b.FDiv(tF, NdotV, b.FAdd(tF, b.FMul(tF, NdotV, b.FSub(tF, one, k)), k));
                uint gl_ = b.FDiv(tF, NdotL, b.FAdd(tF, b.FMul(tF, NdotL, b.FSub(tF, one, k)), k));
                uint G = b.FMul(tF, gv, gl_);

                uint spec = b.Scale(tV3, F, b.FMul(tF, D, G));
                spec = b.Scale(tV3, spec, b.FDiv(tF, one,
                    b.FMax(glsl, tF, b.FMul(tF, b.F(tF, 4f), b.FMul(tF, NdotV, NdotL)), eps)));

                uint kd = b.Scale(tV3, b.FSub(tV3, ones, F), b.FSub(tF, one, metallic));
                uint diffuse = b.Scale(tV3, b.FMul(tV3, kd, albedo), b.FDiv(tF, one, pi));

                uint shadowTerm = li == 0 ? sh : one;
                uint lobe = b.Scale(tV3, b.FAdd(tV3, diffuse, spec), b.FMul(tF, NdotL, b.FMul(tF, shadowTerm, atten)));
                lobe = b.FMul(tV3, lobe, lightCol);
                direct = b.FAdd(tV3, direct, lobe);

                if (li == 0)
                {
                    // ClearCoat on primary light only.
                    uint coatRough = b.F(tF, 0.1f);
                    uint ca = b.FMul(tF, coatRough, coatRough);
                    uint ca2 = b.FMul(tF, ca, ca);
                    uint cDenom = b.FAdd(tF, b.FMul(tF, nh2, b.FSub(tF, ca2, one)), one);
                    uint cD = b.FDiv(tF, ca2, b.FMul(tF, pi, b.FMul(tF, cDenom, cDenom)));
                    uint cF0 = b.Comp(tV3, b.F(tF, 0.04f), b.F(tF, 0.04f), b.F(tF, 0.04f));
                    uint cF = b.FAdd(tV3, cF0, b.Scale(tV3, b.FSub(tV3, ones, cF0), vh5));
                    uint crp1 = b.FAdd(tF, coatRough, one);
                    uint ck = b.FDiv(tF, b.FMul(tF, crp1, crp1), b.F(tF, 8f));
                    uint cgv = b.FDiv(tF, NdotV, b.FAdd(tF, b.FMul(tF, NdotV, b.FSub(tF, one, ck)), ck));
                    uint cgl = b.FDiv(tF, NdotL, b.FAdd(tF, b.FMul(tF, NdotL, b.FSub(tF, one, ck)), ck));
                    uint cG = b.FMul(tF, cgv, cgl);
                    uint coatSpec = b.Scale(tV3, cF, b.FMul(tF, cD, cG));
                    coatSpec = b.Scale(tV3, coatSpec, b.FDiv(tF, one,
                        b.FMax(glsl, tF, b.FMul(tF, b.F(tF, 4f), b.FMul(tF, NdotV, NdotL)), eps)));
                    coatSpec = b.Scale(tV3, coatSpec, b.FMul(tF, clearCoat, b.FMul(tF, NdotL, sh)));
                    direct = b.FAdd(tV3, direct, b.FMul(tV3, coatSpec, lightCol));
                }
            }

            uint ny = b.FAdd(tF, b.FMul(tF, b.Extract(tF, N, 1), half), half);
            uint sky = b.Comp(tV3, b.F(tF, 0.42f), b.F(tF, 0.55f), b.F(tF, 0.78f));
            uint ground = b.Comp(tV3, b.F(tF, 0.12f), b.F(tF, 0.10f), b.F(tF, 0.08f));
            uint hemi = b.Mix(glsl, tV3, ground, sky, b.Splat(tV3, ny));
            // Flat ambient kept tiny — L-DNN GI texture is the main ambient/indirect term (Lumen-like).
            uint ambScale = b.F(tF, ambient + 0.045f);
            uint ambientRgb = b.Scale(tV3, b.FMul(tV3, albedo, hemi), b.FMul(tF, ambScale, ao));

            // Strong multi-bounce GI: irradiance × albedo × AO, plus a cheap 2nd-bounce proxy.
            uint giRgb = b.Rgb(tF, tV3, gi4);
            uint gi = b.Scale(tV3, b.FMul(tV3, giRgb, albedo), ao);
            gi = b.Scale(tV3, gi, b.F(tF, 2.85f + giBoost * 0.9f));
            uint giLum = b.Dot(tF, giRgb, b.Comp(tV3, b.F(tF, 0.299f), b.F(tF, 0.587f), b.F(tF, 0.114f)));
            uint bounce2 = b.Scale(tV3, b.FMul(tV3, gi, albedo), b.FMul(tF, giLum, b.F(tF, 0.35f)));
            gi = b.FAdd(tV3, gi, bounce2);

            uint invV = b.Scale(tV3, V, b.F(tF, -1f));
            uint R = b.Normalize(glsl, tV3, b.FSub(tV3, invV, b.Scale(tV3, N, b.FMul(tF, b.F(tF, 2f), b.Dot(tF, N, invV)))));
            uint ry = b.FAdd(tF, b.FMul(tF, b.Extract(tF, R, 1), half), half);
            uint env = b.Mix(glsl, tV3, ground, sky, b.Splat(tV3, ry));
            uint gloss = b.FMul(tF, b.FSub(tF, one, roughness), b.FSub(tF, one, roughness));
            uint iblSpec = b.Scale(tV3, b.FMul(tV3, env, F0), b.FMul(tF, gloss, ao));
            // Stronger SSR / reflection proxy from GI (Lumen-like specular).
            uint rOff = b.Scale(tV2, b.Comp(tV2, b.Extract(tF, R, 0), b.Extract(tF, R, 2)), b.FMul(tF, b.F(tF, 0.12f), gloss));
            uint ssrUv = b.FAdd(tV2, uv, rOff);
            uint ssrGi = b.Rgb(tF, tV3, b.Sample(tV4, b.Load(tSImg, tex4), ssrUv));
            uint ssr = b.Scale(tV3, b.FMul(tV3, ssrGi, F0), b.FMul(tF, gloss, b.F(tF, 0.95f)));

            uint emissive = b.Scale(tV3, albedo, emissiveI);
            uint hdr = b.FAdd(tV3, b.FAdd(tV3, b.FAdd(tV3, direct, ambientRgb), gi), emissive);
            hdr = b.FAdd(tV3, hdr, iblSpec);
            hdr = b.FAdd(tV3, hdr, ssr);
            hdr = b.FAdd(tV3, hdr, b.Rgb(tF, tV3, fog4));

            _ = bloomStrength;
            b.Store(outC, b.Comp(tV4,
                b.Extract(tF, hdr, 0), b.Extract(tF, hdr, 1), b.Extract(tF, hdr, 2), one));
            b.Return();
            b.EndMain();
            return b.Build();
        }

        /// <summary>
        /// Fullscreen HDR post: multi-tap threshold bloom + ACES + soft gamma → LDR,
        /// plus temporal AA using velocity (binding 1) and history LDR (binding 2).
        /// Binding 0 = HDR scene. <paramref name="texelW"/>/<paramref name="texelH"/> = 1/resolution.
        /// </summary>
        public static byte[] CompileHdrPostFragment(
            float bloomStrength = 0.42f,
            float exposure = 1.2f,
            float texelW = 1f / 1920f,
            float texelH = 1f / 1080f,
            float taaBlend = 0.88f)
        {
            var b = new Spirv();
            b.CapShader();
            b.ImportGlsl();
            b.MemoryModel();

            uint tVoid = b.TVoid();
            uint tF = b.TFloat();
            uint tV2 = b.TVec(tF, 2);
            uint tV3 = b.TVec(tF, 3);
            uint tV4 = b.TVec(tF, 4);
            uint tImg = b.TImage(tF, depth: false);
            uint tSImg = b.TSampled(tImg);
            uint pInV2 = b.TPtr(Spirv.Input, tV2);
            uint pOutV4 = b.TPtr(Spirv.Output, tV4);
            uint pSI = b.TPtr(Spirv.UniformConstant, tSImg);

            b.BeginMain(tVoid, fragment: true);
            uint inUV = b.Var(pInV2, Spirv.Input); b.DecLocation(inUV, 0);
            uint tex0 = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(tex0, 0, 0);
            uint texVel = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texVel, 0, 1);
            uint texHist = b.Var(pSI, Spirv.UniformConstant); b.DecSetBinding(texHist, 0, 2);
            uint outC = b.Var(pOutV4, Spirv.Output); b.DecLocation(outC, 0);

            uint glsl = b.GlslId;
            b.Label();
            uint uv = b.Load(tV2, inUV);
            uint one = b.F(tF, 1f);
            uint zero = b.F(tF, 0f);
            uint samp = b.Load(tSImg, tex0);

            uint SampleRgb(uint du, uint dv)
            {
                uint c = b.Sample(tV4, samp, b.Comp(tV2, b.FAdd(tF, b.Extract(tF, uv, 0), du), b.FAdd(tF, b.Extract(tF, uv, 1), dv)));
                return b.Rgb(tF, tV3, c);
            }

            uint Bright(uint rgb)
            {
                uint lum = b.Dot(tF, rgb, b.Comp(tV3, b.F(tF, 0.2126f), b.F(tF, 0.7152f), b.F(tF, 0.0722f)));
                uint t = b.FMax(glsl, tF, b.FSub(tF, lum, b.F(tF, 0.9f)), zero);
                return b.Scale(tV3, rgb, t);
            }

            uint hdr = SampleRgb(zero, zero);

            // 13-tap weighted bloom (center + cross + diagonals at 1x and 2x texel).
            uint tw = b.F(tF, texelW);
            uint th = b.F(tF, texelH);
            uint ntw = b.FSub(tF, zero, tw);
            uint nth = b.FSub(tF, zero, th);
            uint tw2 = b.FMul(tF, tw, b.F(tF, 2.5f));
            uint th2 = b.FMul(tF, th, b.F(tF, 2.5f));
            uint ntw2 = b.FSub(tF, zero, tw2);
            uint nth2 = b.FSub(tF, zero, th2);
            uint bloom = Bright(hdr);
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(tw, zero)), b.F(tF, 0.9f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(ntw, zero)), b.F(tF, 0.9f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(zero, th)), b.F(tF, 0.9f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(zero, nth)), b.F(tF, 0.9f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(tw, th)), b.F(tF, 0.65f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(ntw, th)), b.F(tF, 0.65f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(tw, nth)), b.F(tF, 0.65f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(ntw, nth)), b.F(tF, 0.65f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(tw2, zero)), b.F(tF, 0.45f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(ntw2, zero)), b.F(tF, 0.45f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(zero, th2)), b.F(tF, 0.45f)));
            bloom = b.FAdd(tV3, bloom, b.Scale(tV3, Bright(SampleRgb(zero, nth2)), b.F(tF, 0.45f)));
            bloom = b.Scale(tV3, bloom, b.F(tF, 1f / 9.1f));

            hdr = b.FAdd(tV3, hdr, b.Scale(tV3, bloom, b.F(tF, bloomStrength)));

            // Mild saturation lift before tonemap.
            uint grey = b.Splat(tV3, b.Dot(tF, hdr, b.Comp(tV3, b.F(tF, 0.2126f), b.F(tF, 0.7152f), b.F(tF, 0.0722f))));
            hdr = b.Mix(glsl, tV3, grey, hdr, b.Splat(tV3, b.F(tF, 1.12f)));

            uint x = b.Scale(tV3, hdr, b.F(tF, exposure));
            uint aA = b.Splat(tV3, b.F(tF, 2.51f));
            uint bB = b.Splat(tV3, b.F(tF, 0.03f));
            uint cC = b.Splat(tV3, b.F(tF, 2.43f));
            uint dD = b.Splat(tV3, b.F(tF, 0.59f));
            uint eE = b.Splat(tV3, b.F(tF, 0.14f));
            uint num = b.FAdd(tV3, b.FMul(tV3, x, b.FAdd(tV3, b.FMul(tV3, x, aA), bB)), bB);
            uint den = b.FAdd(tV3, b.FMul(tV3, x, b.FAdd(tV3, b.FMul(tV3, x, cC), dD)), eE);
            uint tm = b.FDiv(tV3, num, den);
            // Smooth approx of sRGB encode without pow.
            uint gamma = b.Sqrt(glsl, tV3, tm);
            uint ldr = b.Mix(glsl, tV3, tm, gamma, b.Splat(tV3, b.F(tF, 0.78f)));

            // TAA: reproject history with velocity (NDC → UV), blend with current LDR.
            uint vel4 = b.Sample(tV4, b.Load(tSImg, texVel), uv);
            uint half = b.F(tF, 0.5f);
            uint histUv = b.Comp(tV2,
                b.FSub(tF, b.Extract(tF, uv, 0), b.FMul(tF, b.Extract(tF, vel4, 0), half)),
                b.FSub(tF, b.Extract(tF, uv, 1), b.FMul(tF, b.Extract(tF, vel4, 1), half)));
            uint hist = b.Rgb(tF, tV3, b.Sample(tV4, b.Load(tSImg, texHist), histUv));
            uint velLen = b.Length(glsl, tF, b.Comp(tV2, b.Extract(tF, vel4, 0), b.Extract(tF, vel4, 1)));
            uint motionKill = b.Clamp(glsl, tF, b.FMul(tF, velLen, b.F(tF, 8f)), zero, one);
            uint blend = b.FMul(tF, b.F(tF, taaBlend), b.FSub(tF, one, motionKill));
            ldr = b.Mix(glsl, tV3, ldr, hist, b.Splat(tV3, blend));

            b.Store(outC, b.Comp(tV4,
                b.Extract(tF, ldr, 0), b.Extract(tF, ldr, 1), b.Extract(tF, ldr, 2), one));
            b.Return();
            b.EndMain();
            return b.Build();
        }

        private static uint CascadeShadow(
            Spirv b, uint glsl, uint tF, uint tV2, uint tV3, uint tV4, uint tBool, uint tU, uint tSDImg,
            uint shadowSamp, uint lightVP, uint P, uint one, uint half, uint zero, uint eps)
        {
            _ = (tV3, tU, tSDImg);
            uint clip = b.MulMV(tV4, lightVP, b.Comp(tV4,
                b.Extract(tF, P, 0), b.Extract(tF, P, 1), b.Extract(tF, P, 2), one));
            uint cw = b.Extract(tF, clip, 3);
            uint invW = b.FDiv(tF, one, b.FMax(glsl, tF, b.FAbs(glsl, tF, cw), eps));
            uint ndcX = b.FMul(tF, b.Extract(tF, clip, 0), invW);
            uint ndcY = b.FMul(tF, b.Extract(tF, clip, 1), invW);
            uint ndcZ = b.FMul(tF, b.Extract(tF, clip, 2), invW);
            uint sU = b.FAdd(tF, b.FMul(tF, ndcX, half), half);
            uint sV = b.FAdd(tF, b.FMul(tF, ndcY, half), half);
            uint sZ = b.FSub(tF, b.FAdd(tF, b.FMul(tF, ndcZ, half), half), b.F(tF, 0.002f));

            uint texel = b.F(tF, 1f / 2048f);
            uint ntex = b.FSub(tF, zero, texel);
            uint sh = zero;
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, zero, zero));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, texel, zero));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, ntex, zero));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, zero, texel));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, zero, ntex));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, texel, texel));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, ntex, texel));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, texel, ntex));
            sh = b.FAdd(tF, sh, Tap(b, tF, tV2, tV4, tBool, shadowSamp, sU, sV, sZ, ntex, ntex));
            sh = b.FMul(tF, sh, b.F(tF, 1f / 9f));

            uint inMap = b.And(tBool,
                b.And(tBool, b.FOrdGe(tBool, sU, zero), b.FOrdLe(tBool, sU, one)),
                b.And(tBool, b.FOrdGe(tBool, sV, zero), b.FOrdLe(tBool, sV, one)));
            return b.Select(tF, inMap, sh, one);
        }

        private static uint Tap(
            Spirv b, uint tF, uint tV2, uint tV4, uint tBool,
            uint samp, uint u, uint v, uint refZ, uint du, uint dv)
        {
            uint coord = b.Comp(tV2, b.FAdd(tF, u, du), b.FAdd(tF, v, dv));
            uint depth = b.Extract(tF, b.Sample(tV4, samp, coord), 0);
            return b.Select(tF, b.FOrdLe(tBool, refZ, depth), b.F(tF, 1f), b.F(tF, 0f));
        }

        private sealed class Spirv
        {
            public const uint UniformConstant = 0, Input = 1, Uniform = 2, Output = 3;

            private readonly List<uint> _cap = new();
            private readonly List<uint> _ext = new();
            private readonly List<uint> _entry = new();
            private readonly List<uint> _mode = new();
            private readonly List<uint> _dec = new();
            private readonly List<uint> _types = new();
            private readonly List<uint> _code = new();
            private uint _bound = 1;
            private bool _mem;
            private uint? _boolType;
            private uint? _fnTypeVoid;
            private uint _mainFn;
            private bool _fragment;
            private readonly List<uint> _interfaces = new();
            public uint GlslId { get; private set; }

            private uint Id() => _bound++;

            private static void Emit(List<uint> dst, uint op, params uint[] ops)
            {
                dst.Add(op | ((uint)(ops.Length + 1) << 16));
                dst.AddRange(ops);
            }

            private static void EmitStr(List<uint> dst, uint op, uint[] prefix, string s)
            {
                var bytes = Encoding.UTF8.GetBytes(s + "\0");
                int nw = (bytes.Length + 3) / 4;
                var all = new List<uint>(prefix);
                for (int i = 0; i < nw; i++)
                {
                    uint w = 0;
                    for (int j = 0; j < 4 && i * 4 + j < bytes.Length; j++)
                        w |= (uint)bytes[i * 4 + j] << (j * 8);
                    all.Add(w);
                }
                dst.Add(op | ((uint)(all.Count + 1) << 16));
                dst.AddRange(all);
            }

            public void CapShader() => Emit(_cap, 17, 1);
            public void ImportGlsl()
            {
                GlslId = Id();
                EmitStr(_ext, 11, new[] { GlslId }, "GLSL.std.450");
            }
            public void MemoryModel() => _mem = true;

            public void BeginMain(uint voidT, bool fragment)
            {
                _fragment = fragment;
                _fnTypeVoid ??= TFunc(voidT);
                _mainFn = Id();
                Emit(_code, 54, voidT, _mainFn, 0, _fnTypeVoid.Value);
                if (fragment)
                    Emit(_mode, 16, _mainFn, 7);
            }

            public void EndMain()
            {
                Emit(_code, 56);
                var nameBytes = Encoding.UTF8.GetBytes("main\0");
                int nw = (nameBytes.Length + 3) / 4;
                var all = new List<uint> { _fragment ? 4u : 0u, _mainFn };
                for (int i = 0; i < nw; i++)
                {
                    uint w = 0;
                    for (int j = 0; j < 4 && i * 4 + j < nameBytes.Length; j++)
                        w |= (uint)nameBytes[i * 4 + j] << (j * 8);
                    all.Add(w);
                }
                all.AddRange(_interfaces);
                Emit(_entry, 15, all.ToArray());
            }

            public void Label() { uint id = Id(); Emit(_code, 248, id); }
            public void Return() => Emit(_code, 253);

            public void DecLocation(uint id, uint loc)
            {
                Emit(_dec, 71, id, 30, loc);
                _interfaces.Add(id);
            }
            public void DecBuiltIn(uint id, uint builtin)
            {
                Emit(_dec, 71, id, 11, builtin);
                _interfaces.Add(id);
            }
            public void DecSetBinding(uint id, uint set, uint binding)
            {
                Emit(_dec, 71, id, 33, set);
                Emit(_dec, 71, id, 34, binding);
            }
            public void DecBlock(uint id) => Emit(_dec, 71, id, 2);
            public void DecMemberOffset(uint str, uint member, uint offset)
                => Emit(_dec, 72, str, member, 35, offset);
            public void DecMemberMatrix(uint str, uint member)
            {
                Emit(_dec, 72, str, member, 5, 1);
                Emit(_dec, 72, str, member, 7, 16);
            }

            public uint TVoid() { uint id = Id(); Emit(_types, 19, id); return id; }
            public uint TBool()
            {
                if (_boolType.HasValue) return _boolType.Value;
                uint id = Id(); Emit(_types, 20, id); _boolType = id; return id;
            }
            public uint TIntU() { uint id = Id(); Emit(_types, 21, id, 32, 0); return id; }
            public uint TFloat() { uint id = Id(); Emit(_types, 22, id, 32); return id; }
            public uint TVec(uint c, uint n) { uint id = Id(); Emit(_types, 23, id, c, n); return id; }
            public uint TMat(uint col, uint n) { uint id = Id(); Emit(_types, 24, id, col, n); return id; }
            public uint TImage(uint sampled, bool depth)
            {
                uint id = Id();
                Emit(_types, 25, id, sampled, 1u, depth ? 1u : 0u, 0, 0, 1, 0);
                return id;
            }
            public uint TSampled(uint img) { uint id = Id(); Emit(_types, 27, id, img); return id; }
            public uint TStruct(params uint[] members)
            {
                uint id = Id();
                var ops = new List<uint> { id };
                ops.AddRange(members);
                Emit(_types, 30, ops.ToArray());
                return id;
            }
            public uint TPtr(uint sc, uint t) { uint id = Id(); Emit(_types, 32, id, sc, t); return id; }
            public uint TFunc(uint ret) { uint id = Id(); Emit(_types, 33, id, ret); return id; }

            public uint F(uint t, float v)
            {
                uint id = Id();
                Emit(_types, 43, t, id, BitConverter.ToUInt32(BitConverter.GetBytes(v)));
                return id;
            }
            public uint U(uint t, uint v) { uint id = Id(); Emit(_types, 43, t, id, v); return id; }

            public uint Var(uint ptrT, uint sc)
            {
                uint id = Id();
                Emit(_types, 59, ptrT, id, sc);
                return id;
            }

            public uint Load(uint t, uint p) { uint id = Id(); Emit(_code, 61, t, id, p); return id; }
            public void Store(uint p, uint v) => Emit(_code, 62, p, v);
            public uint Access(uint resultT, uint bas, params uint[] idx)
            {
                uint id = Id();
                var ops = new List<uint> { resultT, id, bas };
                ops.AddRange(idx);
                Emit(_code, 65, ops.ToArray());
                return id;
            }

            public uint Comp(uint t, params uint[] parts)
            {
                uint id = Id();
                var ops = new List<uint> { t, id };
                ops.AddRange(parts);
                Emit(_code, 80, ops.ToArray());
                return id;
            }
            public uint Extract(uint t, uint v, uint i) { uint id = Id(); Emit(_code, 81, t, id, v, i); return id; }
            public uint Rgb(uint tF, uint tV3, uint v4)
                => Comp(tV3, Extract(tF, v4, 0), Extract(tF, v4, 1), Extract(tF, v4, 2));
            public uint Splat(uint tV3, uint s) => Comp(tV3, s, s, s);

            public uint FAdd(uint t, uint a, uint b) { uint id = Id(); Emit(_code, 129, t, id, a, b); return id; }
            public uint FSub(uint t, uint a, uint b) { uint id = Id(); Emit(_code, 130, t, id, a, b); return id; }
            public uint FMul(uint t, uint a, uint b) { uint id = Id(); Emit(_code, 133, t, id, a, b); return id; }
            public uint FDiv(uint t, uint a, uint b) { uint id = Id(); Emit(_code, 134, t, id, a, b); return id; }
            public uint Dot(uint t, uint a, uint b) { uint id = Id(); Emit(_code, 148, t, id, a, b); return id; }
            public uint Scale(uint t, uint v, uint s) { uint id = Id(); Emit(_code, 142, t, id, v, s); return id; }
            public uint MulMV(uint t, uint m, uint v) { uint id = Id(); Emit(_code, 145, t, id, m, v); return id; }
            public uint MulVM(uint t, uint v, uint m) { uint id = Id(); Emit(_code, 146, t, id, v, m); return id; }

            public uint Ext(uint t, uint set, uint inst, params uint[] ops)
            {
                uint id = Id();
                var all = new List<uint> { t, id, set, inst };
                all.AddRange(ops);
                Emit(_code, 12, all.ToArray());
                return id;
            }

            public uint Normalize(uint glsl, uint t, uint v) => Ext(t, glsl, 69, v);
            public uint Length(uint glsl, uint t, uint v) => Ext(t, glsl, 66, v);
            public uint FMax(uint glsl, uint t, uint a, uint b) => Ext(t, glsl, 40, a, b);
            public uint FAbs(uint glsl, uint t, uint a) => Ext(t, glsl, 4, a);
            public uint Clamp(uint glsl, uint t, uint x, uint lo, uint hi) => Ext(t, glsl, 43, x, lo, hi);
            public uint Mix(uint glsl, uint t, uint x, uint y, uint a) => Ext(t, glsl, 46, x, y, a);
            public uint Sqrt(uint glsl, uint t, uint v) => Ext(t, glsl, 31, v);

            public uint Sample(uint t, uint img, uint uv) { uint id = Id(); Emit(_code, 87, t, id, img, uv); return id; }
            public uint Select(uint t, uint c, uint th, uint el) { uint id = Id(); Emit(_code, 169, t, id, c, th, el); return id; }
            public uint FOrdLt(uint tb, uint a, uint b) { uint id = Id(); Emit(_code, 180, tb, id, a, b); return id; }
            public uint FOrdLe(uint tb, uint a, uint b) { uint id = Id(); Emit(_code, 184, tb, id, a, b); return id; }
            public uint FOrdGe(uint tb, uint a, uint b) { uint id = Id(); Emit(_code, 185, tb, id, a, b); return id; }
            public uint And(uint tb, uint a, uint b) { uint id = Id(); Emit(_code, 167, tb, id, a, b); return id; }

            public byte[] Build()
            {
                var w = new List<uint> { 0x07230203, 0x00010000, 0x0008000B, _bound, 0 };
                w.AddRange(_cap);
                w.AddRange(_ext);
                if (_mem) Emit(w, 14, 0, 1);
                w.AddRange(_entry);
                w.AddRange(_mode);
                w.AddRange(_dec);
                w.AddRange(_types);
                w.AddRange(_code);
                w[3] = _bound;
                var bytes = new byte[w.Count * 4];
                Buffer.BlockCopy(w.ToArray(), 0, bytes, 0, bytes.Length);
                return bytes;
            }
        }
    }
}
