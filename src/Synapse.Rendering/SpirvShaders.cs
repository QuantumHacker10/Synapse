// =============================================================================
// SpirvShaders.cs - GDNN Engine: SPIR-V Shader Builder + Embedded Shaders
// Supports G-Buffer, deferred lighting, shadow mapping, post-processing
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace GDNN.Rendering.Shaders
{
    public static class EmbeddedShaders
    {
        private const uint MAGIC = 0x07230203;
        private const uint VERSION_1_0 = 0x00010000;
        private const uint GENERATOR = 0x00080008;

        // =====================================================================
        // Original basic shaders (kept for fallback)
        // =====================================================================
        public static byte[] CompileVertexShader() => BuildBasicVertex();
        public static byte[] CompileFragmentShader() => BuildBasicFragment();

        // =====================================================================
        // G-Buffer Pass Shaders (4 MRT outputs)
        // =====================================================================
        public static byte[] CompileGBufferVertex()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);
            var mat4T = b.TypeMatrix(vec4T, 4);

            var ptrInputVec3 = b.TypePointer(StorageClass.Input, vec3T);
            var ptrInputVec2 = b.TypePointer(StorageClass.Input, vec2T);
            var ptrOutputVec3 = b.TypePointer(StorageClass.Output, vec3T);
            var ptrOutputVec2 = b.TypePointer(StorageClass.Output, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);
            var ptrUniformMat4 = b.TypePointer(StorageClass.Uniform, mat4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();

            var inPos = b.Variable(ptrInputVec3, StorageClass.Input, "inPosition");
            b.Decorate(inPos, Decoration.Location, 0);
            var inNormal = b.Variable(ptrInputVec3, StorageClass.Input, "inNormal");
            b.Decorate(inNormal, Decoration.Location, 1);
            var inUV = b.Variable(ptrInputVec2, StorageClass.Input, "inUV");
            b.Decorate(inUV, Decoration.Location, 2);
            var inColor = b.Variable(ptrInputVec3, StorageClass.Input, "inColor");
            b.Decorate(inColor, Decoration.Location, 3);

            var outWorldPos = b.Variable(ptrOutputVec3, StorageClass.Output, "worldPos");
            b.Decorate(outWorldPos, Decoration.Location, 0);
            var outWorldNormal = b.Variable(ptrOutputVec3, StorageClass.Output, "worldNormal");
            b.Decorate(outWorldNormal, Decoration.Location, 1);
            var outUV = b.Variable(ptrOutputVec2, StorageClass.Output, "fragUV");
            b.Decorate(outUV, Decoration.Location, 2);
            var outColor = b.Variable(ptrOutputVec3, StorageClass.Output, "fragColor");
            b.Decorate(outColor, Decoration.Location, 3);

            var mvpPtr = b.Variable(ptrUniformMat4, StorageClass.Uniform, "ubo_camera");
            b.Decorate(mvpPtr, Decoration.DescriptorSet, 0);
            b.Decorate(mvpPtr, Decoration.Binding, 0);
            var modelPtr = b.Variable(ptrUniformMat4, StorageClass.Uniform, "ubo_model");
            b.Decorate(modelPtr, Decoration.DescriptorSet, 0);
            b.Decorate(modelPtr, Decoration.Binding, 1);

            var pos = b.Load(vec3T, inPos);
            var one = b.Constant(floatT, 1.0f);
            var pos4 = b.CompositeConstruct(vec4T, pos, one);
            var model = b.Load(mat4T, modelPtr);
            var worldPos4 = b.VectorTimesMatrix(vec4T, pos4, model);
            var worldPos3 = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, worldPos4, 0),
                b.CompositeExtract(floatT, worldPos4, 1),
                b.CompositeExtract(floatT, worldPos4, 2));
            b.Store(outWorldPos, worldPos3);

            var mvp = b.Load(mat4T, mvpPtr);
            var clipPos = b.VectorTimesMatrix(vec4T, pos4, mvp);
            var glPos = b.Variable(ptrOutputVec4, StorageClass.Output, "gl_Position");
            b.Decorate(glPos, Decoration.BuiltIn, (uint)BuiltIn.Position);
            b.Store(glPos, clipPos);

            var normal = b.Load(vec3T, inNormal);
            var normal4 = b.VectorTimesMatrix(vec4T, b.CompositeConstruct(vec4T, normal, b.Constant(floatT, 0.0f)), model);
            var norm3 = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, normal4, 0),
                b.CompositeExtract(floatT, normal4, 1),
                b.CompositeExtract(floatT, normal4, 2));
            b.Store(outWorldNormal, norm3);

            var uv = b.Load(vec2T, inUV);
            b.Store(outUV, uv);
            var color = b.Load(vec3T, inColor);
            b.Store(outColor, color);

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        public static byte[] CompileGBufferFragment()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);

            var ptrInputVec3 = b.TypePointer(StorageClass.Input, vec3T);
            var ptrInputVec2 = b.TypePointer(StorageClass.Input, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.ExecutionMode(entry, ExecutionMode.OriginUpperLeft);
            b.Label();

            var worldPos = b.Variable(ptrInputVec3, StorageClass.Input, "worldPos");
            b.Decorate(worldPos, Decoration.Location, 0);
            var worldNormal = b.Variable(ptrInputVec3, StorageClass.Input, "worldNormal");
            b.Decorate(worldNormal, Decoration.Location, 1);
            var fragUV = b.Variable(ptrInputVec2, StorageClass.Input, "fragUV");
            b.Decorate(fragUV, Decoration.Location, 2);
            var fragColor = b.Variable(ptrInputVec3, StorageClass.Input, "fragColor");
            b.Decorate(fragColor, Decoration.Location, 3);

            var outAlbedo = b.Variable(ptrOutputVec4, StorageClass.Output, "gbuffer_albedo");
            b.Decorate(outAlbedo, Decoration.Location, 0);
            var outNormals = b.Variable(ptrOutputVec4, StorageClass.Output, "gbuffer_normals");
            b.Decorate(outNormals, Decoration.Location, 1);
            var outDepth = b.Variable(ptrOutputVec4, StorageClass.Output, "gbuffer_depth");
            b.Decorate(outDepth, Decoration.Location, 2);
            var outMaterial = b.Variable(ptrOutputVec4, StorageClass.Output, "gbuffer_material");
            b.Decorate(outMaterial, Decoration.Location, 3);
            var outVelocity = b.Variable(ptrOutputVec4, StorageClass.Output, "gbuffer_velocity");
            b.Decorate(outVelocity, Decoration.Location, 4);

            var one = b.Constant(floatT, 1.0f);
            var zero = b.Constant(floatT, 0.0f);
            var half = b.Constant(floatT, 0.5f);
            var materialId = b.Constant(floatT, 1.0f);

            var color = b.Load(vec3T, fragColor);
            b.Store(outAlbedo, b.CompositeConstruct(vec4T, color, one));

            var normal = b.Load(vec3T, worldNormal);
            b.Store(outNormals, b.CompositeConstruct(vec4T, normal, one));

            var wp = b.Load(vec3T, worldPos);
            var depthVal = b.CompositeExtract(floatT, wp, 2);
            b.Store(outDepth, b.CompositeConstruct(vec4T, depthVal, zero, zero, one));

            // Material: roughness, metallic, materialId, translucency
            b.Store(outMaterial, b.CompositeConstruct(vec4T, half, zero, materialId, zero));

            // Velocity: static geometry → zero motion vectors (XY in RG)
            b.Store(outVelocity, b.CompositeConstruct(vec4T, zero, zero, zero, one));

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        // =====================================================================
        // Deferred Lighting Pass Shaders (full-screen quad + G-Buffer sampling)
        // =====================================================================
        public static byte[] CompileLightingVertex()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec4T = b.TypeVector(floatT, 4);
            var ptrOutputVec2 = b.TypePointer(StorageClass.Output, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();

            var glPos = b.Variable(ptrOutputVec4, StorageClass.Output, "gl_Position");
            b.Decorate(glPos, Decoration.BuiltIn, (uint)BuiltIn.Position);
            var outUV = b.Variable(ptrOutputVec2, StorageClass.Output, "fragUV");
            b.Decorate(outUV, Decoration.Location, 0);

            var glVertexIndex = b.Variable(b.TypePointer(StorageClass.Input, b.TypeInt(32, 0)), StorageClass.Input, "gl_VertexIndex");
            b.Decorate(glVertexIndex, Decoration.BuiltIn, (uint)BuiltIn.VertexIndex);

            var idx = b.Load(b.TypeInt(32, 0), glVertexIndex);

            var posX = b.Constant(floatT, 0.0f);
            var posY = b.Constant(floatT, 0.0f);
            var posZ = b.Constant(floatT, 0.0f);
            var posW = b.Constant(floatT, 1.0f);

            var intT = b.TypeInt(32, 0);
            var zeroI = b.Constant(intT, 0);
            var oneI = b.Constant(intT, 1);
            var twoI = b.Constant(intT, 2);

            var isBottomLeft = b.IEqual(idx, zeroI);
            var isBottomRight = b.IEqual(idx, oneI);
            var isTop = b.IEqual(idx, twoI);

            posX = b.Select(floatT, isBottomLeft, b.Constant(floatT, -1.0f),
                b.Select(floatT, isBottomRight, b.Constant(floatT, 3.0f),
                b.Constant(floatT, -1.0f)));
            posY = b.Select(floatT, isBottomLeft, b.Constant(floatT, -1.0f),
                b.Select(floatT, isBottomRight, b.Constant(floatT, -1.0f),
                b.Constant(floatT, 3.0f)));

            var uvX = b.Select(floatT, isBottomLeft, b.Constant(floatT, 0.0f),
                b.Select(floatT, isBottomRight, b.Constant(floatT, 2.0f),
                b.Constant(floatT, 0.0f)));
            var uvY = b.Select(floatT, isBottomLeft, b.Constant(floatT, 0.0f),
                b.Select(floatT, isBottomRight, b.Constant(floatT, 0.0f),
                b.Constant(floatT, 2.0f)));

            b.Store(glPos, b.CompositeConstruct(vec4T, posX, posY, posZ, posW));
            b.Store(outUV, b.CompositeConstruct(vec2T, uvX, uvY));

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        public static byte[] CompileLightingFragment()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.Capability(Capability.ImageQuery);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var intT = b.TypeInt(32, 0);
            var vec2T = b.TypeVector(floatT, 2);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);

            var image2dT = b.TypeImage(floatT, "2D", 0, 0, 0, 1, 0);
            var samplerT = b.TypeSampler();
            var sampledImageT = b.TypeSampledImage(image2dT);

            var ptrInputVec2 = b.TypePointer(StorageClass.Input, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);
            var ptrUniformConstantSampledImage = b.TypePointer(StorageClass.UniformConstant, sampledImageT);

            var entry = b.Function(voidT, FunctionControl.None);
            b.ExecutionMode(entry, ExecutionMode.OriginUpperLeft);
            b.Label();

            var inUV = b.Variable(ptrInputVec2, StorageClass.Input, "fragUV");
            b.Decorate(inUV, Decoration.Location, 0);

            var gbufAlbedo = b.Variable(ptrUniformConstantSampledImage, StorageClass.UniformConstant, "gbuf_albedo");
            b.Decorate(gbufAlbedo, Decoration.DescriptorSet, 0);
            b.Decorate(gbufAlbedo, Decoration.Binding, 0);
            var gbufNormals = b.Variable(ptrUniformConstantSampledImage, StorageClass.UniformConstant, "gbuf_normals");
            b.Decorate(gbufNormals, Decoration.DescriptorSet, 0);
            b.Decorate(gbufNormals, Decoration.Binding, 1);
            var gbufDepth = b.Variable(ptrUniformConstantSampledImage, StorageClass.UniformConstant, "gbuf_depth");
            b.Decorate(gbufDepth, Decoration.DescriptorSet, 0);
            b.Decorate(gbufDepth, Decoration.Binding, 2);
            var gbufMaterial = b.Variable(ptrUniformConstantSampledImage, StorageClass.UniformConstant, "gbuf_material");
            b.Decorate(gbufMaterial, Decoration.DescriptorSet, 0);
            b.Decorate(gbufMaterial, Decoration.Binding, 3);

            var outColor = b.Variable(ptrOutputVec4, StorageClass.Output, "outColor");
            b.Decorate(outColor, Decoration.Location, 0);

            var uv = b.Load(vec2T, inUV);
            var one = b.Constant(floatT, 1.0f);
            var two = b.Constant(floatT, 2.0f);
            var half = b.Constant(floatT, 0.5f);

            var sampledAlbedo = b.Load(sampledImageT, gbufAlbedo);
            var albedoColor = b.ImageSampleImplicitLod(vec4T, sampledAlbedo, uv);

            var sampledNormals = b.Load(sampledImageT, gbufNormals);
            var normalColor = b.ImageSampleImplicitLod(vec4T, sampledNormals, uv);

            var sampledDepth = b.Load(sampledImageT, gbufDepth);
            var depthColor = b.ImageSampleImplicitLod(vec4T, sampledDepth, uv);

            var sampledMaterial = b.Load(sampledImageT, gbufMaterial);
            var matColor = b.ImageSampleImplicitLod(vec4T, sampledMaterial, uv);

            var albedo = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, albedoColor, 0),
                b.CompositeExtract(floatT, albedoColor, 1),
                b.CompositeExtract(floatT, albedoColor, 2));
            var normal = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, normalColor, 0),
                b.CompositeExtract(floatT, normalColor, 1),
                b.CompositeExtract(floatT, normalColor, 2));
            var roughness = b.CompositeExtract(floatT, matColor, 0);
            var metallic = b.CompositeExtract(floatT, matColor, 1);

            var lightDir = b.Normalize(vec3T, b.ConstantComposite(vec3T, b.Constant(floatT, 0.577f), b.Constant(floatT, 0.577f), b.Constant(floatT, 0.577f)), floatT);
            var nDotL = b.FMax(floatT, b.DotProduct(floatT, normal, lightDir), b.Constant(floatT, 0.1f));

            var diffuse = b.VectorTimesScalar(vec3T, albedo, nDotL);
            var ambient = b.VectorTimesScalar(vec3T, albedo, b.Constant(floatT, 0.15f));
            var lit = b.FAdd(vec3T, diffuse, ambient);

            var finalColor = b.CompositeConstruct(vec4T, lit, one);
            b.Store(outColor, finalColor);

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        // =====================================================================
        // Shadow Map Pass Shaders
        // =====================================================================
        public static byte[] CompileShadowVertex()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);
            var mat4T = b.TypeMatrix(vec4T, 4);

            var ptrInputVec3 = b.TypePointer(StorageClass.Input, vec3T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);
            var ptrUniformMat4 = b.TypePointer(StorageClass.Uniform, mat4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();

            var inPos = b.Variable(ptrInputVec3, StorageClass.Input, "inPosition");
            b.Decorate(inPos, Decoration.Location, 0);

            var glPos = b.Variable(ptrOutputVec4, StorageClass.Output, "gl_Position");
            b.Decorate(glPos, Decoration.BuiltIn, (uint)BuiltIn.Position);

            var lightMVP = b.Variable(ptrUniformMat4, StorageClass.Uniform, "lightMVP");
            b.Decorate(lightMVP, Decoration.DescriptorSet, 0);
            b.Decorate(lightMVP, Decoration.Binding, 0);

            var pos = b.Load(vec3T, inPos);
            var one = b.Constant(floatT, 1.0f);
            var pos4 = b.CompositeConstruct(vec4T, pos, one);
            var mvp = b.Load(mat4T, lightMVP);
            var result = b.VectorTimesMatrix(vec4T, pos4, mvp);
            b.Store(glPos, result);

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        public static byte[] CompileShadowFragment()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();
            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        // =====================================================================
        // Post-Process: Tonemap + Bloom Composite
        // =====================================================================
        public static byte[] CompilePostProcessVertex()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec4T = b.TypeVector(floatT, 4);
            var ptrOutputVec2 = b.TypePointer(StorageClass.Output, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();

            var glPos = b.Variable(ptrOutputVec4, StorageClass.Output, "gl_Position");
            b.Decorate(glPos, Decoration.BuiltIn, (uint)BuiltIn.Position);
            var outUV = b.Variable(ptrOutputVec2, StorageClass.Output, "fragUV");
            b.Decorate(outUV, Decoration.Location, 0);

            var glVertexIndex = b.Variable(b.TypePointer(StorageClass.Input, b.TypeInt(32, 0)), StorageClass.Input, "gl_VertexIndex");
            b.Decorate(glVertexIndex, Decoration.BuiltIn, (uint)BuiltIn.VertexIndex);

            var idx = b.Load(b.TypeInt(32, 0), glVertexIndex);
            var intT = b.TypeInt(32, 0);
            var zeroI = b.Constant(intT, 0);
            var oneI = b.Constant(intT, 1);
            var twoI = b.Constant(intT, 2);

            var isBL = b.IEqual(idx, zeroI);
            var isBR = b.IEqual(idx, oneI);

            var posX = b.Select(floatT, isBL, b.Constant(floatT, -1.0f),
                b.Select(floatT, isBR, b.Constant(floatT, 3.0f), b.Constant(floatT, -1.0f)));
            var posY = b.Select(floatT, isBL, b.Constant(floatT, -1.0f),
                b.Select(floatT, isBR, b.Constant(floatT, -1.0f), b.Constant(floatT, 3.0f)));
            var uvX = b.Select(floatT, isBL, b.Constant(floatT, 0.0f),
                b.Select(floatT, isBR, b.Constant(floatT, 2.0f), b.Constant(floatT, 0.0f)));
            var uvY = b.Select(floatT, isBL, b.Constant(floatT, 0.0f),
                b.Select(floatT, isBR, b.Constant(floatT, 0.0f), b.Constant(floatT, 2.0f)));

            b.Store(glPos, b.CompositeConstruct(vec4T, posX, posY, b.Constant(floatT, 0.0f), b.Constant(floatT, 1.0f)));
            b.Store(outUV, b.CompositeConstruct(vec2T, uvX, uvY));

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        public static byte[] CompileTonemapFragment()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);

            var image2dT = b.TypeImage(floatT, "2D", 0, 0, 0, 1, 0);
            var samplerT = b.TypeSampler();
            var sampledImageT = b.TypeSampledImage(image2dT);

            var ptrInputVec2 = b.TypePointer(StorageClass.Input, vec2T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);
            var ptrUniformConstantSI = b.TypePointer(StorageClass.UniformConstant, sampledImageT);

            var entry = b.Function(voidT, FunctionControl.None);
            b.ExecutionMode(entry, ExecutionMode.OriginUpperLeft);
            b.Label();

            var inUV = b.Variable(ptrInputVec2, StorageClass.Input, "fragUV");
            b.Decorate(inUV, Decoration.Location, 0);

            var sceneTex = b.Variable(ptrUniformConstantSI, StorageClass.UniformConstant, "sceneTexture");
            b.Decorate(sceneTex, Decoration.DescriptorSet, 0);
            b.Decorate(sceneTex, Decoration.Binding, 0);

            var bloomTex = b.Variable(ptrUniformConstantSI, StorageClass.UniformConstant, "bloomTexture");
            b.Decorate(bloomTex, Decoration.DescriptorSet, 0);
            b.Decorate(bloomTex, Decoration.Binding, 1);

            var outColor = b.Variable(ptrOutputVec4, StorageClass.Output, "outColor");
            b.Decorate(outColor, Decoration.Location, 0);

            var uv = b.Load(vec2T, inUV);

            var sampledScene = b.Load(sampledImageT, sceneTex);
            var sceneColor = b.ImageSampleImplicitLod(vec4T, sampledScene, uv);

            var sampledBloom = b.Load(sampledImageT, bloomTex);
            var bloomColor = b.ImageSampleImplicitLod(vec4T, sampledBloom, uv);

            var sceneRGB = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, sceneColor, 0),
                b.CompositeExtract(floatT, sceneColor, 1),
                b.CompositeExtract(floatT, sceneColor, 2));
            var bloomRGB = b.CompositeConstruct(vec3T,
                b.CompositeExtract(floatT, bloomColor, 0),
                b.CompositeExtract(floatT, bloomColor, 1),
                b.CompositeExtract(floatT, bloomColor, 2));

            var combined = b.FAdd(vec3T, sceneRGB, b.VectorTimesScalar(vec3T, bloomRGB, b.Constant(floatT, 0.4f)));
            var exposure = b.Constant(floatT, 1.2f);
            var exposed = b.VectorTimesScalar(vec3T, combined, exposure);

            var a = b.ConstantComposite(vec3T, b.Constant(floatT, 2.51f), b.Constant(floatT, 2.51f), b.Constant(floatT, 2.51f));
            var bC = b.ConstantComposite(vec3T, b.Constant(floatT, 0.03f), b.Constant(floatT, 0.03f), b.Constant(floatT, 0.03f));
            var c = b.ConstantComposite(vec3T, b.Constant(floatT, 2.43f), b.Constant(floatT, 2.43f), b.Constant(floatT, 2.43f));
            var d = b.ConstantComposite(vec3T, b.Constant(floatT, 0.59f), b.Constant(floatT, 0.59f), b.Constant(floatT, 0.59f));
            var e = b.ConstantComposite(vec3T, b.Constant(floatT, 0.14f), b.Constant(floatT, 0.14f), b.Constant(floatT, 0.14f));

            var num = b.FAdd(vec3T, b.FMul(vec3T, exposed, b.FAdd(vec3T, b.FMul(vec3T, exposed, a), bC)), bC);
            var den = b.FAdd(vec3T, b.FMul(vec3T, exposed, b.FAdd(vec3T, b.FMul(vec3T, exposed, c), d)), e);
            var tonemapped = b.FDiv(vec3T, num, den);

            var one = b.Constant(floatT, 1.0f);
            b.Store(outColor, b.CompositeConstruct(vec4T, tonemapped, one));

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        // =====================================================================
        // Basic fallback shaders (original)
        // =====================================================================
        private static byte[] BuildBasicVertex()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec2T = b.TypeVector(floatT, 2);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);
            var mat4T = b.TypeMatrix(vec4T, 4);
            var ptrInputVec3 = b.TypePointer(StorageClass.Input, vec3T);
            var ptrInputVec2 = b.TypePointer(StorageClass.Input, vec2T);
            var ptrOutputVec3 = b.TypePointer(StorageClass.Output, vec3T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);
            var ptrUniformMat4 = b.TypePointer(StorageClass.Uniform, mat4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.Label();

            var glPos = b.Variable(ptrOutputVec4, StorageClass.Output, "gl_Position");
            b.Decorate(glPos, Decoration.BuiltIn, (uint)BuiltIn.Position);
            var inPos = b.Variable(ptrInputVec3, StorageClass.Input, "inPosition");
            b.Decorate(inPos, Decoration.Location, 0);
            var inNormal = b.Variable(ptrInputVec3, StorageClass.Input, "inNormal");
            b.Decorate(inNormal, Decoration.Location, 1);
            var inUV = b.Variable(ptrInputVec2, StorageClass.Input, "inUV");
            b.Decorate(inUV, Decoration.Location, 2);
            var inColor = b.Variable(ptrInputVec3, StorageClass.Input, "inColor");
            b.Decorate(inColor, Decoration.Location, 3);

            var outNormal = b.Variable(ptrOutputVec3, StorageClass.Output, "fragNormal");
            b.Decorate(outNormal, Decoration.Location, 0);
            var outUV = b.Variable(ptrOutputVec3, StorageClass.Output, "fragUV");
            b.Decorate(outUV, Decoration.Location, 1);
            var outColor = b.Variable(ptrOutputVec3, StorageClass.Output, "fragColor");
            b.Decorate(outColor, Decoration.Location, 2);

            var mvpPtr = b.Variable(ptrUniformMat4, StorageClass.Uniform, "ubo_mvp");
            b.Decorate(mvpPtr, Decoration.DescriptorSet, 0);
            b.Decorate(mvpPtr, Decoration.Binding, 0);
            var modelPtr = b.Variable(ptrUniformMat4, StorageClass.Uniform, "ubo_model");
            b.Decorate(modelPtr, Decoration.DescriptorSet, 0);
            b.Decorate(modelPtr, Decoration.Binding, 1);

            var pos = b.Load(vec3T, inPos);
            var one = b.Constant(floatT, 1.0f);
            var pos4 = b.CompositeConstruct(vec4T, pos, one);
            var mvp = b.Load(mat4T, mvpPtr);
            var result = b.VectorTimesMatrix(vec4T, pos4, mvp);
            b.Store(glPos, result);
            var normal = b.Load(vec3T, inNormal);
            b.Store(outNormal, normal);
            var uv = b.Load(vec2T, inUV);
            var uv3 = b.CompositeConstruct(vec3T, uv, one);
            b.Store(outUV, uv3);
            var color = b.Load(vec3T, inColor);
            b.Store(outColor, color);

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        private static byte[] BuildBasicFragment()
        {
            var b = new SpirvBuilder();
            b.Capability(Capability.Shader);
            b.MemoryModel(AddressingModel.Logical, ExecutionModel.GLSL450);

            var voidT = b.TypeVoid();
            var floatT = b.TypeFloat(32);
            var vec3T = b.TypeVector(floatT, 3);
            var vec4T = b.TypeVector(floatT, 4);
            var ptrInputVec3 = b.TypePointer(StorageClass.Input, vec3T);
            var ptrOutputVec4 = b.TypePointer(StorageClass.Output, vec4T);

            var entry = b.Function(voidT, FunctionControl.None);
            b.ExecutionMode(entry, ExecutionMode.OriginUpperLeft);
            b.Label();

            var fragNormal = b.Variable(ptrInputVec3, StorageClass.Input, "fragNormal");
            b.Decorate(fragNormal, Decoration.Location, 0);
            var fragUV = b.Variable(ptrInputVec3, StorageClass.Input, "fragUV");
            b.Decorate(fragUV, Decoration.Location, 1);
            var fragColor = b.Variable(ptrInputVec3, StorageClass.Input, "fragColor");
            b.Decorate(fragColor, Decoration.Location, 2);
            var outColor = b.Variable(ptrOutputVec4, StorageClass.Output, "outColor");
            b.Decorate(outColor, Decoration.Location, 0);

            var normal = b.Load(vec3T, fragNormal);
            var lightDir = b.ConstantComposite(vec3T,
                b.Constant(floatT, 0.577f), b.Constant(floatT, 0.577f), b.Constant(floatT, 0.577f));
            var ndotl = b.DotProduct(floatT, normal, lightDir);
            var ambient = b.Constant(floatT, 0.3f);
            var diffuse = b.FMax(floatT, ndotl, ambient);
            var color = b.Load(vec3T, fragColor);
            var lit = b.VectorTimesScalar(vec3T, color, diffuse);
            var one = b.Constant(floatT, 1.0f);
            var out4 = b.CompositeConstruct(vec4T, lit, one);
            b.Store(outColor, out4);

            b.Return();
            b.FunctionEnd();
            return b.ToBytes();
        }

        // =====================================================================
        // SPIR-V Builder - Full Featured
        // =====================================================================

        public enum Capability : uint { Shader = 1, ImageQuery = 6, Image = 6 }
        public enum AddressingModel : uint { Logical = 0 }
        public enum ExecutionModel : uint { Vertex = 0, Fragment = 4, GLCompute = 1, GLSL450 = 3 }
        public enum StorageClass : uint { UniformConstant = 0, Input = 1, Uniform = 2, Output = 3, Function = 7, Private = 9 }
        public enum Decoration : uint { BuiltIn = 11, Location = 30, DescriptorSet = 33, Binding = 34, NonReadable = 25, NonWritable = 24 }
        public enum BuiltIn : uint { Position = 0, FragmentCoord = 15, VertexIndex = 42 }
        public enum FunctionControl : uint { None = 0 }
        public enum ExecutionMode : uint { OriginUpperLeft = 8 }

        public enum Op : uint
        {
            OpCapability = 19, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
            OpName = 5, OpDecorate = 71, OpMemberDecorate = 72,
            OpTypeVoid = 19, OpTypeFloat = 22, OpTypeInt = 21, OpTypeVector = 23, OpTypeMatrix = 24,
            OpTypePointer = 32, OpTypeFunction = 33,
            OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27,
            OpTypeStruct = 30,
            OpConstant = 43, OpConstantComposite = 44, OpConstantNull = 46, OpConstantTrue = 41, OpConstantFalse = 42,
            OpVariable = 59, OpLoad = 61, OpStore = 62,
            OpAccessChain = 65, OpDecorateString = 5382,
            OpFunction = 54, OpFunctionEnd = 56, OpLabel = 248, OpReturn = 253,
            OpFAdd = 129, OpFSub = 130, OpFMul = 133, OpFDiv = 134,
            OpFMax = 41, OpFMin = 40, OpFNegate = 127,
            OpFAbs = 136, OpFSqrt = 32, OpFClamp = 43,
            OpFRound = 4, OpFFloor = 10, OpFCeil = 8,
            OpFMix = 45,
            OpVectorTimesScalar = 142, OpVectorTimesMatrix = 144, OpMatrixTimesVector = 145,
            OpDotProduct = 148,
            OpDot = 149,
            OpIAdd = 128,
            OpTypeBool = 20,
            OpSelectionMerge = 247,
            OpPhi = 245,
            OpVectorShuffle = 77,
            OpCompositeConstruct = 80, OpCompositeExtract = 81, OpCompositeInsert = 82,
            OpLogicalAnd = 56, OpLogicalOr = 57, OpLogicalNot = 148, OpLogicalNotEqual = 149,
            OpSelect = 162,
            OpBranch = 249, OpBranchConditional = 250,
            OpIEqual = 167, OpINotEqual = 168,
            OpFOrdLessThan = 181, OpFOrdGreaterThan = 183,
            OpFOrdEqual = 180, OpFOrdNotEqual = 182,
            OpFOrdLessThanEqual = 184, OpFOrdGreaterThanEqual = 185,
            OpSampledImage = 444, OpImageSampleImplicitLod = 37,
            OpImageQuerySizeLod = 103, OpImageQuerySize = 102,
            OpSNegate = 127,
            OpBitcast = 124,
        }

        public class SpirvBuilder
        {
            private List<uint> _words = new();
            private uint _bound = 1;
            private List<(uint id, string name)> _names = new();
            private List<(uint id, Decoration dec, uint[] args)> _decorations = new();
            private List<(uint structId, uint member, Decoration dec, uint[] args)> _memberDecorations = new();
            private List<(uint id, uint[] args)> _executionModes = new();
            private bool _headerWritten;

            private uint Id() => _bound++;

            private void Emit(Op op, params uint[] operands)
            {
                uint wordCount = (uint)operands.Length + 1;
                _words.Add(((uint)op) | (wordCount << 16));
                _words.AddRange(operands);
            }

            public void Capability(Capability cap) => Emit(Op.OpCapability, (uint)cap);
            public void MemoryModel(AddressingModel am, ExecutionModel em) => Emit(Op.OpMemoryModel, (uint)am, (uint)em);

            // Types
            public uint TypeVoid() { var id = Id(); Emit(Op.OpTypeVoid, id); return id; }
            public uint TypeFloat(uint width) { var id = Id(); Emit(Op.OpTypeFloat, id, width); return id; }
            public uint TypeInt(uint width, uint signedness) { var id = Id(); Emit(Op.OpTypeInt, id, width, signedness); return id; }
            public uint TypeVector(uint componentType, uint componentCount) { var id = Id(); Emit(Op.OpTypeVector, id, componentType, componentCount); return id; }
            public uint TypeMatrix(uint columnType, uint columnCount) { var id = Id(); Emit(Op.OpTypeMatrix, id, columnType, columnCount); return id; }
            public uint TypePointer(StorageClass sc, uint pointeeType) { var id = Id(); Emit(Op.OpTypePointer, id, (uint)sc, pointeeType); return id; }
            public uint TypeFunction(uint returnType, params uint[] paramTypes)
            {
                var id = Id();
                _words.Add(((uint)Op.OpTypeFunction) | ((4u + (uint)paramTypes.Length) << 16));
                _words.Add(id); _words.Add(returnType);
                foreach (var p in paramTypes) _words.Add(p);
                return id;
            }

            public uint TypeImage(uint sampledType, string dim, uint depth, uint arrayed, uint multisampled, uint sampled, uint format)
            {
                var id = Id();
                uint dimVal = dim switch { "1D" => 0, "2D" => 1, "3D" => 2, "Cube" => 3, _ => 1 };
                Emit(Op.OpTypeImage, id, sampledType, dimVal, depth, arrayed, multisampled, sampled, format);
                return id;
            }

            public uint TypeSampler() { var id = Id(); Emit(Op.OpTypeSampler, id); return id; }
            public uint TypeSampledImage(uint imageType) { var id = Id(); Emit(Op.OpTypeSampledImage, id, imageType); return id; }
            public uint TypeStruct(params uint[] members) { var id = Id(); Emit(Op.OpTypeStruct, id); foreach (var m in members) _words.Add(m); return id; }
            public uint TypeBool() { var id = Id(); Emit(Op.OpTypeBool, id); return id; }

            // Constants
            public uint Constant(uint type, float value)
            {
                var id = Id();
                Emit(Op.OpConstant, type, id);
                _words.Add(BitConverter.ToUInt32(BitConverter.GetBytes(value)));
                return id;
            }

            public uint ConstantInt(uint type, int value)
            {
                var id = Id();
                Emit(Op.OpConstant, type, id);
                _words.Add(BitConverter.ToUInt32(BitConverter.GetBytes(value)));
                return id;
            }

            public uint ConstantTrue() { var id = Id(); Emit(Op.OpConstantTrue, id); return id; }
            public uint ConstantFalse() { var id = Id(); Emit(Op.OpConstantFalse, id); return id; }

            public uint ConstantComposite(uint type, params uint[] constituents)
            {
                var id = Id();
                Emit(Op.OpConstantComposite, type, id);
                foreach (var c in constituents) _words.Add(c);
                return id;
            }

            public uint ConstantNull(uint type) { var id = Id(); Emit(Op.OpConstantNull, type, id); return id; }

            // Variables
            public uint Variable(uint pointerType, StorageClass sc, string name = null)
            {
                var id = Id();
                Emit(Op.OpVariable, pointerType, id, (uint)sc);
                if (name != null) _names.Add((id, name));
                return id;
            }

            public uint Load(uint resultType, uint pointer)
            {
                var id = Id();
                Emit(Op.OpLoad, resultType, id, pointer);
                return id;
            }

            public void Store(uint pointer, uint value) => Emit(Op.OpStore, pointer, value);

            public uint AccessChain(uint resultType, uint baseId, params uint[] indices)
            {
                var id = Id();
                Emit(Op.OpAccessChain, resultType, id, baseId);
                foreach (var idx in indices) _words.Add(idx);
                return id;
            }

            // Functions
            public uint Function(uint returnType, FunctionControl control)
            {
                var id = Id();
                var funcType = TypeFunction(returnType);
                Emit(Op.OpFunction, returnType, id, (uint)control, funcType);
                return id;
            }

            public void FunctionEnd() => Emit(Op.OpFunctionEnd);
            public uint Label() { var id = Id(); Emit(Op.OpLabel, id); return id; }
            public void Return() => Emit(Op.OpReturn);

            // Control flow
            public void Branch(uint targetLabel) => Emit(Op.OpBranch, targetLabel);

            public void BranchConditional(uint condition, uint trueLabel, uint falseLabel)
                => Emit(Op.OpBranchConditional, condition, trueLabel, falseLabel);

            public void SelectionMerge(uint mergeLabel, uint selectionControl = 0)
                => Emit(Op.OpSelectionMerge, mergeLabel, selectionControl);

            public uint Phi(uint type, params (uint value, uint label)[] args)
            {
                var id = Id();
                Emit(Op.OpPhi, type, id);
                foreach (var (val, lbl) in args) { _words.Add(val); _words.Add(lbl); }
                return id;
            }

            // Math - Float
            public uint FAdd(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFAdd, type, id, a, b); return id; }
            public uint FSub(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFSub, type, id, a, b); return id; }
            public uint FMul(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFMul, type, id, a, b); return id; }
            public uint FDiv(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFDiv, type, id, a, b); return id; }
            public uint FMax(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFMax, type, id, a, b); return id; }
            public uint FMin(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpFMin, type, id, a, b); return id; }
            public uint FAbs(uint type, uint operand) { var id = Id(); Emit(Op.OpFAbs, type, id, operand); return id; }
            public uint FSqrt(uint type, uint operand) { var id = Id(); Emit(Op.OpFSqrt, type, id, operand); return id; }
            public uint FClamp(uint type, uint x, uint minVal, uint maxVal)
            { var id = Id(); Emit(Op.OpFClamp, type, id, x, minVal, maxVal); return id; }
            public uint FMix(uint type, uint x, uint y, uint a)
            { var id = Id(); Emit(Op.OpFMix, type, id, x, y, a); return id; }
            public uint FRound(uint type, uint operand) { var id = Id(); Emit(Op.OpFRound, type, id, operand); return id; }
            public uint FFloor(uint type, uint operand) { var id = Id(); Emit(Op.OpFFloor, type, id, operand); return id; }
            public uint FCeil(uint type, uint operand) { var id = Id(); Emit(Op.OpFCeil, type, id, operand); return id; }
            public uint FNegate(uint type, uint operand) { var id = Id(); Emit(Op.OpFNegate, type, id, operand); return id; }
            public uint DotProduct(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpDot, type, id, a, b); return id; }
            public uint VectorTimesScalar(uint type, uint vec, uint scalar) { var id = Id(); Emit(Op.OpVectorTimesScalar, type, id, vec, scalar); return id; }
            public uint VectorTimesMatrix(uint type, uint vec, uint mat) { var id = Id(); Emit(Op.OpVectorTimesMatrix, type, id, vec, mat); return id; }
            public uint MatrixTimesVector(uint type, uint mat, uint vec) { var id = Id(); Emit(Op.OpMatrixTimesVector, type, id, mat, vec); return id; }

            // Math - Integer
            public uint IAdd(uint type, uint a, uint b) { var id = Id(); Emit(Op.OpIAdd, type, id, a, b); return id; }
            public uint IEqual(uint a, uint b)
            {
                var id = Id();
                Emit(Op.OpIEqual, TypeBool(), id, a, b);
                return id;
            }

            // Vector operations
            public uint CompositeConstruct(uint type, params uint[] constituents)
            {
                var id = Id();
                Emit(Op.OpCompositeConstruct, type, id);
                foreach (var c in constituents) _words.Add(c);
                return id;
            }

            public uint CompositeExtract(uint type, uint composite, uint index)
            {
                var id = Id();
                Emit(Op.OpCompositeExtract, type, id, composite, index);
                return id;
            }

            public uint CompositeInsert(uint type, uint value, uint composite, params uint[] indices)
            {
                var id = Id();
                Emit(Op.OpCompositeInsert, type, id, value, composite);
                foreach (var idx in indices) _words.Add(idx);
                return id;
            }

            public uint VectorShuffle(uint type, uint vec1, uint vec2, params uint[] components)
            {
                var id = Id();
                Emit(Op.OpVectorShuffle, type, id, vec1, vec2);
                foreach (var c in components) _words.Add(c);
                return id;
            }

            // Normalize helper - computes vec / length(vec)
            public uint Normalize(uint vecType, uint vec, uint floatType)
            {
                var lenSq = DotProduct(floatType, vec, vec);
                var len = FSqrt(floatType, lenSq);
                var one = Constant(floatType, 1.0f);
                var invLen = FDiv(floatType, one, len);
                return VectorTimesScalar(vecType, vec, invLen);
            }

            // Comparison
            public uint FOrdLessThan(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdLessThan, resultType, id, a, b); return id; }
            public uint FOrdGreaterThan(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdGreaterThan, resultType, id, a, b); return id; }
            public uint FOrdEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdEqual, resultType, id, a, b); return id; }
            public uint FOrdNotEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdNotEqual, resultType, id, a, b); return id; }
            public uint FOrdLessThanEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdLessThanEqual, resultType, id, a, b); return id; }
            public uint FOrdGreaterThanEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpFOrdGreaterThanEqual, resultType, id, a, b); return id; }
            public uint IEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpIEqual, resultType, id, a, b); return id; }
            public uint INotEqual(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpINotEqual, resultType, id, a, b); return id; }

            // Logic
            public uint LogicalAnd(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpLogicalAnd, resultType, id, a, b); return id; }
            public uint LogicalOr(uint resultType, uint a, uint b) { var id = Id(); Emit(Op.OpLogicalOr, resultType, id, a, b); return id; }
            public uint LogicalNot(uint resultType, uint operand) { var id = Id(); Emit(Op.OpLogicalNot, resultType, id, operand); return id; }

            // Select (ternary)
            public uint Select(uint resultType, uint condition, uint trueValue, uint falseValue)
            {
                var id = Id();
                Emit(Op.OpSelect, resultType, id, condition, trueValue, falseValue);
                return id;
            }

            // Image/Sampler
            public uint SampledImage(uint sampledImageType, uint image, uint sampler)
            {
                var id = Id();
                Emit(Op.OpSampledImage, sampledImageType, id, image, sampler);
                return id;
            }

            public uint ImageSampleImplicitLod(uint resultType, uint sampledImage, uint coordinate)
            {
                var id = Id();
                Emit(Op.OpImageSampleImplicitLod, resultType, id, sampledImage, coordinate);
                return id;
            }

            public uint ImageSampleImplicitLod(uint resultType, uint sampledImage, uint coordinate, uint bias)
            {
                var id = Id();
                Emit(Op.OpImageSampleImplicitLod, resultType, id, sampledImage, coordinate);
                _words.Add(bias);
                return id;
            }

            public uint ImageQuerySizeLod(uint resultType, uint image, uint lod)
            {
                var id = Id();
                Emit(Op.OpImageQuerySizeLod, resultType, id, image, lod);
                return id;
            }

            // Decorations
            public void Decorate(uint target, Decoration decoration, params uint[] args)
            {
                _decorations.Add((target, decoration, args));
            }

            public void MemberDecorate(uint structType, uint member, Decoration decoration, params uint[] args)
            {
                _memberDecorations.Add((structType, member, decoration, args));
            }

            public void EntryPoint(ExecutionModel model, uint entryPointId, string name, params uint[] interfaces)
            {
                var nameBytes = Encoding.UTF8.GetBytes(name + '\0');
                var nameWords = new List<uint>();
                foreach (var bt in nameBytes) { nameWords.Add(bt); }
                Emit(Op.OpEntryPoint, (uint)model, entryPointId);
                foreach (var w in nameWords) _words.Add(w);
                foreach (var iface in interfaces) _words.Add(iface);
            }

            public void ExecutionMode(uint entryPoint, ExecutionMode mode, params uint[] args)
            {
                _executionModes.Add((entryPoint, args.Prepend((uint)mode).ToArray()));
            }

            // ToBytes
            public byte[] ToBytes()
            {
                var header = new List<uint> { MAGIC, VERSION_1_0, GENERATOR, _bound, 0 };
                var allWords = new List<uint>(header);

                foreach (var (id, dec, args) in _decorations)
                {
                    allWords.Add(((uint)Op.OpDecorate) | (((uint)args.Length + 3) << 16));
                    allWords.Add(id); allWords.Add((uint)dec);
                    foreach (var a in args) allWords.Add(a);
                }

                foreach (var (structId, member, dec, args) in _memberDecorations)
                {
                    allWords.Add(((uint)Op.OpMemberDecorate) | (((uint)args.Length + 4) << 16));
                    allWords.Add(structId); allWords.Add(member); allWords.Add((uint)dec);
                    foreach (var a in args) allWords.Add(a);
                }

                foreach (var (id, args) in _executionModes)
                {
                    allWords.Add(((uint)Op.OpExecutionMode) | (((uint)args.Length + 3) << 16));
                    allWords.Add(id);
                    foreach (var a in args) allWords.Add(a);
                }

                foreach (var (id, name) in _names)
                {
                    var nameBytes = Encoding.UTF8.GetBytes(name + '\0');
                    int nameWordCount = (nameBytes.Length + 3) / 4;
                    allWords.Add(((uint)Op.OpName) | ((uint)(2 + nameWordCount) << 16));
                    allWords.Add(id);
                    for (int i = 0; i < nameWordCount; i++)
                    {
                        uint w = 0;
                        for (int j = 0; j < 4 && i * 4 + j < nameBytes.Length; j++)
                            w |= (uint)nameBytes[i * 4 + j] << (j * 8);
                        allWords.Add(w);
                    }
                }

                allWords.AddRange(_words);

                var bytes = new byte[allWords.Count * 4];
                Buffer.BlockCopy(allWords.ToArray(), 0, bytes, 0, bytes.Length);
                return bytes;
            }
        }
    }
}
