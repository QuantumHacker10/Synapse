#version 460 core
// =============================================================================
// MeshMeshShader.glsl - Mesh Shader for Vulkan 1.3+
// Replaces traditional vertex/geometry shaders
// Generates vertices and primitives directly on GPU
// =============================================================================

// Input from task shader
layout(triangles) in;
layout(max_vertices = 256) out;
layout(max_primitives = 128) out;

// Per-vertex output
struct MeshVertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec3 color;
};

// Output primitives
out MeshVertex gl_MeshVertices[];

// Uniform buffer
layout(set = 0, binding = 0) uniform MeshShaderParams {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec3 cameraPosition;
    float time;
} params;

// Task work group ID
layout(location = 0) in uvec3 gl_TaskWorkGroupID;
layout(location = 1) in uvec3 gl_LocalInvocationID;
layout(location = 2) in uvec3 gl_GlobalInvocationID;

// For a simple example: generate a quad mesh
// In a real implementation, this would generate complex geometry

void main() {
    // Simple quad generation (2 triangles = 6 vertices)
    // This is just an example - real mesh shaders would generate more complex geometry
    
    vec3 center = vec3(gl_TaskWorkGroupID.xy * 2.0, 0.0);
    float scale = 1.0;
    
    // Triangle 1
    gl_MeshVertices[0].position = center + vec3(-scale, -scale, 0.0);
    gl_MeshVertices[0].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[0].uv = vec2(0.0, 0.0);
    gl_MeshVertices[0].color = vec3(1.0, 0.0, 0.0);
    
    gl_MeshVertices[1].position = center + vec3(scale, -scale, 0.0);
    gl_MeshVertices[1].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[1].uv = vec2(1.0, 0.0);
    gl_MeshVertices[1].color = vec3(0.0, 1.0, 0.0);
    
    gl_MeshVertices[2].position = center + vec3(-scale, scale, 0.0);
    gl_MeshVertices[2].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[2].uv = vec2(0.0, 1.0);
    gl_MeshVertices[2].color = vec3(0.0, 0.0, 1.0);
    
    // Triangle 2
    gl_MeshVertices[3].position = center + vec3(scale, -scale, 0.0);
    gl_MeshVertices[3].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[3].uv = vec2(1.0, 0.0);
    gl_MeshVertices[3].color = vec3(0.0, 1.0, 0.0);
    
    gl_MeshVertices[4].position = center + vec3(scale, scale, 0.0);
    gl_MeshVertices[4].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[4].uv = vec2(1.0, 1.0);
    gl_MeshVertices[4].color = vec3(1.0, 1.0, 0.0);
    
    gl_MeshVertices[5].position = center + vec3(-scale, scale, 0.0);
    gl_MeshVertices[5].normal = vec3(0.0, 0.0, 1.0);
    gl_MeshVertices[5].uv = vec2(0.0, 1.0);
    gl_MeshVertices[5].color = vec3(0.0, 0.0, 1.0);
    
    // Emit 2 triangles (6 vertices)
    gl_PrimitiveCountNV = 2;
    
    // Set primitive indices (2 triangles)
    gl_PrimitiveIndicesNV[0] = 0;
    gl_PrimitiveIndicesNV[1] = 1;
    gl_PrimitiveIndicesNV[2] = 2;
    gl_PrimitiveIndicesNV[3] = 3;
    gl_PrimitiveIndicesNV[4] = 4;
    gl_PrimitiveIndicesNV[5] = 5;
}