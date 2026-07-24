#version 460 core
// =============================================================================
// MeshTaskShader.glsl - Task Shader for Mesh Shader Pipeline
// Vulkan 1.3+ with VK_EXT_mesh_shader
// Generates work items for mesh processing
// =============================================================================

// Input: Task shader doesn't take input vertices
// Output: gl_TaskWorkGroupID, gl_LocalInvocationID, gl_GlobalInvocationID

// Task shader output
struct TaskPayload {
    vec4 data[4]; // Can be used to pass data to mesh shader
};

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 256) out;
layout(max_primitives = 128) out;

// This task shader generates one work item per cluster
// Each work item will invoke the mesh shader

// For a simple implementation, we generate one mesh per task work group
// In a more complex scenario, this would generate multiple meshes

void main() {
    // Generate a single work item
    gl_WorkGroupSize = uvec3(1, 1, 1);
    
    // The mesh shader will be invoked once for this work item
    // We can pass data through gl_TaskWorkGroupID
    
    // For now, just emit one work item
    gl_PrimitiveCountNV = 0; // Will be set by mesh shader
}