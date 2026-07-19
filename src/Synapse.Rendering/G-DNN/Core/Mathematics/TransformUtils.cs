using System;
// ============================================================
// FILE: TransformUtils.cs
// PATH: Core/Mathematics/TransformUtils.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.Mathematics;

/// <summary>
/// Transform hierarchy utilities for the G-DNN neural geometry engine.
/// Provides local-to-world/world-to-local conversions, skeletal animation support,
/// keyframe interpolation, transform stream compression, and matrix palette generation.
/// </summary>
public static class TransformUtils
{
    private const float Epsilon = 1e-6f;
    private const int MaxJoints = 256;

    #region Transform Hierarchy

    /// <summary>
    /// Represents a node in a transform hierarchy with parent reference.
    /// </summary>
    public struct TransformNode
    {
        /// <summary>Local transform relative to parent.</summary>
        public Matrix4x4 LocalTransform;

        /// <summary>Index of the parent node (-1 for root).</summary>
        public int ParentIndex;

        /// <summary>Cached world transform.</summary>
        public Matrix4x4 WorldTransform;

        /// <summary>Whether the world transform needs recalculation.</summary>
        public bool IsDirty;

        /// <summary>Node name for debugging.</summary>
        public string Name;
    }

    /// <summary>
    /// Evaluate a transform hierarchy from local transforms to world transforms.
    /// Parents are evaluated before children.
    /// </summary>
    public static void EvaluateHierarchy(Span<TransformNode> nodes)
    {
        // First pass: compute world transforms
        for (int i = 0; i < nodes.Length; i++)
        {
            if (!nodes[i].IsDirty)
                continue;

            if (nodes[i].ParentIndex < 0)
            {
                nodes[i].WorldTransform = nodes[i].LocalTransform;
            }
            else
            {
                nodes[i].WorldTransform = Matrix4x4.Multiply(
                    nodes[nodes[i].ParentIndex].WorldTransform,
                    nodes[i].LocalTransform);
            }

            nodes[i].IsDirty = false;
        }
    }

    /// <summary>
    /// Mark a node and all its descendants as dirty.
    /// </summary>
    public static void MarkDirty(Span<TransformNode> nodes, int nodeIndex)
    {
        if (nodeIndex < 0 || nodeIndex >= nodes.Length)
            return;

        nodes[nodeIndex].IsDirty = true;

        for (int i = 0; i < nodes.Length; i++)
        {
            if (nodes[i].ParentIndex == nodeIndex)
                MarkDirty(nodes, i);
        }
    }

    /// <summary>
    /// Compute the local transform from a world transform and parent world transform.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 WorldToLocal(Matrix4x4 worldTransform, Matrix4x4 parentWorldTransform)
    {
        Matrix4x4.Invert(parentWorldTransform, out Matrix4x4 invParent);
        return Matrix4x4.Multiply(invParent, worldTransform);
    }

    /// <summary>
    /// Compute the world transform from a local transform and parent world transform.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 LocalToWorld(Matrix4x4 localTransform, Matrix4x4 parentWorldTransform) =>
        Matrix4x4.Multiply(parentWorldTransform, localTransform);

    /// <summary>
    /// Compute the world position of a point given a chain of parent transforms.
    /// </summary>
    public static Vector3 TransformPointHierarchy(ReadOnlySpan<Matrix4x4> transformChain, Vector3 localPoint)
    {
        Vector3 point = localPoint;
        for (int i = 0; i < transformChain.Length; i++)
            point = Vector3.Transform(point, transformChain[i]);
        return point;
    }

    /// <summary>
    /// Compute the world normal of a direction given a chain of parent transforms.
    /// Uses the inverse transpose for correct normals under non-uniform scaling.
    /// </summary>
    public static Vector3 TransformNormalHierarchy(ReadOnlySpan<Matrix4x4> transformChain, Vector3 localNormal)
    {
        Vector3 normal = localNormal;
        for (int i = transformChain.Length - 1; i >= 0; i--)
        {
            Matrix4x4.Invert(transformChain[i], out Matrix4x4 inv);
            Matrix4x4 invT = Matrix4x4.Transpose(inv);
            normal = Vector3.TransformNormal(normal, invT);
        }
        return normal;
    }

    /// <summary>
    /// Build a transform chain from a node to the root.
    /// Returns the number of transforms written to the output span.
    /// </summary>
    public static int BuildTransformChain(ReadOnlySpan<TransformNode> nodes, int nodeIndex,
        Span<Matrix4x4> chain)
    {
        Debug.Assert(nodeIndex >= 0 && nodeIndex < nodes.Length);

        int count = 0;
        int current = nodeIndex;

        while (current >= 0 && count < chain.Length)
        {
            chain[count++] = nodes[current].WorldTransform;
            current = nodes[current].ParentIndex;
        }

        return count;
    }

    /// <summary>
    /// Find the lowest common ancestor of two nodes.
    /// </summary>
    public static int FindLowestCommonAncestor(ReadOnlySpan<TransformNode> nodes, int nodeA, int nodeB)
    {
        Span<bool> visitedA = stackalloc bool[nodes.Length];
        int current = nodeA;

        while (current >= 0)
        {
            visitedA[current] = true;
            current = nodes[current].ParentIndex;
        }

        current = nodeB;
        while (current >= 0)
        {
            if (visitedA[current])
                return current;
            current = nodes[current].ParentIndex;
        }

        return -1;
    }

    /// <summary>
    /// Compute the relative transform between two nodes.
    /// </summary>
    public static Matrix4x4 RelativeTransform(ReadOnlySpan<TransformNode> nodes, int fromNode, int toNode)
    {
        nodes[fromNode].WorldTransform.DecomposeTRS(out Vector3 tA, out Quaternion rA, out Vector3 sA);
        nodes[toNode].WorldTransform.DecomposeTRS(out Vector3 tB, out Quaternion rB, out Vector3 sB);

        if (!Matrix4x4.Invert(nodes[fromNode].WorldTransform, out Matrix4x4 invA))
            invA = Matrix4x4.Identity;
        return Matrix4x4.Multiply(invA, nodes[toNode].WorldTransform);
    }

    #endregion

    #region Inverse Bind Pose

    /// <summary>
    /// Compute inverse bind pose matrices for skeletal animation.
    /// </summary>
    public static void ComputeInverseBindPoses(ReadOnlySpan<Matrix4x4> bindPoseMatrices,
        Span<Matrix4x4> inverseBindPoseMatrices)
    {
        Debug.Assert(bindPoseMatrices.Length == inverseBindPoseMatrices.Length);

        for (int i = 0; i < bindPoseMatrices.Length; i++)
        {
            if (!Matrix4x4.Invert(bindPoseMatrices[i], out inverseBindPoseMatrices[i]))
                inverseBindPoseMatrices[i] = Matrix4x4.Identity;
        }
    }

    /// <summary>
    /// Compute the skinning matrix for a single joint: inverseBindPose * currentWorld.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 ComputeSkinningMatrix(Matrix4x4 currentWorldPose, Matrix4x4 inverseBindPose) =>
        Matrix4x4.Multiply(inverseBindPose, currentWorldPose);

    /// <summary>
    /// Compute all skinning matrices for a skeleton.
    /// </summary>
    public static void ComputeSkinningMatrices(ReadOnlySpan<Matrix4x4> currentWorldPose,
        ReadOnlySpan<Matrix4x4> inverseBindPose, Span<Matrix4x4> skinningMatrices)
    {
        int count = Math.Min(currentWorldPose.Length, Math.Min(inverseBindPose.Length, skinningMatrices.Length));

        for (int i = 0; i < count; i++)
            skinningMatrices[i] = ComputeSkinningMatrix(currentWorldPose[i], inverseBindPose[i]);
    }

    /// <summary>
    /// Compute the bind pose from a reference pose and current pose.
    /// </summary>
    public static void ComputeBindPoseFromReference(ReadOnlySpan<Matrix4x4> referencePose,
        ReadOnlySpan<Matrix4x4> currentPose, Span<Matrix4x4> bindPose)
    {
        int count = Math.Min(referencePose.Length, Math.Min(currentPose.Length, bindPose.Length));

        for (int i = 0; i < count; i++)
        {
            Matrix4x4.Invert(referencePose[i], out Matrix4x4 invRef);
            bindPose[i] = Matrix4x4.Multiply(invRef, currentPose[i]);
        }
    }

    #endregion

    #region Joint Space Transformations

    /// <summary>
    /// Transform a vertex from object space through a skinning matrix palette.
    /// </summary>
    public static Vector3 SkinningTransform(Vector3 position, Vector3 normal,
        ReadOnlySpan<Matrix4x4> skinningMatrices,
        ReadOnlySpan<int> jointIndices, ReadOnlySpan<float> jointWeights)
    {
        Vector3 skinnedPosition = Vector3.Zero;
        Vector3 skinnedNormal = Vector3.Zero;

        for (int i = 0; i < jointIndices.Length; i++)
        {
            int idx = jointIndices[i];
            float weight = jointWeights[i];

            if (weight < Epsilon)
                continue;

            Vector3 transformedPos = Vector3.Transform(position, skinningMatrices[idx]);
            Vector3 transformedNormal = Vector3.TransformNormal(normal, skinningMatrices[idx]);

            skinnedPosition += transformedPos * weight;
            skinnedNormal += transformedNormal * weight;
        }

        return skinnedPosition;
    }

    /// <summary>
    /// Transform a batch of vertices through skinning matrices.
    /// </summary>
    public static void SkinningTransformBatch(
        ReadOnlySpan<Vector3> positions, ReadOnlySpan<Vector3> normals,
        ReadOnlySpan<Matrix4x4> skinningMatrices,
        ReadOnlySpan<JointWeight[]> jointWeights,
        Span<Vector3> outPositions, Span<Vector3> outNormals)
    {
        int count = Math.Min(positions.Length, Math.Min(outPositions.Length, outNormals.Length));

        for (int v = 0; v < count; v++)
        {
            Vector3 pos = Vector3.Zero;
            Vector3 norm = Vector3.Zero;
            JointWeight[] weights = jointWeights[v];

            for (int j = 0; j < weights.Length; j++)
            {
                float w = weights[j].Weight;
                if (w < Epsilon)
                    continue;

                pos += Vector3.Transform(positions[v], skinningMatrices[weights[j].JointIndex]) * w;
                norm += Vector3.TransformNormal(normals[v], skinningMatrices[weights[j].JointIndex]) * w;
            }

            outPositions[v] = pos;
            outNormals[v] = Vector3.Normalize(norm);
        }
    }

    /// <summary>
    /// Joint weight data structure for skinning.
    /// </summary>
    public struct JointWeight
    {
        /// <summary>Index of the joint in the skeleton.</summary>
        public int JointIndex;

        /// <summary>Weight of this joint's influence.</summary>
        public float Weight;
    }

    /// <summary>
    /// Normalize joint weights to sum to 1.0.
    /// </summary>
    public static void NormalizeJointWeights(Span<JointWeight> weights)
    {
        float sum = 0f;
        for (int i = 0; i < weights.Length; i++)
            sum += weights[i].Weight;

        if (sum < Epsilon)
            return;

        float invSum = 1f / sum;
        for (int i = 0; i < weights.Length; i++)
            weights[i].Weight *= invSum;
    }

    /// <summary>
    /// Limit the number of joint influences per vertex to a maximum.
    /// Keeps the top N weights and renormalizes.
    /// </summary>
    public static void LimitJointInfluences(Span<JointWeight> weights, int maxInfluences)
    {
        if (weights.Length <= maxInfluences)
            return;

        // Sort by weight descending
        for (int i = 0; i < maxInfluences; i++)
        {
            int maxIdx = i;
            for (int j = i + 1; j < weights.Length; j++)
            {
                if (weights[j].Weight > weights[maxIdx].Weight)
                    maxIdx = j;
            }
            if (maxIdx != i)
                (weights[i], weights[maxIdx]) = (weights[maxIdx], weights[i]);
        }

        float sum = 0f;
        for (int i = 0; i < maxInfluences; i++)
            sum += weights[i].Weight;

        if (sum > Epsilon)
        {
            float invSum = 1f / sum;
            for (int i = 0; i < maxInfluences; i++)
                weights[i].Weight *= invSum;
        }
    }

    #endregion

    #region Curve Evaluation for Animation Paths

    /// <summary>
    /// Evaluate a position along a keyframed animation curve.
    /// </summary>
    public static Vector3 EvaluatePositionCurve(ReadOnlySpan<AnimationKeyframe> keyframes, float time)
    {
        if (keyframes.Length == 0)
            return Vector3.Zero;
        if (keyframes.Length == 1)
            return keyframes[0].Position;

        if (time <= keyframes[0].Time)
            return keyframes[0].Position;
        if (time >= keyframes[^1].Time)
            return keyframes[^1].Position;

        int segment = FindSegment(keyframes, time);
        float t = (time - keyframes[segment].Time) /
                  (keyframes[segment + 1].Time - keyframes[segment].Time);

        return keyframes[segment].InterpolationType switch
        {
            InterpolationType.Linear => Vector3.Lerp(keyframes[segment].Position, keyframes[segment + 1].Position, t),
            InterpolationType.Hermite => VectorMath.HermiteInterpolate(
                keyframes[segment].Position, keyframes[segment + 1].Position,
                keyframes[segment].OutTangent, keyframes[segment + 1].InTangent, t),
            InterpolationType.CatmullRom => EvaluateCatmullRomPosition(keyframes, segment, t),
            _ => Vector3.Lerp(keyframes[segment].Position, keyframes[segment + 1].Position, t),
        };
    }

    /// <summary>
    /// Evaluate a rotation along a keyframed animation curve.
    /// </summary>
    public static Quaternion EvaluateRotationCurve(ReadOnlySpan<AnimationKeyframe> keyframes, float time)
    {
        if (keyframes.Length == 0)
            return Quaternion.Identity;
        if (keyframes.Length == 1)
            return keyframes[0].Rotation;

        if (time <= keyframes[0].Time)
            return keyframes[0].Rotation;
        if (time >= keyframes[^1].Time)
            return keyframes[^1].Rotation;

        int segment = FindSegment(keyframes, time);
        float t = (time - keyframes[segment].Time) /
                  (keyframes[segment + 1].Time - keyframes[segment].Time);

        Quaternion a = keyframes[segment].Rotation;
        Quaternion b = keyframes[segment + 1].Rotation;

        if (Quaternion.Dot(a, b) < 0)
            b = -b;

        return Quaternion.Slerp(a, b, t);
    }

    /// <summary>
    /// Evaluate a scale along a keyframed animation curve.
    /// </summary>
    public static Vector3 EvaluateScaleCurve(ReadOnlySpan<AnimationKeyframe> keyframes, float time)
    {
        if (keyframes.Length == 0)
            return Vector3.One;
        if (keyframes.Length == 1)
            return keyframes[0].Scale;

        if (time <= keyframes[0].Time)
            return keyframes[0].Scale;
        if (time >= keyframes[^1].Time)
            return keyframes[^1].Scale;

        int segment = FindSegment(keyframes, time);
        float t = (time - keyframes[segment].Time) /
                  (keyframes[segment + 1].Time - keyframes[segment].Time);

        return Vector3.Lerp(keyframes[segment].Scale, keyframes[segment + 1].Scale, t);
    }

    /// <summary>
    /// Evaluate a complete TRS transform from keyframe curves.
    /// </summary>
    public static Matrix4x4 EvaluateTransformCurve(ReadOnlySpan<AnimationKeyframe> positionKeyframes,
        ReadOnlySpan<AnimationKeyframe> rotationKeyframes, ReadOnlySpan<AnimationKeyframe> scaleKeyframes,
        float time)
    {
        Vector3 pos = EvaluatePositionCurve(positionKeyframes, time);
        Quaternion rot = EvaluateRotationCurve(rotationKeyframes, time);
        Vector3 scale = EvaluateScaleCurve(scaleKeyframes, time);

        return TransformUtils.BuildTRS(pos, rot, scale);
    }

    private static int FindSegment(ReadOnlySpan<AnimationKeyframe> keyframes, float time)
    {
        for (int i = 0; i < keyframes.Length - 1; i++)
        {
            if (time >= keyframes[i].Time && time <= keyframes[i + 1].Time)
                return i;
        }
        return keyframes.Length - 2;
    }

    private static Vector3 EvaluateCatmullRomPosition(ReadOnlySpan<AnimationKeyframe> keyframes, int segment, float t)
    {
        Vector3 p0 = segment > 0 ? keyframes[segment - 1].Position : keyframes[segment].Position;
        Vector3 p1 = keyframes[segment].Position;
        Vector3 p2 = keyframes[segment + 1].Position;
        Vector3 p3 = segment + 2 < keyframes.Length ? keyframes[segment + 2].Position : keyframes[segment + 1].Position;

        return VectorMath.CatmullRom(p0, p1, p2, p3, t);
    }

    #endregion

    #region Keyframe Interpolation

    /// <summary>
    /// Animation keyframe data structure.
    /// </summary>
    public struct AnimationKeyframe
    {
        /// <summary>Time of this keyframe in seconds.</summary>
        public float Time;

        /// <summary>Position at this keyframe.</summary>
        public Vector3 Position;

        /// <summary>Rotation at this keyframe.</summary>
        public Quaternion Rotation;

        /// <summary>Scale at this keyframe.</summary>
        public Vector3 Scale;

        /// <summary>Incoming tangent (for Hermite interpolation).</summary>
        public Vector3 InTangent;

        /// <summary>Outgoing tangent (for Hermite interpolation).</summary>
        public Vector3 OutTangent;

        /// <summary>Interpolation type for this segment.</summary>
        public InterpolationType InterpolationType;
    }

    /// <summary>
    /// Supported interpolation types for animation curves.
    /// </summary>
    public enum InterpolationType
    {
        /// <summary>Linear interpolation.</summary>
        Linear,

        /// <summary>Hermite spline interpolation.</summary>
        Hermite,

        /// <summary>Catmull-Rom spline interpolation.</summary>
        CatmullRom,

        /// <summary>Cubic Bezier interpolation.</summary>
        Bezier,

        /// <summary>Step (constant) interpolation.</summary>
        Step
    }

    /// <summary>
    /// Interpolate between two keyframes using linear interpolation.
    /// </summary>
    public static void InterpolateLinear(AnimationKeyframe a, AnimationKeyframe b, float t,
        out Vector3 position, out Quaternion rotation, out Vector3 scale)
    {
        position = Vector3.Lerp(a.Position, b.Position, t);
        rotation = Quaternion.Slerp(a.Rotation, b.Rotation, t);
        scale = Vector3.Lerp(a.Scale, b.Scale, t);
    }

    /// <summary>
    /// Interpolate between keyframes using Catmull-Rom spline.
    /// </summary>
    public static void InterpolateCatmullRom(ReadOnlySpan<AnimationKeyframe> keyframes, int segment, float t,
        out Vector3 position, out Quaternion rotation, out Vector3 scale)
    {
        position = EvaluateCatmullRomPosition(keyframes, segment, t);

        // Catmull-Rom for rotation
        Quaternion r0 = segment > 0 ? keyframes[segment - 1].Rotation : keyframes[segment].Rotation;
        Quaternion r1 = keyframes[segment].Rotation;
        Quaternion r2 = keyframes[segment + 1].Rotation;
        Quaternion r3 = segment + 2 < keyframes.Length ? keyframes[segment + 2].Rotation : keyframes[segment + 1].Rotation;

        // Simple SLERP approximation for catmull-rom rotation
        Quaternion q12 = Quaternion.Slerp(r1, r2, t);
        rotation = q12;

        // Catmull-Rom for scale
        Vector3 s0 = segment > 0 ? keyframes[segment - 1].Scale : keyframes[segment].Scale;
        Vector3 s1 = keyframes[segment].Scale;
        Vector3 s2 = keyframes[segment + 1].Scale;
        Vector3 s3 = segment + 2 < keyframes.Length ? keyframes[segment + 2].Scale : keyframes[segment + 1].Scale;
        scale = VectorMath.CatmullRom(s0, s1, s2, s3, t);
    }

    /// <summary>
    /// Compute tangents for Hermite interpolation from keyframe positions.
    /// </summary>
    public static void ComputeHermiteTangents(Span<AnimationKeyframe> keyframes)
    {
        for (int i = 0; i < keyframes.Length; i++)
        {
            Vector3 prev = i > 0 ? keyframes[i - 1].Position : keyframes[i].Position;
            Vector3 next = i < keyframes.Length - 1 ? keyframes[i + 1].Position : keyframes[i].Position;

            keyframes[i].OutTangent = (next - prev) * 0.5f;
            keyframes[i].InTangent = keyframes[i].OutTangent;
        }
    }

    /// <summary>
    /// Compute tangents using automatic tangent mode (similar to Unity's AnimationCurve).
    /// </summary>
    public static void ComputeAutoTangents(Span<AnimationKeyframe> keyframes)
    {
        for (int i = 0; i < keyframes.Length; i++)
        {
            Vector3 tangent;

            if (i == 0)
            {
                tangent = keyframes[1].Position - keyframes[0].Position;
            }
            else if (i == keyframes.Length - 1)
            {
                tangent = keyframes[^1].Position - keyframes[^2].Position;
            }
            else
            {
                tangent = (keyframes[i + 1].Position - keyframes[i - 1].Position) * 0.5f;
            }

            keyframes[i].InTangent = tangent;
            keyframes[i].OutTangent = tangent;
        }
    }

    #endregion

    #region Transform Stream Compression

    /// <summary>
    /// Compression format for transform streams.
    /// </summary>
    public enum CompressionFormat
    {
        /// <summary>No compression (full precision).</summary>
        None,

        /// <summary>Quantized to 16-bit per component.</summary>
        Quantized16,

        /// <summary>Quaternion compression using smallest-three encoding.</summary>
        QuaternionSmallestThree,

        /// <summary>Full TRS compression with quantized position and scale.</summary>
        FullCompressed
    }

    /// <summary>
    /// Compressed transform data.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct CompressedTransform
    {
        public Half PosX, PosY, PosZ;
        public Half RotX, RotY, RotZ, RotW;
        public Half ScaleX, ScaleY, ScaleZ;
        public CompressionFormat Format;
    }

    /// <summary>
    /// Compress a Matrix4x4 into a compact TRS representation.
    /// </summary>
    public static CompressedTransform CompressTransform(Matrix4x4 matrix, CompressionFormat format = CompressionFormat.Quantized16)
    {
        matrix.DecomposeTRS(out Vector3 translation, out Quaternion rotation, out Vector3 scale);

        return new CompressedTransform
        {
            PosX = (Half)translation.X,
            PosY = (Half)translation.Y,
            PosZ = (Half)translation.Z,
            RotX = (Half)rotation.X,
            RotY = (Half)rotation.Y,
            RotZ = (Half)rotation.Z,
            RotW = (Half)rotation.W,
            ScaleX = (Half)scale.X,
            ScaleY = (Half)scale.Y,
            ScaleZ = (Half)scale.Z,
            Format = format
        };
    }

    /// <summary>
    /// Decompress a CompressedTransform back to a Matrix4x4.
    /// </summary>
    public static Matrix4x4 DecompressTransform(CompressedTransform compressed)
    {
        Vector3 translation = new((float)compressed.PosX, (float)compressed.PosY, (float)compressed.PosZ);
        Quaternion rotation = Quaternion.Normalize(new Quaternion(
            (float)compressed.RotX, (float)compressed.RotY,
            (float)compressed.RotZ, (float)compressed.RotW));
        Vector3 scale = new((float)compressed.ScaleX, (float)compressed.ScaleY, (float)compressed.ScaleZ);

        return BuildTRS(translation, rotation, scale);
    }

    /// <summary>
    /// Compress a batch of transforms.
    /// </summary>
    public static void CompressTransforms(ReadOnlySpan<Matrix4x4> matrices, Span<CompressedTransform> compressed,
        CompressionFormat format = CompressionFormat.Quantized16)
    {
        int count = Math.Min(matrices.Length, compressed.Length);
        for (int i = 0; i < count; i++)
            compressed[i] = CompressTransform(matrices[i], format);
    }

    /// <summary>
    /// Decompress a batch of transforms.
    /// </summary>
    public static void DecompressTransforms(ReadOnlySpan<CompressedTransform> compressed, Span<Matrix4x4> matrices)
    {
        int count = Math.Min(compressed.Length, matrices.Length);
        for (int i = 0; i < count; i++)
            matrices[i] = DecompressTransform(compressed[i]);
    }

    /// <summary>
    /// Compute the byte size of a compressed transform stream.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CompressedTransformSize(int transformCount) =>
        transformCount * Unsafe.SizeOf<CompressedTransform>();

    /// <summary>
    /// Delta-compress a transform stream by storing only the difference from the previous frame.
    /// </summary>
    public static void DeltaCompress(ReadOnlySpan<Matrix4x4> current, ReadOnlySpan<Matrix4x4> previous,
        Span<CompressedTransform> deltas)
    {
        int count = Math.Min(current.Length, Math.Min(previous.Length, deltas.Length));

        for (int i = 0; i < count; i++)
        {
            current[i].DecomposeTRS(out Vector3 tC, out Quaternion rC, out Vector3 sC);
            previous[i].DecomposeTRS(out Vector3 tP, out Quaternion rP, out Vector3 sP);

            Vector3 deltaPos = tC - tP;
            Quaternion deltaRot = Quaternion.Normalize(Quaternion.Inverse(rP) * rC);
            Vector3 deltaScale = sC - sP;

            deltas[i] = new CompressedTransform
            {
                PosX = (Half)deltaPos.X,
                PosY = (Half)deltaPos.Y,
                PosZ = (Half)deltaPos.Z,
                RotX = (Half)deltaRot.X,
                RotY = (Half)deltaRot.Y,
                RotZ = (Half)deltaRot.Z,
                RotW = (Half)deltaRot.W,
                ScaleX = (Half)deltaScale.X,
                ScaleY = (Half)deltaScale.Y,
                ScaleZ = (Half)deltaScale.Z,
                Format = CompressionFormat.FullCompressed
            };
        }
    }

    /// <summary>
    /// Apply delta-compressed transforms to reconstruct absolute transforms.
    /// </summary>
    public static void DeltaDecompress(ReadOnlySpan<CompressedTransform> deltas,
        ReadOnlySpan<Matrix4x4> previous, Span<Matrix4x4> current)
    {
        int count = Math.Min(deltas.Length, Math.Min(previous.Length, current.Length));

        for (int i = 0; i < count; i++)
        {
            previous[i].DecomposeTRS(out Vector3 tP, out Quaternion rP, out Vector3 sP);

            Vector3 deltaPos = new((float)deltas[i].PosX, (float)deltas[i].PosY, (float)deltas[i].PosZ);
            Quaternion deltaRot = Quaternion.Normalize(new Quaternion(
                (float)deltas[i].RotX, (float)deltas[i].RotY,
                (float)deltas[i].RotZ, (float)deltas[i].RotW));
            Vector3 deltaScale = new((float)deltas[i].ScaleX, (float)deltas[i].ScaleY, (float)deltas[i].ScaleZ);

            current[i] = BuildTRS(tP + deltaPos, Quaternion.Normalize(rP * deltaRot), sP + deltaScale);
        }
    }

    #endregion

    #region Matrix Palette Generation for Skinning

    /// <summary>
    /// Generate a matrix palette for GPU skinning.
    /// Each matrix is: inverseBindPose * currentWorldPose.
    /// </summary>
    public static void GenerateMatrixPalette(
        ReadOnlySpan<Matrix4x4> currentWorldPose,
        ReadOnlySpan<Matrix4x4> inverseBindPose,
        ReadOnlySpan<int> jointMap,
        Span<Matrix4x4> palette)
    {
        int count = Math.Min(jointMap.Length, palette.Length);

        for (int i = 0; i < count; i++)
        {
            int jointIdx = jointMap[i];
            palette[i] = Matrix4x4.Multiply(inverseBindPose[jointIdx], currentWorldPose[jointIdx]);
        }
    }

    /// <summary>
    /// Generate a matrix palette with combined transform for LOD transitions.
    /// Blends between two skeleton poses.
    /// </summary>
    public static void GenerateBlendedMatrixPalette(
        ReadOnlySpan<Matrix4x4> worldPoseA, ReadOnlySpan<Matrix4x4> worldPoseB,
        ReadOnlySpan<Matrix4x4> inverseBindPose,
        ReadOnlySpan<int> jointMap, float blendWeight,
        Span<Matrix4x4> palette)
    {
        int count = Math.Min(jointMap.Length, palette.Length);

        for (int i = 0; i < count; i++)
        {
            int jointIdx = jointMap[i];

            worldPoseA[jointIdx].DecomposeTRS(out Vector3 tA, out Quaternion rA, out Vector3 sA);
            worldPoseB[jointIdx].DecomposeTRS(out Vector3 tB, out Quaternion rB, out Vector3 sB);

            Vector3 t = Vector3.Lerp(tA, tB, blendWeight);
            Quaternion r = Quaternion.Slerp(rA, rB, blendWeight);
            Vector3 s = Vector3.Lerp(sA, sB, blendWeight);

            Matrix4x4 blendedPose = BuildTRS(t, r, s);
            palette[i] = Matrix4x4.Multiply(inverseBindPose[jointIdx], blendedPose);
        }
    }

    /// <summary>
    /// Generate a matrix palette for multi-material skinning with per-material joint offsets.
    /// </summary>
    public static void GenerateMultiMaterialPalette(
        ReadOnlySpan<Matrix4x4> worldPose,
        ReadOnlySpan<Matrix4x4> inverseBindPose,
        ReadOnlySpan<int> jointMap,
        ReadOnlySpan<Matrix4x4> materialOffsets,
        int materialIndex,
        Span<Matrix4x4> palette)
    {
        int count = Math.Min(jointMap.Length, palette.Length);
        Matrix4x4 matOffset = materialIndex < materialOffsets.Length
            ? materialOffsets[materialIndex]
            : Matrix4x4.Identity;

        for (int i = 0; i < count; i++)
        {
            int jointIdx = jointMap[i];
            palette[i] = Matrix4x4.Multiply(
                Matrix4x4.Multiply(inverseBindPose[jointIdx], worldPose[jointIdx]),
                matOffset);
        }
    }

    /// <summary>
    /// Pack matrix palette into a texture-friendly format (transposed for GPU sampling).
    /// </summary>
    public static void PackPaletteForGPU(ReadOnlySpan<Matrix4x4> palette, Span<float> packedData)
    {
        Debug.Assert(packedData.Length >= palette.Length * 16);

        for (int i = 0; i < palette.Length; i++)
        {
            int offset = i * 16;
            Matrix4x4 m = Matrix4x4.Transpose(palette[i]);
            packedData[offset + 0] = m.M11;
            packedData[offset + 1] = m.M12;
            packedData[offset + 2] = m.M13;
            packedData[offset + 3] = m.M14;
            packedData[offset + 4] = m.M21;
            packedData[offset + 5] = m.M22;
            packedData[offset + 6] = m.M23;
            packedData[offset + 7] = m.M24;
            packedData[offset + 8] = m.M31;
            packedData[offset + 9] = m.M32;
            packedData[offset + 10] = m.M33;
            packedData[offset + 11] = m.M34;
            packedData[offset + 12] = m.M41;
            packedData[offset + 13] = m.M42;
            packedData[offset + 14] = m.M43;
            packedData[offset + 15] = m.M44;
        }
    }

    /// <summary>
    /// Compute the bounding box of a skinned mesh at a given pose.
    /// </summary>
    public static void ComputeSkinnedBounds(
        ReadOnlySpan<Vector3> restPositions,
        ReadOnlySpan<Matrix4x4> skinningMatrices,
        ReadOnlySpan<JointWeight[]> jointWeights,
        out Vector3 boundsMin, out Vector3 boundsMax)
    {
        boundsMin = new Vector3(float.MaxValue);
        boundsMax = new Vector3(float.MinValue);

        for (int v = 0; v < restPositions.Length; v++)
        {
            Vector3 skinnedPos = Vector3.Zero;
            JointWeight[] weights = jointWeights[v];

            for (int j = 0; j < weights.Length; j++)
            {
                float w = weights[j].Weight;
                if (w < Epsilon)
                    continue;
                skinnedPos += Vector3.Transform(restPositions[v], skinningMatrices[weights[j].JointIndex]) * w;
            }

            boundsMin = Vector3.Min(boundsMin, skinnedPos);
            boundsMax = Vector3.Max(boundsMax, skinnedPos);
        }
    }

    #endregion

    #region Additional Transform Utilities

    /// <summary>
    /// Build a TRS matrix from individual components.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 BuildTRS(Vector3 translation, Quaternion rotation, Vector3 scale)
    {
        Matrix4x4 result = Matrix4x4.CreateFromQuaternion(rotation);
        result.M11 *= scale.X;
        result.M12 *= scale.X;
        result.M13 *= scale.X;
        result.M21 *= scale.Y;
        result.M22 *= scale.Y;
        result.M23 *= scale.Y;
        result.M31 *= scale.Z;
        result.M32 *= scale.Z;
        result.M33 *= scale.Z;
        result.M41 = translation.X;
        result.M42 = translation.Y;
        result.M43 = translation.Z;
        return result;
    }

    /// <summary>
    /// Interpolate between two TRS transforms.
    /// </summary>
    public static void InterpolateTransforms(
        Matrix4x4 from, Matrix4x4 to, float t,
        out Vector3 position, out Quaternion rotation, out Vector3 scale)
    {
        from.DecomposeTRS(out Vector3 tA, out Quaternion rA, out Vector3 sA);
        to.DecomposeTRS(out Vector3 tB, out Quaternion rB, out Vector3 sB);

        position = Vector3.Lerp(tA, tB, t);
        rotation = Quaternion.Slerp(rA, rB, t);
        scale = Vector3.Lerp(sA, sB, t);
    }

    /// <summary>
    /// Get the distance between two transforms in world space.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float TransformDistance(Matrix4x4 a, Matrix4x4 b)
    {
        Vector3 posA = new(a.M41, a.M42, a.M43);
        Vector3 posB = new(b.M41, b.M42, b.M43);
        return Vector3.Distance(posA, posB);
    }

    /// <summary>
    /// Get the angular difference between two transforms in radians.
    /// </summary>
    public static float TransformAngle(Matrix4x4 a, Matrix4x4 b)
    {
        Quaternion rA = Quaternion.CreateFromRotationMatrix(a);
        Quaternion rB = Quaternion.CreateFromRotationMatrix(b);
        return QuaternionMath.AngularDistance(rA, rB);
    }

    /// <summary>
    /// Smoothly interpolate transforms with frame-rate independence.
    /// </summary>
    public static Matrix4x4 SmoothDampTransform(
        Matrix4x4 current, Matrix4x4 target, ref Vector3 velocity,
        float smoothTime, float maxSpeed, float deltaTime)
    {
        current.DecomposeTRS(out Vector3 curPos, out Quaternion curRot, out Vector3 curScale);
        target.DecomposeTRS(out Vector3 tarPos, out Quaternion tarRot, out Vector3 tarScale);

        Vector3 newPos = VectorMath.SmoothDamp(curPos, tarPos, ref velocity, smoothTime, maxSpeed, deltaTime);
        Quaternion newRot = QuaternionMath.SlerpShortest(curRot, tarRot,
            MathF.Min(1f, deltaTime / MathF.Max(smoothTime, Epsilon)));
        Vector3 newScale = Vector3.Lerp(curScale, tarScale,
            MathF.Min(1f, deltaTime / MathF.Max(smoothTime, Epsilon)));

        return BuildTRS(newPos, newRot, newScale);
    }

    /// <summary>
    /// Decompose a matrix into a transform hierarchy node.
    /// </summary>
    public static TransformNode ToTransformNode(this Matrix4x4 matrix, int parentIndex = -1, string? name = null)
    {
        return new TransformNode
        {
            LocalTransform = matrix,
            WorldTransform = matrix,
            ParentIndex = parentIndex,
            IsDirty = true,
            Name = name ?? string.Empty
        };
    }

    /// <summary>
    /// Compute the velocity (change in position per second) between two transforms.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComputeVelocity(Matrix4x4 previous, Matrix4x4 current, float deltaTime)
    {
        if (deltaTime < Epsilon)
            return Vector3.Zero;
        Vector3 prevPos = new(previous.M41, previous.M42, previous.M43);
        Vector3 currPos = new(current.M41, current.M42, current.M43);
        return (currPos - prevPos) / deltaTime;
    }

    /// <summary>
    /// Compute the angular velocity between two rotations.
    /// </summary>
    public static Vector3 ComputeAngularVelocity(Quaternion previous, Quaternion current, float deltaTime)
    {
        if (deltaTime < Epsilon)
            return Vector3.Zero;

        Quaternion delta = Quaternion.Normalize(Quaternion.Inverse(previous) * current);
        if (delta.W < 0)
            delta = -delta;

        float angle = 2f * MathF.Acos(Math.Clamp(MathF.Abs(delta.W), 0f, 1f));
        float s = MathF.Sqrt(1f - delta.W * delta.W);

        if (s < Epsilon)
            return Vector3.Zero;

        Vector3 axis = new Vector3(delta.X, delta.Y, delta.Z) / s;
        return axis * angle / deltaTime;
    }

    /// <summary>
    /// Build a look-at transform matrix.
    /// </summary>
    public static Matrix4x4 BuildLookAt(Vector3 position, Vector3 target, Vector3 up)
    {
        Vector3 forward = Vector3.Normalize(target - position);
        Vector3 right = Vector3.Normalize(Vector3.Cross(up, forward));
        Vector3 actualUp = Vector3.Cross(forward, right);

        return new Matrix4x4(
            right.X, right.Y, right.Z, 0,
            actualUp.X, actualUp.Y, actualUp.Z, 0,
            forward.X, forward.Y, forward.Z, 0,
            position.X, position.Y, position.Z, 1);
    }

    /// <summary>
    /// Compute the projection of a point onto a plane defined by a transform's local XY plane.
    /// </summary>
    public static Vector3 ProjectOntoPlane(Matrix4x4 planeTransform, Vector3 worldPoint)
    {
        Matrix4x4.Invert(planeTransform, out Matrix4x4 invTransform);
        Vector3 localPoint = Vector3.Transform(worldPoint, invTransform);
        localPoint.Z = 0;
        return Vector3.Transform(localPoint, planeTransform);
    }

    #endregion
}
