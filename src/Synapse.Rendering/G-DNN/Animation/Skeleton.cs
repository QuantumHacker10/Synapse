using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: Skeleton.cs
// PATH: Animation/Skeleton.cs
// ============================================================


using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace GDNN.Animation
{
    /// <summary>
    /// Represents a hierarchical skeleton composed of named joints arranged in a tree structure.
    /// Provides bind-pose management, inverse-bind-pose computation, pose evaluation,
    /// pose blending, clip management, and binary serialization.
    /// </summary>
    public sealed class Skeleton : IDisposable
    {
        private JointData[] _joints;
        private int _jointCount;
        private AnimationClipReference[] _clips;
        private int _clipCount;
        private bool _disposed;

        /// <summary>Maximum number of joints supported by a single skeleton.</summary>
        public const int MaxJoints = 256;

        /// <summary>Maximum number of animation clips that can be registered.</summary>
        public const int MaxClips = 64;

        /// <summary>
        /// A compact value type that stores all per-joint data in a cache-friendly layout.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct JointData
        {
            /// <summary>Index of the parent joint (-1 for root joints).</summary>
            public int ParentIndex;

            /// <summary>Number of direct children of this joint.</summary>
            public int ChildCount;

            /// <summary>Index of the first child joint in the joint array.</summary>
            public int FirstChildIndex;

            /// <summary>Depth of this joint in the hierarchy (0 for roots).</summary>
            public int Depth;

            /// <summary>Name offset into the shared string table.</summary>
            public int NameOffset;

            /// <summary>Local-space translation of the bind pose.</summary>
            public Vector3 BindTranslation;

            /// <summary>Local-space rotation quaternion of the bind pose.</summary>
            public Quaternion BindRotation;

            /// <summary>Local-space scale of the bind pose.</summary>
            public Vector3 BindScale;

            /// <summary>Current local-space translation during animation.</summary>
            public Vector3 LocalTranslation;

            /// <summary>Current local-space rotation quaternion during animation.</summary>
            public Quaternion LocalRotation;

            /// <summary>Current local-space scale during animation.</summary>
            public Vector3 LocalScale;

            /// <summary>Cached world-space transform matrix.</summary>
            public Matrix4x4 WorldMatrix;

            /// <summary>Cached inverse world-space transform matrix.</summary>
            public Matrix4x4 InverseWorldMatrix;

            /// <summary>Bind-pose world matrix (constant after initialization).</summary>
            public Matrix4x4 BindPoseMatrix;

            /// <summary>Inverse bind-pose world matrix (constant after initialization).</summary>
            public Matrix4x4 InverseBindPoseMatrix;

            /// <summary>Whether this joint has been flagged dirty and needs re-evaluation.</summary>
            public byte IsDirty;

            /// <summary>Whether this joint is active (enabled) in the current pose.</summary>
            public byte IsActive;

            /// <summary>Padding for alignment.</summary>
            public ushort Padding;
        }

        /// <summary>
        /// Represents a single animation clip registered with this skeleton.
        /// </summary>
        public sealed class AnimationClipReference
        {
            /// <summary>The animation clip data.</summary>
            public AnimationClip Clip;

            /// <summary>Weight for blending when this clip is active.</summary>
            public float Weight;

            /// <summary>Whether this clip is currently enabled.</summary>
            public bool Enabled;

            /// <summary>Time offset into the clip for scheduling.</summary>
            public float TimeOffset;

            /// <summary>Playback speed multiplier.</summary>
            public float Speed;

            /// <summary>Number of times to loop (0 = infinite).</summary>
            public int LoopCount;

            /// <summary>Elapsed playback time.</summary>
            internal float ElapsedTime;

            /// <summary>Current loop iteration.</summary>
            internal int CurrentLoop;

            /// <summary>Normalized time [0,1) within the current loop.</summary>
            public float NormalizedTime => Clip != null && Clip.Duration > 0f
                ? (ElapsedTime % Clip.Duration) / Clip.Duration
                : 0f;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Skeleton"/> class with the specified capacity.
        /// </summary>
        /// <param name="maxJoints">Maximum number of joints the skeleton can hold.</param>
        public Skeleton(int maxJoints = MaxJoints)
        {
            if (maxJoints <= 0 || maxJoints > MaxJoints)
                throw new ArgumentOutOfRangeException(nameof(maxJoints), $"Must be between 1 and {MaxJoints}.");

            _joints = new JointData[maxJoints];
            _jointCount = 0;
            _clips = new AnimationClipReference[MaxClips];
            _clipCount = 0;
            _disposed = false;
        }

        /// <summary>Gets the number of joints currently in the skeleton.</summary>
        public int JointCount => _jointCount;

        /// <summary>Gets the capacity (maximum joints) of this skeleton.</summary>
        public int Capacity => _joints.Length;

        /// <summary>Gets the number of registered animation clips.</summary>
        public int ClipCount => _clipCount;

        /// <summary>
        /// Gets a read-only span over all joint data.
        /// </summary>
        public ReadOnlySpan<JointData> Joints => new ReadOnlySpan<JointData>(_joints, 0, _jointCount);

        /// <summary>
        /// Gets a mutable span over all joint data for direct manipulation.
        /// </summary>
        public Span<JointData> MutableJoints => new Span<JointData>(_joints, 0, _jointCount);

        /// <summary>
        /// Gets a reference to a specific joint by index.
        /// </summary>
        /// <param name="index">Joint index (0-based).</param>
        public ref JointData GetJoint(int index)
        {
            if ((uint)index >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(index), $"Joint index {index} is out of range [0, {_jointCount}).");
            return ref _joints[index];
        }

        /// <summary>
        /// Adds a new joint to the skeleton.
        /// </summary>
        /// <param name="name">Name of the joint.</param>
        /// <param name="parentIndex">Parent joint index (-1 for root).</param>
        /// <param name="localTranslation">Local bind-pose translation.</param>
        /// <param name="localRotation">Local bind-pose rotation quaternion.</param>
        /// <param name="localScale">Local bind-pose scale.</param>
        /// <returns>Index of the newly added joint.</returns>
        public int AddJoint(
            string name,
            int parentIndex,
            Vector3 localTranslation,
            Quaternion localRotation,
            Vector3 localScale)
        {
            if (_jointCount >= _joints.Length)
                throw new InvalidOperationException($"Skeleton capacity ({_joints.Length}) exceeded.");

            if (parentIndex < -1 || parentIndex >= _jointCount)
                throw new ArgumentOutOfRangeException(nameof(parentIndex), $"Parent index {parentIndex} is invalid.");

            int index = _jointCount++;
            ref JointData joint = ref _joints[index];

            joint.ParentIndex = parentIndex;
            joint.ChildCount = 0;
            joint.FirstChildIndex = -1;
            joint.NameOffset = 0;
            joint.BindTranslation = localTranslation;
            joint.BindRotation = localRotation;
            joint.BindScale = localScale;
            joint.LocalTranslation = localTranslation;
            joint.LocalRotation = localRotation;
            joint.LocalScale = localScale;
            joint.WorldMatrix = Matrix4x4.Identity;
            joint.InverseWorldMatrix = Matrix4x4.Identity;
            joint.BindPoseMatrix = Matrix4x4.Identity;
            joint.InverseBindPoseMatrix = Matrix4x4.Identity;
            joint.IsDirty = 1;
            joint.IsActive = 1;
            joint.Padding = 0;

            if (parentIndex >= 0)
            {
                ref JointData parent = ref _joints[parentIndex];
                if (parent.ChildCount == 0)
                {
                    parent.FirstChildIndex = index;
                }
                parent.ChildCount++;
            }

            joint.Depth = parentIndex >= 0 ? _joints[parentIndex].Depth + 1 : 0;

            return index;
        }

        /// <summary>
        /// Adds a new joint with identity transform as bind pose.
        /// </summary>
        /// <param name="name">Name of the joint.</param>
        /// <param name="parentIndex">Parent joint index (-1 for root).</param>
        /// <returns>Index of the newly added joint.</returns>
        public int AddJoint(string name, int parentIndex)
        {
            return AddJoint(name, parentIndex, Vector3.Zero, Quaternion.Identity, Vector3.One);
        }

        /// <summary>
        /// Removes a joint and all its descendants from the skeleton.
        /// Re-indexes remaining joints to maintain a contiguous array.
        /// </summary>
        /// <param name="index">Index of the joint to remove.</param>
        public void RemoveJoint(int index)
        {
            if ((uint)index >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(index));

            int removedCount = GetSubtreeCount(index);

            for (int i = index; i < _jointCount - removedCount; i++)
            {
                _joints[i] = _joints[i + removedCount];
            }

            _jointCount -= removedCount;

            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];
                if (joint.ParentIndex >= index)
                {
                    joint.ParentIndex -= removedCount;
                }
                else if (joint.ParentIndex >= index + removedCount)
                {
                    joint.ParentIndex -= removedCount;
                }
            }

            RebuildHierarchy();
        }

        /// <summary>
        /// Returns the parent joint index of the specified joint.
        /// </summary>
        /// <param name="jointIndex">Joint index to query.</param>
        /// <returns>Parent index, or -1 if the joint is a root.</returns>
        public int GetParentIndex(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            return _joints[jointIndex].ParentIndex;
        }

        /// <summary>
        /// Returns a span of child joint indices for the specified joint.
        /// </summary>
        /// <param name="jointIndex">Joint index to query.</param>
        /// <returns>Span containing child indices.</returns>
        public ReadOnlySpan<int> GetChildIndices(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            ref JointData joint = ref _joints[jointIndex];
            if (joint.ChildCount == 0)
                return ReadOnlySpan<int>.Empty;

            int[] children = new int[joint.ChildCount];
            int childIdx = joint.FirstChildIndex;
            for (int i = 0; i < joint.ChildCount; i++)
            {
                children[i] = childIdx;
                childIdx = GetNextSibling(childIdx);
            }
            return children;
        }

        /// <summary>
        /// Gets the next sibling joint index, or -1 if there is no next sibling.
        /// </summary>
        /// <param name="jointIndex">Current joint index.</param>
        /// <returns>Next sibling index or -1.</returns>
        public int GetNextSibling(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                return -1;

            ref JointData joint = ref _joints[jointIndex];
            if (joint.ParentIndex < 0)
                return -1;

            ref JointData parent = ref _joints[joint.ParentIndex];
            int childIdx = parent.FirstChildIndex;
            int siblingIndex = childIdx;

            for (int i = 0; i < parent.ChildCount; i++)
            {
                if (siblingIndex == jointIndex)
                {
                    if (i + 1 < parent.ChildCount)
                        return GetNextSiblingInList(childIdx, i + 1);
                    return -1;
                }
                siblingIndex = GetNextSiblingInList(childIdx, i + 1);
            }

            return -1;
        }

        private int GetNextSiblingInList(int firstChildIndex, int skipCount)
        {
            int current = firstChildIndex;
            for (int i = 0; i < skipCount; i++)
            {
                if (current < 0 || current >= _jointCount)
                    return -1;
                current = _joints[current].ParentIndex >= 0
                    ? current + 1
                    : -1;
            }
            return current;
        }

        /// <summary>
        /// Returns the depth of the specified joint in the hierarchy.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <returns>Depth (0 for root joints).</returns>
        public int GetDepth(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            return _joints[jointIndex].Depth;
        }

        /// <summary>
        /// Returns the total number of joints in the subtree rooted at the specified joint.
        /// </summary>
        /// <param name="jointIndex">Root of the subtree.</param>
        /// <returns>Number of joints in the subtree (including the root).</returns>
        public int GetSubtreeCount(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            int count = 0;
            int stackPtr = 0;
            int[] stack = new int[MaxJoints];
            stack[stackPtr++] = jointIndex;

            while (stackPtr > 0)
            {
                int current = stack[--stackPtr];
                count++;

                ref JointData joint = ref _joints[current];
                if (joint.ChildCount > 0)
                {
                    int childIdx = joint.FirstChildIndex;
                    for (int i = 0; i < joint.ChildCount; i++)
                    {
                        stack[stackPtr++] = childIdx;
                        childIdx = GetNextSiblingSafe(childIdx);
                    }
                }
            }

            return count;
        }

        private int GetNextSiblingSafe(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                return -1;

            ref JointData joint = ref _joints[jointIndex];
            if (joint.ParentIndex < 0)
                return -1;

            ref JointData parent = ref _joints[joint.ParentIndex];
            int childIdx = parent.FirstChildIndex;
            int idx = 0;

            while (idx < parent.ChildCount)
            {
                if (childIdx == jointIndex && idx + 1 < parent.ChildCount)
                {
                    return childIdx + 1;
                }
                childIdx++;
                idx++;
            }

            return -1;
        }

        /// <summary>
        /// Returns all joint indices in the subtree rooted at the specified joint,
        /// in depth-first order.
        /// </summary>
        /// <param name="jointIndex">Root of the subtree.</param>
        /// <returns>Array of joint indices.</returns>
        public int[] GetSubtreeIndices(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            List<int> result = new List<int>();
            int stackPtr = 0;
            int[] stack = new int[MaxJoints];
            stack[stackPtr++] = jointIndex;

            while (stackPtr > 0)
            {
                int current = stack[--stackPtr];
                result.Add(current);

                ref JointData joint = ref _joints[current];
                if (joint.ChildCount > 0)
                {
                    int childIdx = joint.FirstChildIndex;
                    for (int i = 0; i < joint.ChildCount; i++)
                    {
                        stack[stackPtr++] = childIdx;
                        childIdx++;
                    }
                }
            }

            return result.ToArray();
        }

        /// <summary>
        /// Rebuilds the internal hierarchy after bulk modifications.
        /// Recomputes depths and child relationships.
        /// </summary>
        public void RebuildHierarchy()
        {
            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];
                joint.ChildCount = 0;
                joint.FirstChildIndex = -1;
                joint.Depth = 0;
            }

            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];
                if (joint.ParentIndex >= 0 && joint.ParentIndex < _jointCount)
                {
                    ref JointData parent = ref _joints[joint.ParentIndex];
                    if (parent.ChildCount == 0)
                    {
                        parent.FirstChildIndex = i;
                    }
                    parent.ChildCount++;
                    joint.Depth = parent.Depth + 1;
                }
            }
        }

        /// <summary>
        /// Computes the bind pose and inverse bind pose matrices for all joints.
        /// Must be called after the skeleton hierarchy is finalized.
        /// </summary>
        public void ComputeBindPoseMatrices()
        {
            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];

                Matrix4x4 localMatrix = CreateLocalMatrix(
                    joint.BindTranslation,
                    joint.BindRotation,
                    joint.BindScale);

                if (joint.ParentIndex >= 0 && joint.ParentIndex < _jointCount)
                {
                    joint.BindPoseMatrix = localMatrix * _joints[joint.ParentIndex].BindPoseMatrix;
                }
                else
                {
                    joint.BindPoseMatrix = localMatrix;
                }

                Matrix4x4.Invert(joint.BindPoseMatrix, out joint.InverseBindPoseMatrix);
            }
        }

        /// <summary>
        /// Evaluates the current pose from the local transforms stored in each joint,
        /// computing world-space matrices and inverse bind-pose skinning matrices.
        /// </summary>
        /// <param name="output">Pre-allocated span for skinning matrices (length must be >= JointCount).</param>
        public unsafe void EvaluatePose(Span<Matrix4x4> output)
        {
            if (output.Length < _jointCount)
                throw new ArgumentException($"Output span must have at least {_jointCount} elements.", nameof(output));

            fixed (JointData* jointsPtr = _joints)
            {
                for (int i = 0; i < _jointCount; i++)
                {
                    JointData* joint = &jointsPtr[i];

                    Matrix4x4 localMatrix = CreateLocalMatrix(
                        joint->LocalTranslation,
                        joint->LocalRotation,
                        joint->LocalScale);

                    if (joint->ParentIndex >= 0 && joint->ParentIndex < _jointCount)
                    {
                        joint->WorldMatrix = localMatrix * jointsPtr[joint->ParentIndex].WorldMatrix;
                    }
                    else
                    {
                        joint->WorldMatrix = localMatrix;
                    }

                    output[i] = joint->InverseBindPoseMatrix * joint->WorldMatrix;
                }
            }
        }

        /// <summary>
        /// Evaluates the current pose without allocating output, writing directly to joint data.
        /// </summary>
        public unsafe void EvaluatePoseInPlace()
        {
            fixed (JointData* jointsPtr = _joints)
            {
                for (int i = 0; i < _jointCount; i++)
                {
                    JointData* joint = &jointsPtr[i];

                    Matrix4x4 localMatrix = CreateLocalMatrix(
                        joint->LocalTranslation,
                        joint->LocalRotation,
                        joint->LocalScale);

                    if (joint->ParentIndex >= 0 && joint->ParentIndex < _jointCount)
                    {
                        joint->WorldMatrix = localMatrix * jointsPtr[joint->ParentIndex].WorldMatrix;
                    }
                    else
                    {
                        joint->WorldMatrix = localMatrix;
                    }

                    joint->InverseWorldMatrix = joint->WorldMatrix;
                    Matrix4x4.Invert(joint->InverseWorldMatrix, out joint->InverseWorldMatrix);
                }
            }
        }

        /// <summary>
        /// Sets the local transform of a joint and marks it and its descendants dirty.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <param name="translation">Local translation.</param>
        /// <param name="rotation">Local rotation.</param>
        /// <param name="scale">Local scale.</param>
        public void SetLocalTransform(int jointIndex, Vector3 translation, Quaternion rotation, Vector3 scale)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            ref JointData joint = ref _joints[jointIndex];
            joint.LocalTranslation = translation;
            joint.LocalRotation = rotation;
            joint.LocalScale = scale;
            joint.IsDirty = 1;

            MarkDescendantsDirty(jointIndex);
        }

        /// <summary>
        /// Sets the local translation of a joint.
        /// </summary>
        public void SetLocalTranslation(int jointIndex, Vector3 translation)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            _joints[jointIndex].LocalTranslation = translation;
            _joints[jointIndex].IsDirty = 1;
            MarkDescendantsDirty(jointIndex);
        }

        /// <summary>
        /// Sets the local rotation of a joint.
        /// </summary>
        public void SetLocalRotation(int jointIndex, Quaternion rotation)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            _joints[jointIndex].LocalRotation = rotation;
            _joints[jointIndex].IsDirty = 1;
            MarkDescendantsDirty(jointIndex);
        }

        /// <summary>
        /// Sets the local scale of a joint.
        /// </summary>
        public void SetLocalScale(int jointIndex, Vector3 scale)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            _joints[jointIndex].LocalScale = scale;
            _joints[jointIndex].IsDirty = 1;
            MarkDescendantsDirty(jointIndex);
        }

        /// <summary>
        /// Gets the world-space transform matrix of a joint.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <returns>World-space transform matrix.</returns>
        public Matrix4x4 GetWorldMatrix(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            return _joints[jointIndex].WorldMatrix;
        }

        /// <summary>
        /// Gets the inverse bind-pose matrix of a joint (used for skinning).
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <returns>Inverse bind-pose matrix.</returns>
        public Matrix4x4 GetInverseBindPoseMatrix(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            return _joints[jointIndex].InverseBindPoseMatrix;
        }

        /// <summary>
        /// Gets the bind-pose matrix of a joint.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <returns>Bind-pose matrix.</returns>
        public Matrix4x4 GetBindPoseMatrix(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));
            return _joints[jointIndex].BindPoseMatrix;
        }

        /// <summary>
        /// Marks all descendants of the specified joint as dirty.
        /// </summary>
        private void MarkDescendantsDirty(int jointIndex)
        {
            int stackPtr = 0;
            int[] stack = stackPool;
            ref JointData joint = ref _joints[jointIndex];

            if (joint.ChildCount > 0)
            {
                stack[stackPtr++] = joint.FirstChildIndex;
            }

            while (stackPtr > 0)
            {
                int current = stack[--stackPtr];
                _joints[current].IsDirty = 1;

                ref JointData j = ref _joints[current];
                if (j.ChildCount > 0)
                {
                    int childIdx = j.FirstChildIndex;
                    for (int i = 0; i < j.ChildCount; i++)
                    {
                        stack[stackPtr++] = childIdx;
                        childIdx++;
                    }
                }
            }
        }

        [ThreadStatic]
        private static int[] stackPool = new int[MaxJoints];

        // ── Animation Clip Management ────────────────────────────────────

        /// <summary>
        /// Registers an animation clip with this skeleton.
        /// </summary>
        /// <param name="clip">The animation clip to register.</param>
        /// <param name="weight">Initial blending weight.</param>
        /// <returns>Index of the registered clip.</returns>
        public int RegisterClip(AnimationClip clip, float weight = 1.0f)
        {
            if (clip == null) throw new ArgumentNullException(nameof(clip));
            if (_clipCount >= MaxClips)
                throw new InvalidOperationException($"Maximum clip count ({MaxClips}) exceeded.");

            int index = _clipCount++;
            _clips[index] = new AnimationClipReference
            {
                Clip = clip,
                Weight = weight,
                Enabled = true,
                TimeOffset = 0f,
                Speed = 1f,
                LoopCount = 0,
                ElapsedTime = 0f,
                CurrentLoop = 0
            };

            return index;
        }

        /// <summary>
        /// Removes a registered animation clip.
        /// </summary>
        /// <param name="clipIndex">Index of the clip to remove.</param>
        public void UnregisterClip(int clipIndex)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));

            for (int i = clipIndex; i < _clipCount - 1; i++)
            {
                _clips[i] = _clips[i + 1];
            }

            _clipCount--;
        }

        /// <summary>
        /// Gets a reference to a registered clip by index.
        /// </summary>
        public ref AnimationClipReference GetClipReference(int clipIndex)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));
            return ref _clips[clipIndex];
        }

        /// <summary>
        /// Finds a registered clip by name.
        /// </summary>
        /// <param name="clipName">Name of the animation clip.</param>
        /// <returns>Index of the clip, or -1 if not found.</returns>
        public int FindClip(string clipName)
        {
            for (int i = 0; i < _clipCount; i++)
            {
                if (_clips[i].Clip != null &&
                    string.Equals(_clips[i].Clip.Name, clipName, StringComparison.Ordinal))
                {
                    return i;
                }
            }
            return -1;
        }

        /// <summary>
        /// Enables or disables a registered clip.
        /// </summary>
        public void SetClipEnabled(int clipIndex, bool enabled)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));
            _clips[clipIndex].Enabled = enabled;
        }

        /// <summary>
        /// Sets the playback speed of a registered clip.
        /// </summary>
        public void SetClipSpeed(int clipIndex, float speed)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));
            _clips[clipIndex].Speed = speed;
        }

        /// <summary>
        /// Sets the blending weight of a registered clip.
        /// </summary>
        public void SetClipWeight(int clipIndex, float weight)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));
            _clips[clipIndex].Weight = weight;
        }

        /// <summary>
        /// Resets the playback time of a registered clip.
        /// </summary>
        public void ResetClipTime(int clipIndex)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));
            _clips[clipIndex].ElapsedTime = 0f;
            _clips[clipIndex].CurrentLoop = 0;
        }

        // ── Pose Evaluation at Arbitrary Time ────────────────────────────

        /// <summary>
        /// Evaluates the pose at a specific time for a registered clip and stores results
        /// in the skeleton's local transforms.
        /// </summary>
        /// <param name="clipIndex">Registered clip index.</param>
        /// <param name="time">Time in seconds to evaluate at.</param>
        public void EvaluateClipAtTime(int clipIndex, float time)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));

            ref AnimationClipReference clipRef = ref _clips[clipIndex];
            if (clipRef.Clip == null || !clipRef.Enabled)
                return;

            clipRef.Clip.SampleAtTime(time, this);
        }

        /// <summary>
        /// Advances the playback time of a clip and evaluates the pose.
        /// </summary>
        /// <param name="clipIndex">Registered clip index.</param>
        /// <param name="deltaTime">Time elapsed since last update.</param>
        public void UpdateClip(int clipIndex, float deltaTime)
        {
            if ((uint)clipIndex >= (uint)_clipCount)
                throw new ArgumentOutOfRangeException(nameof(clipIndex));

            ref AnimationClipReference clipRef = ref _clips[clipIndex];
            if (clipRef.Clip == null || !clipRef.Enabled)
                return;

            clipRef.ElapsedTime += deltaTime * clipRef.Speed;

            if (clipRef.LoopCount > 0 && clipRef.CurrentLoop >= clipRef.LoopCount)
                return;

            float duration = clipRef.Clip.Duration;
            if (duration > 0f)
            {
                while (clipRef.ElapsedTime >= duration)
                {
                    clipRef.ElapsedTime -= duration;
                    clipRef.CurrentLoop++;

                    if (clipRef.LoopCount > 0 && clipRef.CurrentLoop >= clipRef.LoopCount)
                    {
                        clipRef.ElapsedTime = duration;
                        break;
                    }
                }
            }

            clipRef.Clip.SampleAtTime(clipRef.ElapsedTime, this);
        }

        /// <summary>
        /// Updates all enabled clips and blends their results into the skeleton.
        /// </summary>
        /// <param name="deltaTime">Time elapsed since last update.</param>
        public void UpdateAllClips(float deltaTime)
        {
            for (int i = 0; i < _clipCount; i++)
            {
                if (_clips[i].Enabled)
                {
                    UpdateClip(i, deltaTime);
                }
            }
        }

        // ── Pose Blending ────────────────────────────────────────────────

        /// <summary>
        /// Blends two poses using additive blending. The source pose is added to the target.
        /// </summary>
        /// <param name="targetIndex">Index of the target joint.</param>
        /// <param name="sourceTranslation">Source translation to add.</param>
        /// <param name="sourceRotation">Source rotation to add (as quaternion difference).</param>
        /// <param name="sourceScale">Source scale to add.</param>
        /// <param name="weight">Blend weight [0,1].</param>
        public void BlendAdditive(
            int targetIndex,
            Vector3 sourceTranslation,
            Quaternion sourceRotation,
            Vector3 sourceScale,
            float weight)
        {
            if ((uint)targetIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(targetIndex));

            ref JointData joint = ref _joints[targetIndex];
            joint.LocalTranslation += sourceTranslation * weight;
            joint.LocalRotation = Quaternion.Normalize(
                Quaternion.Slerp(Quaternion.Identity, sourceRotation, weight) * joint.LocalRotation);
            joint.LocalScale += (sourceScale - Vector3.One) * weight;
            joint.IsDirty = 1;
        }

        /// <summary>
        /// Blends two poses using override blending. The source pose overrides the target
        /// based on weight.
        /// </summary>
        /// <param name="targetIndex">Index of the target joint.</param>
        /// <param name="sourceTranslation">Source translation to blend toward.</param>
        /// <param name="sourceRotation">Source rotation to blend toward.</param>
        /// <param name="sourceScale">Source scale to blend toward.</param>
        /// <param name="weight">Blend weight [0 = target, 1 = source].</param>
        public void BlendOverride(
            int targetIndex,
            Vector3 sourceTranslation,
            Quaternion sourceRotation,
            Vector3 sourceScale,
            float weight)
        {
            if ((uint)targetIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(targetIndex));

            ref JointData joint = ref _joints[targetIndex];
            joint.LocalTranslation = Vector3.Lerp(joint.LocalTranslation, sourceTranslation, weight);
            joint.LocalRotation = Quaternion.Slerp(joint.LocalRotation, sourceRotation, weight);
            joint.LocalScale = Vector3.Lerp(joint.LocalScale, sourceScale, weight);
            joint.IsDirty = 1;
        }

        /// <summary>
        /// Performs layered blending where each layer has a specified weight and mask.
        /// </summary>
        /// <param name="basePose">Base pose clip to evaluate first.</param>
        /// <param name="layers">Array of (clip, weight, jointMask) tuples for each layer.</param>
        /// <param name="layerCount">Number of layers.</param>
        public void BlendLayers(
            AnimationClip basePose,
            ReadOnlySpan<(AnimationClip Clip, float Weight, ReadOnlyMemory<bool> JointMask)> layers,
            int layerCount)
        {
            if (basePose != null)
            {
                basePose.SampleAtTime(0f, this);
            }

            for (int layer = 0; layer < layerCount; layer++)
            {
                var (clip, weight, jointMask) = layers[layer];
                if (clip == null || weight <= 0f)
                    continue;

                AnimationClip tempClip = clip;

                for (int j = 0; j < _jointCount; j++)
                {
                    if ((uint)j < (uint)jointMask.Length && !jointMask.Span[j])
                        continue;

                    if (clip.TryGetChannelData(j, out Vector3 pos, out Quaternion rot, out Vector3 scl))
                    {
                        BlendOverride(j, pos, rot, scl, weight);
                    }
                }
            }
        }

        /// <summary>
        /// Blends the current pose with a target pose at a specific joint.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <param name="targetTranslation">Target translation.</param>
        /// <param name="targetRotation">Target rotation.</param>
        /// <param name="targetScale">Target scale.</param>
        /// <param name="blendFactor">Blend factor [0=current, 1=target].</param>
        public void BlendJoint(
            int jointIndex,
            Vector3 targetTranslation,
            Quaternion targetRotation,
            Vector3 targetScale,
            float blendFactor)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            blendFactor = Math.Clamp(blendFactor, 0f, 1f);

            ref JointData joint = ref _joints[jointIndex];
            joint.LocalTranslation = Vector3.Lerp(joint.LocalTranslation, targetTranslation, blendFactor);
            joint.LocalRotation = Quaternion.Slerp(joint.LocalRotation, targetRotation, blendFactor);
            joint.LocalScale = Vector3.Lerp(joint.LocalScale, targetScale, blendFactor);
            joint.IsDirty = 1;
        }

        /// <summary>
        /// Performs a partial-body blend where only joints matching the mask are affected.
        /// </summary>
        /// <param name="source">Source skeleton to blend from.</param>
        /// <param name="jointMask">Boolean mask per joint (true = blend this joint).</param>
        /// <param name="weight">Blend weight [0,1].</param>
        public void BlendPartialBody(Skeleton source, ReadOnlySpan<bool> jointMask, float weight)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (source._jointCount != _jointCount)
                throw new ArgumentException("Source skeleton must have the same joint count.");

            weight = Math.Clamp(weight, 0f, 1f);

            for (int i = 0; i < _jointCount; i++)
            {
                if ((uint)i < (uint)jointMask.Length && !jointMask[i])
                    continue;

                ref JointData target = ref _joints[i];
                ref readonly JointData src = ref source._joints[i];

                target.LocalTranslation = Vector3.Lerp(target.LocalTranslation, src.LocalTranslation, weight);
                target.LocalRotation = Quaternion.Slerp(target.LocalRotation, src.LocalRotation, weight);
                target.LocalScale = Vector3.Lerp(target.LocalScale, src.LocalScale, weight);
                target.IsDirty = 1;
            }
        }

        // ── Serialization ────────────────────────────────────────────────

        /// <summary>
        /// Serializes the skeleton to a byte array in a compact binary format.
        /// </summary>
        /// <returns>Binary representation of the skeleton.</returns>
        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

            writer.Write((uint)0x534B454C); // Magic: "SKEL"
            writer.Write((uint)1); // Version
            writer.Write((uint)_jointCount);
            writer.Write((uint)_clipCount);

            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData j = ref _joints[i];
                writer.Write(j.ParentIndex);
                writer.Write(j.ChildCount);
                writer.Write(j.FirstChildIndex);
                writer.Write(j.Depth);
                writer.Write(j.NameOffset);
                WriteVector3(writer, j.BindTranslation);
                WriteQuaternion(writer, j.BindRotation);
                WriteVector3(writer, j.BindScale);
                WriteVector3(writer, j.LocalTranslation);
                WriteQuaternion(writer, j.LocalRotation);
                WriteVector3(writer, j.LocalScale);
                WriteMatrix4x4(writer, j.BindPoseMatrix);
                WriteMatrix4x4(writer, j.InverseBindPoseMatrix);
                writer.Write(j.IsDirty);
                writer.Write(j.IsActive);
                writer.Write(j.Padding);
            }

            writer.Flush();
            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes a skeleton from a byte array.
        /// </summary>
        /// <param name="data">Binary data previously produced by <see cref="Serialize"/>.</param>
        /// <returns>A new <see cref="Skeleton"/> instance.</returns>
        public static Skeleton Deserialize(ReadOnlySpan<byte> data)
        {
            if (data.Length < 16)
                throw new ArgumentException("Data too short to contain a valid skeleton header.");

            int offset = 0;

            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset));
            offset += 4;
            if (magic != 0x534B454C)
                throw new InvalidDataException("Invalid skeleton magic number.");

            uint version = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset));
            offset += 4;
            if (version != 1)
                throw new InvalidDataException($"Unsupported skeleton version {version}.");

            uint jointCount = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset));
            offset += 4;
            uint clipCount = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset));
            offset += 4;

            if (jointCount > MaxJoints)
                throw new InvalidDataException($"Joint count {jointCount} exceeds maximum {MaxJoints}.");

            Skeleton skeleton = new Skeleton((int)jointCount);
            skeleton._jointCount = (int)jointCount;

            for (int i = 0; i < (int)jointCount; i++)
            {
                ref JointData j = ref skeleton._joints[i];
                j.ParentIndex = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                j.ChildCount = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                j.FirstChildIndex = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                j.Depth = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                j.NameOffset = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                j.BindTranslation = ReadVector3(data, ref offset);
                j.BindRotation = ReadQuaternion(data, ref offset);
                j.BindScale = ReadVector3(data, ref offset);
                j.LocalTranslation = ReadVector3(data, ref offset);
                j.LocalRotation = ReadQuaternion(data, ref offset);
                j.LocalScale = ReadVector3(data, ref offset);
                j.BindPoseMatrix = ReadMatrix4x4(data, ref offset);
                j.InverseBindPoseMatrix = ReadMatrix4x4(data, ref offset);
                j.IsDirty = data[offset]; offset += 1;
                j.IsActive = data[offset]; offset += 1;
                j.Padding = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(offset)); offset += 2;
                j.WorldMatrix = Matrix4x4.Identity;
                j.InverseWorldMatrix = Matrix4x4.Identity;
            }

            skeleton._clipCount = (int)clipCount;

            return skeleton;
        }

        // ── Utility Methods ──────────────────────────────────────────────

        /// <summary>
        /// Creates a local-space transform matrix from TRS components.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Matrix4x4 CreateLocalMatrix(Vector3 translation, Quaternion rotation, Vector3 scale)
        {
            Matrix4x4 matrix = Matrix4x4.CreateScale(scale);
            matrix *= Matrix4x4.CreateFromQuaternion(rotation);
            matrix *= Matrix4x4.CreateTranslation(translation);
            return matrix;
        }

        /// <summary>
        /// Resets all joints to their bind pose.
        /// </summary>
        public void ResetToBindPose()
        {
            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];
                joint.LocalTranslation = joint.BindTranslation;
                joint.LocalRotation = joint.BindRotation;
                joint.LocalScale = joint.BindScale;
                joint.IsDirty = 1;
            }
        }

        /// <summary>
        /// Sets all joints to identity local transforms.
        /// </summary>
        public void ResetToIdentity()
        {
            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];
                joint.LocalTranslation = Vector3.Zero;
                joint.LocalRotation = Quaternion.Identity;
                joint.LocalScale = Vector3.One;
                joint.IsDirty = 1;
            }
        }

        /// <summary>
        /// Copies the current local transforms from another skeleton.
        /// </summary>
        /// <param name="other">Source skeleton to copy from.</param>
        public void CopyLocalTransformsFrom(Skeleton other)
        {
            if (other == null) throw new ArgumentNullException(nameof(other));
            int count = Math.Min(_jointCount, other._jointCount);

            for (int i = 0; i < count; i++)
            {
                _joints[i].LocalTranslation = other._joints[i].LocalTranslation;
                _joints[i].LocalRotation = other._joints[i].LocalRotation;
                _joints[i].LocalScale = other._joints[i].LocalScale;
                _joints[i].IsDirty = 1;
            }
        }

        /// <summary>
        /// Returns the total number of joints that are descendants of the specified joint.
        /// </summary>
        public int GetDescendantCount(int jointIndex)
        {
            return GetSubtreeCount(jointIndex) - 1;
        }

        /// <summary>
        /// Checks if one joint is an ancestor of another.
        /// </summary>
        public bool IsAncestor(int ancestorIndex, int descendantIndex)
        {
            if ((uint)ancestorIndex >= (uint)_jointCount ||
                (uint)descendantIndex >= (uint)_jointCount)
                return false;

            int current = _joints[descendantIndex].ParentIndex;
            while (current >= 0)
            {
                if (current == ancestorIndex)
                    return true;
                current = _joints[current].ParentIndex;
            }
            return false;
        }

        /// <summary>
        /// Finds the lowest common ancestor of two joints.
        /// </summary>
        /// <returns>Index of the LCA, or -1 if none found.</returns>
        public int FindLowestCommonAncestor(int jointA, int jointB)
        {
            if ((uint)jointA >= (uint)_jointCount ||
                (uint)jointB >= (uint)_jointCount)
                return -1;

            int depthA = _joints[jointA].Depth;
            int depthB = _joints[jointB].Depth;

            int a = jointA;
            int b = jointB;

            while (depthA > depthB)
            {
                a = _joints[a].ParentIndex;
                depthA--;
            }
            while (depthB > depthA)
            {
                b = _joints[b].ParentIndex;
                depthB--;
            }

            while (a != b && a >= 0 && b >= 0)
            {
                a = _joints[a].ParentIndex;
                b = _joints[b].ParentIndex;
            }

            return a == b ? a : -1;
        }

        /// <summary>
        /// Gets the path string for a joint (e.g., "Hips/Spine/Chest/Head").
        /// </summary>
        public string GetJointPath(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)_jointCount)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            Stack<int> path = new Stack<int>();
            int current = jointIndex;
            while (current >= 0)
            {
                path.Push(current);
                current = _joints[current].ParentIndex;
            }

            var sb = new StringBuilder();
            bool first = true;
            foreach (int idx in path)
            {
                if (!first) sb.Append('/');
                sb.Append($"Joint_{idx}");
                first = false;
            }
            return sb.ToString();
        }

        /// <summary>
        /// Finds a joint by name. Linear search through registered names.
        /// </summary>
        /// <param name="name">Joint name to find.</param>
        /// <returns>Joint index, or -1 if not found.</returns>
        public int FindJointByName(ReadOnlySpan<char> name)
        {
            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].NameOffset >= 0)
                {
                    // In a full implementation this would look up the string table.
                    // For now we compare the offset as a proxy.
                }
            }
            return -1;
        }

        /// <summary>
        /// Gets all root joints (joints with no parent).
        /// </summary>
        public List<int> GetRootJoints()
        {
            List<int> roots = new List<int>();
            for (int i = 0; i < _jointCount; i++)
            {
                if (_joints[i].ParentIndex < 0)
                    roots.Add(i);
            }
            return roots;
        }

        /// <summary>
        /// Validates the skeleton structure for consistency.
        /// </summary>
        /// <returns>True if valid; false otherwise.</returns>
        public bool Validate()
        {
            for (int i = 0; i < _jointCount; i++)
            {
                ref JointData joint = ref _joints[i];

                if (joint.ParentIndex >= i)
                    return false;

                if (joint.ParentIndex < -1)
                    return false;

                if (joint.Depth < 0)
                    return false;

                if (joint.ParentIndex >= 0)
                {
                    if (_joints[joint.ParentIndex].Depth + 1 != joint.Depth)
                        return false;
                }
                else if (joint.Depth != 0)
                {
                    return false;
                }
            }
            return true;
        }

        // ── Binary Helpers ───────────────────────────────────────────────

        private static unsafe void WriteVector3(System.IO.BinaryWriter writer, Vector3 v)
        {
            byte* bytes = stackalloc byte[12];
            Unsafe.WriteUnaligned(ref bytes[0], v.X);
            Unsafe.WriteUnaligned(ref bytes[4], v.Y);
            Unsafe.WriteUnaligned(ref bytes[8], v.Z);
            writer.Write(new ReadOnlySpan<byte>(bytes, 12));
        }

        private static unsafe void WriteQuaternion(System.IO.BinaryWriter writer, Quaternion q)
        {
            byte* bytes = stackalloc byte[16];
            Unsafe.WriteUnaligned(ref bytes[0], q.X);
            Unsafe.WriteUnaligned(ref bytes[4], q.Y);
            Unsafe.WriteUnaligned(ref bytes[8], q.Z);
            Unsafe.WriteUnaligned(ref bytes[12], q.W);
            writer.Write(new ReadOnlySpan<byte>(bytes, 16));
        }

        private static unsafe void WriteMatrix4x4(System.IO.BinaryWriter writer, Matrix4x4 m)
        {
            byte[] bytes = new byte[64];
            MemoryMarshal.Write(bytes.AsSpan(), ref m);
            writer.Write(bytes);
        }

        private static unsafe Vector3 ReadVector3(ReadOnlySpan<byte> data, ref int offset)
        {
            Vector3 v = default;
            v.X = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)));
            v.Y = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset + 4)));
            v.Z = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset + 8)));
            offset += 12;
            return v;
        }

        private static unsafe Quaternion ReadQuaternion(ReadOnlySpan<byte> data, ref int offset)
        {
            Quaternion q = default;
            q.X = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)));
            q.Y = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset + 4)));
            q.Z = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset + 8)));
            q.W = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset + 12)));
            offset += 16;
            return q;
        }

        private static unsafe Matrix4x4 ReadMatrix4x4(ReadOnlySpan<byte> data, ref int offset)
        {
            Matrix4x4 m = MemoryMarshal.Read<Matrix4x4>(data.Slice(offset, 64));
            offset += 64;
            return m;
        }

        /// <summary>
        /// Disposes the skeleton and releases internal arrays.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _joints = Array.Empty<JointData>();
                _clips = Array.Empty<AnimationClipReference>();
                _jointCount = 0;
                _clipCount = 0;
                _disposed = true;
            }
        }
    }
}
