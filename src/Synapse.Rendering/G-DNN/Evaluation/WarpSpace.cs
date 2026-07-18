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
// FILE: WarpSpace.cs
// PATH: Evaluation/WarpSpace.cs
// ============================================================


using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using GDNN.Rendering.Compat;

namespace GDNN.Evaluation
{
    /// <summary>
    /// Represents a single joint in a skeletal hierarchy.
    /// </summary>
    public sealed class Joint
    {
        /// <summary>Unique joint identifier.</summary>
        public int Id { get; set; }

        /// <summary>Joint name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Parent joint index (-1 for root).</summary>
        public int ParentIndex { get; set; } = -1;

        /// <summary>Bind pose local transform.</summary>
        public Matrix4x4 BindPoseLocal { get; set; } = Matrix4x4.Identity;

        /// <summary>Bind pose world transform (computed).</summary>
        public Matrix4x4 BindPoseWorld { get; set; } = Matrix4x4.Identity;

        /// <summary>Inverse bind pose world transform (precomputed).</summary>
        public Matrix4x4 InverseBindPoseWorld { get; set; } = Matrix4x4.Identity;

        /// <summary>Current local animation transform.</summary>
        public Matrix4x4 CurrentLocal { get; set; } = Matrix4x4.Identity;

        /// <summary>Current world transform (computed).</summary>
        public Matrix4x4 CurrentWorld { get; set; } = Matrix4x4.Identity;

        /// <summary>Children joint indices.</summary>
        public List<int> Children { get; set; } = new();
    }

    /// <summary>
    /// Represents a bone weight for linear blend skinning.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BoneWeight
    {
        /// <summary>Joint indices (up to 4 influences).</summary>
        public int Joint0, Joint1, Joint2, Joint3;

        /// <summary>Corresponding weights.</summary>
        public float Weight0, Weight1, Weight2, Weight3;

        /// <summary>Gets the total weight (should sum to 1).</summary>
        public readonly float TotalWeight => Weight0 + Weight1 + Weight2 + Weight3;

        /// <summary>Normalizes the weights to sum to 1.</summary>
        public void Normalize()
        {
            float total = TotalWeight;
            if (total > 1e-8f)
            {
                Weight0 /= total;
                Weight1 /= total;
                Weight2 /= total;
                Weight3 /= total;
            }
        }

        /// <summary>Creates a single-influence bone weight.</summary>
        public static BoneWeight Single(int jointIndex, float weight = 1.0f)
        {
            return new BoneWeight
            {
                Joint0 = jointIndex,
                Weight0 = weight
            };
        }

        /// <summary>Creates a dual-influence bone weight.</summary>
        public static BoneWeight Dual(int joint0, float weight0, int joint1, float weight1)
        {
            return new BoneWeight
            {
                Joint0 = joint0,
                Weight0 = weight0,
                Joint1 = joint1,
                Weight1 = weight1
            };
        }
    }

    /// <summary>
    /// Represents a dual quaternion for DQS skinning.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DualQuaternion
    {
        /// <summary>Real part (quaternion).</summary>
        public Quaternion Real;

        /// <summary>Dual part (quaternion).</summary>
        public Quaternion Dual;

        /// <summary>Creates a dual quaternion from a translation and rotation.</summary>
        public static DualQuaternion FromTranslationRotation(Vector3 translation, Quaternion rotation)
        {
            var dq = new DualQuaternion
            {
                Real = rotation,
                Dual = new Quaternion(
                    translation.X * 0.5f * rotation.W + translation.Y * 0.5f * rotation.Z - translation.Z * 0.5f * rotation.Y,
                    -translation.X * 0.5f * rotation.Z + translation.Y * 0.5f * rotation.W + translation.Z * 0.5f * rotation.X,
                    translation.X * 0.5f * rotation.Y - translation.Y * 0.5f * rotation.X + translation.Z * 0.5f * rotation.W,
                    -translation.X * 0.5f * rotation.X - translation.Y * 0.5f * rotation.Y - translation.Z * 0.5f * rotation.Z
                )
            };
            return dq;
        }

        /// <summary>Creates a dual quaternion from a matrix.</summary>
        public static DualQuaternion FromMatrix(Matrix4x4 matrix)
        {
            Vector3 translation = matrix.Translation;
            Quaternion rotation = Quaternion.CreateFromRotationMatrix(matrix);
            return FromTranslationRotation(translation, rotation);
        }

        /// <summary>Normalizes the dual quaternion.</summary>
        public void Normalize()
        {
            float len = Real.Length();
            if (len > 1e-8f)
            {
                Real = Quaternion.Normalize(Real);
                Dual = Quaternion.Normalize(Dual);
            }
        }

        /// <summary>Conjugates the dual quaternion.</summary>
        public DualQuaternion Conjugate()
        {
            return new DualQuaternion
            {
                Real = Quaternion.Conjugate(Real),
                Dual = Quaternion.Conjugate(Dual)
            };
        }

        /// <summary>Transforms a point by the dual quaternion.</summary>
        public Vector3 TransformPoint(Vector3 point)
        {
            Vector3 t = 2.0f * new Vector3(Dual.X, Dual.Y, Dual.Z);
            Vector3 rxyz = new Vector3(Real.X, Real.Y, Real.Z);

            Vector3 rotated = point + 2.0f * Vector3.Cross(rxyz, Vector3.Cross(rxyz, point) + Real.W * point) + t;
            return rotated;
        }

        /// <summary>Transforms a direction by the dual quaternion (no translation).</summary>
        public Vector3 TransformDirection(Vector3 direction)
        {
            Vector3 rxyz = new Vector3(Real.X, Real.Y, Real.Z);
            return direction + 2.0f * Vector3.Cross(rxyz, Vector3.Cross(rxyz, direction) + Real.W * direction);
        }

        /// <summary>Multiplies two dual quaternions.</summary>
        public static DualQuaternion operator *(DualQuaternion a, DualQuaternion b)
        {
            return new DualQuaternion
            {
                Real = a.Real * b.Real,
                Dual = a.Real * b.Dual + a.Dual * b.Real
            };
        }

        /// <summary>Linearly interpolates between two dual quaternions.</summary>
        public static DualQuaternion Slerp(DualQuaternion a, DualQuaternion b, float t)
        {
            float dot = a.Real.X * b.Real.X + a.Real.Y * b.Real.Y +
                        a.Real.Z * b.Real.Z + a.Real.W * b.Real.W;

            DualQuaternion bAdj = b;
            if (dot < 0)
            {
                bAdj.Real = -bAdj.Real;
                bAdj.Dual = -bAdj.Dual;
                dot = -dot;
            }

            if (dot > 0.9995f)
            {
                // Linear interpolation for nearly parallel quaternions
                return new DualQuaternion
                {
                    Real = Quaternion.Normalize(Quaternion.Lerp(a.Real, bAdj.Real, t)),
                    Dual = Quaternion.Lerp(a.Dual, bAdj.Dual, t)
                };
            }

            float angle = MathF.Acos(MathF.Min(1, MathF.Max(-1, dot)));
            float sinAngle = MathF.Sin(angle);

            float wa = MathF.Sin((1 - t) * angle) / sinAngle;
            float wb = MathF.Sin(t * angle) / sinAngle;

            return new DualQuaternion
            {
                Real = Quaternion.Normalize(a.Real * wa + bAdj.Real * wb),
                Dual = a.Dual * wa + bAdj.Dual * wb
            };
        }
    }

    /// <summary>
    /// Represents a pose (collection of joint transforms).
    /// </summary>
    public sealed class Pose
    {
        /// <summary>Pose name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Local transforms for each joint.</summary>
        public Matrix4x4[] LocalTransforms { get; set; } = Array.Empty<Matrix4x4>();

        /// <summary>Timestamp in seconds (for animation clips).</summary>
        public float Timestamp { get; set; }

        /// <summary>Gets the number of joints in this pose.</summary>
        public int JointCount => LocalTransforms.Length;
    }

    /// <summary>
    /// Represents an animation clip with multiple keyframes.
    /// </summary>
    public sealed class AnimationClip
    {
        /// <summary>Animation name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Keyframe poses.</summary>
        public List<Pose> Keyframes { get; set; } = new();

        /// <summary>Animation duration in seconds.</summary>
        public float Duration => Keyframes.Count > 0 ? Keyframes[^1].Timestamp : 0;

        /// <summary>Whether the animation loops.</summary>
        public bool Loop { get; set; } = true;

        /// <summary>Gets interpolated pose at a given time.</summary>
        public Pose Interpolate(float time)
        {
            if (Keyframes.Count == 0) return new Pose();
            if (Keyframes.Count == 1) return Keyframes[0];

            // Clamp or wrap time
            if (Loop)
            {
                time = time % Duration;
                if (time < 0) time += Duration;
            }
            else
            {
                time = Math.Clamp(time, 0, Duration);
            }

            // Find surrounding keyframes
            int prev = 0;
            for (int i = 0; i < Keyframes.Count - 1; i++)
            {
                if (Keyframes[i + 1].Timestamp > time)
                {
                    prev = i;
                    break;
                }
                prev = i;
            }

            int next = Math.Min(prev + 1, Keyframes.Count - 1);
            float t = 0;

            float range = Keyframes[next].Timestamp - Keyframes[prev].Timestamp;
            if (range > 1e-6f)
                t = (time - Keyframes[prev].Timestamp) / range;

            return PoseLerp(Keyframes[prev], Keyframes[next], t);
        }

        /// <summary>Interpolates between two poses.</summary>
        private static Pose PoseLerp(Pose a, Pose b, float t)
        {
            int count = Math.Max(a.JointCount, b.JointCount);
            var result = new Pose
            {
                Name = a.Name,
                Timestamp = MathHelper.Lerp(a.Timestamp, b.Timestamp, t),
                LocalTransforms = new Matrix4x4[count]
            };

            for (int i = 0; i < count; i++)
            {
                if (i < a.JointCount && i < b.JointCount)
                {
                    result.LocalTransforms[i] = Matrix4x4.Lerp(a.LocalTransforms[i], b.LocalTransforms[i], t);
                }
                else if (i < a.JointCount)
                {
                    result.LocalTransforms[i] = a.LocalTransforms[i];
                }
                else
                {
                    result.LocalTransforms[i] = b.LocalTransforms[i];
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Helper for math operations.
    /// </summary>
    internal static class MathHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Lerp(float a, float b, float t) => a + (b - a) * t;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float InverseLerp(float a, float b, float value)
        {
            if (MathF.Abs(b - a) < 1e-8f) return 0;
            return (value - a) / (b - a);
        }
    }

    /// <summary>
    /// Inverse warping for skeletal animation.
    /// Provides LBS and DQS inverse computation, joint influence evaluation,
    /// bone weight interpolation, bind/world pose transforms, and animation blending.
    /// </summary>
    public sealed class WarpSpace : IDisposable
    {
        private readonly List<Joint> _joints;
        private readonly List<BoneWeight> _vertexWeights;
        private readonly List<AnimationClip> _animations;
        private Pose[]? _blendBuffer;
        private bool _disposed;

        /// <summary>Gets the number of joints.</summary>
        public int JointCount => _joints.Count;

        /// <summary>Gets the number of vertex weight entries.</summary>
        public int VertexWeightCount => _vertexWeights.Count;

        /// <summary>Gets the number of loaded animations.</summary>
        public int AnimationCount => _animations.Count;

        /// <summary>
        /// Initializes a new WarpSpace.
        /// </summary>
        public WarpSpace()
        {
            _joints = new List<Joint>();
            _vertexWeights = new List<BoneWeight>();
            _animations = new List<AnimationClip>();
        }

        /// <summary>
        /// Adds a joint to the skeleton.
        /// </summary>
        /// <param name="joint">The joint to add.</param>
        /// <returns>Index of the added joint.</returns>
        public int AddJoint(Joint joint)
        {
            if (joint == null) throw new ArgumentNullException(nameof(joint));
            int index = _joints.Count;
            joint.Id = index;
            _joints.Add(joint);

            if (joint.ParentIndex >= 0 && joint.ParentIndex < _joints.Count)
            {
                _joints[joint.ParentIndex].Children.Add(index);
            }

            return index;
        }

        /// <summary>
        /// Gets a joint by index.
        /// </summary>
        public Joint GetJoint(int index) => _joints[index];

        /// <summary>
        /// Gets all joints.
        /// </summary>
        public List<Joint> Joints => _joints;

        /// <summary>
        /// Computes bind pose world transforms for the entire skeleton.
        /// </summary>
        public void ComputeBindPoseWorldTransforms()
        {
            for (int i = 0; i < _joints.Count; i++)
            {
                ComputeJointWorldTransform(i, isBindPose: true);
            }

            // Precompute inverse bind pose
            for (int i = 0; i < _joints.Count; i++)
            {
                if (Matrix4x4.Invert(_joints[i].BindPoseWorld, out var inv))
                {
                    _joints[i].InverseBindPoseWorld = inv;
                }
            }
        }

        /// <summary>
        /// Computes world transform for a joint by walking up the hierarchy.
        /// </summary>
        private void ComputeJointWorldTransform(int jointIndex, bool isBindPose)
        {
            var joint = _joints[jointIndex];

            if (isBindPose)
            {
                if (joint.ParentIndex < 0)
                {
                    joint.BindPoseWorld = joint.BindPoseLocal;
                }
                else
                {
                    ComputeJointWorldTransform(joint.ParentIndex, true);
                    joint.BindPoseWorld = joint.BindPoseLocal * _joints[joint.ParentIndex].BindPoseWorld;
                }
            }
            else
            {
                if (joint.ParentIndex < 0)
                {
                    joint.CurrentWorld = joint.CurrentLocal;
                }
                else
                {
                    ComputeJointWorldTransform(joint.ParentIndex, false);
                    joint.CurrentWorld = joint.CurrentLocal * _joints[joint.ParentIndex].CurrentWorld;
                }
            }
        }

        /// <summary>
        /// Updates all current world transforms from local transforms.
        /// </summary>
        public void UpdateWorldTransforms()
        {
            for (int i = 0; i < _joints.Count; i++)
            {
                ComputeJointWorldTransform(i, isBindPose: false);
            }
        }

        /// <summary>
        /// Sets local transform for a joint and updates the hierarchy.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <param name="localTransform">New local transform.</param>
        public void SetJointLocalTransform(int jointIndex, Matrix4x4 localTransform)
        {
            _joints[jointIndex].CurrentLocal = localTransform;
            UpdateWorldTransforms();
        }

        /// <summary>
        /// Sets the current pose from a Pose object.
        /// </summary>
        /// <param name="pose">Pose to apply.</param>
        public void ApplyPose(Pose pose)
        {
            int count = Math.Min(pose.JointCount, _joints.Count);
            for (int i = 0; i < count; i++)
            {
                _joints[i].CurrentLocal = pose.LocalTransforms[i];
            }
            UpdateWorldTransforms();
        }

        /// <summary>
        /// Adds a vertex weight entry.
        /// </summary>
        /// <param name="weight">Bone weight for the vertex.</param>
        public void AddVertexWeight(BoneWeight weight)
        {
            weight.Normalize();
            _vertexWeights.Add(weight);
        }

        /// <summary>
        /// Gets the bone weight for a vertex.
        /// </summary>
        public BoneWeight GetVertexWeight(int vertexIndex) => _vertexWeights[vertexIndex];

        /// <summary>
        /// Performs Linear Blend Skinning (LBS) on a vertex position.
        /// </summary>
        /// <param name="vertex">Original bind-pose vertex position.</param>
        /// <param name="weight">Bone weight.</param>
        /// <returns>Skinned vertex position.</returns>
        public Vector3 SkinVertexLBS(Vector3 vertex, BoneWeight weight)
        {
            Vector3 result = Vector3.Zero;

            SkinVertexLBSContribution(vertex, weight.Joint0, weight.Weight0, ref result);
            SkinVertexLBSContribution(vertex, weight.Joint1, weight.Weight1, ref result);
            SkinVertexLBSContribution(vertex, weight.Joint2, weight.Weight2, ref result);
            SkinVertexLBSContribution(vertex, weight.Joint3, weight.Weight3, ref result);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SkinVertexLBSContribution(Vector3 vertex, int jointIndex, float weight, ref Vector3 result)
        {
            if (weight < 1e-8f || jointIndex < 0 || jointIndex >= _joints.Count) return;

            var joint = _joints[jointIndex];
            // Transform: bindInverse * currentWorld * vertex
            Vector3 transformed = Vector3.Transform(
                Vector3.Transform(vertex, joint.InverseBindPoseWorld),
                joint.CurrentWorld);

            result += transformed * weight;
        }

        /// <summary>
        /// Performs Linear Blend Skinning on a vertex normal.
        /// Uses the inverse transpose of the skinned matrix.
        /// </summary>
        /// <param name="normal">Original bind-pose normal.</param>
        /// <param name="weight">Bone weight.</param>
        /// <returns>Skinned normal.</returns>
        public Vector3 SkinNormalLBS(Vector3 normal, BoneWeight weight)
        {
            Vector3 result = Vector3.Zero;

            SkinNormalLBSContribution(normal, weight.Joint0, weight.Weight0, ref result);
            SkinNormalLBSContribution(normal, weight.Joint1, weight.Weight1, ref result);
            SkinNormalLBSContribution(normal, weight.Joint2, weight.Weight2, ref result);
            SkinNormalLBSContribution(normal, weight.Joint3, weight.Weight3, ref result);

            if (result.LengthSquared() > 1e-10f)
                return Vector3.Normalize(result);
            return normal;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SkinNormalLBSContribution(Vector3 normal, int jointIndex, float weight, ref Vector3 result)
        {
            if (weight < 1e-8f || jointIndex < 0 || jointIndex >= _joints.Count) return;

            var joint = _joints[jointIndex];
            // Normal transform: transpose(inverse(currentWorld * bindInverse))
            // Simplified: use the rotation part of the skinning matrix
            Matrix4x4 skinMatrix = joint.InverseBindPoseWorld * joint.CurrentWorld;

            if (Matrix4x4.Invert(skinMatrix, out var invSkin))
            {
                Matrix4x4 normalMatrix = Matrix4x4.Transpose(invSkin);
                Vector3 transformed = Vector3.TransformNormal(normal, normalMatrix);
                result += transformed * weight;
            }
        }

        /// <summary>
        /// Performs Dual Quaternion Skinning (DQS) on a vertex position.
        /// </summary>
        /// <param name="vertex">Original bind-pose vertex position.</param>
        /// <param name="weight">Bone weight.</param>
        /// <returns>Skinned vertex position.</returns>
        public Vector3 SkinVertexDQS(Vector3 vertex, BoneWeight weight)
        {
            var blendedDQ = BlendDualQuaternions(weight);
            return blendedDQ.TransformPoint(vertex);
        }

        /// <summary>
        /// Performs Dual Quaternion Skinning on a vertex normal.
        /// </summary>
        /// <param name="normal">Original bind-pose normal.</param>
        /// <param name="weight">Bone weight.</param>
        /// <returns>Skinned normal.</returns>
        public Vector3 SkinNormalDQS(Vector3 normal, BoneWeight weight)
        {
            var blendedDQ = BlendDualQuaternions(weight);
            return blendedDQ.TransformDirection(normal);
        }

        /// <summary>
        /// Blends dual quaternions from bone weights.
        /// </summary>
        /// <param name="weight">Bone weight.</param>
        /// <returns>Blended dual quaternion.</returns>
        public DualQuaternion BlendDualQuaternions(BoneWeight weight)
        {
            var result = new DualQuaternion();
            float totalWeight = 0;

            AddDQSContribution(weight.Joint0, weight.Weight0, ref result, ref totalWeight);
            AddDQSContribution(weight.Joint1, weight.Weight1, ref result, ref totalWeight);
            AddDQSContribution(weight.Joint2, weight.Weight2, ref result, ref totalWeight);
            AddDQSContribution(weight.Joint3, weight.Weight3, ref result, ref totalWeight);

            if (totalWeight > 1e-8f)
            {
                result.Normalize();
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AddDQSContribution(int jointIndex, float weight,
            ref DualQuaternion result, ref float totalWeight)
        {
            if (weight < 1e-8f || jointIndex < 0 || jointIndex >= _joints.Count) return;

            var joint = _joints[jointIndex];
            var bindDQ = DualQuaternion.FromMatrix(joint.BindPoseWorld);
            var currentDQ = DualQuaternion.FromMatrix(joint.CurrentWorld);

            // DQS: currentDQ * bindInverseDQ
            var skinDQ = currentDQ * bindDQ.Conjugate();

            // Ensure consistent hemisphere for blending
            if (totalWeight > 1e-8f && skinDQ.Real.W < 0)
            {
                skinDQ.Real = -skinDQ.Real;
                skinDQ.Dual = -skinDQ.Dual;
            }

            result.Real += skinDQ.Real * weight;
            result.Dual += skinDQ.Dual * weight;
            totalWeight += weight;
        }

        /// <summary>
        /// Computes the influence of each joint on a given world-space point.
        /// </summary>
        /// <param name="point">World-space query point.</param>
        /// <param name="maxInfluenceDistance">Maximum distance for influence.</param>
        /// <returns>Array of (jointIndex, weight) pairs.</returns>
        public List<(int JointIndex, float Weight)> EvaluateJointInfluence(Vector3 point,
            float maxInfluenceDistance)
        {
            var influences = new List<(int, float)>();
            float totalWeight = 0;

            for (int i = 0; i < _joints.Count; i++)
            {
                Vector3 jointPos = _joints[i].CurrentWorld.Translation;
                float dist = Vector3.Distance(point, jointPos);

                if (dist < maxInfluenceDistance)
                {
                    // Distance-based influence with smooth falloff
                    float normalizedDist = dist / maxInfluenceDistance;
                    float w = 1.0f - normalizedDist * normalizedDist;
                    w = MathF.Max(0, w);
                    influences.Add((i, w));
                    totalWeight += w;
                }
            }

            // Normalize weights
            if (totalWeight > 1e-8f)
            {
                for (int i = 0; i < influences.Count; i++)
                {
                    var influence = influences[i];
                    influences[i] = (influence.Item1, influence.Item2 / totalWeight);
                }
            }

            return influences;
        }

        /// <summary>
        /// Normalizes bone weights for all vertices.
        /// </summary>
        public void NormalizeAllWeights()
        {
            for (int i = 0; i < _vertexWeights.Count; i++)
            {
                var w = _vertexWeights[i];
                w.Normalize();
                _vertexWeights[i] = w;
            }
        }

        /// <summary>
        /// Interpolates bone weights between two sets.
        /// </summary>
        /// <param name="a">First weight set.</param>
        /// <param name="b">Second weight set.</param>
        /// <param name="t">Interpolation factor [0,1].</param>
        /// <returns>Interpolated bone weight.</returns>
        public static BoneWeight InterpolateWeights(BoneWeight a, BoneWeight b, float t)
        {
            // Collect all unique joint indices
            var joints = new HashSet<int>();
            AddUniqueJoint(a, joints);
            AddUniqueJoint(b, joints);

            var result = new BoneWeight();
            float[] weights = new float[4];
            int[] jointIndices = new int[4];
            int count = 0;

            foreach (int joint in joints)
            {
                if (count >= 4) break;

                float wA = GetWeightForJoint(a, joint);
                float wB = GetWeightForJoint(b, joint);
                float w = MathHelper.Lerp(wA, wB, t);

                if (w > 1e-6f)
                {
                    jointIndices[count] = joint;
                    weights[count] = w;
                    count++;
                }
            }

            // Take the top 4 weights
            result.Joint0 = count > 0 ? jointIndices[0] : 0;
            result.Weight0 = count > 0 ? weights[0] : 0;
            result.Joint1 = count > 1 ? jointIndices[1] : 0;
            result.Weight1 = count > 1 ? weights[1] : 0;
            result.Joint2 = count > 2 ? jointIndices[2] : 0;
            result.Weight2 = count > 2 ? weights[2] : 0;
            result.Joint3 = count > 3 ? jointIndices[3] : 0;
            result.Weight3 = count > 3 ? weights[3] : 0;

            result.Normalize();
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddUniqueJoint(BoneWeight weight, HashSet<int> joints)
        {
            if (weight.Weight0 > 1e-6f) joints.Add(weight.Joint0);
            if (weight.Weight1 > 1e-6f) joints.Add(weight.Joint1);
            if (weight.Weight2 > 1e-6f) joints.Add(weight.Joint2);
            if (weight.Weight3 > 1e-6f) joints.Add(weight.Joint3);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GetWeightForJoint(BoneWeight weight, int joint)
        {
            if (weight.Joint0 == joint) return weight.Weight0;
            if (weight.Joint1 == joint) return weight.Weight1;
            if (weight.Joint2 == joint) return weight.Weight2;
            if (weight.Joint3 == joint) return weight.Weight3;
            return 0;
        }

        /// <summary>
        /// Adds an animation clip.
        /// </summary>
        /// <param name="clip">Animation clip to add.</param>
        public void AddAnimation(AnimationClip clip)
        {
            if (clip == null) throw new ArgumentNullException(nameof(clip));
            _animations.Add(clip);
        }

        /// <summary>
        /// Gets an animation clip by index.
        /// </summary>
        public AnimationClip GetAnimation(int index) => _animations[index];

        /// <summary>
        /// Blends two poses with a given factor.
        /// </summary>
        /// <param name="poseA">First pose.</param>
        /// <param name="poseB">Second pose.</param>
        /// <param name="factor">Blend factor [0,1].</param>
        /// <returns>Blended pose.</returns>
        public Pose BlendPoses(Pose poseA, Pose poseB, float factor)
        {
            factor = Math.Clamp(factor, 0, 1);
            int count = Math.Max(poseA.JointCount, poseB.JointCount);
            var result = new Pose
            {
                Name = $"Blend({poseA.Name}, {poseB.Name})",
                LocalTransforms = new Matrix4x4[count],
                Timestamp = MathHelper.Lerp(poseA.Timestamp, poseB.Timestamp, factor)
            };

            for (int i = 0; i < count; i++)
            {
                var tA = i < poseA.JointCount ? poseA.LocalTransforms[i] : Matrix4x4.Identity;
                var tB = i < poseB.JointCount ? poseB.LocalTransforms[i] : Matrix4x4.Identity;
                result.LocalTransforms[i] = Matrix4x4.Lerp(tA, tB, factor);
            }

            return result;
        }

        /// <summary>
        /// Multi-way blend of poses with weights.
        /// </summary>
        /// <param name="poses">Poses to blend.</param>
        /// <param name="weights">Corresponding weights.</param>
        /// <returns>Blended pose.</returns>
        public Pose BlendPosesMulti(ReadOnlySpan<Pose> poses, ReadOnlySpan<float> weights)
        {
            if (poses.Length != weights.Length)
                throw new ArgumentException("Poses and weights must have the same length.");

            if (poses.Length == 0) return new Pose();
            if (poses.Length == 1) return poses[0];

            // Normalize weights
            float totalWeight = 0;
            for (int i = 0; i < weights.Length; i++) totalWeight += weights[i];

            if (totalWeight < 1e-8f) return poses[0];

            int maxJoints = 0;
            for (int i = 0; i < poses.Length; i++)
                maxJoints = Math.Max(maxJoints, poses[i].JointCount);

            var result = new Pose
            {
                Name = "MultiBlend",
                LocalTransforms = new Matrix4x4[maxJoints]
            };

            for (int j = 0; j < maxJoints; j++)
            {
                var blended = RenderingMath.ZeroMatrix;
                for (int i = 0; i < poses.Length; i++)
                {
                    if (j < poses[i].JointCount)
                    {
                        blended += poses[i].LocalTransforms[j] * (weights[i] / totalWeight);
                    }
                }
                result.LocalTransforms[j] = blended;
            }

            return result;
        }

        /// <summary>
        /// Applies additive animation blending.
        /// </summary>
        /// <param name="basePose">Base pose.</param>
        /// <param name="additivePose">Additive pose (relative to identity).</param>
        /// <param name="weight">Additive blend weight.</param>
        /// <returns>Result pose.</returns>
        public Pose BlendAdditive(Pose basePose, Pose additivePose, float weight)
        {
            int count = Math.Max(basePose.JointCount, additivePose.JointCount);
            var result = new Pose
            {
                Name = $"Additive({basePose.Name})",
                LocalTransforms = new Matrix4x4[count],
                Timestamp = basePose.Timestamp
            };

            for (int i = 0; i < count; i++)
            {
                var baseT = i < basePose.JointCount ? basePose.LocalTransforms[i] : Matrix4x4.Identity;
                var addT = i < additivePose.JointCount ? additivePose.LocalTransforms[i] : Matrix4x4.Identity;

                // Decompose, add rotation, recompose
                Vector3 baseScale = ExtractScale(baseT);
                Quaternion baseRot = Quaternion.CreateFromRotationMatrix(baseT);
                Vector3 baseTrans = baseT.Translation;

                Quaternion addRot = Quaternion.CreateFromRotationMatrix(addT);
                Quaternion blendedRot = Quaternion.Slerp(Quaternion.Identity, addRot, weight) * baseRot;

                result.LocalTransforms[i] = Matrix4x4.CreateScale(baseScale)
                    * Matrix4x4.CreateFromQuaternion(blendedRot)
                    * Matrix4x4.CreateTranslation(baseTrans);
            }

            return result;
        }

        /// <summary>
        /// Extracts scale from a transformation matrix.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 ExtractScale(Matrix4x4 matrix)
        {
            float sx = new Vector3(matrix.M11, matrix.M12, matrix.M13).Length();
            float sy = new Vector3(matrix.M21, matrix.M22, matrix.M23).Length();
            float sz = new Vector3(matrix.M31, matrix.M32, matrix.M33).Length();
            return new Vector3(sx, sy, sz);
        }

        /// <summary>
        /// Transforms a point from bind space to world space using LBS.
        /// </summary>
        public Vector3 BindToWorldLBS(Vector3 bindPoint, BoneWeight weight)
        {
            return SkinVertexLBS(bindPoint, weight);
        }

        /// <summary>
        /// Transforms a point from bind space to world space using DQS.
        /// </summary>
        public Vector3 BindToWorldDQS(Vector3 bindPoint, BoneWeight weight)
        {
            return SkinVertexDQS(bindPoint, weight);
        }

        /// <summary>
        /// Transforms a point from world space to bind space (inverse skinning) using LBS.
        /// This is an approximation using iterative refinement.
        /// </summary>
        /// <param name="worldPoint">World-space point.</param>
        /// <param name="weight">Bone weight.</param>
        /// <param name="maxIterations">Maximum refinement iterations.</param>
        /// <returns>Approximate bind-space point.</returns>
        public Vector3 WorldToBindLBS(Vector3 worldPoint, BoneWeight weight, int maxIterations = 8)
        {
            // Initial estimate using first joint
            int primaryJoint = weight.Joint0;
            Vector3 bindEstimate = Vector3.Transform(
                worldPoint,
                _joints[primaryJoint].InverseBindPoseWorld);

            // Iterative refinement
            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Forward skin the current estimate
                Vector3 skinned = SkinVertexLBS(bindEstimate, weight);
                // Compute error
                Vector3 error = worldPoint - skinned;
                // Apply correction (approximate inverse Jacobian)
                Vector3 correction = Vector3.Transform(
                    error,
                    _joints[primaryJoint].CurrentWorld);
                bindEstimate += correction * 0.5f;
            }

            return bindEstimate;
        }

        /// <summary>
        /// Transforms a point from world space to bind space using DQS.
        /// </summary>
        public Vector3 WorldToBindDQS(Vector3 worldPoint, BoneWeight weight, int maxIterations = 8)
        {
            int primaryJoint = weight.Joint0;
            Vector3 bindEstimate = Vector3.Transform(
                worldPoint,
                _joints[primaryJoint].InverseBindPoseWorld);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                Vector3 skinned = SkinVertexDQS(bindEstimate, weight);
                Vector3 error = worldPoint - skinned;
                Vector3 correction = Vector3.Transform(
                    error,
                    _joints[primaryJoint].CurrentWorld);
                bindEstimate += correction * 0.5f;
            }

            return bindEstimate;
        }

        /// <summary>
        /// Computes the skinning matrix for a specific joint.
        /// </summary>
        /// <param name="jointIndex">Joint index.</param>
        /// <returns>4x4 skinning matrix.</returns>
        public Matrix4x4 GetSkinningMatrix(int jointIndex)
        {
            var joint = _joints[jointIndex];
            return joint.InverseBindPoseWorld * joint.CurrentWorld;
        }

        /// <summary>
        /// Computes all skinning matrices.
        /// </summary>
        /// <returns>Array of skinning matrices.</returns>
        public Matrix4x4[] GetAllSkinningMatrices()
        {
            var matrices = new Matrix4x4[_joints.Count];
            for (int i = 0; i < _joints.Count; i++)
            {
                matrices[i] = GetSkinningMatrix(i);
            }
            return matrices;
        }

        /// <summary>
        /// Evaluates the Jacobian of the skinning function at a point.
        /// Used for gradient-based optimization of inverse skinning.
        /// </summary>
        /// <param name="point">Bind-space point.</param>
        /// <param name="weight">Bone weight.</param>
        /// <param name="epsilon">Finite difference step.</param>
        /// <returns>3x3 Jacobian matrix (stored as 3 Vector3 rows).</returns>
        public (Vector3 Row0, Vector3 Row1, Vector3 Row2) ComputeSkinningJacobian(
            Vector3 point, BoneWeight weight, float epsilon = 0.001f)
        {
            Vector3 fx0 = SkinVertexLBS(point - new Vector3(epsilon, 0, 0), weight);
            Vector3 fx1 = SkinVertexLBS(point + new Vector3(epsilon, 0, 0), weight);
            Vector3 fy0 = SkinVertexLBS(point - new Vector3(0, epsilon, 0), weight);
            Vector3 fy1 = SkinVertexLBS(point + new Vector3(0, epsilon, 0), weight);
            Vector3 fz0 = SkinVertexLBS(point - new Vector3(0, 0, epsilon), weight);
            Vector3 fz1 = SkinVertexLBS(point + new Vector3(0, 0, epsilon), weight);

            float inv2e = 1.0f / (2.0f * epsilon);

            Vector3 col0 = (fx1 - fx0) * inv2e;
            Vector3 col1 = (fy1 - fy0) * inv2e;
            Vector3 col2 = (fz1 - fz0) * inv2e;

            return (col0, col1, col2);
        }

        /// <summary>
        /// Performs animation at a given time.
        /// </summary>
        /// <param name="animationIndex">Animation clip index.</param>
        /// <param name="time">Time in seconds.</param>
        public void Animate(int animationIndex, float time)
        {
            var clip = _animations[animationIndex];
            var pose = clip.Interpolate(time);
            ApplyPose(pose);
        }

        /// <summary>
        /// Blends between two animations at given times.
        /// </summary>
        /// <param name="animIndexA">First animation index.</param>
        /// <param name="timeA">Time in first animation.</param>
        /// <param name="animIndexB">Second animation index.</param>
        /// <param name="timeB">Time in second animation.</param>
        /// <param name="blendFactor">Blend factor [0,1].</param>
        public void AnimateBlended(int animIndexA, float timeA, int animIndexB, float timeB, float blendFactor)
        {
            var clipA = _animations[animIndexA];
            var clipB = _animations[animIndexB];
            var poseA = clipA.Interpolate(timeA);
            var poseB = clipB.Interpolate(timeB);
            var blended = BlendPoses(poseA, poseB, blendFactor);
            ApplyPose(blended);
        }

        /// <summary>
        /// Disposes the WarpSpace.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _joints.Clear();
            _vertexWeights.Clear();
            _animations.Clear();
            _blendBuffer = null;
            GC.SuppressFinalize(this);
        }
    }
}
