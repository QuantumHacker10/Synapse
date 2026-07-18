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
// FILE: AnimationBlender.cs
// PATH: Animation/AnimationBlender.cs
// ============================================================


using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GDNN.Animation
{
    /// <summary>
    /// Manages multi-clip blending with weights, cross-fade transitions, a state machine
    /// for animation states, additive blending, layer-based blending, aim IK target blending,
    /// and a blend tree for locomotion.
    /// </summary>
    public sealed class AnimationBlender : IDisposable
    {
        public const int MaxBlendLayers = 16;
        public const int MaxStateTransitions = 64;
        public const int MaxBlendTreeNodes = 32;
        public const int MaxAnimationStates = 32;
        public const int MaxCrossfades = 8;

        private BlendLayer[] _layers;
        private int _layerCount;
        private AnimationState[] _states;
        private int _stateCount;
        private int _currentStateIndex;
        private CrossfadeInfo[] _crossfades;
        private int _crossfadeCount;
        private BlendTree _blendTree;
        private AimIKTarget _aimIK;
        private Skeleton? _ownerSkeleton;
        private bool _disposed;

        /// <summary>
        /// Represents a single blend layer containing a clip reference and weight.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct BlendLayer
        {
            public AnimationClip? Clip;
            public float Weight;
            public float PlaybackSpeed;
            public float ElapsedTime;
            public bool Enabled;
            public bool Additive;
            public int Order;
            public byte Padding;

            public float NormalizedTime => Clip != null && Clip.Duration > 0f
                ? (ElapsedTime % Clip.Duration) / Clip.Duration
                : 0f;
        }

        /// <summary>
        /// Represents a transition between two animation states.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct StateTransition
        {
            public int FromState;
            public int ToState;
            public float Duration;
            public int ConditionType;
            public float ConditionValue;
            public bool HasExitTime;
            public float ExitTime;
            public int Padding;
        }

        /// <summary>
        /// Represents an animation state in the state machine.
        /// </summary>
        public sealed class AnimationState
        {
            public string Name;
            public AnimationClip? Clip;
            public float Speed;
            public float Weight;
            public int TransitionCount;
            public StateTransition[] Transitions;
            public bool IsDefault;
            public object? UserData;

            public float PlaybackTime;

            public float NormalizedTime => Clip != null && Clip.Duration > 0f
                ? (PlaybackTime % Clip.Duration) / Clip.Duration
                : 0f;

            public AnimationState(string name)
            {
                Name = name ?? "Unnamed";
                Clip = null;
                Speed = 1f;
                Weight = 0f;
                TransitionCount = 0;
                Transitions = new StateTransition[8];
                IsDefault = false;
                UserData = null;
            }
        }

        /// <summary>
        /// Holds information about an active crossfade transition.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct CrossfadeInfo
        {
            public int SourceStateIndex;
            public int TargetStateIndex;
            public float Duration;
            public float Elapsed;
            public bool IsComplete;

            public readonly float Progress => Duration > 0f ? Math.Clamp(Elapsed / Duration, 0f, 1f) : 1f;
            public readonly float Weight => Progress;
        }

        /// <summary>
        /// Represents a single node in a blend tree.
        /// </summary>
        public sealed class BlendTreeNode
        {
            public AnimationClip? Clip;
            public Vector2 MotionParameter;
            public float Threshold;
            public float Weight;
            public int ChildCount;
            public BlendTreeNode[] Children;
            public BlendTreeBlendMode BlendMode;

            public BlendTreeNode()
            {
                Clip = null;
                MotionParameter = Vector2.Zero;
                Threshold = 0f;
                Weight = 1f;
                ChildCount = 0;
                Children = Array.Empty<BlendTreeNode>();
                BlendMode = BlendTreeBlendMode.Simple1D;
            }
        }

        /// <summary>
        /// Blend modes for blend tree nodes.
        /// </summary>
        public enum BlendTreeBlendMode : byte
        {
            Simple1D = 0,
            SimpleDirectional2D = 1,
            FreeformDirectional2D = 2,
            FreeformCartesian2D = 3
        }

        /// <summary>
        /// Represents the aim IK target for directional blending.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct AimIKTarget
        {
            public Vector3 TargetPosition;
            public Vector3 CurrentAimDirection;
            public Vector3 DesiredAimDirection;
            public float Weight;
            public float Damping;
            public int AimJointIndex;
            public int LookAtJointIndex;
            public bool Enabled;

            public static AimIKTarget Identity => new AimIKTarget
            {
                TargetPosition = Vector3.UnitZ,
                CurrentAimDirection = Vector3.UnitZ,
                DesiredAimDirection = Vector3.UnitZ,
                Weight = 0f,
                Damping = 8f,
                AimJointIndex = -1,
                LookAtJointIndex = -1,
                Enabled = false
            };
        }

        public AnimationBlender()
        {
            _layers = new BlendLayer[MaxBlendLayers];
            _layerCount = 0;
            _states = new AnimationState[MaxAnimationStates];
            _stateCount = 0;
            _currentStateIndex = -1;
            _crossfades = new CrossfadeInfo[MaxCrossfades];
            _crossfadeCount = 0;
            _blendTree = new BlendTree();
            _aimIK = AimIKTarget.Identity;
            _ownerSkeleton = null;
            _disposed = false;
        }

        public int LayerCount => _layerCount;
        public int StateCount => _stateCount;
        public int CurrentStateIndex => _currentStateIndex;
        public int CrossfadeCount => _crossfadeCount;
        public BlendTree BlendTreeRoot => _blendTree;

        public ref AimIKTarget AimIK => ref _aimIK;

        public void SetOwnerSkeleton(Skeleton skeleton)
        {
            _ownerSkeleton = skeleton;
        }

        // ── Multi-Clip Blending with Weights ─────────────────────────────

        public int AddLayer(AnimationClip clip, float weight = 1.0f, bool additive = false)
        {
            if (_layerCount >= MaxBlendLayers)
                throw new InvalidOperationException($"Maximum layer count ({MaxBlendLayers}) exceeded.");

            int index = _layerCount++;
            _layers[index] = new BlendLayer
            {
                Clip = clip,
                Weight = weight,
                PlaybackSpeed = 1f,
                ElapsedTime = 0f,
                Enabled = true,
                Additive = additive,
                Order = _layerCount,
                Padding = 0
            };

            return index;
        }

        public void RemoveLayer(int layerIndex)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));

            for (int i = layerIndex; i < _layerCount - 1; i++)
            {
                _layers[i] = _layers[i + 1];
            }
            _layerCount--;
        }

        public ref BlendLayer GetLayer(int layerIndex)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            return ref _layers[layerIndex];
        }

        public void SetLayerWeight(int layerIndex, float weight)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            _layers[layerIndex].Weight = weight;
        }

        public void SetLayerEnabled(int layerIndex, bool enabled)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            _layers[layerIndex].Enabled = enabled;
        }

        public void UpdateLayers(float deltaTime)
        {
            for (int i = 0; i < _layerCount; i++)
            {
                if (!_layers[i].Enabled || _layers[i].Clip == null) continue;

                ref BlendLayer layer = ref _layers[i];
                layer.ElapsedTime += deltaTime * layer.PlaybackSpeed;

                if (layer.Clip.Loop)
                {
                    while (layer.ElapsedTime >= layer.Clip.Duration && layer.Clip.Duration > 0f)
                        layer.ElapsedTime -= layer.Clip.Duration;
                }
                else
                {
                    layer.ElapsedTime = Math.Min(layer.ElapsedTime, layer.Clip.Duration);
                }
            }
        }

        public void BlendAllLayers(Skeleton targetSkeleton)
        {
            if (targetSkeleton == null) throw new ArgumentNullException(nameof(targetSkeleton));

            bool first = true;

            for (int i = 0; i < _layerCount; i++)
            {
                if (!_layers[i].Enabled || _layers[i].Clip == null || _layers[i].Weight <= 0f)
                    continue;

                ref BlendLayer layer = ref _layers[i];

                if (_layers[i].Additive)
                {
                    ApplyAdditiveLayer(targetSkeleton, layer.Clip, layer.ElapsedTime, layer.Weight);
                }
                else
                {
                    if (first)
                    {
                        layer.Clip.SampleAtTime(layer.ElapsedTime, targetSkeleton);
                        first = false;
                    }
                    else
                    {
                        ApplyOverrideLayer(targetSkeleton, layer.Clip, layer.ElapsedTime, layer.Weight);
                    }
                }
            }
        }

        private void ApplyOverrideLayer(Skeleton skeleton, AnimationClip clip, float time, float weight)
        {
            for (int j = 0; j < skeleton.JointCount; j++)
            {
                if (clip.TryGetJointData(j, out var data))
                {
                    Vector3 pos = clip.SamplePosition(data, time);
                    Quaternion rot = clip.SampleRotation(data, time);
                    Vector3 scl = clip.SampleScale(data, time);
                    skeleton.BlendJoint(j, pos, rot, scl, weight);
                }
            }
        }

        private void ApplyAdditiveLayer(Skeleton skeleton, AnimationClip clip, float time, float weight)
        {
            for (int j = 0; j < skeleton.JointCount; j++)
            {
                if (clip.TryGetJointData(j, out var data))
                {
                    Vector3 pos = clip.SamplePosition(data, time);
                    Quaternion rot = clip.SampleRotation(data, time);
                    Vector3 scl = clip.SampleScale(data, time);
                    skeleton.BlendAdditive(j, pos, rot, scl, weight);
                }
            }
        }

        // ── Cross-Fade Transitions ───────────────────────────────────────

        public void CrossfadeTo(int targetStateIndex, float fadeDuration)
        {
            if ((uint)targetStateIndex >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(targetStateIndex));
            if (_crossfadeCount >= MaxCrossfades)
                throw new InvalidOperationException("Maximum crossfade count exceeded.");

            int idx = _crossfadeCount++;
            _crossfades[idx] = new CrossfadeInfo
            {
                SourceStateIndex = _currentStateIndex,
                TargetStateIndex = targetStateIndex,
                Duration = Math.Max(0.01f, fadeDuration),
                Elapsed = 0f,
                IsComplete = false
            };
        }

        public void CrossfadeTo(string stateName, float fadeDuration)
        {
            int idx = FindState(stateName);
            if (idx < 0) throw new ArgumentException($"State '{stateName}' not found.");
            CrossfadeTo(idx, fadeDuration);
        }

        public void UpdateCrossfades(float deltaTime)
        {
            for (int i = _crossfadeCount - 1; i >= 0; i--)
            {
                ref CrossfadeInfo cf = ref _crossfades[i];
                cf.Elapsed += deltaTime;

                if (cf.Elapsed >= cf.Duration)
                {
                    _currentStateIndex = cf.TargetStateIndex;
                    cf.IsComplete = true;

                    for (int j = i; j < _crossfadeCount - 1; j++)
                        _crossfades[j] = _crossfades[j + 1];
                    _crossfadeCount--;
                }
            }
        }

        public void ApplyCrossfades(Skeleton targetSkeleton)
        {
            if (targetSkeleton == null) return;

            if (_crossfadeCount == 0)
            {
                if (_currentStateIndex >= 0 && _currentStateIndex < _stateCount)
                {
                    var state = _states[_currentStateIndex];
                    if (state.Clip != null)
                    {
                        state.Clip.SampleAtTime(state.Clip.Duration * state.NormalizedTime, targetSkeleton);
                    }
                }
                return;
            }

            for (int i = 0; i < _crossfadeCount; i++)
            {
                ref CrossfadeInfo cf = ref _crossfades[i];

                if (cf.SourceStateIndex >= 0 && cf.SourceStateIndex < _stateCount)
                {
                    var srcState = _states[cf.SourceStateIndex];
                    float srcWeight = 1.0f - cf.Weight;

                    if (srcState.Clip != null && srcWeight > 0.001f)
                    {
                        ApplyOverrideLayer(targetSkeleton, srcState.Clip,
                            srcState.Clip.Duration * srcState.NormalizedTime, srcWeight);
                    }
                }

                if (cf.TargetStateIndex >= 0 && cf.TargetStateIndex < _stateCount)
                {
                    var tgtState = _states[cf.TargetStateIndex];
                    float tgtWeight = cf.Weight;

                    if (tgtState.Clip != null && tgtWeight > 0.001f)
                    {
                        ApplyOverrideLayer(targetSkeleton, tgtState.Clip,
                            tgtState.Clip.Duration * tgtState.NormalizedTime, tgtWeight);
                    }
                }
            }
        }

        // ── State Machine ────────────────────────────────────────────────

        public int AddState(string name, AnimationClip? clip = null)
        {
            if (_stateCount >= MaxAnimationStates)
                throw new InvalidOperationException($"Maximum state count ({MaxAnimationStates}) exceeded.");

            int index = _stateCount++;
            _states[index] = new AnimationState(name) { Clip = clip };

            if (_currentStateIndex < 0)
                _currentStateIndex = index;

            return index;
        }

        public void RemoveState(int stateIndex)
        {
            if ((uint)stateIndex >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(stateIndex));

            for (int i = stateIndex; i < _stateCount - 1; i++)
            {
                _states[i] = _states[i + 1];
                _states[i].TransitionCount = 0;
                for (int t = 0; t < _states[i].Transitions.Length; t++)
                {
                    ref StateTransition tr = ref _states[i].Transitions[t];
                    if (tr.FromState > stateIndex) tr.FromState--;
                    if (tr.ToState > stateIndex) tr.ToState--;
                }
            }

            _stateCount--;

            if (_currentStateIndex == stateIndex)
                _currentStateIndex = _stateCount > 0 ? 0 : -1;
            else if (_currentStateIndex > stateIndex)
                _currentStateIndex--;
        }

        public int FindState(string name)
        {
            for (int i = 0; i < _stateCount; i++)
            {
                if (string.Equals(_states[i].Name, name, StringComparison.Ordinal))
                    return i;
            }
            return -1;
        }

        public ref AnimationState GetState(int stateIndex)
        {
            if ((uint)stateIndex >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(stateIndex));
            return ref _states[stateIndex];
        }

        public void SetCurrentState(int stateIndex)
        {
            if ((uint)stateIndex >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(stateIndex));
            _currentStateIndex = stateIndex;
        }

        public void SetCurrentState(string name)
        {
            int idx = FindState(name);
            if (idx < 0) throw new ArgumentException($"State '{name}' not found.");
            _currentStateIndex = idx;
        }

        public void AddTransition(int fromState, int toState, float duration,
            bool hasExitTime = false, float exitTime = 0f,
            int conditionType = 0, float conditionValue = 0f)
        {
            if ((uint)fromState >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(fromState));
            if ((uint)toState >= (uint)_stateCount)
                throw new ArgumentOutOfRangeException(nameof(toState));

            var state = _states[fromState];
            if (state.TransitionCount >= state.Transitions.Length)
                Array.Resize(ref state.Transitions, state.Transitions.Length * 2);

            state.Transitions[state.TransitionCount++] = new StateTransition
            {
                FromState = fromState,
                ToState = toState,
                Duration = duration,
                ConditionType = conditionType,
                ConditionValue = conditionValue,
                HasExitTime = hasExitTime,
                ExitTime = exitTime,
                Padding = 0
            };
        }

        public void UpdateStateMachine(float deltaTime, float parameterValue = 0f)
        {
            UpdateCrossfades(deltaTime);

            if (_currentStateIndex < 0 || _currentStateIndex >= _stateCount) return;

            ref AnimationState current = ref _states[_currentStateIndex];
            if (current.Clip != null)
            {
                current.Clip.SampleAtTime(
                    current.Clip.Duration * current.NormalizedTime,
                    _ownerSkeleton!);
            }

            for (int t = 0; t < current.TransitionCount; t++)
            {
                ref StateTransition tr = ref current.Transitions[t];

                bool shouldTransition = false;

                if (tr.HasExitTime)
                {
                    float normalizedTime = current.Clip != null && current.Clip.Duration > 0f
                        ? (current.Clip.Duration * current.NormalizedTime) / current.Clip.Duration
                        : 0f;
                    shouldTransition = normalizedTime >= tr.ExitTime;
                }

                if (tr.ConditionType > 0)
                {
                    shouldTransition = tr.ConditionType switch
                    {
                        1 => parameterValue > tr.ConditionValue,
                        2 => parameterValue < tr.ConditionValue,
                        3 => Math.Abs(parameterValue - tr.ConditionValue) < 0.01f,
                        4 => Math.Abs(parameterValue - tr.ConditionValue) >= 0.01f,
                        _ => shouldTransition
                    };
                }
                else
                {
                    shouldTransition = true;
                }

                if (shouldTransition && _crossfadeCount == 0)
                {
                    CrossfadeTo(tr.ToState, tr.Duration);
                    break;
                }
            }
        }

        // ── Additive Animation Blending ──────────────────────────────────

        public void ApplyAdditiveClip(Skeleton targetSkeleton, AnimationClip additiveClip, float time, float weight)
        {
            if (targetSkeleton == null) throw new ArgumentNullException(nameof(targetSkeleton));
            if (additiveClip == null) throw new ArgumentNullException(nameof(additiveClip));

            for (int j = 0; j < targetSkeleton.JointCount; j++)
            {
                if (additiveClip.TryGetJointData(j, out var data))
                {
                    Vector3 pos = additiveClip.SamplePosition(data, time);
                    Quaternion rot = additiveClip.SampleRotation(data, time);
                    Vector3 scl = additiveClip.SampleScale(data, time);
                    targetSkeleton.BlendAdditive(j, pos, rot, scl, weight);
                }
            }
        }

        public static void BlendAdditiveTransforms(
            JointTransform baseTransform,
            JointTransform additiveTransform,
            float weight,
            out JointTransform result)
        {
            result = JointTransform.BlendAdditive(baseTransform, additiveTransform, weight);
        }

        public static void ComputeAdditiveDelta(
            JointTransform baseTransform,
            JointTransform currentTransform,
            out JointTransform additiveDelta)
        {
            additiveDelta = currentTransform.Delta(baseTransform);
        }

        // ── Layer-Based Blending ─────────────────────────────────────────

        public void ApplyLayeredBlending(Skeleton targetSkeleton)
        {
            if (targetSkeleton == null) return;

            int[] sortedLayers = new int[_layerCount];
            for (int i = 0; i < _layerCount; i++) sortedLayers[i] = i;
            Array.Sort(sortedLayers, (a, b) => _layers[a].Order.CompareTo(_layers[b].Order));

            bool hasBase = false;

            for (int idx = 0; idx < _layerCount; idx++)
            {
                int layerIdx = sortedLayers[idx];
                if (!_layers[layerIdx].Enabled || _layers[layerIdx].Clip == null) continue;

                ref BlendLayer layer = ref _layers[layerIdx];

                if (!hasBase)
                {
                    layer.Clip.SampleAtTime(layer.ElapsedTime, targetSkeleton);
                    hasBase = true;
                }
                else if (layer.Additive)
                {
                    ApplyAdditiveLayer(targetSkeleton, layer.Clip, layer.ElapsedTime, layer.Weight);
                }
                else
                {
                    ApplyOverrideLayer(targetSkeleton, layer.Clip, layer.ElapsedTime, layer.Weight);
                }
            }
        }

        public void SetLayerOrder(int layerIndex, int order)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            _layers[layerIndex].Order = order;
        }

        public void SetLayerAdditive(int layerIndex, bool additive)
        {
            if ((uint)layerIndex >= (uint)_layerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            _layers[layerIndex].Additive = additive;
        }

        // ── Aim IK Target Blending ──────────────────────────────────────

        public void UpdateAimIK(float deltaTime)
        {
            if (!_aimIK.Enabled || _ownerSkeleton == null) return;
            if (_aimIK.AimJointIndex < 0 || _aimIK.AimJointIndex >= _ownerSkeleton.JointCount) return;

            _aimIK.DesiredAimDirection = Vector3.Normalize(
                _aimIK.TargetPosition - _ownerSkeleton.GetJoint(_aimIK.AimJointIndex).LocalTranslation);

            if (_aimIK.DesiredAimDirection.LengthSquared() < 1e-6f)
                return;

            float dampingFactor = 1f - MathF.Exp(-_aimIK.Damping * deltaTime);
            _aimIK.CurrentAimDirection = Vector3.Lerp(
                _aimIK.CurrentAimDirection,
                _aimIK.DesiredAimDirection,
                dampingFactor);

            _aimIK.CurrentAimDirection = Vector3.Normalize(_aimIK.CurrentAimDirection);
        }

        public void ApplyAimIK(Skeleton targetSkeleton)
        {
            if (!_aimIK.Enabled || _ownerSkeleton == null) return;
            if (targetSkeleton == null) return;

            if (_aimIK.AimJointIndex < 0 || _aimIK.AimJointIndex >= targetSkeleton.JointCount) return;

            float weight = _aimIK.Weight;
            if (weight <= 0f) return;

            Vector3 aimDir = Vector3.Normalize(_aimIK.CurrentAimDirection);

            Vector3 forward = Vector3.UnitZ;
            float dot = Vector3.Dot(forward, aimDir);
            dot = Math.Clamp(dot, -1f, 1f);
            float angle = MathF.Acos(dot);

            if (angle < 0.001f) return;

            Vector3 axis = Vector3.Cross(forward, aimDir);
            if (axis.LengthSquared() < 1e-6f) return;

            axis = Vector3.Normalize(axis);
            Quaternion aimRotation = Quaternion.CreateFromAxisAngle(axis, angle * weight);

            ref var joint = ref targetSkeleton.GetJoint(_aimIK.AimJointIndex);
            joint.LocalRotation = Quaternion.Normalize(aimRotation * joint.LocalRotation);
        }

        public void SetAimIKTarget(Vector3 targetPosition)
        {
            _aimIK.TargetPosition = targetPosition;
        }

        public void SetAimIKParameters(int aimJointIndex, int lookAtJointIndex, float weight, float damping)
        {
            _aimIK.AimJointIndex = aimJointIndex;
            _aimIK.LookAtJointIndex = lookAtJointIndex;
            _aimIK.Weight = weight;
            _aimIK.Damping = damping;
        }

        // ── Blend Tree for Locomotion ────────────────────────────────────

        public sealed class BlendTree
        {
            public BlendTreeNode Root;
            public int NodeCount;
            public BlendTreeBlendMode DefaultMode;

            public BlendTree()
            {
                Root = new BlendTreeNode();
                NodeCount = 0;
                DefaultMode = BlendTreeBlendMode.Simple1D;
            }

            public BlendTreeNode AddChild(AnimationClip? clip, float threshold = 0f)
            {
                if (Root.ChildCount >= MaxBlendTreeNodes)
                    throw new InvalidOperationException("Maximum blend tree node count exceeded.");

                var child = new BlendTreeNode
                {
                    Clip = clip,
                    Threshold = threshold,
                    BlendMode = DefaultMode
                };

                var newChildren = new BlendTreeNode[Root.ChildCount + 1];
                Array.Copy(Root.Children, newChildren, Root.ChildCount);
                newChildren[Root.ChildCount] = child;
                Root.Children = newChildren;
                Root.ChildCount++;
                NodeCount++;

                return child;
            }

            public BlendTreeNode AddChild2D(AnimationClip? clip, Vector2 motionParameter)
            {
                var child = AddChild(clip, 0f);
                child.MotionParameter = motionParameter;
                return child;
            }

            public void Clear()
            {
                Root.Children = Array.Empty<BlendTreeNode>();
                Root.ChildCount = 0;
                NodeCount = 0;
            }
        }

        public void ConfigureBlendTree(BlendTreeBlendMode mode)
        {
            _blendTree.DefaultMode = mode;
            _blendTree.Root.BlendMode = mode;
        }

        public BlendTreeNode AddBlendTreeClip(AnimationClip? clip, float threshold = 0f)
        {
            return _blendTree.AddChild(clip, threshold);
        }

        public BlendTreeNode AddBlendTreeClip2D(AnimationClip? clip, Vector2 motionParameter)
        {
            return _blendTree.AddChild2D(clip, motionParameter);
        }

        public void SampleBlendTree(float parameter, Skeleton targetSkeleton)
        {
            if (targetSkeleton == null || _blendTree.Root.ChildCount == 0) return;

            float[] weights = new float[_blendTree.Root.ChildCount];
            ComputeBlendTreeWeights1D(parameter, weights);
            ApplyBlendTreeWeights(targetSkeleton, weights);
        }

        public void SampleBlendTree2D(Vector2 parameter, Skeleton targetSkeleton)
        {
            if (targetSkeleton == null || _blendTree.Root.ChildCount == 0) return;

            float[] weights = new float[_blendTree.Root.ChildCount];

            switch (_blendTree.DefaultMode)
            {
                case BlendTreeBlendMode.SimpleDirectional2D:
                    ComputeBlendTreeWeightsSimpleDirectional(parameter, weights);
                    break;
                case BlendTreeBlendMode.FreeformDirectional2D:
                    ComputeBlendTreeWeightsFreeformDirectional(parameter, weights);
                    break;
                case BlendTreeBlendMode.FreeformCartesian2D:
                    ComputeBlendTreeWeightsFreeformCartesian(parameter, weights);
                    break;
                default:
                    ComputeBlendTreeWeights1D(parameter.X, weights);
                    break;
            }

            ApplyBlendTreeWeights(targetSkeleton, weights);
        }

        private void ComputeBlendTreeWeights1D(float parameter, float[] weights)
        {
            var children = _blendTree.Root.Children;
            int count = _blendTree.Root.ChildCount;
            if (count == 0) return;

            if (count == 1)
            {
                weights[0] = 1f;
                return;
            }

            float minThreshold = children[0].Threshold;
            float maxThreshold = children[count - 1].Threshold;
            float range = maxThreshold - minThreshold;

            if (range < 1e-6f)
            {
                for (int i = 0; i < count; i++) weights[i] = 1f / count;
                return;
            }

            float normalized = (parameter - minThreshold) / range;
            normalized = Math.Clamp(normalized, 0f, 1f);

            for (int i = 0; i < count - 1; i++)
            {
                float t0 = (children[i].Threshold - minThreshold) / range;
                float t1 = (children[i + 1].Threshold - minThreshold) / range;
                float segmentRange = t1 - t0;

                if (segmentRange < 1e-6f) continue;

                if (normalized >= t0 && normalized <= t1)
                {
                    float localT = (normalized - t0) / segmentRange;
                    weights[i] = 1f - localT;
                    weights[i + 1] = localT;
                    return;
                }
            }

            if (normalized <= (children[0].Threshold - minThreshold) / range)
                weights[0] = 1f;
            else
                weights[count - 1] = 1f;
        }

        private void ComputeBlendTreeWeightsSimpleDirectional(Vector2 parameter, float[] weights)
        {
            var children = _blendTree.Root.Children;
            int count = _blendTree.Root.ChildCount;
            if (count == 0) return;

            float paramLen = parameter.Length();
            if (paramLen < 1e-6f)
            {
                weights[0] = 1f;
                return;
            }

            Vector2 paramDir = parameter / paramLen;
            float totalWeight = 0f;

            for (int i = 0; i < count; i++)
            {
                Vector2 motionDir = children[i].MotionParameter;
                float motionLen = motionDir.Length();

                if (motionLen < 1e-6f)
                {
                    weights[i] = 1f / count;
                }
                else
                {
                    motionDir /= motionLen;
                    float dot = Vector2.Dot(paramDir, motionDir);
                    weights[i] = Math.Max(0f, dot) * paramLen;
                }

                totalWeight += weights[i];
            }

            if (totalWeight > 1e-6f)
            {
                for (int i = 0; i < count; i++)
                    weights[i] /= totalWeight;
            }
        }

        private void ComputeBlendTreeWeightsFreeformDirectional(Vector2 parameter, float[] weights)
        {
            var children = _blendTree.Root.Children;
            int count = _blendTree.Root.ChildCount;
            if (count == 0) return;

            float paramLen = parameter.Length();
            if (paramLen < 1e-6f)
            {
                weights[0] = 1f;
                return;
            }

            float totalWeight = 0f;

            for (int i = 0; i < count; i++)
            {
                Vector2 motion = children[i].MotionParameter;
                float motionLen = motion.Length();

                if (motionLen < 1e-6f)
                {
                    weights[i] = 1f;
                }
                else
                {
                    float dot = Vector2.Dot(parameter, motion) / (paramLen * motionLen);
                    weights[i] = Math.Max(0f, dot) * paramLen;
                }

                totalWeight += weights[i];
            }

            if (totalWeight > 1e-6f)
            {
                for (int i = 0; i < count; i++)
                    weights[i] /= totalWeight;
            }
        }

        private void ComputeBlendTreeWeightsFreeformCartesian(Vector2 parameter, float[] weights)
        {
            var children = _blendTree.Root.Children;
            int count = _blendTree.Root.ChildCount;
            if (count == 0) return;

            float totalWeight = 0f;

            for (int i = 0; i < count; i++)
            {
                float dist = Vector2.Distance(parameter, children[i].MotionParameter);
                weights[i] = 1f / (dist + 0.001f);
                totalWeight += weights[i];
            }

            if (totalWeight > 1e-6f)
            {
                for (int i = 0; i < count; i++)
                    weights[i] /= totalWeight;
            }
        }

        private void ApplyBlendTreeWeights(Skeleton targetSkeleton, float[] weights)
        {
            var children = _blendTree.Root.Children;
            int count = _blendTree.Root.ChildCount;

            bool first = true;

            for (int i = 0; i < count; i++)
            {
                if (weights[i] <= 0.001f || children[i].Clip == null) continue;

                if (first)
                {
                    children[i].Clip.SampleAtTime(0f, targetSkeleton);
                    first = false;
                }
                else
                {
                    ApplyOverrideLayer(targetSkeleton, children[i].Clip, 0f, weights[i]);
                }
            }
        }

        // ── Serialization ────────────────────────────────────────────────

        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(ms);

            writer.Write((uint)0x424C4E44);
            writer.Write((uint)1);
            writer.Write((uint)_layerCount);
            writer.Write((uint)_stateCount);
            writer.Write((uint)_currentStateIndex);
            writer.Write((uint)_crossfadeCount);

            for (int i = 0; i < _layerCount; i++)
            {
                writer.Write(_layers[i].Weight);
                writer.Write(_layers[i].PlaybackSpeed);
                writer.Write(_layers[i].ElapsedTime);
                writer.Write(_layers[i].Enabled);
                writer.Write(_layers[i].Additive);
                writer.Write((uint)_layers[i].Order);
            }

            for (int i = 0; i < _stateCount; i++)
            {
                writer.Write((uint)_states[i].TransitionCount);
                writer.Write(_states[i].Speed);
                writer.Write(_states[i].Weight);
                writer.Write(_states[i].IsDefault);

                for (int t = 0; t < _states[i].TransitionCount; t++)
                {
                    ref StateTransition tr = ref _states[i].Transitions[t];
                    writer.Write((uint)tr.FromState);
                    writer.Write((uint)tr.ToState);
                    writer.Write(tr.Duration);
                    writer.Write((uint)tr.ConditionType);
                    writer.Write(tr.ConditionValue);
                    writer.Write(tr.HasExitTime);
                    writer.Write(tr.ExitTime);
                }
            }

            for (int i = 0; i < _crossfadeCount; i++)
            {
                writer.Write((uint)_crossfades[i].SourceStateIndex);
                writer.Write((uint)_crossfades[i].TargetStateIndex);
                writer.Write(_crossfades[i].Duration);
                writer.Write(_crossfades[i].Elapsed);
                writer.Write(_crossfades[i].IsComplete);
            }

            writer.Write(_aimIK.Enabled);
            writer.Write(_aimIK.Weight);
            writer.Write(_aimIK.Damping);
            writer.Write((uint)_aimIK.AimJointIndex);
            writer.Write((uint)_aimIK.LookAtJointIndex);
            writer.Write(_aimIK.TargetPosition.X);
            writer.Write(_aimIK.TargetPosition.Y);
            writer.Write(_aimIK.TargetPosition.Z);

            writer.Flush();
            return ms.ToArray();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _layers = Array.Empty<BlendLayer>();
                _states = Array.Empty<AnimationState>();
                _crossfades = Array.Empty<CrossfadeInfo>();
                _layerCount = 0;
                _stateCount = 0;
                _crossfadeCount = 0;
                _currentStateIndex = -1;
                _disposed = true;
            }
        }
    }
}
