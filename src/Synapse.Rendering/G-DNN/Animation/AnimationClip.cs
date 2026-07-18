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
// FILE: AnimationClip.cs
// PATH: Animation/AnimationClip.cs
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
    /// Represents a single animation clip containing keyframe channels for position,
    /// rotation, and scale of each joint. Supports sampling at arbitrary times,
    /// keyframe compression, animation events, root motion extraction, and curve simplification.
    /// </summary>
    public sealed class AnimationClip : IDisposable
    {
        public const int MaxKeyframesPerChannel = 4096;
        public const int MaxChannels = 256;
        public const int MaxEvents = 64;

        public string Name;
        public float Duration;
        public float FrameRate;
        public bool Loop;
        public bool ExtractRootMotion;
        public int RootMotionJointIndex;

        private JointAnimationData[] _jointData;
        private int[] _jointIndices;
        private int _channelCount;
        private AnimationEvent[] _events;
        private int _eventCount;
        private bool _disposed;

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct Keyframe
        {
            public float Time;
            public float Value;
            public float InTangent;
            public float OutTangent;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Keyframe(float time, float value, float inTangent = 0f, float outTangent = 0f)
            {
                Time = time;
                Value = value;
                InTangent = inTangent;
                OutTangent = outTangent;
            }

            public const int SizeInBytes = 16;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct Vector3Keyframe
        {
            public float Time;
            public float X;
            public float Y;
            public float Z;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Vector3Keyframe(float time, Vector3 value)
            {
                Time = time;
                X = value.X;
                Y = value.Y;
                Z = value.Z;
            }

            public readonly Vector3 Value => new Vector3(X, Y, Z);
            public const int SizeInBytes = 16;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct QuaternionKeyframe
        {
            public float Time;
            public float X;
            public float Y;
            public float Z;
            public float W;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public QuaternionKeyframe(float time, Quaternion value)
            {
                Time = time;
                X = value.X;
                Y = value.Y;
                Z = value.Z;
                W = value.W;
            }

            public readonly Quaternion Value => new Quaternion(X, Y, Z, W);
            public const int SizeInBytes = 20;
        }

        public sealed class KeyframeChannel
        {
            public float[] Times;
            public float[] Values;
            public float[] InTangents;
            public float[] OutTangents;
            public int KeyframeCount;
            public InterpolationMode Interpolation;

            public KeyframeChannel(int capacity = 64)
            {
                Times = new float[capacity];
                Values = new float[capacity];
                InTangents = new float[capacity];
                OutTangents = new float[capacity];
                KeyframeCount = 0;
                Interpolation = InterpolationMode.Linear;
            }
        }

        public sealed class PositionChannel
        {
            public KeyframeChannel X;
            public KeyframeChannel Y;
            public KeyframeChannel Z;

            public PositionChannel(int capacity = 64)
            {
                X = new KeyframeChannel(capacity);
                Y = new KeyframeChannel(capacity);
                Z = new KeyframeChannel(capacity);
            }

            public int KeyframeCount => Math.Max(X.KeyframeCount, Math.Max(Y.KeyframeCount, Z.KeyframeCount));
        }

        public sealed class RotationChannel
        {
            public KeyframeChannel X;
            public KeyframeChannel Y;
            public KeyframeChannel Z;
            public KeyframeChannel W;

            public RotationChannel(int capacity = 64)
            {
                X = new KeyframeChannel(capacity);
                Y = new KeyframeChannel(capacity);
                Z = new KeyframeChannel(capacity);
                W = new KeyframeChannel(capacity);
            }

            public int KeyframeCount => Math.Max(
                Math.Max(X.KeyframeCount, Y.KeyframeCount),
                Math.Max(Z.KeyframeCount, W.KeyframeCount));
        }

        public sealed class ScaleChannel
        {
            public KeyframeChannel X;
            public KeyframeChannel Y;
            public KeyframeChannel Z;

            public ScaleChannel(int capacity = 64)
            {
                X = new KeyframeChannel(capacity);
                Y = new KeyframeChannel(capacity);
                Z = new KeyframeChannel(capacity);
            }

            public int KeyframeCount => Math.Max(X.KeyframeCount, Math.Max(Y.KeyframeCount, Z.KeyframeCount));
        }

        public sealed class JointAnimationData
        {
            public PositionChannel Position;
            public RotationChannel Rotation;
            public ScaleChannel Scale;
            public bool HasPosition;
            public bool HasRotation;
            public bool HasScale;

            public JointAnimationData()
            {
                Position = new PositionChannel();
                Rotation = new RotationChannel();
                Scale = new ScaleChannel();
                HasPosition = false;
                HasRotation = false;
                HasScale = false;
            }
        }

        public enum InterpolationMode : byte
        {
            Constant = 0,
            Linear = 1,
            Cubic = 2,
            CatmullRom = 3
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct AnimationEvent
        {
            public float Time;
            public int EventType;
            public int IntParameter;
            public float FloatParameter;
            public int StringOffset;
            internal byte HasFired;
            public const int SizeInBytes = 20;
        }

        public AnimationClip(string name, float duration, float frameRate = 30f)
        {
            Name = name ?? "Unnamed";
            Duration = Math.Max(0f, duration);
            FrameRate = Math.Max(1f, frameRate);
            Loop = false;
            ExtractRootMotion = false;
            RootMotionJointIndex = 0;
            _jointData = new JointAnimationData[MaxChannels];
            _jointIndices = new int[MaxChannels];
            for (int i = 0; i < MaxChannels; i++)
            {
                _jointIndices[i] = -1;
            }
            _channelCount = 0;
            _events = new AnimationEvent[MaxEvents];
            _eventCount = 0;
            _disposed = false;
        }

        public int ChannelCount => _channelCount;
        public int EventCount => _eventCount;

        public JointAnimationData GetOrCreateJointData(int jointIndex)
        {
            if ((uint)jointIndex >= (uint)MaxChannels)
                throw new ArgumentOutOfRangeException(nameof(jointIndex));

            int channelIdx = FindChannelIndex(jointIndex);
            if (channelIdx >= 0)
                return _jointData[channelIdx];

            if (_channelCount >= MaxChannels)
                throw new InvalidOperationException("Maximum channel count exceeded.");

            channelIdx = _channelCount++;
            _jointData[channelIdx] = new JointAnimationData();
            _jointIndices[channelIdx] = jointIndex;
            return _jointData[channelIdx];
        }

        public JointAnimationData? GetJointData(int jointIndex)
        {
            int channelIdx = FindChannelIndex(jointIndex);
            return channelIdx >= 0 ? _jointData[channelIdx] : null;
        }

        public bool TryGetJointData(int jointIndex, out JointAnimationData data)
        {
            data = GetJointData(jointIndex)!;
            return data != null;
        }

        public bool TryGetChannelData(int jointIndex, out Vector3 position, out Quaternion rotation, out Vector3 scale)
        {
            JointAnimationData? data = GetJointData(jointIndex);
            if (data == null)
            {
                position = Vector3.Zero;
                rotation = Quaternion.Identity;
                scale = Vector3.One;
                return false;
            }

            float time = 0f;
            position = SamplePosition(data, time);
            rotation = SampleRotation(data, time);
            scale = SampleScale(data, time);
            return data.HasPosition || data.HasRotation || data.HasScale;
        }

        public int FindChannelIndex(int jointIndex)
        {
            for (int i = 0; i < _channelCount; i++)
            {
                if (_jointIndices[i] == jointIndex)
                    return i;
            }
            return -1;
        }

        // ── Keyframe Adding ──────────────────────────────────────────────

        public void AddPositionKeyframe(int jointIndex, float time, Vector3 value)
        {
            JointAnimationData data = GetOrCreateJointData(jointIndex);
            AddScalarKeyframe(data.Position.X, time, value.X);
            AddScalarKeyframe(data.Position.Y, time, value.Y);
            AddScalarKeyframe(data.Position.Z, time, value.Z);
            data.HasPosition = true;
        }

        public void AddRotationKeyframe(int jointIndex, float time, Quaternion value)
        {
            JointAnimationData data = GetOrCreateJointData(jointIndex);
            AddScalarKeyframe(data.Rotation.X, time, value.X);
            AddScalarKeyframe(data.Rotation.Y, time, value.Y);
            AddScalarKeyframe(data.Rotation.Z, time, value.Z);
            AddScalarKeyframe(data.Rotation.W, time, value.W);
            data.HasRotation = true;
        }

        public void AddScaleKeyframe(int jointIndex, float time, Vector3 value)
        {
            JointAnimationData data = GetOrCreateJointData(jointIndex);
            AddScalarKeyframe(data.Scale.X, time, value.X);
            AddScalarKeyframe(data.Scale.Y, time, value.Y);
            AddScalarKeyframe(data.Scale.Z, time, value.Z);
            data.HasScale = true;
        }

        public void SetPositionKeyframes(int jointIndex, ReadOnlySpan<Vector3Keyframe> keyframes)
        {
            JointAnimationData data = GetOrCreateJointData(jointIndex);
            int count = Math.Min(keyframes.Length, MaxKeyframesPerChannel);

            SetChannelKeyframes(data.Position.X, keyframes, k => k.X);
            SetChannelKeyframes(data.Position.Y, keyframes, k => k.Y);
            SetChannelKeyframes(data.Position.Z, keyframes, k => k.Z);
            data.HasPosition = count > 0;
        }

        public void SetRotationKeyframes(int jointIndex, ReadOnlySpan<QuaternionKeyframe> keyframes)
        {
            JointAnimationData data = GetOrCreateJointData(jointIndex);
            int count = Math.Min(keyframes.Length, MaxKeyframesPerChannel);

            SetChannelKeyframes(data.Rotation.X, keyframes, k => k.X);
            SetChannelKeyframes(data.Rotation.Y, keyframes, k => k.Y);
            SetChannelKeyframes(data.Rotation.Z, keyframes, k => k.Z);
            SetChannelKeyframes(data.Rotation.W, keyframes, k => k.W);
            data.HasRotation = count > 0;
        }

        private static void AddScalarKeyframe(KeyframeChannel channel, float time, float value)
        {
            if (channel.KeyframeCount >= MaxKeyframesPerChannel)
                return;

            int idx = channel.KeyframeCount;
            channel.Times[idx] = time;
            channel.Values[idx] = value;
            channel.InTangents[idx] = 0f;
            channel.OutTangents[idx] = 0f;
            channel.KeyframeCount++;

            SortChannelByKeyframes(channel);
        }

        private static void SetChannelKeyframes<T>(KeyframeChannel channel, ReadOnlySpan<T> keyframes, Func<T, float> selector)
        {
            int count = Math.Min(keyframes.Length, MaxKeyframesPerChannel);
            for (int i = 0; i < count; i++)
            {
                channel.Times[i] = GetTime(keyframes[i]);
                channel.Values[i] = selector(keyframes[i]);
                channel.InTangents[i] = 0f;
                channel.OutTangents[i] = 0f;
            }
            channel.KeyframeCount = count;
            ComputeTangents(channel);
        }

        private static float GetTime<T>(T keyframe)
        {
            if (keyframe is Vector3Keyframe v3) return v3.Time;
            if (keyframe is QuaternionKeyframe q) return q.Time;
            if (keyframe is Keyframe k) return k.Time;
            return 0f;
        }

        private static void SortChannelByKeyframes(KeyframeChannel channel)
        {
            for (int i = 1; i < channel.KeyframeCount; i++)
            {
                float t = channel.Times[i];
                float v = channel.Values[i];
                float tin = channel.InTangents[i];
                float tout = channel.OutTangents[i];
                int j = i - 1;

                while (j >= 0 && channel.Times[j] > t)
                {
                    channel.Times[j + 1] = channel.Times[j];
                    channel.Values[j + 1] = channel.Values[j];
                    channel.InTangents[j + 1] = channel.InTangents[j];
                    channel.OutTangents[j + 1] = channel.OutTangents[j];
                    j--;
                }

                channel.Times[j + 1] = t;
                channel.Values[j + 1] = v;
                channel.InTangents[j + 1] = tin;
                channel.OutTangents[j + 1] = tout;
            }
        }

        // ── Sampling ─────────────────────────────────────────────────────

        public void SampleAtTime(float time, Skeleton skeleton)
        {
            if (skeleton == null) throw new ArgumentNullException(nameof(skeleton));

            time = ClampTime(time);

            for (int ch = 0; ch < _channelCount; ch++)
            {
                int jointIdx = _jointIndices[ch];
                if (jointIdx < 0 || jointIdx >= skeleton.JointCount) continue;

                JointAnimationData data = _jointData[ch];
                Vector3 pos = data.HasPosition ? SamplePosition(data, time) : skeleton.GetJoint(jointIdx).LocalTranslation;
                Quaternion rot = data.HasRotation ? SampleRotation(data, time) : skeleton.GetJoint(jointIdx).LocalRotation;
                Vector3 scl = data.HasScale ? SampleScale(data, time) : skeleton.GetJoint(jointIdx).LocalScale;

                skeleton.SetLocalTransform(jointIdx, pos, rot, scl);
            }
        }

        public Vector3 SamplePosition(JointAnimationData data, float time)
        {
            float x = SampleChannel(data.Position.X, time);
            float y = SampleChannel(data.Position.Y, time);
            float z = SampleChannel(data.Position.Z, time);
            return new Vector3(x, y, z);
        }

        public Quaternion SampleRotation(JointAnimationData data, float time)
        {
            float x = SampleChannel(data.Rotation.X, time);
            float y = SampleChannel(data.Rotation.Y, time);
            float z = SampleChannel(data.Rotation.Z, time);
            float w = SampleChannel(data.Rotation.W, time);

            Quaternion q = Quaternion.Normalize(new Quaternion(x, y, z, w));
            return q;
        }

        public Vector3 SampleScale(JointAnimationData data, float time)
        {
            float x = SampleChannel(data.Scale.X, time);
            float y = SampleChannel(data.Scale.Y, time);
            float z = SampleChannel(data.Scale.Z, time);
            return new Vector3(x, y, z);
        }

        public float SampleChannel(KeyframeChannel channel, float time)
        {
            if (channel.KeyframeCount == 0)
                return 0f;

            if (channel.KeyframeCount == 1)
                return channel.Values[0];

            if (time <= channel.Times[0])
                return channel.Values[0];

            if (time >= channel.Times[channel.KeyframeCount - 1])
                return channel.Values[channel.KeyframeCount - 1];

            int idx = FindKeyframeIndex(channel, time);
            if (idx < 0) return channel.Values[0];
            if (idx >= channel.KeyframeCount - 1) return channel.Values[channel.KeyframeCount - 1];

            float t0 = channel.Times[idx];
            float t1 = channel.Times[idx + 1];
            float dt = t1 - t0;

            if (dt < 1e-6f)
                return channel.Values[idx];

            float t = (time - t0) / dt;

            return channel.Interpolation switch
            {
                InterpolationMode.Constant => channel.Values[idx],
                InterpolationMode.Linear => Lerp(channel.Values[idx], channel.Values[idx + 1], t),
                InterpolationMode.Cubic => HermiteInterpolate(
                    channel.Values[idx], channel.Values[idx + 1],
                    channel.OutTangents[idx] * dt, channel.InTangents[idx + 1] * dt, t),
                InterpolationMode.CatmullRom => channel.Values[idx],
                _ => Lerp(channel.Values[idx], channel.Values[idx + 1], t)
            };
        }

        private static int FindKeyframeIndex(KeyframeChannel channel, float time)
        {
            int lo = 0;
            int hi = channel.KeyframeCount - 1;

            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                if (channel.Times[mid] < time)
                    lo = mid + 1;
                else if (channel.Times[mid] > time)
                    hi = mid - 1;
                else
                    return mid;
            }

            return Math.Max(0, hi);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Lerp(float a, float b, float t) => a + (b - a) * t;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HermiteInterpolate(float p0, float p1, float m0, float m1, float t)
        {
            float t2 = t * t;
            float t3 = t2 * t;

            float h00 = 2f * t3 - 3f * t2 + 1f;
            float h10 = t3 - 2f * t2 + t;
            float h01 = -2f * t3 + 3f * t2;
            float h11 = t3 - t2;

            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ClampTime(float time)
        {
            if (Duration <= 0f) return 0f;
            if (Loop)
            {
                time = time % Duration;
                if (time < 0f) time += Duration;
            }
            else
            {
                time = Math.Clamp(time, 0f, Duration);
            }
            return time;
        }

        // ── Tangent Computation ──────────────────────────────────────────

        public static void ComputeTangents(KeyframeChannel channel)
        {
            if (channel.KeyframeCount < 2) return;

            for (int i = 0; i < channel.KeyframeCount; i++)
            {
                float tangent = 0f;

                if (i == 0)
                {
                    if (channel.KeyframeCount > 1)
                    {
                        float dt = channel.Times[1] - channel.Times[0];
                        tangent = dt > 1e-6f ? (channel.Values[1] - channel.Values[0]) / dt : 0f;
                    }
                }
                else if (i == channel.KeyframeCount - 1)
                {
                    float dt = channel.Times[i] - channel.Times[i - 1];
                    tangent = dt > 1e-6f ? (channel.Values[i] - channel.Values[i - 1]) / dt : 0f;
                }
                else
                {
                    float dt = channel.Times[i + 1] - channel.Times[i - 1];
                    tangent = dt > 1e-6f ? (channel.Values[i + 1] - channel.Values[i - 1]) / dt : 0f;
                }

                channel.OutTangents[i] = tangent;
                channel.InTangents[i] = tangent;
            }
        }

        public void ComputeAllTangents()
        {
            for (int ch = 0; ch < _channelCount; ch++)
            {
                JointAnimationData data = _jointData[ch];
                if (data.HasPosition)
                {
                    ComputeTangents(data.Position.X);
                    ComputeTangents(data.Position.Y);
                    ComputeTangents(data.Position.Z);
                }
                if (data.HasRotation)
                {
                    ComputeTangents(data.Rotation.X);
                    ComputeTangents(data.Rotation.Y);
                    ComputeTangents(data.Rotation.Z);
                    ComputeTangents(data.Rotation.W);
                }
                if (data.HasScale)
                {
                    ComputeTangents(data.Scale.X);
                    ComputeTangents(data.Scale.Y);
                    ComputeTangents(data.Scale.Z);
                }
            }
        }

        // ── Keyframe Compression ─────────────────────────────────────────

        public void CompressUniformSampling(float samplingRate)
        {
            if (samplingRate <= 0f || Duration <= 0f) return;

            int sampleCount = (int)(Duration * samplingRate) + 1;

            for (int ch = 0; ch < _channelCount; ch++)
            {
                JointAnimationData data = _jointData[ch];

                if (data.HasPosition)
                {
                    ResampleChannel(data.Position.X, sampleCount);
                    ResampleChannel(data.Position.Y, sampleCount);
                    ResampleChannel(data.Position.Z, sampleCount);
                }
                if (data.HasRotation)
                {
                    ResampleChannel(data.Rotation.X, sampleCount);
                    ResampleChannel(data.Rotation.Y, sampleCount);
                    ResampleChannel(data.Rotation.Z, sampleCount);
                    ResampleChannel(data.Rotation.W, sampleCount);
                }
                if (data.HasScale)
                {
                    ResampleChannel(data.Scale.X, sampleCount);
                    ResampleChannel(data.Scale.Y, sampleCount);
                    ResampleChannel(data.Scale.Z, sampleCount);
                }
            }
        }

        private void ResampleChannel(KeyframeChannel channel, int targetCount)
        {
            if (channel.KeyframeCount == 0) return;
            targetCount = Math.Min(targetCount, MaxKeyframesPerChannel);

            float[] newTimes = new float[targetCount];
            float[] newValues = new float[targetCount];
            float[] newIn = new float[targetCount];
            float[] newOut = new float[targetCount];

            float step = Duration / (targetCount - 1);

            for (int i = 0; i < targetCount; i++)
            {
                float t = i * step;
                newTimes[i] = t;
                newValues[i] = SampleChannel(channel, t);
                newIn[i] = 0f;
                newOut[i] = 0f;
            }

            channel.Times = newTimes;
            channel.Values = newValues;
            channel.InTangents = newIn;
            channel.OutTangents = newOut;
            channel.KeyframeCount = targetCount;
            ComputeTangents(channel);
        }

        public void SimplifyCurves(float tolerance)
        {
            for (int ch = 0; ch < _channelCount; ch++)
            {
                JointAnimationData data = _jointData[ch];
                if (data.HasPosition)
                {
                    data.Position.X = SimplifyChannel(data.Position.X, tolerance);
                    data.Position.Y = SimplifyChannel(data.Position.Y, tolerance);
                    data.Position.Z = SimplifyChannel(data.Position.Z, tolerance);
                }
                if (data.HasRotation)
                {
                    data.Rotation.X = SimplifyChannel(data.Rotation.X, tolerance);
                    data.Rotation.Y = SimplifyChannel(data.Rotation.Y, tolerance);
                    data.Rotation.Z = SimplifyChannel(data.Rotation.Z, tolerance);
                    data.Rotation.W = SimplifyChannel(data.Rotation.W, tolerance);
                }
                if (data.HasScale)
                {
                    data.Scale.X = SimplifyChannel(data.Scale.X, tolerance);
                    data.Scale.Y = SimplifyChannel(data.Scale.Y, tolerance);
                    data.Scale.Z = SimplifyChannel(data.Scale.Z, tolerance);
                }
            }
        }

        private static KeyframeChannel SimplifyChannel(KeyframeChannel channel, float tolerance)
        {
            if (channel.KeyframeCount <= 2) return channel;

            bool[] keep = new bool[channel.KeyframeCount];
            keep[0] = true;
            keep[channel.KeyframeCount - 1] = true;

            SimplifyRecursive(channel, 0, channel.KeyframeCount - 1, tolerance, keep);

            int newCount = 0;
            for (int i = 0; i < channel.KeyframeCount; i++)
            {
                if (keep[i]) newCount++;
            }

            if (newCount >= channel.KeyframeCount) return channel;

            KeyframeChannel result = new KeyframeChannel(newCount);
            result.Interpolation = channel.Interpolation;
            int idx = 0;
            for (int i = 0; i < channel.KeyframeCount; i++)
            {
                if (keep[i])
                {
                    result.Times[idx] = channel.Times[i];
                    result.Values[idx] = channel.Values[i];
                    result.InTangents[idx] = channel.InTangents[i];
                    result.OutTangents[idx] = channel.OutTangents[i];
                    idx++;
                }
            }
            result.KeyframeCount = newCount;
            ComputeTangents(result);
            return result;
        }

        private static void SimplifyRecursive(KeyframeChannel channel, int start, int end, float tolerance, bool[] keep)
        {
            if (end - start < 2) return;

            float maxError = 0f;
            int maxIndex = start;

            float t0 = channel.Times[start];
            float t1 = channel.Times[end];
            float v0 = channel.Values[start];
            float v1 = channel.Values[end];
            float dt = t1 - t0;

            for (int i = start + 1; i < end; i++)
            {
                float t = dt > 1e-6f ? (channel.Times[i] - t0) / dt : 0f;
                float expected = Lerp(v0, v1, t);
                float error = Math.Abs(channel.Values[i] - expected);

                if (error > maxError)
                {
                    maxError = error;
                    maxIndex = i;
                }
            }

            if (maxError > tolerance)
            {
                keep[maxIndex] = true;
                SimplifyRecursive(channel, start, maxIndex, tolerance, keep);
                SimplifyRecursive(channel, maxIndex, end, tolerance, keep);
            }
        }

        // ── Animation Events ─────────────────────────────────────────────

        public void AddEvent(float time, int eventType, int intParam = 0, float floatParam = 0f, int stringOffset = 0)
        {
            if (_eventCount >= MaxEvents)
                throw new InvalidOperationException("Maximum event count exceeded.");

            int idx = _eventCount++;
            _events[idx] = new AnimationEvent
            {
                Time = time,
                EventType = eventType,
                IntParameter = intParam,
                FloatParameter = floatParam,
                StringOffset = stringOffset,
                HasFired = 0
            };

            for (int i = _eventCount - 1; i > 0; i--)
            {
                if (_events[i].Time < _events[i - 1].Time)
                {
                    (_events[i], _events[i - 1]) = (_events[i - 1], _events[i]);
                }
                else break;
            }
        }

        public void ClearEvents()
        {
            _eventCount = 0;
        }

        public ReadOnlySpan<AnimationEvent> GetEvents()
        {
            return new ReadOnlySpan<AnimationEvent>(_events, 0, _eventCount);
        }

        public void ResetEventFiring()
        {
            for (int i = 0; i < _eventCount; i++)
            {
                _events[i].HasFired = 0;
            }
        }

        public int CheckEvents(float previousTime, float currentTime, Span<AnimationEvent> triggeredEvents)
        {
            int count = 0;
            for (int i = 0; i < _eventCount && count < triggeredEvents.Length; i++)
            {
                if (_events[i].HasFired != 0) continue;

                bool wasBefore = previousTime < _events[i].Time;
                bool isAfter = currentTime >= _events[i].Time;

                if (Loop)
                {
                    wasBefore = previousTime <= currentTime
                        ? (previousTime < _events[i].Time && currentTime >= _events[i].Time)
                        : true;
                    isAfter = true;
                }

                if (wasBefore && isAfter)
                {
                    _events[i].HasFired = 1;
                    triggeredEvents[count++] = _events[i];
                }
            }
            return count;
        }

        // ── Root Motion Extraction ───────────────────────────────────────

        public Vector3 ExtractRootMotionDelta(float previousTime, float currentTime)
        {
            if (!ExtractRootMotion) return Vector3.Zero;

            JointAnimationData? data = GetJointData(RootMotionJointIndex);
            if (data == null || !data.HasPosition) return Vector3.Zero;

            Vector3 previousPos = SamplePosition(data, ClampTime(previousTime));
            Vector3 currentPos = SamplePosition(data, ClampTime(currentTime));

            return currentPos - previousPos;
        }

        public Quaternion ExtractRootRotationDelta(float previousTime, float currentTime)
        {
            if (!ExtractRootMotion) return Quaternion.Identity;

            JointAnimationData? data = GetJointData(RootMotionJointIndex);
            if (data == null || !data.HasRotation) return Quaternion.Identity;

            Quaternion previousRot = SampleRotation(data, ClampTime(previousTime));
            Quaternion currentRot = SampleRotation(data, ClampTime(currentTime));

            Quaternion delta = currentRot * Quaternion.Conjugate(previousRot);
            return Quaternion.Normalize(delta);
        }

        public void RemoveRootMotionFromData()
        {
            if (!ExtractRootMotion) return;

            JointAnimationData? data = GetJointData(RootMotionJointIndex);
            if (data == null) return;

            data.HasPosition = false;
            data.Position.X.KeyframeCount = 0;
            data.Position.Y.KeyframeCount = 0;
            data.Position.Z.KeyframeCount = 0;
        }

        // ── Animation Layering and Blending ──────────────────────────────

        public void BlendWith(AnimationClip other, float weight, float time)
        {
            if (other == null) throw new ArgumentNullException(nameof(other));
            weight = Math.Clamp(weight, 0f, 1f);

            for (int ch = 0; ch < _channelCount; ch++)
            {
                int jointIdx = _jointIndices[ch];
                int otherCh = other.FindChannelIndex(jointIdx);
                if (otherCh < 0) continue;

                JointAnimationData thisData = _jointData[ch];
                JointAnimationData otherData = other._jointData[otherCh];

                if (thisData.HasPosition && otherData.HasPosition)
                {
                    BlendChannelVector(thisData.Position, otherData.Position, weight, time);
                }
                if (thisData.HasRotation && otherData.HasRotation)
                {
                    BlendChannelQuaternion(thisData.Rotation, otherData.Rotation, weight, time);
                }
                if (thisData.HasScale && otherData.HasScale)
                {
                    BlendChannelVector(thisData.Scale, otherData.Scale, weight, time);
                }
            }
        }

        private void BlendChannelVector(PositionChannel target, PositionChannel source, float weight, float time)
        {
            float tx = SampleChannel(target.X, time);
            float ty = SampleChannel(target.Y, time);
            float tz = SampleChannel(target.Z, time);

            float sx = SampleChannel(source.X, time);
            float sy = SampleChannel(source.Y, time);
            float sz = SampleChannel(source.Z, time);

            float rx = Lerp(tx, sx, weight);
            float ry = Lerp(ty, sy, weight);
            float rz = Lerp(tz, sz, weight);

            target.X.Values[0] = rx;
            target.Y.Values[0] = ry;
            target.Z.Values[0] = rz;
            target.X.KeyframeCount = 1;
            target.Y.KeyframeCount = 1;
            target.Z.KeyframeCount = 1;
            target.X.Times[0] = 0f;
            target.Y.Times[0] = 0f;
            target.Z.Times[0] = 0f;
        }

        private void BlendChannelVector(ScaleChannel target, ScaleChannel source, float weight, float time)
        {
            float tx = SampleChannel(target.X, time);
            float ty = SampleChannel(target.Y, time);
            float tz = SampleChannel(target.Z, time);

            float sx = SampleChannel(source.X, time);
            float sy = SampleChannel(source.Y, time);
            float sz = SampleChannel(source.Z, time);

            target.X.Values[0] = Lerp(tx, sx, weight);
            target.Y.Values[0] = Lerp(ty, sy, weight);
            target.Z.Values[0] = Lerp(tz, sz, weight);
            target.X.KeyframeCount = 1;
            target.Y.KeyframeCount = 1;
            target.Z.KeyframeCount = 1;
            target.X.Times[0] = 0f;
            target.Y.Times[0] = 0f;
            target.Z.Times[0] = 0f;
        }

        private void BlendChannelQuaternion(RotationChannel target, RotationChannel source, float weight, float time)
        {
            float tx = SampleChannel(target.X, time);
            float ty = SampleChannel(target.Y, time);
            float tz = SampleChannel(target.Z, time);
            float tw = SampleChannel(target.W, time);

            float sx = SampleChannel(source.X, time);
            float sy = SampleChannel(source.Y, time);
            float sz = SampleChannel(source.Z, time);
            float sw = SampleChannel(source.W, time);

            Quaternion t = Quaternion.Normalize(new Quaternion(tx, ty, tz, tw));
            Quaternion s = Quaternion.Normalize(new Quaternion(sx, sy, sz, sw));
            Quaternion blended = Quaternion.Slerp(t, s, weight);

            target.X.Values[0] = blended.X;
            target.Y.Values[0] = blended.Y;
            target.Z.Values[0] = blended.Z;
            target.W.Values[0] = blended.W;
            target.X.KeyframeCount = 1;
            target.Y.KeyframeCount = 1;
            target.Z.KeyframeCount = 1;
            target.W.KeyframeCount = 1;
            target.X.Times[0] = 0f;
            target.Y.Times[0] = 0f;
            target.Z.Times[0] = 0f;
            target.W.Times[0] = 0f;
        }

        // ── Total Keyframe Count ─────────────────────────────────────────

        public int GetTotalKeyframeCount()
        {
            int total = 0;
            for (int ch = 0; ch < _channelCount; ch++)
            {
                JointAnimationData data = _jointData[ch];
                if (data.HasPosition)
                {
                    total += data.Position.X.KeyframeCount + data.Position.Y.KeyframeCount + data.Position.Z.KeyframeCount;
                }
                if (data.HasRotation)
                {
                    total += data.Rotation.X.KeyframeCount + data.Rotation.Y.KeyframeCount
                           + data.Rotation.Z.KeyframeCount + data.Rotation.W.KeyframeCount;
                }
                if (data.HasScale)
                {
                    total += data.Scale.X.KeyframeCount + data.Scale.Y.KeyframeCount + data.Scale.Z.KeyframeCount;
                }
            }
            return total;
        }

        public int GetMemoryUsage()
        {
            int total = GetTotalKeyframeCount();
            return total * Keyframe.SizeInBytes + _channelCount * 64 + _eventCount * AnimationEvent.SizeInBytes;
        }

        // ── Serialization ────────────────────────────────────────────────

        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

            writer.Write((uint)0x41434C50);
            writer.Write((uint)2);
            writer.Write(Duration);
            writer.Write(FrameRate);
            writer.Write(Loop);
            writer.Write(ExtractRootMotion);
            writer.Write(RootMotionJointIndex);
            writer.Write((uint)_channelCount);
            writer.Write((uint)_eventCount);

            for (int ch = 0; ch < _channelCount; ch++)
            {
                writer.Write((uint)_jointIndices[ch]);
                JointAnimationData data = _jointData[ch];
                writer.Write(data.HasPosition);
                writer.Write(data.HasRotation);
                writer.Write(data.HasScale);

                if (data.HasPosition) WritePositionChannel(writer, data.Position);
                if (data.HasRotation) WriteRotationChannel(writer, data.Rotation);
                if (data.HasScale) WriteScaleChannel(writer, data.Scale);
            }

            for (int i = 0; i < _eventCount; i++)
            {
                writer.Write(_events[i].Time);
                writer.Write(_events[i].EventType);
                writer.Write(_events[i].IntParameter);
                writer.Write(_events[i].FloatParameter);
                writer.Write(_events[i].StringOffset);
            }

            writer.Flush();
            return ms.ToArray();
        }

        public static AnimationClip Deserialize(ReadOnlySpan<byte> data)
        {
            if (data.Length < 32)
                throw new ArgumentException("Data too short.");

            int offset = 0;
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            if (magic != 0x41434C50)
                throw new InvalidDataException("Invalid animation clip magic.");

            uint version = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            float duration = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
            float frameRate = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
            bool loop = data[offset] != 0; offset += 1;
            bool extractRoot = data[offset] != 0; offset += 1;
            int rootJoint = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
            uint channelCount = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            uint eventCount = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;

            var clip = new AnimationClip("Deserialized", duration, frameRate)
            {
                Loop = loop,
                ExtractRootMotion = extractRoot,
                RootMotionJointIndex = rootJoint
            };

            for (int ch = 0; ch < (int)channelCount; ch++)
            {
                int jointIdx = (int)BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
                bool hasPos = data[offset] != 0; offset += 1;
                bool hasRot = data[offset] != 0; offset += 1;
                bool hasScl = data[offset] != 0; offset += 1;

                var jointData = clip.GetOrCreateJointData(jointIdx);
                jointData.HasPosition = hasPos;
                jointData.HasRotation = hasRot;
                jointData.HasScale = hasScl;

                if (hasPos) { ReadPositionChannel(data, ref offset, jointData.Position); }
                if (hasRot) { ReadRotationChannel(data, ref offset, jointData.Rotation); }
                if (hasScl) { ReadScaleChannel(data, ref offset, jointData.Scale); }
            }

            for (int i = 0; i < (int)eventCount; i++)
            {
                float time = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
                int eventType = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                int intParam = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;
                float floatParam = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
                int strOffset = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)); offset += 4;

                clip.AddEvent(time, eventType, intParam, floatParam, strOffset);
            }

            return clip;
        }

        private static void WritePositionChannel(System.IO.BinaryWriter writer, PositionChannel ch)
        {
            WriteScalarChannel(writer, ch.X);
            WriteScalarChannel(writer, ch.Y);
            WriteScalarChannel(writer, ch.Z);
        }

        private static void WriteRotationChannel(System.IO.BinaryWriter writer, RotationChannel ch)
        {
            WriteScalarChannel(writer, ch.X);
            WriteScalarChannel(writer, ch.Y);
            WriteScalarChannel(writer, ch.Z);
            WriteScalarChannel(writer, ch.W);
        }

        private static void WriteScaleChannel(System.IO.BinaryWriter writer, ScaleChannel ch)
        {
            WriteScalarChannel(writer, ch.X);
            WriteScalarChannel(writer, ch.Y);
            WriteScalarChannel(writer, ch.Z);
        }

        private static void WriteScalarChannel(System.IO.BinaryWriter writer, KeyframeChannel ch)
        {
            writer.Write((uint)ch.KeyframeCount);
            writer.Write((byte)ch.Interpolation);
            for (int i = 0; i < ch.KeyframeCount; i++)
            {
                writer.Write(ch.Times[i]);
                writer.Write(ch.Values[i]);
                writer.Write(ch.InTangents[i]);
                writer.Write(ch.OutTangents[i]);
            }
        }

        private static void ReadPositionChannel(ReadOnlySpan<byte> data, ref int offset, PositionChannel ch)
        {
            ReadScalarChannel(data, ref offset, ch.X);
            ReadScalarChannel(data, ref offset, ch.Y);
            ReadScalarChannel(data, ref offset, ch.Z);
        }

        private static void ReadRotationChannel(ReadOnlySpan<byte> data, ref int offset, RotationChannel ch)
        {
            ReadScalarChannel(data, ref offset, ch.X);
            ReadScalarChannel(data, ref offset, ch.Y);
            ReadScalarChannel(data, ref offset, ch.Z);
            ReadScalarChannel(data, ref offset, ch.W);
        }

        private static void ReadScaleChannel(ReadOnlySpan<byte> data, ref int offset, ScaleChannel ch)
        {
            ReadScalarChannel(data, ref offset, ch.X);
            ReadScalarChannel(data, ref offset, ch.Y);
            ReadScalarChannel(data, ref offset, ch.Z);
        }

        private static void ReadScalarChannel(ReadOnlySpan<byte> data, ref int offset, KeyframeChannel ch)
        {
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            ch.Interpolation = (InterpolationMode)data[offset]; offset += 1;
            ch.KeyframeCount = (int)count;

            for (int i = 0; i < (int)count; i++)
            {
                ch.Times[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
                ch.Values[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
                ch.InTangents[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
                ch.OutTangents[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset))); offset += 4;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _jointData = Array.Empty<JointAnimationData>();
                _jointIndices = Array.Empty<int>();
                _events = Array.Empty<AnimationEvent>();
                _channelCount = 0;
                _eventCount = 0;
                _disposed = true;
            }
        }
    }
}
