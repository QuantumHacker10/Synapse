// =============================================================================
// MotionMatching.cs - GDNN Engine: Motion Matching Animation System
// Database-driven procedural animation with feature matching and blending
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

using GDNN.Rendering.Compat;

namespace GDNN.Rendering.Animation
{
    public enum MatchCostFunction { Euclidean = 0, Manhattan = 1, Cosine = 2 }
    public enum BlendMode { Linear = 0, Cubic = 1, Bezier = 2 }

    public record MotionMatchingConfig
    {
        public int MaxDatabaseFrames { get; init; } = 10000;
        public int SearchWindow { get; init; } = 30;
        public int PredictionHorizon { get; init; } = 10;
        public int FeatureCount { get; init; } = 12;
        public float BlendTime { get; init; } = 0.25f;
        public MatchCostFunction CostFunction { get; init; } = MatchCostFunction.Euclidean;
        public BlendMode BlendMode { get; init; } = BlendMode.Cubic;
        public bool EnableFootIK { get; init; } = true;
        public bool EnableTrajectoryPrediction { get; init; } = true;
        public int TrajectoryPoints { get; init; } = 8;
        public float TrajectoryInterval { get; init; } = 0.2f;
    }

    public struct MotionFrame
    {
        public Vector3 RootPosition;
        public Quaternion RootRotation;
        public Vector3 RootVelocity;
        public Vector3 RootAcceleration;
        public Vector3[] JointPositions;
        public Quaternion[] JointRotations;
        public Vector3[] JointVelocities;
        public float[] JointAngles;
        public float Time;
        public int PoseIndex;
    }

    public struct MotionFeature
    {
        public Vector3 TrajectoryPosition;
        public Vector3 TrajectoryDirection;
        public Vector3 HipPosition;
        public Vector3 LeftFootPosition;
        public Vector3 RightFootPosition;
        public Vector3 LeftHandPosition;
        public Vector3 RightHandPosition;
        public float LeftFootPhase;
        public float RightFootPhase;
        public Vector3 Velocity;
        public float Speed;
        public float Direction;
    }

    public struct MotionPose
    {
        public Quaternion[] JointRotations;
        public Vector3[] JointPositions;
        public float Time;
    }

    public struct TrajectoryPoint
    {
        public Vector3 Position;
        public Vector3 Direction;
        public float Time;
    }

    public struct BlendState
    {
        public int FromPoseIndex;
        public int ToPoseIndex;
        public float BlendAlpha;
        public float BlendDuration;
        public float BlendElapsed;
        public bool IsBlending;
        public MotionPose FromPose;
        public MotionPose ToPose;
    }

    public struct FootIKResult
    {
        public Vector3 LeftFootPosition;
        public Vector3 RightFootPosition;
        public Quaternion LeftFootRotation;
        public Quaternion RightFootRotation;
        public float LeftFootHeight;
        public float RightFootHeight;
        public bool LeftFootGrounded;
        public bool RightFootGrounded;
    }

    public class MotionMatchingDatabase
    {
        public MotionFrame[] Frames { get; set; }
        public MotionFeature[] Features { get; set; }
        public TrajectoryPoint[,] Trajectories { get; set; }
        public int FrameCount { get; set; }
        public float TotalDuration { get; set; }
        public int JointCount { get; set; }

        public void Initialize(int frameCount, int jointCount, float frameRate)
        {
            FrameCount = frameCount;
            JointCount = jointCount;
            TotalDuration = frameCount / frameRate;
            Frames = new MotionFrame[frameCount];
            Features = new MotionFeature[frameCount];
            Trajectories = new TrajectoryPoint[frameCount, 8];

            for (int i = 0; i < frameCount; i++)
            {
                Frames[i].JointPositions = new Vector3[jointCount];
                Frames[i].JointRotations = new Quaternion[jointCount];
                Frames[i].JointVelocities = new Vector3[jointCount];
                Frames[i].JointAngles = new float[jointCount];
            }
        }

        public void AddFrame(MotionFrame frame, int index)
        {
            if (index < 0 || index >= FrameCount) return;
            Frames[index] = frame;
            Features[index] = ExtractFeature(frame);
        }

        private MotionFeature ExtractFeature(MotionFrame frame)
        {
            return new MotionFeature
            {
                TrajectoryPosition = frame.RootPosition,
                TrajectoryDirection = RenderingMath.Rotate(frame.RootRotation, RenderingMath.Forward),
                HipPosition = frame.JointPositions.Length > 0 ? frame.JointPositions[0] : frame.RootPosition,
                LeftFootPosition = frame.JointPositions.Length > 14 ? frame.JointPositions[14] : Vector3.Zero,
                RightFootPosition = frame.JointPositions.Length > 18 ? frame.JointPositions[18] : Vector3.Zero,
                LeftHandPosition = frame.JointPositions.Length > 22 ? frame.JointPositions[22] : Vector3.Zero,
                RightHandPosition = frame.JointPositions.Length > 26 ? frame.JointPositions[26] : Vector3.Zero,
                Velocity = frame.RootVelocity,
                Speed = frame.RootVelocity.Length(),
                Direction = MathF.Atan2(frame.RootVelocity.Z, frame.RootVelocity.X)
            };
        }
    }

    public class MotionMatcher : IDisposable
    {
        private MotionMatchingConfig _config;
        private MotionMatchingDatabase _database;
        private BlendState _blendState;
        private FootIKResult _footIK;

        private int _currentFrameIndex;
        private float _currentTime;
        private float _playbackSpeed;
        private bool _isPlaying;
        private bool _disposed;

        private TrajectoryPoint[] _currentTrajectory;
        private TrajectoryPoint[] _desiredTrajectory;
        private Vector3 _inputVelocity;
        private float _inputDirection;
        private float _inputSpeed;

        private int _lastSearchStart;
        private float _matchCooldown;
        private float _matchInterval;

        public int CurrentFrame => _currentFrameIndex;
        public float CurrentTime => _currentTime;
        public bool IsPlaying => _isPlaying;
        public FootIKResult FootIK => _footIK;
        public BlendState CurrentBlend => _blendState;

        public MotionMatcher(MotionMatchingConfig? config = null)
        {
            _config = config ?? new MotionMatchingConfig();
            _currentTrajectory = new TrajectoryPoint[_config.TrajectoryPoints];
            _desiredTrajectory = new TrajectoryPoint[_config.TrajectoryPoints];
            _playbackSpeed = 1.0f;
            _matchInterval = 0.05f;
        }

        public void LoadDatabase(MotionMatchingDatabase database)
        {
            _database = database;
            if (database.FrameCount > 0)
            {
                _currentFrameIndex = 0;
                _currentTime = 0;
                _isPlaying = true;
            }
        }

        public void SetInput(Vector3 velocity, float direction, float speed)
        {
            _inputVelocity = velocity;
            _inputDirection = direction;
            _inputSpeed = speed;
        }

        public MotionPose Update(float deltaTime, Vector3 rootPosition, Quaternion rootRotation)
        {
            if (_database == null || !_isPlaying) return default;

            _currentTime += deltaTime * _playbackSpeed;
            _matchCooldown -= deltaTime;

            UpdateTrajectory(rootPosition, rootRotation);
            UpdateDesiredTrajectory();

            if (_matchCooldown <= 0 && ShouldSearchForMatch())
            {
                int bestFrame = FindBestMatchingFrame();
                if (bestFrame >= 0 && bestFrame != _currentFrameIndex)
                {
                    StartBlend(bestFrame);
                    _matchCooldown = _matchInterval;
                }
            }

            UpdateBlend(deltaTime);

            var pose = SamplePose(_currentFrameIndex);

            if (_config.EnableFootIK)
                _footIK = ComputeFootIK(pose);

            return pose;
        }

        private bool ShouldSearchForMatch()
        {
            if (_database == null) return false;
            float currentSpeed = _inputSpeed;
            return currentSpeed > 0.1f || MathF.Abs(_inputDirection) > 0.1f;
        }

        private void UpdateTrajectory(Vector3 rootPosition, Quaternion rootRotation)
        {
            for (int i = 0; i < _config.TrajectoryPoints; i++)
            {
                float t = i * _config.TrajectoryInterval;
                Vector3 forward = RenderingMath.Rotate(rootRotation, RenderingMath.Forward);
                Vector3 right = RenderingMath.Rotate(rootRotation, RenderingMath.Right);

                _currentTrajectory[i] = new TrajectoryPoint
                {
                    Position = rootPosition + forward * t * _inputSpeed + right * MathF.Sin(_inputDirection * t) * _inputSpeed,
                    Direction = forward,
                    Time = t
                };
            }
        }

        private void UpdateDesiredTrajectory()
        {
            for (int i = 0; i < _config.TrajectoryPoints; i++)
            {
                float t = i * _config.TrajectoryInterval;
                Vector3 forward = new Vector3(MathF.Cos(_inputDirection), 0, MathF.Sin(_inputDirection));

                _desiredTrajectory[i] = new TrajectoryPoint
                {
                    Position = forward * t * _inputSpeed,
                    Direction = forward,
                    Time = t
                };
            }
        }

        private int FindBestMatchingFrame()
        {
            if (_database == null || _database.FrameCount == 0) return -1;

            int searchStart = Math.Max(0, _lastSearchStart - _config.SearchWindow / 2);
            int searchEnd = Math.Min(_database.FrameCount, _lastSearchStart + _config.SearchWindow);

            float bestCost = float.MaxValue;
            int bestFrame = -1;

            for (int i = searchStart; i < searchEnd; i++)
            {
                float cost = ComputeMatchCost(i);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestFrame = i;
                }
            }

            int fullSearchStart = 0;
            int fullSearchEnd = _database.FrameCount;
            int step = Math.Max(1, _database.FrameCount / 500);

            for (int i = fullSearchStart; i < fullSearchEnd; i += step)
            {
                float cost = ComputeMatchCost(i);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestFrame = i;
                }
            }

            _lastSearchStart = bestFrame >= 0 ? bestFrame : _currentFrameIndex;
            return bestFrame;
        }

        private float ComputeMatchCost(int frameIndex)
        {
            if (frameIndex < 0 || frameIndex >= _database.FrameCount) return float.MaxValue;

            var feature = _database.Features[frameIndex];

            float cost = 0;

            float trajectoryWeight = 2.0f;
            for (int i = 0; i < _config.TrajectoryPoints && i < _database.Trajectories.GetLength(1); i++)
            {
                var dbTraj = _database.Trajectories[frameIndex, i];
                var desired = _desiredTrajectory[i];

                cost += Vector3.Distance(dbTraj.Position, desired.Position) * trajectoryWeight;
                cost += (1.0f - Vector3.Dot(dbTraj.Direction, desired.Direction)) * trajectoryWeight;
            }

            cost += feature.Speed * 0.5f;
            cost += MathF.Abs(feature.Direction - _inputDirection) * 1.0f;

            float footWeight = 1.0f;
            cost += feature.LeftFootPosition.Y * footWeight;
            cost += feature.RightFootPosition.Y * footWeight;

            return cost;
        }

        private void StartBlend(int targetFrame)
        {
            _blendState = new BlendState
            {
                FromPoseIndex = _currentFrameIndex,
                ToPoseIndex = targetFrame,
                BlendAlpha = 0,
                BlendDuration = _config.BlendTime,
                BlendElapsed = 0,
                IsBlending = true,
                FromPose = SamplePose(_currentFrameIndex),
                ToPose = SamplePose(targetFrame)
            };
        }

        private void UpdateBlend(float deltaTime)
        {
            if (!_blendState.IsBlending) return;

            _blendState.BlendElapsed += deltaTime;
            _blendState.BlendAlpha = MathF.Min(_blendState.BlendElapsed / _blendState.BlendDuration, 1.0f);

            float t = _blendState.BlendAlpha;
            t = _config.BlendMode switch
            {
                BlendMode.Linear => t,
                BlendMode.Cubic => t * t * (3 - 2 * t),
                BlendMode.Bezier => t * t * t * (t * (6 * t - 15) + 10),
                _ => t
            };

            _blendState.BlendAlpha = t;

            if (_blendState.BlendAlpha >= 1.0f)
            {
                _currentFrameIndex = _blendState.ToPoseIndex;
                _currentTime = _database.Frames[_currentFrameIndex].Time;
                _blendState.IsBlending = false;
            }
        }

        private MotionPose SamplePose(int frameIndex)
        {
            if (_database == null || frameIndex < 0 || frameIndex >= _database.FrameCount)
                return default;

            var frame = _database.Frames[frameIndex];
            return new MotionPose
            {
                JointRotations = (Quaternion[])frame.JointRotations.Clone(),
                JointPositions = (Vector3[])frame.JointPositions.Clone(),
                Time = frame.Time
            };
        }

        public MotionPose GetBlendedPose()
        {
            if (!_blendState.IsBlending) return SamplePose(_currentFrameIndex);

            var from = _blendState.FromPose;
            var to = _blendState.ToPose;
            float t = _blendState.BlendAlpha;

            int jointCount = Math.Min(from.JointRotations?.Length ?? 0, to.JointRotations?.Length ?? 0);
            var result = new MotionPose
            {
                JointRotations = new Quaternion[jointCount],
                JointPositions = new Vector3[jointCount],
                Time = float.Lerp(from.Time, to.Time, t)
            };

            for (int i = 0; i < jointCount; i++)
            {
                result.JointRotations[i] = Quaternion.Slerp(from.JointRotations[i], to.JointRotations[i], t);
                result.JointPositions[i] = Vector3.Lerp(from.JointPositions[i], to.JointPositions[i], t);
            }

            return result;
        }

        private FootIKResult ComputeFootIK(MotionPose pose)
        {
            var result = new FootIKResult();

            if (pose.JointPositions == null || pose.JointPositions.Length < 20)
                return result;

            result.LeftFootPosition = pose.JointPositions[14];
            result.RightFootPosition = pose.JointPositions[18];
            result.LeftFootRotation = pose.JointRotations.Length > 14 ? pose.JointRotations[14] : Quaternion.Identity;
            result.RightFootRotation = pose.JointRotations.Length > 18 ? pose.JointRotations[18] : Quaternion.Identity;
            result.LeftFootHeight = result.LeftFootPosition.Y;
            result.RightFootHeight = result.RightFootPosition.Y;
            result.LeftFootGrounded = result.LeftFootHeight < 0.1f;
            result.RightFootGrounded = result.RightFootHeight < 0.1f;

            return result;
        }

        public void SetPlaybackSpeed(float speed) => _playbackSpeed = speed;
        public void Play() => _isPlaying = true;
        public void Pause() => _isPlaying = false;
        public void Seek(float time)
        {
            if (_database == null) return;
            _currentTime = time;
            for (int i = 0; i < _database.FrameCount; i++)
            {
                if (_database.Frames[i].Time >= time)
                {
                    _currentFrameIndex = i;
                    break;
                }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
        }
    }
}
