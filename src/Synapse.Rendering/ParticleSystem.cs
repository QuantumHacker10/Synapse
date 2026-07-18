// =============================================================================
// ParticleSystem.cs - GDNN Engine: Niagara-like Procedural Particle System
// GPU-driven particle simulation with modules, emitters, and renderers
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace GDNN.Rendering.Particles
{
    public enum ParticleEmitterType { Point = 0, Box = 1, Sphere = 2, Cone = 3, Cylinder = 4, Mesh = 5 }
    public enum ParticleSortMode { None = 0, Distance = 1, OldestFirst = 2, YoungestFirst = 3 }
    public enum ParticleRenderMode { Billboard = 0, Mesh = 1, Ribbon = 2, Beam = 3 }

    public record ParticleSystemConfig
    {
        public int MaxParticles { get; init; } = 100000;
        public int MaxEmitters { get; init; } = 64;
        public bool EnableGPU { get; init; } = true;
        public bool EnableCollision { get; init; } = true;
        public bool EnableLOD { get; init; } = true;
        public float LODDistance { get; init; } = 100.0f;
    }

    public class ParticleData
    {
        public Vector3 Position;
        public Vector3 Velocity;
        public Vector3 Acceleration;
        public Vector3 PreviousPosition;
        public Vector4 Color;
        public float Size;
        public float Rotation;
        public float AngularVelocity;
        public float Lifetime;
        public float Age;
        public float NormalizedAge;
        public float Mass;
        public float Drag;
        public int EmitterIndex;
        public bool IsAlive;
        public int SortKey;
    }

    public class ParticleEmitter
    {
        public string Name { get; set; } = "Emitter";
        public ParticleEmitterType Type { get; set; } = ParticleEmitterType.Point;
        public bool IsActive { get; set; } = true;
        public bool IsLooping { get; set; } = true;
        public float Duration { get; set; } = 5.0f;
        public float ElapsedTime { get; set; }

        public int BurstCount { get; set; } = 0;
        public float BurstTime { get; set; } = 0;

        public float Rate { get; set; } = 100.0f;
        public float RateOverDistance { get; set; }
        public int MaxParticles { get; set; } = 10000;

        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; } = Quaternion.Identity;
        public Vector3 Scale { get; set; } = Vector3.One;

        public Vector3 EmitterShapeMin { get; set; } = new Vector3(-0.5f);
        public Vector3 EmitterShapeMax { get; set; } = new Vector3(0.5f);
        public float ConeAngle { get; set; } = 25.0f;
        public float ConeRadius { get; set; } = 1.0f;

        public Vector3 StartVelocityMin { get; set; } = new Vector3(0, 1, 0);
        public Vector3 StartVelocityMax { get; set; } = new Vector3(0, 2, 0);
        public float LifetimeMin { get; set; } = 1.0f;
        public float LifetimeMax { get; set; } = 3.0f;
        public float StartSizeMin { get; set; } = 0.1f;
        public float StartSizeMax { get; set; } = 0.5f;
        public float StartRotationMin { get; set; } = 0;
        public float StartRotationMax { get; set; } = 360.0f;
        public float MassMin { get; set; } = 1.0f;
        public float MassMax { get; set; } = 1.0f;
        public float Drag { get; set; } = 0.1f;

        public Vector4 StartColorMin { get; set; } = new Vector4(1, 1, 1, 1);
        public Vector4 StartColorMax { get; set; } = new Vector4(1, 1, 1, 1);

        public List<ParticleModule> Modules { get; set; } = new();
    }

    public abstract class ParticleModule
    {
        public bool Enabled { get; set; } = true;
        public int Order { get; set; }
        public abstract void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter);
    }

    public class ParticleModuleGravity : ParticleModule
    {
        public Vector3 Gravity { get; set; } = new Vector3(0, -9.81f, 0);
        public float Strength { get; set; } = 1.0f;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                particles[i].Acceleration += Gravity * Strength;
            }
        }
    }

    public class ParticleModuleDrag : ParticleModule
    {
        public float DragCoefficient { get; set; } = 0.5f;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                float drag = 1.0f / (1.0f + deltaTime * DragCoefficient * particles[i].Mass);
                particles[i].Velocity *= drag;
            }
        }
    }

    public class ParticleModuleColorOverLife : ParticleModule
    {
        public Vector4 ColorStart { get; set; } = new Vector4(1, 1, 1, 1);
        public Vector4 ColorEnd { get; set; } = new Vector4(1, 0, 0, 0);

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                particles[i].Color = Vector4.Lerp(ColorStart, ColorEnd, particles[i].NormalizedAge);
            }
        }
    }

    public class ParticleModuleSizeOverLife : ParticleModule
    {
        public float SizeStart { get; set; } = 1.0f;
        public float SizeEnd { get; set; } = 0.0f;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                float size = float.Lerp(SizeStart, SizeEnd, particles[i].NormalizedAge);
                particles[i].Size *= size;
            }
        }
    }

    public class ParticleModuleRotationOverLife : ParticleModule
    {
        public float RotationSpeed { get; set; } = 90.0f;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                particles[i].Rotation += RotationSpeed * deltaTime;
            }
        }
    }

    public class ParticleModuleCurlNoise : ParticleModule
    {
        public float Strength { get; set; } = 1.0f;
        public float Frequency { get; set; } = 1.0f;
        private float _time;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            _time += deltaTime;
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                Vector3 noise = CurlNoise(particles[i].Position * Frequency + new Vector3(_time));
                particles[i].Velocity += noise * Strength * deltaTime;
            }
        }

        private Vector3 CurlNoise(Vector3 p)
        {
            float e = 0.01f;
            float n1 = SimpleNoise(p + new Vector3(e, 0, 0)) - SimpleNoise(p - new Vector3(e, 0, 0));
            float n2 = SimpleNoise(p + new Vector3(0, e, 0)) - SimpleNoise(p - new Vector3(0, e, 0));
            float n3 = SimpleNoise(p + new Vector3(0, 0, e)) - SimpleNoise(p - new Vector3(0, 0, e));
            return Vector3.Normalize(new Vector3(n1, n2, n3) / (2 * e));
        }

        private float SimpleNoise(Vector3 p)
        {
            return MathF.Sin(p.X * 12.9898f + p.Y * 78.233f + p.Z * 45.164f) * 0.5f + 0.5f;
        }
    }

    public class ParticleModuleCollision : ParticleModule
    {
        public float Restitution { get; set; } = 0.5f;
        public float Friction { get; set; } = 0.1f;
        public float GroundHeight { get; set; } = 0.0f;

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                if (particles[i].Position.Y < GroundHeight)
                {
                    particles[i].Position.Y = GroundHeight;
                    particles[i].Velocity.Y = MathF.Abs(particles[i].Velocity.Y) * Restitution;
                    particles[i].Velocity.X *= (1.0f - Friction);
                    particles[i].Velocity.Z *= (1.0f - Friction);
                }
            }
        }
    }

    public class ParticleModuleSubUV : ParticleModule
    {
        public int GridX { get; set; } = 4;
        public int GridY { get; set; } = 4;
        public bool RandomStartFrame { get; set; }

        public override void Update(ParticleData[] particles, int count, float deltaTime, ParticleEmitter emitter)
        {
            int totalFrames = GridX * GridY;
            for (int i = 0; i < count; i++)
            {
                if (!particles[i].IsAlive) continue;
                int frame = (int)(particles[i].NormalizedAge * totalFrames);
                frame = Math.Clamp(frame, 0, totalFrames - 1);
                particles[i].SortKey = frame;
            }
        }
    }

    public class ParticleRenderer
    {
        public ParticleRenderMode Mode { get; set; } = ParticleRenderMode.Billboard;
        public ParticleSortMode SortMode { get; set; } = ParticleSortMode.Distance;
        public bool SortEnabled { get; set; } = true;
        public float SortUpdateInterval { get; set; } = 0.1f;
        public float SortTimer { get; set; }
    }

    public class GPUParticleBuffer
    {
        public float[] PositionBuffer { get; set; }
        public float[] VelocityBuffer { get; set; }
        public float[] ColorBuffer { get; set; }
        public float[] SizeBuffer { get; set; }
        public float[] RotationBuffer { get; set; }
        public float[] AgeBuffer { get; set; }
        public float[] LifetimeBuffer { get; set; }
        public int[] AliveIndices { get; set; }
        public int[] DeadIndices { get; set; }
        public int AliveCount { get; set; }
        public int DeadCount { get; set; }

        public GPUParticleBuffer(int maxParticles)
        {
            PositionBuffer = new float[maxParticles * 3];
            VelocityBuffer = new float[maxParticles * 3];
            ColorBuffer = new float[maxParticles * 4];
            SizeBuffer = new float[maxParticles];
            RotationBuffer = new float[maxParticles];
            AgeBuffer = new float[maxParticles];
            LifetimeBuffer = new float[maxParticles];
            AliveIndices = new int[maxParticles];
            DeadIndices = new int[maxParticles];

            for (int i = 0; i < maxParticles; i++)
                DeadIndices[i] = i;
            DeadCount = maxParticles;
        }
    }

    public class ParticleSystem : IDisposable
    {
        private ParticleSystemConfig _config;
        private List<ParticleEmitter> _emitters;
        private ParticleData[] _particles;
        private GPUParticleBuffer _gpuBuffer;
        private int _activeParticleCount;
        private float _totalTime;
        private bool _disposed;
        private readonly object _lock = new();
        private Random _rng = new();

        private Vector3 _cameraPosition;

        public int ActiveParticles => _activeParticleCount;
        public int MaxParticles => _config.MaxParticles;
        public List<ParticleEmitter> Emitters => _emitters;
        public GPUParticleBuffer GPUBuffer => _gpuBuffer;

        public ParticleSystem(ParticleSystemConfig? config = null)
        {
            _config = config ?? new ParticleSystemConfig();
            _emitters = new List<ParticleEmitter>();
            _particles = new ParticleData[_config.MaxParticles];
            _gpuBuffer = new GPUParticleBuffer(_config.MaxParticles);

            for (int i = 0; i < _config.MaxParticles; i++)
                _particles[i] = new ParticleData();
        }

        public int AddEmitter(ParticleEmitter emitter)
        {
            _emitters.Add(emitter);
            return _emitters.Count - 1;
        }

        public void RemoveEmitter(int index)
        {
            if (index >= 0 && index < _emitters.Count)
                _emitters.RemoveAt(index);
        }

        public void Update(float deltaTime, Vector3 cameraPosition)
        {
            lock (_lock)
            {
                _totalTime += deltaTime;
                _cameraPosition = cameraPosition;

                _activeParticleCount = 0;

                for (int e = 0; e < _emitters.Count; e++)
                {
                    var emitter = _emitters[e];
                    if (!emitter.IsActive) continue;

                    emitter.ElapsedTime += deltaTime;
                    if (!emitter.IsLooping && emitter.ElapsedTime > emitter.Duration)
                    {
                        emitter.IsActive = false;
                        continue;
                    }

                    EmitParticles(emitter, deltaTime);
                    UpdateModules(emitter, deltaTime);
                    IntegrateParticles(deltaTime);
                    AgeParticles(deltaTime);
                    CompactAliveParticles();
                }

                UpdateGPUBuffer();
            }
        }

        private void EmitParticles(ParticleEmitter emitter, float deltaTime)
        {
            int particlesToEmit = (int)(emitter.Rate * deltaTime);

            if (emitter.BurstCount > 0 && MathF.Abs(emitter.ElapsedTime - emitter.BurstTime) < deltaTime)
                particlesToEmit += emitter.BurstCount;

            for (int i = 0; i < particlesToEmit && _activeParticleCount < _config.MaxParticles; i++)
            {
                int slot = FindDeadParticle();
                if (slot < 0) break;

                var p = _particles[slot];
                p.IsAlive = true;
                p.Age = 0;
                p.Lifetime = float.Lerp(emitter.LifetimeMin, emitter.LifetimeMax, (float)_rng.NextDouble());
                p.EmitterIndex = _emitters.IndexOf(emitter);

                p.Position = GeneratePosition(emitter);
                p.Velocity = GenerateVelocity(emitter);
                p.Acceleration = Vector3.Zero;
                p.Size = float.Lerp(emitter.StartSizeMin, emitter.StartSizeMax, (float)_rng.NextDouble());
                p.Rotation = float.Lerp(emitter.StartRotationMin, emitter.StartRotationMax, (float)_rng.NextDouble());
                p.AngularVelocity = 0;
                p.Mass = float.Lerp(emitter.MassMin, emitter.MassMax, (float)_rng.NextDouble());
                p.Drag = emitter.Drag;
                p.Color = Vector4.Lerp(emitter.StartColorMin, emitter.StartColorMax, (float)_rng.NextDouble());
                p.PreviousPosition = p.Position;

                _gpuBuffer.DeadIndices[slot] = -1;
                _gpuBuffer.AliveIndices[_activeParticleCount] = slot;
                _activeParticleCount++;
            }
        }

        private Vector3 GeneratePosition(ParticleEmitter emitter)
        {
            Vector3 localPos = emitter.Type switch
            {
                ParticleEmitterType.Point => Vector3.Zero,
                ParticleEmitterType.Box => new Vector3(
                    float.Lerp(emitter.EmitterShapeMin.X, emitter.EmitterShapeMax.X, (float)_rng.NextDouble()),
                    float.Lerp(emitter.EmitterShapeMin.Y, emitter.EmitterShapeMax.Y, (float)_rng.NextDouble()),
                    float.Lerp(emitter.EmitterShapeMin.Z, emitter.EmitterShapeMax.Z, (float)_rng.NextDouble())),
                ParticleEmitterType.Sphere => GenerateSpherePoint(emitter.ConeRadius),
                ParticleEmitterType.Cone => GenerateConePoint(emitter),
                ParticleEmitterType.Cylinder => GenerateCylinderPoint(emitter),
                _ => Vector3.Zero
            };

            Vector3 rotated = Vector3.Transform(localPos, emitter.Rotation);
            return emitter.Position + rotated;
        }

        private Vector3 GenerateSpherePoint(float radius)
        {
            float theta = (float)_rng.NextDouble() * MathF.PI * 2;
            float phi = MathF.Acos(2 * (float)_rng.NextDouble() - 1);
            float r = radius * MathF.Pow((float)_rng.NextDouble(), 1.0f / 3);
            return new Vector3(
                r * MathF.Sin(phi) * MathF.Cos(theta),
                r * MathF.Sin(phi) * MathF.Sin(theta),
                r * MathF.Cos(phi));
        }

        private Vector3 GenerateConePoint(ParticleEmitter emitter)
        {
            float angle = emitter.ConeAngle * MathF.PI / 180.0f;
            float r = (float)_rng.NextDouble() * emitter.ConeRadius;
            float theta = (float)_rng.NextDouble() * MathF.PI * 2;
            float spread = (float)_rng.NextDouble() * MathF.Tan(angle);
            return new Vector3(MathF.Cos(theta) * spread * r, 1.0f, MathF.Sin(theta) * spread * r);
        }

        private Vector3 GenerateCylinderPoint(ParticleEmitter emitter)
        {
            float theta = (float)_rng.NextDouble() * MathF.PI * 2;
            float r = emitter.ConeRadius * MathF.Sqrt((float)_rng.NextDouble());
            float y = float.Lerp(emitter.EmitterShapeMin.Y, emitter.EmitterShapeMax.Y, (float)_rng.NextDouble());
            return new Vector3(MathF.Cos(theta) * r, y, MathF.Sin(theta) * r);
        }

        private Vector3 GenerateVelocity(ParticleEmitter emitter)
        {
            Vector3 minV = emitter.StartVelocityMin;
            Vector3 maxV = emitter.StartVelocityMax;
            return new Vector3(
                float.Lerp(minV.X, maxV.X, (float)_rng.NextDouble()),
                float.Lerp(minV.Y, maxV.Y, (float)_rng.NextDouble()),
                float.Lerp(minV.Z, maxV.Z, (float)_rng.NextDouble()));
        }

        private int FindDeadParticle()
        {
            for (int i = 0; i < _config.MaxParticles; i++)
            {
                if (!_particles[i].IsAlive)
                {
                    _gpuBuffer.DeadIndices[i] = -1;
                    return i;
                }
            }
            return -1;
        }

        private void UpdateModules(ParticleEmitter emitter, float deltaTime)
        {
            var sortedModules = emitter.Modules;
            sortedModules.Sort((a, b) => a.Order.CompareTo(b.Order));

            foreach (var module in sortedModules)
            {
                if (!module.Enabled) continue;
                module.Update(_particles, _config.MaxParticles, deltaTime, emitter);
            }
        }

        private void IntegrateParticles(float deltaTime)
        {
            for (int i = 0; i < _config.MaxParticles; i++)
            {
                if (!_particles[i].IsAlive) continue;

                ref var p = ref _particles[i];
                p.PreviousPosition = p.Position;

                p.Velocity += p.Acceleration * deltaTime;
                p.Position += p.Velocity * deltaTime;
                p.Acceleration = Vector3.Zero;
            }
        }

        private void AgeParticles(float deltaTime)
        {
            for (int i = 0; i < _config.MaxParticles; i++)
            {
                if (!_particles[i].IsAlive) continue;

                ref var p = ref _particles[i];
                p.Age += deltaTime;
                p.NormalizedAge = MathF.Min(p.Age / p.Lifetime, 1.0f);

                if (p.Age >= p.Lifetime)
                {
                    p.IsAlive = false;
                    _gpuBuffer.DeadIndices[_gpuBuffer.DeadCount++] = i;
                }
            }
        }

        private void CompactAliveParticles()
        {
            int writeIdx = 0;
            for (int i = 0; i < _activeParticleCount; i++)
            {
                int particleIdx = _gpuBuffer.AliveIndices[i];
                if (_particles[particleIdx].IsAlive)
                {
                    _gpuBuffer.AliveIndices[writeIdx++] = particleIdx;
                }
            }
            _activeParticleCount = writeIdx;
        }

        private void UpdateGPUBuffer()
        {
            for (int i = 0; i < _activeParticleCount; i++)
            {
                int idx = _gpuBuffer.AliveIndices[i];
                var p = _particles[idx];

                int posIdx = i * 3;
                _gpuBuffer.PositionBuffer[posIdx] = p.Position.X;
                _gpuBuffer.PositionBuffer[posIdx + 1] = p.Position.Y;
                _gpuBuffer.PositionBuffer[posIdx + 2] = p.Position.Z;

                _gpuBuffer.VelocityBuffer[posIdx] = p.Velocity.X;
                _gpuBuffer.VelocityBuffer[posIdx + 1] = p.Velocity.Y;
                _gpuBuffer.VelocityBuffer[posIdx + 2] = p.Velocity.Z;

                int colIdx = i * 4;
                _gpuBuffer.ColorBuffer[colIdx] = p.Color.X;
                _gpuBuffer.ColorBuffer[colIdx + 1] = p.Color.Y;
                _gpuBuffer.ColorBuffer[colIdx + 2] = p.Color.Z;
                _gpuBuffer.ColorBuffer[colIdx + 3] = p.Color.W;

                _gpuBuffer.SizeBuffer[i] = p.Size;
                _gpuBuffer.RotationBuffer[i] = p.Rotation;
                _gpuBuffer.AgeBuffer[i] = p.Age;
                _gpuBuffer.LifetimeBuffer[i] = p.Lifetime;
            }

            _gpuBuffer.AliveCount = _activeParticleCount;
        }

        public void GetParticleData(int index, out Vector3 position, out Vector4 color, out float size)
        {
            if (index >= 0 && index < _activeParticleCount)
            {
                int idx = _gpuBuffer.AliveIndices[index];
                var p = _particles[idx];
                position = p.Position;
                color = p.Color;
                size = p.Size;
            }
            else
            {
                position = Vector3.Zero;
                color = Vector4.Zero;
                size = 0;
            }
        }

        public void SetParticleData(int index, Vector3 position, Vector4 color, float size)
        {
            if (index >= 0 && index < _activeParticleCount)
            {
                int idx = _gpuBuffer.AliveIndices[index];
                _particles[idx].Position = position;
                _particles[idx].Color = color;
                _particles[idx].Size = size;
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                for (int i = 0; i < _config.MaxParticles; i++)
                    _particles[i].IsAlive = false;
                _activeParticleCount = 0;
                _totalTime = 0;

                foreach (var emitter in _emitters)
                {
                    emitter.IsActive = true;
                    emitter.ElapsedTime = 0;
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
