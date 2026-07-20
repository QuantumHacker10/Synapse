// ============================================================================
// Synapse Omnia — Physics/MultiphysicsOrchestrator.cs
// Couples living laws, rigid-body dynamics, and optional continuum solvers
// into a single industrial frame-step pipeline.
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;

namespace Synapse.Physics;

/// <summary>Which continuum modules run alongside living laws / rigid bodies.</summary>
[Flags]
public enum ContinuumModules
{
    None = 0,
    LivingLaws = 1,
    RigidBodies = 2,
    Sph = 4,
    Elasticity = 8,
    All = LivingLaws | RigidBodies | Sph | Elasticity
}

/// <summary>Configuration for the multiphysics frame pipeline.</summary>
public sealed class MultiphysicsConfig
{
    public ContinuumModules EnabledModules { get; set; } = ContinuumModules.LivingLaws | ContinuumModules.RigidBodies;
    public float FixedTimeStep { get; set; } = 1f / 60f;
    public int MaxSubSteps { get; set; } = 4;
    public bool SyncFieldFromRigidBodies { get; set; } = true;
    public Vector3 Gravity { get; set; } = new(0f, -9.81f, 0f);
}

/// <summary>Per-step telemetry for budgets and industrial diagnostics.</summary>
public sealed class MultiphysicsStepStats
{
    public int SubSteps { get; set; }
    public float RigidBodyMs { get; set; }
    public float LivingLawMs { get; set; }
    public float ContinuumMs { get; set; }
    public float TotalMs { get; set; }
    public PhysicsWorldStats? RigidStats { get; set; }
    public float AverageTemperature { get; set; }
}

/// <summary>
/// Industrial multiphysics orchestrator: fixed-step accumulation, living-law
/// field updates, rigid-body world step, optional SPH/elasticity ticks.
/// </summary>
public sealed class MultiphysicsOrchestrator : IDisposable
{
    private readonly MultiphysicsConfig _config;
    private readonly LivingLawCompiler _lawCompiler;
    private readonly RigidBodyWorld _rigidWorld;
    private PhysicsField _field;
    private string? _activeLawId;
    private SphSolver? _sph;
    private ElasticitySolver? _elasticity;
    private float _accumulator;
    private bool _disposed;

    public MultiphysicsOrchestrator(
        LivingLawCompiler lawCompiler,
        PhysicsField field,
        MultiphysicsConfig? config = null)
    {
        _lawCompiler = lawCompiler ?? throw new ArgumentNullException(nameof(lawCompiler));
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _config = config ?? new MultiphysicsConfig();
        _rigidWorld = new RigidBodyWorld { Gravity = _config.Gravity };
    }

    public RigidBodyWorld RigidWorld => _rigidWorld;
    public PhysicsField Field => _field;
    public string? ActiveLawId => _activeLawId;
    public MultiphysicsStepStats LastStats { get; } = new();
    public MultiphysicsConfig Config => _config;

    public void SetField(PhysicsField field) =>
        _field = field ?? throw new ArgumentNullException(nameof(field));

    public void SetActiveLaw(string? lawId) => _activeLawId = lawId;

    /// <summary>Enables an SPH module with a small industrial default fluid block.</summary>
    public SphSolver EnableSph(SphConfig? config = null)
    {
        _sph = new SphSolver(config ?? new SphConfig { NumParticles = 256, TimeStep = _config.FixedTimeStep });
        _config.EnabledModules |= ContinuumModules.Sph;
        return _sph;
    }

    /// <summary>Enables grid elasticity with contact against a ground plane.</summary>
    public ElasticitySolver EnableElasticity(ElasticityConfig? config = null)
    {
        var cfg = config ?? new ElasticityConfig
        {
            GridSize = (8, 8, 8),
            CellSize = 0.1,
            TimeStep = _config.FixedTimeStep,
            EnableContact = true
        };
        _elasticity = new ElasticitySolver(cfg);
        _config.EnabledModules |= ContinuumModules.Elasticity;
        return _elasticity;
    }

    /// <summary>
    /// Seeds a default industrial demo: ground plane + falling boxes / spheres
    /// derived from scene-like descriptors.
    /// </summary>
    public void SeedDemoBodies()
    {
        _rigidWorld.Clear();

        _rigidWorld.AddBody(new RigidBody
        {
            Name = "Ground",
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Plane, Size = new Vector3(50f, 0.01f, 50f) },
            Position = new Vector3(0f, 0f, 0f),
            Material = new PhysicsMaterial { Friction = 0.7f, Restitution = 0.1f }
        });

        _rigidWorld.AddBody(new RigidBody
        {
            Name = "Crate",
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.5f, 0.5f, 0.5f) },
            Position = new Vector3(0f, 3f, 0f),
            Material = PhysicsMaterial.Default
        }).SetMass(10f);

        _rigidWorld.AddBody(new RigidBody
        {
            Name = "Ball",
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.4f) },
            Position = new Vector3(0.2f, 5f, 0.1f),
            Material = PhysicsMaterial.Bouncy
        }).SetMass(2f);
    }

    /// <summary>
    /// Creates / updates rigid bodies from scene entity descriptors.
    /// Mesh/Ground → static box or plane; others → dynamic box scaled from entity.
    /// </summary>
    public void SyncFromEntities(IEnumerable<PhysicsEntityDesc> entities)
    {
        foreach (var e in entities)
        {
            var existing = _rigidWorld.GetBody(e.Id);
            if (existing != null)
            {
                if (existing.Type != BodyType.Dynamic || existing.IsSleeping == false)
                {
                    // Keep simulation authority for dynamic bodies; only push kinematic/static from scene.
                    if (existing.Type != BodyType.Dynamic)
                    {
                        existing.Position = e.Position;
                        existing.UpdateAabb();
                    }
                }
                continue;
            }

            bool isGround = e.Name.Contains("Ground", StringComparison.OrdinalIgnoreCase)
                || e.Type.Equals("Ground", StringComparison.OrdinalIgnoreCase);
            bool isStatic = isGround || e.IsStatic || e.Type.Equals("Mesh", StringComparison.OrdinalIgnoreCase)
                && e.Scale.Y < 0.5f && e.Scale.X > 2f;

            var body = new RigidBody
            {
                Id = e.Id,
                Name = e.Name,
                Type = isStatic ? BodyType.Static : BodyType.Dynamic,
                Position = e.Position,
                Material = e.Restitution > 0.5f ? PhysicsMaterial.Bouncy : PhysicsMaterial.Default
            };

            if (isGround)
            {
                body.Collider = new Collider { Shape = ColliderShape.Plane, Size = new Vector3(50f, 0.01f, 50f) };
                body.Position = new Vector3(e.Position.X, e.Position.Y, e.Position.Z);
            }
            else if (e.Collider == ColliderShape.Sphere
                     || e.Type.Equals("Genome", StringComparison.OrdinalIgnoreCase)
                     || e.Type.Equals("Character", StringComparison.OrdinalIgnoreCase))
            {
                float r = MathF.Max(0.2f, MathF.Max(e.Scale.X, MathF.Max(e.Scale.Y, e.Scale.Z)) * 0.5f);
                body.Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(r) };
            }
            else
            {
                body.Collider = new Collider
                {
                    Shape = ColliderShape.Box,
                    Size = new Vector3(
                        MathF.Max(0.05f, e.Scale.X * 0.5f),
                        MathF.Max(0.05f, e.Scale.Y * 0.5f),
                        MathF.Max(0.05f, e.Scale.Z * 0.5f))
                };
            }

            if (body.Type == BodyType.Dynamic)
            {
                float mass = e.Mass > 0f ? e.Mass : MathF.Max(0.1f, e.Scale.X * e.Scale.Y * e.Scale.Z * body.Material.Density * 0.001f);
                body.SetMass(mass);
                // Lift dynamic entities slightly so they fall onto the ground plane.
                if (body.Position.Y < body.Collider.Size.Y + 0.1f)
                    body.Position = new Vector3(body.Position.X, body.Collider.Size.Y + 1.5f, body.Position.Z);
            }
            else
            {
                body.SetMass(0f);
            }

            _rigidWorld.AddBody(body);
        }

        // Ensure a ground plane exists even if the scene omitted one.
        bool hasPlane = false;
        for (int i = 0; i < _rigidWorld.Bodies.Count; i++)
        {
            if (_rigidWorld.Bodies[i].Collider.Shape == ColliderShape.Plane)
            {
                hasPlane = true;
                break;
            }
        }
        if (!hasPlane)
        {
            _rigidWorld.AddBody(new RigidBody
            {
                Name = "ImplicitGround",
                Type = BodyType.Static,
                Collider = new Collider { Shape = ColliderShape.Plane },
                Position = Vector3.Zero
            });
        }
    }

    /// <summary>Writes dynamic body transforms back to entity descriptors.</summary>
    public void WriteTransformsTo(IList<PhysicsEntityDesc> entities)
    {
        for (int i = 0; i < entities.Count; i++)
        {
            var e = entities[i];
            var body = _rigidWorld.GetBody(e.Id);
            if (body == null || body.Type != BodyType.Dynamic)
                continue;
            e.Position = body.Position;
            entities[i] = e;
        }
    }

    /// <summary>
    /// Fixed-step multiphysics tick. Returns false if the budget was exceeded mid-substep.
    /// </summary>
    public bool Step(float dt, TimeSpan? budget = null)
    {
        if (_disposed) return false;
        var totalSw = System.Diagnostics.Stopwatch.StartNew();
        var budgetMs = budget?.TotalMilliseconds ?? double.PositiveInfinity;

        _accumulator += Math.Clamp(dt, 0f, _config.FixedTimeStep * _config.MaxSubSteps);
        int steps = 0;
        float rbMs = 0f, lawMs = 0f, contMs = 0f;

        while (_accumulator >= _config.FixedTimeStep && steps < _config.MaxSubSteps)
        {
            if (totalSw.Elapsed.TotalMilliseconds > budgetMs)
                break;

            float h = _config.FixedTimeStep;

            if ((_config.EnabledModules & ContinuumModules.LivingLaws) != 0
                && _lawCompiler != null
                && !string.IsNullOrEmpty(_activeLawId))
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    _lawCompiler.ApplyLaw(_activeLawId!, _field, h);
                    _field.Time += h;
                }
                catch
                {
                    // Living-law failures must not halt rigid-body industrial path.
                }
                sw.Stop();
                lawMs += (float)sw.Elapsed.TotalMilliseconds;
            }

            if ((_config.EnabledModules & ContinuumModules.RigidBodies) != 0)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _rigidWorld.Step(h);
                sw.Stop();
                rbMs += (float)sw.Elapsed.TotalMilliseconds;
            }

            if ((_config.EnabledModules & ContinuumModules.Sph) != 0 && _sph != null)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _sph.Step(h);
                sw.Stop();
                contMs += (float)sw.Elapsed.TotalMilliseconds;
            }

            if ((_config.EnabledModules & ContinuumModules.Elasticity) != 0 && _elasticity != null)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                _elasticity.Step();
                sw.Stop();
                contMs += (float)sw.Elapsed.TotalMilliseconds;
            }

            if (_config.SyncFieldFromRigidBodies)
                InjectKineticHeating();

            _accumulator -= h;
            steps++;
        }

        totalSw.Stop();
        LastStats.SubSteps = steps;
        LastStats.RigidBodyMs = rbMs;
        LastStats.LivingLawMs = lawMs;
        LastStats.ContinuumMs = contMs;
        LastStats.TotalMs = (float)totalSw.Elapsed.TotalMilliseconds;
        LastStats.RigidStats = _rigidWorld.LastStats;
        LastStats.AverageTemperature = SampleAverageTemperature(_field);
        return steps > 0 || dt <= 0f;
    }

    /// <summary>Converts a fraction of rigid-body KE into field temperature (weak coupling).</summary>
    private void InjectKineticHeating()
    {
        float ke = _rigidWorld.ComputeKineticEnergy();
        if (ke < 1e-3f) return;
        float heat = MathF.Min(5f, ke * 0.001f);
        int g = _field.GridSize;
        int cx = g / 2, cy = Math.Clamp(g / 2 + 1, 0, g - 1), cz = g / 2;
        _field.Temperature[cx, cy, cz] += heat;
    }

    private static float SampleAverageTemperature(PhysicsField field)
    {
        float sum = 0f;
        int n = 0;
        int step = Math.Max(1, field.GridSize / 4);
        for (int z = 0; z < field.GridSize; z += step)
            for (int y = 0; y < field.GridSize; y += step)
                for (int x = 0; x < field.GridSize; x += step)
                {
                    sum += field.Temperature[x, y, z];
                    n++;
                }
        return n == 0 ? 0f : sum / n;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _elasticity?.Dispose();
        _sph = null;
    }
}

/// <summary>Lightweight entity descriptor for physics ↔ scene bridging.</summary>
public struct PhysicsEntityDesc
{
    public Guid Id;
    public string Name;
    public string Type;
    public Vector3 Position;
    public Vector3 Scale;
    public bool IsStatic;
    public float Mass;
    public float Restitution;
    public ColliderShape Collider;
}
