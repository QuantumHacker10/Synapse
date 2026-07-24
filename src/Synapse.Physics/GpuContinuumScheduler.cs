// =============================================================================
// Scene-scale GPU-friendly continuum (SPH / LBM / FEM-elastic)
// =============================================================================

using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading.Tasks;

namespace Synapse.Physics;

/// <summary>Scene-scale continuum profile for cinematic / industrial budgets.</summary>
public enum ContinuumScale : byte
{
    Demo = 0,
    Industrial = 1,
    Cinematic = 2
}

/// <summary>
/// Batched continuum scheduler: scales SPH / LBM / elasticity to scene size and
/// exposes SoA spans ready for Vulkan compute upload.
/// </summary>
public sealed class GpuContinuumScheduler : IDisposable
{
    private SphSolver? _sph;
    private LatticeBoltzmannSolver? _lbm;
    private ElasticitySolver? _elasticity;
    private bool _disposed;

    public ContinuumScale Scale { get; private set; } = ContinuumScale.Demo;
    public SphSolver? Sph => _sph;
    public LatticeBoltzmannSolver? Lbm => _lbm;
    public ElasticitySolver? Elasticity => _elasticity;
    public float LastSphMs { get; private set; }
    public float LastLbmMs { get; private set; }
    public float LastElasticityMs { get; private set; }
    public int SphParticleCount => _sph?.Particles.Count ?? 0;
    public (int X, int Y, int Z) LbmGrid { get; private set; }
    public (int X, int Y, int Z) ElasticityGrid { get; private set; }

    public void Configure(ContinuumScale scale, float fixedDt)
    {
        Scale = scale;
        var (sphN, lbm, elas, spacing) = scale switch
        {
            ContinuumScale.Cinematic => (8192, (96, 48, 48), (24, 24, 24), 0.06f),
            ContinuumScale.Industrial => (2048, (64, 32, 32), (16, 16, 16), 0.08f),
            _ => (256, (32, 16, 16), (8, 8, 8), 0.1f)
        };

        _lbm?.Dispose();
        _elasticity?.Dispose();

        _sph = new SphSolver(new SphConfig
        {
            NumParticles = sphN,
            TimeStep = MathF.Min(fixedDt, 0.002f),
            SmoothingRadius = scale == ContinuumScale.Cinematic ? 0.08f : 0.1f
        });
        _sph.InitializeCube(-0.8f, 0.1f, -0.8f, spacing);

        _lbm = new LatticeBoltzmannSolver(new LBMConfig
        {
            GridSize = lbm,
            Lattice = LatticeType.D3Q19,
            Collision = CollisionModel.BGK,
            Relaxation = 0.8,
            InletVelocity = 0.04,
            BodyForce = new[] { 0.0, -1e-5, 0.0 }
        });
        LbmGrid = lbm;

        _elasticity = new ElasticitySolver(new ElasticityConfig
        {
            GridSize = elas,
            CellSize = scale == ContinuumScale.Cinematic ? 0.08 : 0.1,
            TimeStep = Math.Max(1e-5, fixedDt),
            EnableContact = true
        });
        ElasticityGrid = elas;
    }

    /// <summary>One batched continuum substep.</summary>
    public void Step(float dt)
    {
        if (_disposed)
            return;

        if (_sph != null)
        {
            var sw = Stopwatch.StartNew();
            _sph.Step(dt);
            LastSphMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        if (_lbm != null)
        {
            var sw = Stopwatch.StartNew();
            _lbm.Step();
            LastLbmMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        if (_elasticity != null)
        {
            var sw = Stopwatch.StartNew();
            _elasticity.Step();
            LastElasticityMs = (float)sw.Elapsed.TotalMilliseconds;
        }
    }

    /// <summary>Exposes SPH positions as a flat XYZ buffer for GPU upload / rendering.</summary>
    public bool TryGetSphPositions(out float[] xyz)
    {
        if (_sph == null)
        {
            xyz = Array.Empty<float>();
            return false;
        }
        int n = _sph.Particles.Count;
        var buf = new float[n * 3];
        var px = _sph.Particles.Px;
        var py = _sph.Particles.Py;
        var pz = _sph.Particles.Pz;
        Parallel.For(0, n, i =>
        {
            buf[i * 3] = px[i];
            buf[i * 3 + 1] = py[i];
            buf[i * 3 + 2] = pz[i];
        });
        xyz = buf;
        return true;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _lbm?.Dispose();
        _elasticity?.Dispose();
        _sph = null;
        _lbm = null;
        _elasticity = null;
    }
}
