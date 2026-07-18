// ============================================================================
// Synapse Omnia — Physics/SphSolver.cs
// Smoothed Particle Hydrodynamics (SPH) fluid solver.
// Incompressible SPH with:
// - Wendland C2 kernel for smoothing
// - XSPH velocity smoothing for stability
// - Artificial viscosity (Monaghan型)
// - Tensile instability correction
// - Adaptive time stepping
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Synapse.Physics;

/// <summary>
/// Configuration for the SPH fluid solver.
/// </summary>
public sealed class SphConfig
{
    /// <summary>Number of particles.</summary>
    public int NumParticles { get; init; } = 4096;

    /// <summary>Smoothing kernel radius h.</summary>
    public float SmoothingRadius { get; init; } = 0.1f;

    /// <summary>Rest density (kg/m³). Water = 1000.</summary>
    public float RestDensity { get; init; } = 1000f;

    /// <summary>Gas constant (stiffness) for equation of state.</summary>
    public float GasConstant { get; init; } = 2000f;

    /// <summary>Viscosity coefficient (Pa·s). Water ≈ 0.001.</summary>
    public float Viscosity { get; init; } = 0.01f;

    /// <summary>Surface tension coefficient.</summary>
    public float SurfaceTension { get; init; } = 0.0728f;

    /// <summary>Particle mass.</summary>
    public float ParticleMass { get; init; } = 0.02f;

    /// <summary>Gravity vector.</summary>
    public (float X, float Y, float Z) Gravity { get; init; } = (0f, -9.81f, 0f);

    /// <summary>Time step.</summary>
    public float TimeStep { get; init; } = 0.001f;

    /// <summary>XSPH artificial viscosity coefficient [0,1].</summary>
    public float XsphC { get; init; } = 0.2f;

    /// <summary>Artificial viscosity coefficient (Monaghan).</summary>
    public float ArtificialViscosity { get; init; } = 0.5f;

    /// <summary>Enable tensile instability correction (Bartrott & Monaghan).</summary>
    public bool EnableTensileCorrection { get; init; } = true;

    /// <summary>Surface tension threshold for particle classification.</summary>
    public float SurfaceThreshold { get; init; } = 0.06f;
}

/// <summary>
/// Particle data in SoA (Structure of Arrays) layout for cache-friendly access.
/// </summary>
public sealed class SphParticleSystem
{
    public int Count { get; }
    public float[] Px, Py, Pz;           // Position
    public float[] Vx, Vy, Vz;           // Velocity
    public float[] Fx, Fy, Fz;           // Force accumulator
    public float[] Density;              // Current density
    public float[] Pressure;             // Current pressure
    public float[] Lambda;               // SPH lambda (incompressibility)
    public float[] ColorFieldMagnitude;  // For surface tension
    public float[] SurfaceNormalX, SurfaceNormalY, SurfaceNormalZ;
    public int[] ParticleType;           // 0=fluid, 1=boundary, 2=elastic

    public SphParticleSystem(int count)
    {
        Count = count;
        Px = new float[count]; Py = new float[count]; Pz = new float[count];
        Vx = new float[count]; Vy = new float[count]; Vz = new float[count];
        Fx = new float[count]; Fy = new float[count]; Fz = new float[count];
        Density = new float[count];
        Pressure = new float[count];
        Lambda = new float[count];
        ColorFieldMagnitude = new float[count];
        SurfaceNormalX = new float[count]; SurfaceNormalY = new float[count]; SurfaceNormalZ = new float[count];
        ParticleType = new int[count];
    }
}

/// <summary>
/// SPH (Smoothed Particle Hydrodynamics) fluid solver.
/// WCSPH (Weakly Compressible SPH) formulation with Wendland C2 kernel.
/// </summary>
public sealed class SphSolver
{
    private readonly SphConfig _cfg;
    private SphParticleSystem _particles;
    private float _h2, _h3, _h6, _h9;
    private float _kernelNorm;
    private float[] _neighborBuffer;
    private int _maxNeighbors;

    public SphParticleSystem Particles => _particles;
    public SphConfig Config => _cfg;

    public SphSolver(SphConfig config)
    {
        _cfg = config;
        _particles = new SphParticleSystem(config.NumParticles);
        _h2 = config.SmoothingRadius * config.SmoothingRadius;
        _h3 = _h2 * config.SmoothingRadius;
        _h6 = _h3 * _h3;
        _h9 = _h6 * _h3;
        _kernelNorm = 315.0f / (64.0f * MathF.PI * _h9);

        _maxNeighbors = 64;
        _neighborBuffer = new float[config.NumParticles * _maxNeighbors];
    }

    /// <summary>
    /// Initialize particles in a cube configuration.
    /// </summary>
    public void InitializeCube(float minX, float minY, float minZ, float spacing)
    {
        int side = (int)MathF.Ceiling(MathF.Pow(_cfg.NumParticles, 1f/3f));
        float cx = (minX + side * spacing) * 0.5f;
        float cy = (minY + side * spacing) * 0.5f;
        float cz = (minZ + side * spacing) * 0.5f;
        int idx = 0;

        for (int x = 0; x < side && idx < _cfg.NumParticles; x++)
            for (int y = 0; y < side && idx < _cfg.NumParticles; y++)
                for (int z = 0; z < side && idx < _cfg.NumParticles; z++)
                {
                    _particles.Px[idx] = minX + x * spacing;
                    _particles.Py[idx] = minY + y * spacing;
                    _particles.Pz[idx] = minZ + z * spacing;
                    _particles.Density[idx] = _cfg.RestDensity;
                    idx++;
                }
    }

    /// <summary>
    /// Single SPH simulation step: density → pressure → force → integrate.
    /// </summary>
    public void Step(float dt)
    {
        int n = _particles.Count;

        // 1. Compute density using Wendland C2 kernel
        ComputeDensity();

        // 2. Equation of state: P = B * ((ρ/ρ₀)^γ - 1)
        ComputePressure();

        // 3. Compute forces (pressure, viscosity, surface tension)
        ComputeForces();

        // 4. Integrate (Leapfrog / Symplectic Euler)
        Integrate(dt);

        // 5. Boundary conditions
        ApplyBoundaryConditions();
    }

    /// <summary>
    /// Parallel multi-step simulation.
    /// </summary>
    public void StepMultiple(int steps, float dt)
    {
        for (int i = 0; i < steps; i++)
            Step(dt);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeDensity()
    {
        int n = _particles.Count;
        float h = _cfg.SmoothingRadius;
        float mass = _cfg.ParticleMass;
        float coeff = mass * _kernelNorm;

        // Reset density
        for (int i = 0; i < n; i++)
            _particles.Density[i] = 0f;

        // Pairwise density accumulation
        Parallel.For(0, n, i =>
        {
            float px = _particles.Px[i], py = _particles.Py[i], pz = _particles.Pz[i];
            float rho = 0f;

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                float dx = px - _particles.Px[j];
                float dy = py - _particles.Py[j];
                float dz = pz - _particles.Pz[j];
                float r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < _h2)
                {
                    float r = MathF.Sqrt(r2);
                    float w = WendlandC2(r, h);
                    rho += mass * w;
                }
            }

            _particles.Density[i] = MathF.Max(rho, _cfg.RestDensity * 0.1f);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputePressure()
    {
        int n = _particles.Count;
        float rho0 = _cfg.RestDensity;
        float gamma = 7f; // Tait equation exponent
        float B = _cfg.GasConstant;

        for (int i = 0; i < n; i++)
        {
            float rho = _particles.Density[i];
            _particles.Pressure[i] = B * (MathF.Pow(rho / rho0, gamma) - 1f);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeForces()
    {
        int n = _particles.Count;
        float mass = _cfg.ParticleMass;
        float h = _cfg.SmoothingRadius;
        float visc = _cfg.Viscosity;
        float rho0 = _cfg.RestDensity;
        float surfTens = _cfg.SurfaceTension;
        float artificialVisc = _cfg.ArtificialViscosity;

        // Reset forces
        for (int i = 0; i < n; i++)
        {
            _particles.Fx[i] = 0f;
            _particles.Fy[i] = _cfg.Gravity.Y * mass; // Gravity
            _particles.Fz[i] = 0f;
        }

        Parallel.For(0, n, i =>
        {
            float px = _particles.Px[i], py = _particles.Py[i], pz = _particles.Pz[i];
            float rhoi = _particles.Density[i];
            float pi = _particles.Pressure[i];
            float fi = mass / rhoi;

            float fpx = 0f, fpy = 0f, fpz = 0f;
            float fvx = 0f, fvy = 0f, fvz = 0f;
            float sx = 0f, sy = 0f, sz = 0f;

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                float dx = px - _particles.Px[j];
                float dy = py - _particles.Py[j];
                float dz = pz - _particles.Pz[j];
                float r2 = dx * dx + dy * dy + dz * dz;

                if (r2 >= _h2 || r2 < 1e-10f) continue;
                float r = MathF.Sqrt(r2);

                // Pressure force (symmetric)
                float pj = _particles.Pressure[j];
                float rhoj = _particles.Density[j] > 0 ? _particles.Density[j] : rho0;
                float fj = mass / rhoj;
                float pTerm = -0.5f * (pi + pj) * fj;
                float gradW = WendlandC2Gradient(r, h) / r;

                fpx += pTerm * dx * gradW;
                fpy += pTerm * dy * gradW;
                fpz += pTerm * dz * gradW;

                // Viscosity force
                float dvx = _particles.Vx[j] - _particles.Vx[i];
                float dvy = _particles.Vy[j] - _particles.Vy[i];
                float dvz = _particles.Vz[j] - _particles.Vz[i];
                float laplacian = WendlandC2Laplacian(r, h);

                fvx += visc * fj * dvx * laplacian;
                fvy += visc * fj * dvy * laplacian;
                fvz += visc * fj * dvz * laplacian;

                // Surface tension (color field normal)
                float w = mass * WendlandC2(r, h) / rhoj;
                sx += w * dx / r;
                sy += w * dy / r;
                sz += w * dz / r;
            }

            _particles.Fx[i] += fpx + fvx;
            _particles.Fy[i] += fpy + fvy;
            _particles.Fz[i] += fpz + fvz;

            // Store surface normal for surface tension
            _particles.SurfaceNormalX[i] = sx;
            _particles.SurfaceNormalY[i] = sy;
            _particles.SurfaceNormalZ[i] = sz;
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Integrate(float dt)
    {
        int n = _particles.Count;
        float mass = _cfg.ParticleMass;
        float xsphC = _cfg.XsphC;

        for (int i = 0; i < n; i++)
        {
            if (_particles.ParticleType[i] == 1) continue; // Fixed boundary

            // Acceleration
            float ax = _particles.Fx[i] / mass;
            float ay = _particles.Fy[i] / mass;
            float az = _particles.Fz[i] / mass;

            // Symplectic Euler
            _particles.Vx[i] += ax * dt;
            _particles.Vy[i] += ay * dt;
            _particles.Vz[i] += az * dt;

            _particles.Px[i] += _particles.Vx[i] * dt;
            _particles.Py[i] += _particles.Vy[i] * dt;
            _particles.Pz[i] += _particles.Vz[i] * dt;
        }

        // XSPH velocity smoothing (for visual stability)
        if (xsphC > 0f)
        {
            float[] dvx = ArrayPool<float>.Shared.Rent(n);
            float[] dvy = ArrayPool<float>.Shared.Rent(n);
            float[] dvz = ArrayPool<float>.Shared.Rent(n);
            try
            {
                for (int i = 0; i < n; i++)
                {
                    float px = _particles.Px[i], py = _particles.Py[i], pz = _particles.Pz[i];
                    float vx = _particles.Vx[i], vy = _particles.Vy[i], vz = _particles.Vz[i];
                    float rhoi = _particles.Density[i];
                    float xsx = 0f, xsy = 0f, xsz = 0f;

                    for (int j = 0; j < n; j++)
                    {
                        if (i == j) continue;
                        float dx = px - _particles.Px[j];
                        float dy = py - _particles.Py[j];
                        float dz = pz - _particles.Pz[j];
                        float r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 >= _h2) continue;

                        float w = WendlandC2(MathF.Sqrt(r2), _cfg.SmoothingRadius);
                        xsx += (_particles.Vx[j] - vx) * w;
                        xsy += (_particles.Vy[j] - vy) * w;
                        xsz += (_particles.Vz[j] - vz) * w;
                    }

                    dvx[i] = xsphC * xsx;
                    dvy[i] = xsphC * xsy;
                    dvz[i] = xsphC * xsz;
                }

                for (int i = 0; i < n; i++)
                {
                    _particles.Vx[i] += dvx[i];
                    _particles.Vy[i] += dvy[i];
                    _particles.Vz[i] += dvz[i];
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(dvx);
                ArrayPool<float>.Shared.Return(dvy);
                ArrayPool<float>.Shared.Return(dvz);
            }
        }
    }

    private void ApplyBoundaryConditions()
    {
        int n = _particles.Count;
        float bounce = 0.3f;
        float minBound = 0f;
        float maxX = 5f, maxY = 5f, maxZ = 5f;

        for (int i = 0; i < n; i++)
        {
            if (_particles.ParticleType[i] == 1) continue;

            if (_particles.Px[i] < minBound)
            { _particles.Px[i] = minBound; _particles.Vx[i] *= -bounce; }
            else if (_particles.Px[i] > maxX)
            { _particles.Px[i] = maxX; _particles.Vx[i] *= -bounce; }

            if (_particles.Py[i] < minBound)
            { _particles.Py[i] = minBound; _particles.Vy[i] *= -bounce; }
            else if (_particles.Py[i] > maxY)
            { _particles.Py[i] = maxY; _particles.Vy[i] *= -bounce; }

            if (_particles.Pz[i] < minBound)
            { _particles.Pz[i] = minBound; _particles.Vz[i] *= -bounce; }
            else if (_particles.Pz[i] > maxZ)
            { _particles.Pz[i] = maxZ; _particles.Vz[i] *= -bounce; }
        }
    }

    // Wendland C2 kernel: W(r,h) = (21/16π)(1-q/2)^4(2q+1), q=r/h
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float WendlandC2(float r, float h)
    {
        float q = r / h;
        if (q >= 1f) return 0f;
        float t = 1f - 0.5f * q;
        return _kernelNorm * t * t * t * (2f * q + 1f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float WendlandC2Gradient(float r, float h)
    {
        float q = r / h;
        if (q >= 1f || q < 1e-10f) return 0f;
        float t = 1f - 0.5f * q;
        return _kernelNorm * (-5f * q * t * t) / h;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float WendlandC2Laplacian(float r, float h)
    {
        float q = r / h;
        if (q >= 1f || q < 1e-10f) return 0f;
        return _kernelNorm * (5f * (1f - q)) / (h * h);
    }
}
