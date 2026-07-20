// ============================================================================
// Synapse Omnia — Physics Solvers
// Complete implementations of electromagnetic, acoustic, thermodynamic,
// chemical, gravitational, lattice-Boltzmann, quantum, elastic, turbulent,
// and multiphysics solvers.
//
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;
public sealed class NBodySolver : IDisposable
{
    private readonly NBodyConfig _cfg;
    private readonly double _G;
    private readonly double _eps2;   // softening squared
    private NBodyParticle[] _bodies;

    // Barnes-Hut tree.
    private BHNode _root;
    private BHNode[] _nodePool;
    private int _nodePoolIdx;

    // Conserved quantities.
    private double _initialEnergy;
    private double[] _initialAngMom; // (Lx, Ly, Lz)

    // Trajectory storage.
    private List<double[]> _trajectories;

    // Radiation power accumulator.
    private double _totalRadiatedEnergy;

    private bool _disposed;

    public int NumBodies => _bodies.Length;
    public ReadOnlySpan<NBodyParticle> Bodies => _bodies;
    public double TotalRadiatedEnergy => _totalRadiatedEnergy;

    public NBodySolver(NBodyConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _G = config.GravitationalConstant;
        _eps2 = config.Softening * config.Softening;

        _bodies = new NBodyParticle[config.NumBodies];
        _initialAngMom = new double[3];

        if (config.UseBarnesHut)
        {
            // Pre-allocate octree node pool (generous size).
            int maxNodes = Math.Max(config.NumBodies * 4, 1024);
            _nodePool = new BHNode[maxNodes];
            for (int i = 0; i < maxNodes; i++)
                _nodePool[i] = new BHNode();
        }

        if (config.RecordTrajectory)
            _trajectories = new List<double[]>();
    }

    /// <summary>
    /// Initialise bodies with the given positions, velocities, and masses.
    /// </summary>
    public void Initialise(double[] x, double[] y, double[] z,
        double[] vx, double[] vy, double[] vz, double[] mass)
    {
        int n = Math.Min(x.Length, _bodies.Length);
        for (int i = 0; i < n; i++)
        {
            _bodies[i] = new NBodyParticle(x[i], y[i], z[i], mass[i])
            {
                Vx = vx[i],
                Vy = vy[i],
                Vz = vz[i]
            };
        }
        _initialEnergy = ComputeTotalEnergy();
        ComputeAngularMomentum().CopyTo(_initialAngMom, 0);
    }

    /// <summary>
    /// Initialise Plummer model: isotropic sphere with scale radius a.
    /// </summary>
    public void InitialisePlummer(double scaleRadius, double totalMass, int seed = 42)
    {
        var rng = new Random(seed);
        int n = _bodies.Length;
        double mi = totalMass / n;

        for (int i = 0; i < n; i++)
        {
            // Rejection sampling for radius from Plummer density profile.
            double r;
            do
            {
                double u = rng.NextDouble();
                r = scaleRadius / Math.Sqrt(Math.Pow(u, -2.0 / 3.0) - 1.0);
            } while (r > 100.0 * scaleRadius);

            // Random point on sphere.
            double cosTheta = 2.0 * rng.NextDouble() - 1.0;
            double sinTheta = Math.Sqrt(1.0 - cosTheta * cosTheta);
            double phi = 2.0 * Math.PI * rng.NextDouble();

            double x = r * sinTheta * Math.Cos(phi);
            double y = r * sinTheta * Math.Sin(phi);
            double z = r * cosTheta;

            // Velocity from Plummer distribution.
            double q = rng.NextDouble();
            double g = q * q * Math.Pow(1.0 + 1.0 / (scaleRadius * scaleRadius), -0.75);
            double vEsc = Math.Sqrt(2.0 * _G * totalMass / scaleRadius);
            double v = vEsc * g * 0.5;

            double cosThetaV = 2.0 * rng.NextDouble() - 1.0;
            double sinThetaV = Math.Sqrt(1.0 - cosThetaV * cosThetaV);
            double phiV = 2.0 * Math.PI * rng.NextDouble();

            _bodies[i] = new NBodyParticle(x, y, z, mi)
            {
                Vx = v * sinThetaV * Math.Cos(phiV),
                Vy = v * sinThetaV * Math.Sin(phiV),
                Vz = v * cosThetaV
            };
        }

        // Remove centre-of-mass velocity.
        double vcmX = 0, vcmY = 0, vcmZ = 0;
        for (int i = 0; i < n; i++)
        {
            vcmX += _bodies[i].Vx * _bodies[i].Mass;
            vcmY += _bodies[i].Vy * _bodies[i].Mass;
            vcmZ += _bodies[i].Vz * _bodies[i].Mass;
        }
        double totalM = totalMass;
        vcmX /= totalM;
        vcmY /= totalM;
        vcmZ /= totalM;
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx -= vcmX;
            _bodies[i].Vy -= vcmY;
            _bodies[i].Vz -= vcmZ;
        }

        _initialEnergy = ComputeTotalEnergy();
        ComputeAngularMomentum().CopyTo(_initialAngMom, 0);
    }

    // -----------------------------------------------------------------------
    //  Direct O(N²) force computation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute gravitational accelerations using direct pairwise summation.
    /// a_i = G Σ_{j≠i} m_j (r_j − r_i) / |r_j − r_i|³
    /// with softening: replace |r|² → |r|² + ε².
    /// </summary>
    public void ComputeForcesDirect()
    {
        int n = _bodies.Length;

        // Zero accelerations.
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Ax = 0;
            _bodies[i].Ay = 0;
            _bodies[i].Az = 0;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dx = _bodies[j].X - _bodies[i].X;
                double dy = _bodies[j].Y - _bodies[i].Y;
                double dz = _bodies[j].Z - _bodies[i].Z;
                double r2 = dx * dx + dy * dy + dz * dz + _eps2;
                double invR = 1.0 / Math.Sqrt(r2);
                double invR3 = invR * invR * invR;

                double fx = _G * dx * invR3;
                double fy = _G * dy * invR3;
                double fz = _G * dz * invR3;

                _bodies[i].Ax += fx * _bodies[j].Mass;
                _bodies[i].Ay += fy * _bodies[j].Mass;
                _bodies[i].Az += fz * _bodies[j].Mass;

                _bodies[j].Ax -= fx * _bodies[i].Mass;
                _bodies[j].Ay -= fy * _bodies[i].Mass;
                _bodies[j].Az -= fz * _bodies[i].Mass;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Barnes-Hut octree
    // -----------------------------------------------------------------------

    private BHNode AllocateNode()
    {
        if (_nodePoolIdx >= _nodePool.Length)
        {
            // Grow pool.
            var bigger = new BHNode[_nodePool.Length * 2];
            for (int i = _nodePool.Length; i < bigger.Length; i++)
                bigger[i] = new BHNode();
            Array.Copy(_nodePool, bigger, _nodePool.Length);
            _nodePool = bigger;
        }
        var node = _nodePool[_nodePoolIdx++];
        node.BodyIndex = -1;
        node.Children = null;
        node.ChildCount = 0;
        node.TotalMass = 0;
        node.Cx = node.Cy = node.Cz = 0;
        node.HalfSize = 0;
        return node;
    }

    /// <summary>
    /// Build the Barnes-Hut octree from the current body positions.
    /// </summary>
    private BHNode BuildTree()
    {
        _nodePoolIdx = 0;

        // Find bounding box.
        double xmin = double.MaxValue, xmax = double.MinValue;
        double ymin = double.MaxValue, ymax = double.MinValue;
        double zmin = double.MaxValue, zmax = double.MinValue;
        for (int i = 0; i < _bodies.Length; i++)
        {
            if (_bodies[i].X < xmin)
                xmin = _bodies[i].X;
            if (_bodies[i].X > xmax)
                xmax = _bodies[i].X;
            if (_bodies[i].Y < ymin)
                ymin = _bodies[i].Y;
            if (_bodies[i].Y > ymax)
                ymax = _bodies[i].Y;
            if (_bodies[i].Z < zmin)
                zmin = _bodies[i].Z;
            if (_bodies[i].Z > zmax)
                zmax = _bodies[i].Z;
        }

        double cx = 0.5 * (xmin + xmax);
        double cy = 0.5 * (ymin + ymax);
        double cz = 0.5 * (zmin + zmax);
        double halfSize = 0.5 * Math.Max(Math.Max(xmax - xmin, ymax - ymin), zmax - zmin) * 1.01;

        var root = AllocateNode();
        root.Cx = cx;
        root.Cy = cy;
        root.Cz = cz;
        root.HalfSize = halfSize;

        for (int i = 0; i < _bodies.Length; i++)
            InsertBody(root, i);

        return root;
    }

    private void InsertBody(BHNode node, int bodyIdx)
    {
        if (node.BodyIndex == -1 && node.ChildCount == 0)
        {
            // Empty node: make it a leaf.
            node.BodyIndex = bodyIdx;
            node.TotalMass = _bodies[bodyIdx].Mass;
            node.Cx = _bodies[bodyIdx].X;
            node.Cy = _bodies[bodyIdx].Y;
            node.Cz = _bodies[bodyIdx].Z;
            return;
        }

        if (node.BodyIndex != -1 && node.ChildCount == 0)
        {
            // Current leaf: split into children.
            int oldBody = node.BodyIndex;
            node.BodyIndex = -1;
            InsertIntoChild(node, oldBody);
            InsertIntoChild(node, bodyIdx);
            return;
        }

        // Internal node: update centre of mass and recurse.
        double totalM = node.TotalMass + _bodies[bodyIdx].Mass;
        node.Cx = (node.Cx * node.TotalMass + _bodies[bodyIdx].X * _bodies[bodyIdx].Mass) / totalM;
        node.Cy = (node.Cy * node.TotalMass + _bodies[bodyIdx].Y * _bodies[bodyIdx].Mass) / totalM;
        node.Cz = (node.Cz * node.TotalMass + _bodies[bodyIdx].Z * _bodies[bodyIdx].Mass) / totalM;
        node.TotalMass = totalM;

        InsertIntoChild(node, bodyIdx);
    }

    private void InsertIntoChild(BHNode node, int bodyIdx)
    {
        double bx = _bodies[bodyIdx].X;
        double by = _bodies[bodyIdx].Y;
        double bz = _bodies[bodyIdx].Z;

        // Determine octant.
        int octant = 0;
        if (bx >= node.Cx)
            octant |= 1;
        if (by >= node.Cy)
            octant |= 2;
        if (bz >= node.Cz)
            octant |= 4;

        if (node.Children == null)
        {
            node.Children = new BHNode[8];
            node.ChildCount = 0;
        }

        if (node.Children[octant] == null)
        {
            node.Children[octant] = AllocateNode();
            double halfChild = node.HalfSize * 0.5;
            node.Children[octant].HalfSize = halfChild;
            node.Children[octant].Cx = node.Cx + ((octant & 1) == 0 ? -halfChild : halfChild);
            node.Children[octant].Cy = node.Cy + ((octant & 2) == 0 ? -halfChild : halfChild);
            node.Children[octant].Cz = node.Cz + ((octant & 4) == 0 ? -halfChild : halfChild);
            node.ChildCount++;
        }

        InsertBody(node.Children[octant], bodyIdx);
    }

    /// <summary>
    /// Compute acceleration on body i using the Barnes-Hut tree.
    /// </summary>
    private void ComputeAccelerationBH(BHNode node, int bodyIdx)
    {
        if (node == null || node.TotalMass == 0)
            return;

        double dx = node.Cx - _bodies[bodyIdx].X;
        double dy = node.Cy - _bodies[bodyIdx].Y;
        double dz = node.Cz - _bodies[bodyIdx].Z;
        double r2 = dx * dx + dy * dy + dz * dz + _eps2;

        // Check if this node is far enough to treat as a single particle.
        bool isOpen = (node.HalfSize * 2) * (node.HalfSize * 2) > _cfg.Theta * _cfg.Theta * r2;

        if (!isOpen || (node.ChildCount == 0 && node.BodyIndex != -1))
        {
            // Treat as point mass.
            if (node.BodyIndex == bodyIdx)
                return; // skip self

            double invR = 1.0 / Math.Sqrt(r2);
            double invR3 = invR * invR * invR;
            double fx = _G * node.TotalMass * dx * invR3;
            double fy = _G * node.TotalMass * dy * invR3;
            double fz = _G * node.TotalMass * dz * invR3;

            _bodies[bodyIdx].Ax += fx;
            _bodies[bodyIdx].Ay += fy;
            _bodies[bodyIdx].Az += fz;
        }
        else
        {
            // Open the node and recurse.
            if (node.Children != null)
            {
                for (int c = 0; c < 8; c++)
                    if (node.Children[c] != null)
                        ComputeAccelerationBH(node.Children[c], bodyIdx);
            }
        }
    }

    /// <summary>
    /// Compute all forces using Barnes-Hut algorithm.
    /// </summary>
    public void ComputeForcesBH()
    {
        for (int i = 0; i < _bodies.Length; i++)
        {
            _bodies[i].Ax = 0;
            _bodies[i].Ay = 0;
            _bodies[i].Az = 0;
        }

        _root = BuildTree();
        for (int i = 0; i < _bodies.Length; i++)
            ComputeAccelerationBH(_root, i);
    }

    // -----------------------------------------------------------------------
    //  Post-Newtonian gravitational radiation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the leading-order (2.5PN) gravitational radiation reaction
    /// force for each body in a binary system.
    /// F_rad = −(32/5) G⁴ m₁² m₂² (m₁+m₂) / (c⁵ r⁵) v_r
    /// where v_r is the relative velocity projected along the separation.
    /// </summary>
    public void ComputeRadiationReaction(int bodyA, int bodyB)
    {
        if (!_cfg.ComputeRadiation)
            return;

        double c5 = Math.Pow(PhysicsConstants.C0, 5);
        ref var a = ref _bodies[bodyA];
        ref var b = ref _bodies[bodyB];

        double dx = b.X - a.X;
        double dy = b.Y - a.Y;
        double dz = b.Z - a.Z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = Math.Sqrt(r2);
        double r5 = r2 * r2 * r;

        double dvx = b.Vx - a.Vx;
        double dvy = b.Vy - a.Vy;
        double dvz = b.Vz - a.Vz;

        // Radial velocity (positive = separating).
        double vr = (dx * dvx + dy * dvy + dz * dvz) / r;

        double M = a.Mass + b.Mass;
        double m1m2 = a.Mass * b.Mass;
        double coeff = -(32.0 / 5.0) * Math.Pow(_G, 4) *
                       m1m2 * m1m2 * M / (c5 * r5);

        // Radiation reaction force along the line connecting the two bodies.
        double frx = coeff * vr * dx / r;
        double fry = coeff * vr * dy / r;
        double frz = coeff * vr * dz / r;

        a.PNx = frx / a.Mass;
        a.PNy = fry / a.Mass;
        a.PNz = frz / a.Mass;

        b.PNx = -frx / b.Mass;
        b.PNy = -fry / b.Mass;
        b.PNz = -frz / b.Mass;

        // Power radiated: P = (32/5) G⁴ m₁² m₂² (m₁+m₂) / (c⁵ r⁵) v_r²
        double power = (32.0 / 5.0) * Math.Pow(_G, 4) *
                       m1m2 * m1m2 * M / (c5 * r5) * vr * vr;
        _totalRadiatedEnergy += power * _cfg.TimeStep;
    }

    /// <summary>
    /// Compute 1PN correction to accelerations for a pair of bodies.
    /// </summary>
    public void Compute1PNCorrection(int bodyA, int bodyB)
    {
        if (_cfg.RadiationOrder < 1)
            return;

        double c2 = PhysicsConstants.C0 * PhysicsConstants.C0;
        ref var a = ref _bodies[bodyA];
        ref var b = ref _bodies[bodyB];

        double dx = b.X - a.X;
        double dy = b.Y - a.Y;
        double dz = b.Z - a.Z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = Math.Sqrt(r2);

        double dvx = b.Vx - a.Vx;
        double dvy = b.Vy - a.Vy;
        double dvz = b.Vz - a.Vz;
        double v2 = dvx * dvx + dvy * dvy + dvz * dvz;

        double M = a.Mass + b.Mass;
        double nu = a.Mass * b.Mass / (M * M); // symmetric mass ratio

        // 1PN potentials.
        double phi = -_G * M / r;
        double v2OverC2 = v2 / c2;

        // 1PN correction factor: 1 + (3+ν) v²/c² − (4+2ν) GM/(rc²) + ...
        double correction = 1.0 + (3.0 + nu) * v2OverC2 -
                            (4.0 + 2.0 * nu) * phi / c2;

        // Apply correction to Newtonian acceleration.
        double fxN = _G * dx / (r2 * r);
        double fyN = _G * dy / (r2 * r);
        double fzN = _G * dz / (r2 * r);

        a.Ax += fxN * (correction - 1.0) * b.Mass;
        a.Ay += fyN * (correction - 1.0) * b.Mass;
        a.Az += fzN * (correction - 1.0) * b.Mass;
    }

    // -----------------------------------------------------------------------
    //  Leapfrog integration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the system by one time-step using leapfrog (kick-drift-kick) integration.
    /// This is a symplectic integrator that conserves energy to high order.
    /// </summary>
    public void StepLeapfrog()
    {
        int n = _bodies.Length;
        double dt = _cfg.TimeStep;
        double halfDt = 0.5 * dt;

        // Compute forces at current positions.
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();

        // Compute PN radiation reaction if enabled.
        if (_cfg.ComputeRadiation)
        {
            // For simplicity, apply radiation to first two bodies (binary).
            if (n >= 2)
            {
                ComputeRadiationReaction(0, 1);
                if (_cfg.RadiationOrder >= 1)
                {
                    Compute1PNCorrection(0, 1);
                    Compute1PNCorrection(1, 0);
                }
            }
        }

        // Half-kick: v(t+dt/2) = v(t) + (dt/2) a(t)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx += halfDt * (_bodies[i].Ax + _bodies[i].PNx);
            _bodies[i].Vy += halfDt * (_bodies[i].Ay + _bodies[i].PNy);
            _bodies[i].Vz += halfDt * (_bodies[i].Az + _bodies[i].PNz);
        }

        // Drift: r(t+dt) = r(t) + dt v(t+dt/2)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X += dt * _bodies[i].Vx;
            _bodies[i].Y += dt * _bodies[i].Vy;
            _bodies[i].Z += dt * _bodies[i].Vz;
        }

        // Compute forces at new positions.
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();

        if (_cfg.ComputeRadiation && n >= 2)
        {
            ComputeRadiationReaction(0, 1);
            if (_cfg.RadiationOrder >= 1)
            {
                Compute1PNCorrection(0, 1);
                Compute1PNCorrection(1, 0);
            }
        }

        // Half-kick: v(t+dt) = v(t+dt/2) + (dt/2) a(t+dt)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx += halfDt * (_bodies[i].Ax + _bodies[i].PNx);
            _bodies[i].Vy += halfDt * (_bodies[i].Ay + _bodies[i].PNy);
            _bodies[i].Vz += halfDt * (_bodies[i].Az + _bodies[i].PNz);
        }

        // Record trajectory if requested.
        if (_cfg.RecordTrajectory && _trajectories != null &&
            _trajectories.Count % _cfg.TrajectoryInterval == 0)
        {
            double[] traj = new double[n * 3];
            for (int i = 0; i < n; i++)
            {
                traj[i * 3] = _bodies[i].X;
                traj[i * 3 + 1] = _bodies[i].Y;
                traj[i * 3 + 2] = _bodies[i].Z;
            }
            _trajectories.Add(traj);
        }
    }

    // -----------------------------------------------------------------------
    //  Runge-Kutta 4th order (alternative integrator)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the system by one time-step using RK4.
    /// </summary>
    public void StepRK4()
    {
        int n = _bodies.Length;
        double dt = _cfg.TimeStep;

        // Store initial state.
        var saved = new NBodyParticle[n];
        for (int i = 0; i < n; i++)
            saved[i] = _bodies[i];

        double[] k1ax = new double[n], k1ay = new double[n], k1az = new double[n];

        // k1
        ComputeForcesOrBH();
        for (int i = 0; i < n; i++)
        {
            k1ax[i] = _bodies[i].Ax;
            k1ay[i] = _bodies[i].Ay;
            k1az[i] = _bodies[i].Az;
        }

        // k2: advance to midpoint using k1
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + 0.5 * dt * saved[i].Vx;
            _bodies[i].Y = saved[i].Y + 0.5 * dt * saved[i].Vy;
            _bodies[i].Z = saved[i].Z + 0.5 * dt * saved[i].Vz;
            _bodies[i].Vx = saved[i].Vx + 0.5 * dt * k1ax[i];
            _bodies[i].Vy = saved[i].Vy + 0.5 * dt * k1ay[i];
            _bodies[i].Vz = saved[i].Vz + 0.5 * dt * k1az[i];
        }
        ComputeForcesOrBH();
        double[] k2ax = new double[n], k2ay = new double[n], k2az = new double[n];
        double[] k2vx = new double[n], k2vy = new double[n], k2vz = new double[n];
        for (int i = 0; i < n; i++)
        {
            k2ax[i] = _bodies[i].Ax;
            k2ay[i] = _bodies[i].Ay;
            k2az[i] = _bodies[i].Az;
            k2vx[i] = _bodies[i].Vx;
            k2vy[i] = _bodies[i].Vy;
            k2vz[i] = _bodies[i].Vz;
        }

        // k3
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + 0.5 * dt * k2vx[i];
            _bodies[i].Y = saved[i].Y + 0.5 * dt * k2vy[i];
            _bodies[i].Z = saved[i].Z + 0.5 * dt * k2vz[i];
            _bodies[i].Vx = saved[i].Vx + 0.5 * dt * k2ax[i];
            _bodies[i].Vy = saved[i].Vy + 0.5 * dt * k2ay[i];
            _bodies[i].Vz = saved[i].Vz + 0.5 * dt * k2az[i];
        }
        ComputeForcesOrBH();
        double[] k3ax = new double[n], k3ay = new double[n], k3az = new double[n];
        double[] k3vx = new double[n], k3vy = new double[n], k3vz = new double[n];
        for (int i = 0; i < n; i++)
        {
            k3ax[i] = _bodies[i].Ax;
            k3ay[i] = _bodies[i].Ay;
            k3az[i] = _bodies[i].Az;
            k3vx[i] = _bodies[i].Vx;
            k3vy[i] = _bodies[i].Vy;
            k3vz[i] = _bodies[i].Vz;
        }

        // k4
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + dt * k3vx[i];
            _bodies[i].Y = saved[i].Y + dt * k3vy[i];
            _bodies[i].Z = saved[i].Z + dt * k3vz[i];
            _bodies[i].Vx = saved[i].Vx + dt * k3ax[i];
            _bodies[i].Vy = saved[i].Vy + dt * k3ay[i];
            _bodies[i].Vz = saved[i].Vz + dt * k3az[i];
        }
        ComputeForcesOrBH();
        double[] k4ax = new double[n], k4ay = new double[n], k4az = new double[n];
        for (int i = 0; i < n; i++)
        {
            k4ax[i] = _bodies[i].Ax;
            k4ay[i] = _bodies[i].Ay;
            k4az[i] = _bodies[i].Az;
        }

        // Combine.
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx = saved[i].Vx + dt / 6.0 *
                (k1ax[i] + 2 * k2ax[i] + 2 * k3ax[i] + k4ax[i]);
            _bodies[i].Vy = saved[i].Vy + dt / 6.0 *
                (k1ay[i] + 2 * k2ay[i] + 2 * k3ay[i] + k4ay[i]);
            _bodies[i].Vz = saved[i].Vz + dt / 6.0 *
                (k1az[i] + 2 * k2az[i] + 2 * k3az[i] + k4az[i]);
            _bodies[i].X = saved[i].X + dt / 6.0 *
                (saved[i].Vx + 2 * k2vx[i] + 2 * k3vx[i] + _bodies[i].Vx);
            _bodies[i].Y = saved[i].Y + dt / 6.0 *
                (saved[i].Vy + 2 * k2vy[i] + 2 * k3vy[i] + _bodies[i].Vy);
            _bodies[i].Z = saved[i].Z + dt / 6.0 *
                (saved[i].Vz + 2 * k2vz[i] + 2 * k3vz[i] + _bodies[i].Vz);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeForcesOrBH()
    {
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();
    }

    // -----------------------------------------------------------------------
    //  Conserved quantities
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute total energy: KE + PE.
    /// KE = 0.5 Σ m_i |v_i|²
    /// PE = −G Σ_{i&lt;j} m_i m_j / r_ij
    /// </summary>
    public double ComputeTotalEnergy()
    {
        double ke = 0, pe = 0;
        int n = _bodies.Length;
        for (int i = 0; i < n; i++)
        {
            double v2 = _bodies[i].Vx * _bodies[i].Vx +
                        _bodies[i].Vy * _bodies[i].Vy +
                        _bodies[i].Vz * _bodies[i].Vz;
            ke += 0.5 * _bodies[i].Mass * v2;

            for (int j = i + 1; j < n; j++)
            {
                double dx = _bodies[i].X - _bodies[j].X;
                double dy = _bodies[i].Y - _bodies[j].Y;
                double dz = _bodies[i].Z - _bodies[j].Z;
                double r = Math.Sqrt(dx * dx + dy * dy + dz * dz + _eps2);
                pe -= _G * _bodies[i].Mass * _bodies[j].Mass / r;
            }
        }
        return ke + pe;
    }

    /// <summary>
    /// Compute total angular momentum: L = Σ m_i r_i × v_i.
    /// </summary>
    public double[] ComputeAngularMomentum()
    {
        double lx = 0, ly = 0, lz = 0;
        for (int i = 0; i < _bodies.Length; i++)
        {
            double m = _bodies[i].Mass;
            lx += m * (_bodies[i].Y * _bodies[i].Vz - _bodies[i].Z * _bodies[i].Vy);
            ly += m * (_bodies[i].Z * _bodies[i].Vx - _bodies[i].X * _bodies[i].Vz);
            lz += m * (_bodies[i].X * _bodies[i].Vy - _bodies[i].Y * _bodies[i].Vx);
        }
        return new[] { lx, ly, lz };
    }

    /// <summary>
    /// Compute total linear momentum: P = Σ m_i v_i.
    /// </summary>
    public double[] ComputeLinearMomentum()
    {
        double px = 0, py = 0, pz = 0;
        for (int i = 0; i < _bodies.Length; i++)
        {
            px += _bodies[i].Mass * _bodies[i].Vx;
            py += _bodies[i].Mass * _bodies[i].Vy;
            pz += _bodies[i].Mass * _bodies[i].Vz;
        }
        return new[] { px, py, pz };
    }

    /// <summary>
    /// Energy conservation error: (E − E₀) / |E₀|.
    /// </summary>
    public double EnergyError()
    {
        double e = ComputeTotalEnergy();
        return Math.Abs(e - _initialEnergy) / Math.Max(Math.Abs(_initialEnergy), 1e-30);
    }

    /// <summary>
    /// Angular momentum conservation error: |L − L₀| / |L₀|.
    /// </summary>
    public double AngularMomentumError()
    {
        double[] L = ComputeAngularMomentum();
        double magL = Math.Sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        double magL0 = Math.Sqrt(
            _initialAngMom[0] * _initialAngMom[0] +
            _initialAngMom[1] * _initialAngMom[1] +
            _initialAngMom[2] * _initialAngMom[2]);
        return Math.Abs(magL - magL0) / Math.Max(magL0, 1e-30);
    }

    /// <summary>
    /// Get stored trajectories (if RecordTrajectory is enabled).
    /// </summary>
    public IReadOnlyList<double[]> Trajectories => _trajectories;

    /// <summary>
    /// Run the simulation for the configured number of steps using leapfrog.
    /// </summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            StepLeapfrog();
    }

    /// <summary>
    /// Find the two closest bodies (useful for binary detection).
    /// </summary>
    public (int I, int J, double Distance) FindClosestPair()
    {
        double minDist = double.MaxValue;
        int bestI = 0, bestJ = 1;
        for (int i = 0; i < _bodies.Length; i++)
        {
            for (int j = i + 1; j < _bodies.Length; j++)
            {
                double dx = _bodies[i].X - _bodies[j].X;
                double dy = _bodies[i].Y - _bodies[j].Y;
                double dz = _bodies[i].Z - _bodies[j].Z;
                double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (d < minDist)
                {
                    minDist = d;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
        return (bestI, bestJ, minDist);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  6. LatticeBoltzmannSolver — D2Q9, D3Q19, BGK, MRT, multiphase
// ============================================================================

/// <summary>
/// Lattice type for the Boltzmann solver.
/// </summary>
