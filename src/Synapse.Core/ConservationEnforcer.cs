// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

public class ConservationEnforcer
{
    private Vector3D _initialLinearMomentum;
    private Vector3D _initialAngularMomentum;
    private double _initialEnergy;
    private bool _initialized;
    private readonly double _tolerance;
    private int _violationCount;
    private double _maxEnergyDrift;
    private double _maxMomentumDrift;

    public int ViolationCount => _violationCount;
    public double MaxEnergyDrift => _maxEnergyDrift;
    public double MaxMomentumDrift => _maxMomentumDrift;

    public ConservationEnforcer(double tolerance = 1e-10) { _tolerance = tolerance; }

    public void Initialize(Vector3D linearMomentum, Vector3D angularMomentum, double energy)
    {
        _initialLinearMomentum = linearMomentum;
        _initialAngularMomentum = angularMomentum;
        _initialEnergy = energy;
        _initialized = true;
        _violationCount = 0;
        _maxEnergyDrift = 0;
        _maxMomentumDrift = 0;
    }

    public bool CheckEnergy(double currentEnergy, out double drift)
    {
        drift = Math.Abs(currentEnergy - _initialEnergy) / Math.Max(Math.Abs(_initialEnergy), 1.0);
        _maxEnergyDrift = Math.Max(_maxEnergyDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public bool CheckLinearMomentum(Vector3D current, out double drift)
    {
        drift = (current - _initialLinearMomentum).Length() / Math.Max(_initialLinearMomentum.Length(), 1.0);
        _maxMomentumDrift = Math.Max(_maxMomentumDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public bool CheckAngularMomentum(Vector3D current, out double drift)
    {
        drift = (current - _initialAngularMomentum).Length() / Math.Max(_initialAngularMomentum.Length(), 1.0);
        _maxMomentumDrift = Math.Max(_maxMomentumDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public Vector3D CorrectLinearMomentum(Vector3D current, double totalMass)
    {
        if (totalMass < 1e-30)
            return current;
        var correction = (_initialLinearMomentum - current) / totalMass;
        return current + correction;
    }

    public double CorrectEnergy(double currentEnergy) => _initialEnergy;

    public void Reset() { _initialized = false; _violationCount = 0; _maxEnergyDrift = 0; _maxMomentumDrift = 0; }
}

/// <summary>Symmetric positive-definite matrix solver using Cholesky decomposition.</summary>
