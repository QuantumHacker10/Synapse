// Simulation → Physics actuator: sentient agents deposit heat and impulses.

using System;
using System.Numerics;
using Synapse.Physics;

namespace GDNN.Sentience;

/// <summary>
/// Bridges sentient simulation entities into the multiphysics world so agents
/// can heat the continuum and kick rigid bodies (Simulation → Physics).
/// </summary>
public sealed class PhysicsActuator
{
    private MultiphysicsOrchestrator? _physics;

    public const string HeatKey = "physics.heat";
    public const string ImpulseKey = "physics.impulse";

    public void Bind(MultiphysicsOrchestrator? orchestrator) => _physics = orchestrator;

    public bool IsBound => _physics != null;

    /// <summary>Deposits heat at an entity world position.</summary>
    public void DepositHeat(Vector3 position, float joules) =>
        _physics?.DepositHeat(position, joules);

    /// <summary>Applies an impulse near a world position.</summary>
    public bool ApplyImpulse(Vector3 position, Vector3 impulse) =>
        _physics?.ApplyWorldImpulse(position, impulse) ?? false;

    /// <summary>
    /// Per-entity tick: reads property keys <see cref="HeatKey"/> and
    /// <see cref="ImpulseKey"/> then clears them after application.
    /// </summary>
    public void TickEntity(SentientEntity entity)
    {
        if (_physics == null || entity == null)
            return;

        if (TryToFloat(entity[HeatKey], out float heat) && MathF.Abs(heat) > 1e-4f)
        {
            DepositHeat(entity.Position, heat);
            entity[HeatKey] = 0f;
        }

        var impulseObj = entity[ImpulseKey];
        if (impulseObj is Vector3 impulse && impulse.LengthSquared() > 1e-6f)
        {
            ApplyImpulse(entity.Position, impulse);
            entity[ImpulseKey] = Vector3.Zero;
        }
        else if (impulseObj is float[] arr && arr.Length >= 3)
        {
            var v = new Vector3(arr[0], arr[1], arr[2]);
            if (v.LengthSquared() > 1e-6f)
            {
                ApplyImpulse(entity.Position, v);
                entity[ImpulseKey] = Array.Empty<float>();
            }
        }
    }

    private static bool TryToFloat(object? value, out float result)
    {
        switch (value)
        {
            case float f:
                result = f;
                return true;
            case double d:
                result = (float)d;
                return true;
            case int i:
                result = i;
                return true;
            case string s when float.TryParse(s, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var parsed):
                result = parsed;
                return true;
            default:
                result = 0f;
                return false;
        }
    }
}
