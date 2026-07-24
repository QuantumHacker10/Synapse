// Omnia Industrial Pipeline — LLM → Physics → Rendering → Simulation cascade.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using GDNN.Llm;
using GDNN.Rendering;
using GDNN.Scene;
using GDNN.Sentience;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;

namespace Synapse.Runtime;

/// <summary>
/// Result of applying a full world delta through the industrial cascade.
/// </summary>
public sealed class WorldDeltaApplyResult
{
    public List<string> Applied { get; } = new();
    public List<string> Warnings { get; } = new();
    public bool Success => Applied.Count > 0;
    public string Summary
    {
        get
        {
            if (Applied.Count == 0 && Warnings.Count == 0)
                return "No industrial world-delta signals found in the reply.";
            var sb = new StringBuilder();
            if (Applied.Count > 0)
                sb.Append("Applied: ").Append(string.Join(", ", Applied));
            if (Warnings.Count > 0)
            {
                if (sb.Length > 0)
                    sb.Append(" | ");
                sb.Append("Warnings: ").Append(string.Join("; ", Warnings));
            }
            return sb.ToString();
        }
    }
}

/// <summary>
/// Industrial pipeline facade: deterministic order
/// <b>LLM parse → Physics (laws/continuum) → Rendering (L-DNN/G-DNN) → Simulation (BT/agents)</b>.
/// </summary>
public sealed class OmniaIndustrialPipeline
{
    private readonly EngineHost _host;
    private readonly ISynapseLogger _logger;
    private readonly PhysicsFieldGiCoupler _fieldCoupler = new();
    private readonly PhysicsActuator _actuator = new();
    private readonly LumenNeuralSurfaceCacheHolder _surfaceCache = new();

    public OmniaIndustrialPipeline(EngineHost host, ISynapseLogger logger)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public PhysicsFieldGiCoupler FieldCoupler => _fieldCoupler;
    public PhysicsActuator Actuator => _actuator;
    public LumenNeuralSurfaceCacheHolder SurfaceCache => _surfaceCache;

    /// <summary>Binds simulation actuators to the live multiphysics orchestrator.</summary>
    public void BindPhysics(MultiphysicsOrchestrator? physics)
    {
        _actuator.Bind(physics);
    }

    /// <summary>
    /// Parses and applies a full LLM world delta in industrial order.
    /// </summary>
    public WorldDeltaApplyResult ApplyWorldDelta(string llmText)
    {
        var result = new WorldDeltaApplyResult();
        if (string.IsNullOrWhiteSpace(llmText))
        {
            result.Warnings.Add("Empty reply");
            return result;
        }

        var delta = StructuredOutputParser.ParseWorldDelta(llmText);
        if (!delta.HasAny)
        {
            // Preserve legacy behavior: try scene then behavior independently.
            string scene = _host.ApplyLlmSceneHints(llmText);
            if (scene.StartsWith("Applied:", StringComparison.Ordinal))
            {
                result.Applied.Add(scene["Applied:".Length..].Trim());
                return result;
            }

            string behavior = _host.ApplyLlmBehaviorHints(llmText);
            if (behavior.Contains("Registered", StringComparison.Ordinal))
            {
                result.Applied.Add(behavior);
                return result;
            }

            result.Warnings.Add(scene);
            return result;
        }

        // 1) LLM → Physics (living laws + continuum modules)
        if (delta.Law != null)
        {
            string lawStatus = _host.ApplyLlmLawHint(delta.Law);
            if (lawStatus.StartsWith("Applied:", StringComparison.Ordinal) ||
                lawStatus.Contains("active", StringComparison.OrdinalIgnoreCase))
                result.Applied.Add(lawStatus.StartsWith("Applied:", StringComparison.Ordinal)
                    ? lawStatus["Applied:".Length..].Trim()
                    : lawStatus);
            else
                result.Warnings.Add(lawStatus);
        }

        // 2) LLM → Rendering (lighting + SDF / G-DNN)
        if (delta.Lighting != null || delta.Sdf != null)
        {
            string sceneStatus = _host.ApplyLlmSceneHints(llmText);
            if (sceneStatus.StartsWith("Applied:", StringComparison.Ordinal))
                result.Applied.Add(sceneStatus["Applied:".Length..].Trim());
            else
                result.Warnings.Add(sceneStatus);
        }

        // 3) LLM → Rendering materials
        if (delta.Material != null)
        {
            string mat = _host.ApplyLlmMaterialHint(delta.Material);
            if (!string.IsNullOrWhiteSpace(mat))
                result.Applied.Add(mat);
        }

        // 4) LLM → Simulation (behavior trees + impulses)
        if (delta.BehaviorNodes is { Count: > 0 })
        {
            string bt = _host.ApplyLlmBehaviorHints(llmText);
            if (bt.Contains("Registered", StringComparison.Ordinal))
                result.Applied.Add(bt);
            else
                result.Warnings.Add(bt);
        }

        if (delta.Impulse != null)
        {
            string impulseStatus = _host.ApplyLlmImpulseHint(delta.Impulse);
            if (!string.IsNullOrWhiteSpace(impulseStatus))
                result.Applied.Add(impulseStatus);
        }

        _logger.Info("OmniaPipeline", result.Summary);
        return result;
    }

    /// <summary>
    /// Per-frame Physics → Rendering coupling + Simulation → Physics actuator drain.
    /// Called from <see cref="FrameOrchestrator"/> between physics and render.
    /// </summary>
    public void TickCoupling(float dt)
    {
        _ = dt;
        var physics = _host.Multiphysics;
        if (physics == null)
            return;

        BindPhysics(physics);

        var field = physics.Field;
        _fieldCoupler.IngestField(
            field.Temperature.Data,
            field.Density.Data,
            field.GridSize,
            field.GridSize,
            field.GridSize);

        _host.ApplyPhysicsFieldToRenderer(_fieldCoupler);

        // Simulation → Physics: drain agent property actuators.
        var sentience = _host.Sentience;
        if (sentience != null)
        {
            foreach (var entity in sentience.GetAllEntities())
                _actuator.TickEntity(entity);
        }

        _surfaceCache.Cache.TemporalDecay(0.985f);
    }
}

/// <summary>Process-lifetime holder for the Lumen Neural 3.0 surface radiance cache.</summary>
public sealed class LumenNeuralSurfaceCacheHolder
{
    public GDNN.Lighting.LDNN.LumenNeural30.SurfaceRadianceCache Cache { get; } =
        new(resolution: 48, origin: new Vector3(-12f, -2f, -12f), cellSize: 0.5f);
}
