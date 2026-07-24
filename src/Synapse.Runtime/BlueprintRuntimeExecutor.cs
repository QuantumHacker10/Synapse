using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime;

/// <summary>Executes blueprint graphs at simulation runtime with execution budgets.</summary>
public sealed class BlueprintRuntimeExecutor
{
    public const int DefaultMaxNodesPerTick = 32;
    public const int DefaultMaxSpawnsPerGraph = 16;
    public const int DefaultMaxLlmCallsPerGraph = 4;

    private readonly EngineHost _host;
    private readonly ISynapseLogger _logger;
    private readonly Dictionary<Guid, int> _nodeIndex = new();
    private BlueprintDocument? _document;
    private Guid _currentNodeId;
    private bool _running;
    private int _nodesThisTick;
    private int _spawnCount;
    private int _llmCallCount;
    private DateTimeOffset _lastLlmCall = DateTimeOffset.MinValue;
    private string? _lastError;

    public BlueprintRuntimeExecutor(EngineHost host, ISynapseLogger logger)
    {
        _host = host;
        _logger = logger;
    }

    public bool IsRunning => _running;
    public string? LastError => _lastError;
    public int MaxNodesPerTick { get; set; } = DefaultMaxNodesPerTick;
    public int MaxSpawnsPerGraph { get; set; } = DefaultMaxSpawnsPerGraph;
    public int MaxLlmCallsPerGraph { get; set; } = DefaultMaxLlmCallsPerGraph;
    public TimeSpan LlmCooldown { get; set; } = TimeSpan.FromSeconds(2);

    public void Load(BlueprintDocument document)
    {
        ArgumentNullException.ThrowIfNull(document);
        var (ok, msg) = document.Validate();
        if (!ok)
            throw new InvalidOperationException(msg);

        _document = document;
        _nodeIndex.Clear();
        for (int i = 0; i < document.Nodes.Count; i++)
            _nodeIndex[document.Nodes[i].Id] = i;

        var entry = document.Nodes.First(n => n.Kind == BlueprintNodeKind.Entry);
        _currentNodeId = entry.Id;
        _running = true;
        _spawnCount = 0;
        _llmCallCount = 0;
        _lastError = null;
    }

    public void Stop() => _running = false;

    public async Task TickAsync(float deltaTime, CancellationToken cancellationToken = default)
    {
        if (!_running || _document == null)
            return;

        _nodesThisTick = 0;
        while (_running && _nodesThisTick < MaxNodesPerTick)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var node = _document.Nodes.FirstOrDefault(n => n.Id == _currentNodeId);
            if (node == null)
            {
                Fail("Current blueprint node missing.");
                return;
            }

            if (node.Kind == BlueprintNodeKind.Exit)
            {
                _running = false;
                return;
            }

            await ExecuteNodeAsync(node, deltaTime, cancellationToken).ConfigureAwait(false);
            _nodesThisTick++;

            var next = GetNextNodeId(node.Id);
            if (next == null)
            {
                Fail($"Node '{node.Title}' has no outgoing edge.");
                return;
            }

            if (next.Value == _currentNodeId)
            {
                Fail($"Node '{node.Title}' forms a self-loop.");
                return;
            }

            _currentNodeId = next.Value;
            if (_document.Nodes.FirstOrDefault(n => n.Id == _currentNodeId)?.Kind == BlueprintNodeKind.Exit)
            {
                _running = false;
                return;
            }
        }
    }

    private void Fail(string message)
    {
        _lastError = message;
        _running = false;
        _logger.Warn("Blueprint", message);
    }

    private async Task ExecuteNodeAsync(BlueprintNode node, float deltaTime, CancellationToken ct)
    {
        switch (node.Kind)
        {
            case BlueprintNodeKind.Entry:
                break;
            case BlueprintNodeKind.LawApply:
                _host.ApplyLaw(node.Payload ?? "heat_equation");
                break;
            case BlueprintNodeKind.EvolveStep:
                _logger.Debug("Blueprint", "EvolveStep requested (use StartEvolutionAsync for full runs)");
                break;
            case BlueprintNodeKind.SpawnAgent:
                if (_spawnCount >= MaxSpawnsPerGraph)
                {
                    Fail($"SpawnAgent budget exceeded ({MaxSpawnsPerGraph}).");
                    return;
                }
                var pos = new System.Numerics.Vector3(0, 0, 0);
                _host.CompileAndSpawnBlueprint(BlueprintDocument.CreateDefault(), pos);
                _spawnCount++;
                break;
            case BlueprintNodeKind.LlmQuery:
                if (_llmCallCount >= MaxLlmCallsPerGraph)
                {
                    Fail($"LlmQuery budget exceeded ({MaxLlmCallsPerGraph}).");
                    return;
                }
                if (DateTimeOffset.UtcNow - _lastLlmCall < LlmCooldown)
                {
                    _logger.Debug("Blueprint", "LlmQuery skipped (cooldown)");
                    break;
                }
                if (!string.IsNullOrWhiteSpace(node.Payload))
                {
                    await _host.ChatAsync(node.Payload, ct).ConfigureAwait(false);
                    _llmCallCount++;
                    _lastLlmCall = DateTimeOffset.UtcNow;
                }
                break;
            case BlueprintNodeKind.Wait:
                if (float.TryParse(node.Payload, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var wait) &&
                    wait > deltaTime)
                {
                    await Task.Delay(TimeSpan.FromSeconds(Math.Clamp(wait, 0, 1.0)), ct).ConfigureAwait(false);
                }
                break;
            case BlueprintNodeKind.Action:
                _logger.Debug("Blueprint", $"Action: {node.Payload ?? node.Title}");
                break;
            default:
                break;
        }
    }

    private Guid? GetNextNodeId(Guid fromNodeId)
    {
        if (_document == null)
            return null;
        var edge = _document.Edges
            .Where(e => e.FromNodeId == fromNodeId)
            .OrderBy(e => e.FromPin)
            .FirstOrDefault();
        return edge?.ToNodeId;
    }
}
