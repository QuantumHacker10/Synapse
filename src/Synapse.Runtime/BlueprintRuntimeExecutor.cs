using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime;

/// <summary>Executes blueprint graphs at simulation runtime (v2).</summary>
public sealed class BlueprintRuntimeExecutor
{
    private readonly EngineHost _host;
    private readonly ISynapseLogger _logger;
    private readonly Dictionary<Guid, int> _nodeIndex = new();
    private BlueprintDocument? _document;
    private Guid _currentNodeId;
    private bool _running;

    public BlueprintRuntimeExecutor(EngineHost host, ISynapseLogger logger)
    {
        _host = host;
        _logger = logger;
    }

    public bool IsRunning => _running;

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
    }

    public void Stop() => _running = false;

    public async Task TickAsync(float deltaTime, CancellationToken cancellationToken = default)
    {
        if (!_running || _document == null)
            return;

        var node = _document.Nodes.FirstOrDefault(n => n.Id == _currentNodeId);
        if (node == null)
        {
            _running = false;
            return;
        }

        if (node.Kind == BlueprintNodeKind.Exit)
        {
            _running = false;
            return;
        }

        await ExecuteNodeAsync(node, deltaTime, cancellationToken).ConfigureAwait(false);
        _currentNodeId = GetNextNodeId(node.Id) ?? _currentNodeId;

        if (_document.Nodes.FirstOrDefault(n => n.Id == _currentNodeId)?.Kind == BlueprintNodeKind.Exit)
            _running = false;
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
                var pos = new System.Numerics.Vector3(0, 0, 0);
                _host.CompileAndSpawnBlueprint(BlueprintDocument.CreateDefault(), pos);
                break;
            case BlueprintNodeKind.LlmQuery:
                if (!string.IsNullOrWhiteSpace(node.Payload))
                    await _host.ChatAsync(node.Payload, ct).ConfigureAwait(false);
                break;
            case BlueprintNodeKind.Wait:
                if (float.TryParse(node.Payload, out var wait) && wait > deltaTime)
                    await Task.Delay(TimeSpan.FromSeconds(Math.Min(wait, 1.0)), ct).ConfigureAwait(false);
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
