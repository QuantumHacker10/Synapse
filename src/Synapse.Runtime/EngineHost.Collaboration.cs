using System;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Network;
using Synapse.VR;
using Synapse.Web;

namespace Synapse.Runtime;

/// <summary>
/// First-class VR / WAN / Web Studio surfaces owned by <see cref="EngineHost"/>.
/// Driven each frame by <see cref="FrameOrchestrator"/> when enabled.
/// </summary>
public sealed partial class EngineHost
{
    private IVrSession? _vrSession;
    private WanSimulationPeerHub? _wanHub;
    private long _wanPatchSequence;
    private long _wanPatchesSent;
    private long _wanPatchesReceived;
    private int _wanSyncCounter;
    private string _vrStatus = "VR : off";
    private string _wanStatus = "WAN : off";
    private string _webStatus = "Web : prêt";
    private readonly object _collaborationGate = new();

    /// <summary>Active OpenXR session when VR is enabled.</summary>
    public IVrSession? VrSession => _vrSession;

    /// <summary>Active WAN collaboration hub when a session is open.</summary>
    public WanSimulationPeerHub? WanHub => _wanHub;

    public bool IsVrActive => _vrSession is { IsAvailable: true, IsRunning: true };
    public bool IsWanConnected => _wanHub != null;
    public string VrStatusText => _vrStatus;
    public string WanStatusText => _wanStatus;
    public string WebStatusText => _webStatus;
    public long WanPatchesSent => Interlocked.Read(ref _wanPatchesSent);
    public long WanPatchesReceived => Interlocked.Read(ref _wanPatchesReceived);

    /// <summary>Raised after a remote WAN patch is applied to the local scene.</summary>
    public event Action<string>? CollaborationPatchApplied;

    /// <summary>
    /// Starts OpenXR (native when possible). Honors <c>SYNAPSE_VR_SIMULATE</c> for lab synthetic mode.
    /// </summary>
    public async Task<bool> EnableVrAsync(int width = 1280, int height = 720, CancellationToken ct = default)
    {
        InitializeModules();
        if (_vrSession != null)
        {
            await _vrSession.DisposeAsync().ConfigureAwait(false);
            _vrSession = null;
        }

        var session = VrSessionFactory.Create(_logger);
        bool ok = await session.TryInitializeAsync(width, height, cancellationToken: ct).ConfigureAwait(false);
        if (!ok)
        {
            await session.DisposeAsync().ConfigureAwait(false);
            _vrStatus = "VR : indisponible (installez un runtime OpenXR ou SYNAPSE_VR_SIMULATE=1)";
            _logger.Warn("VR", _vrStatus);
            return false;
        }

        _vrSession = session;
        _vrStatus = session.UsesNativeOpenXr
            ? $"VR : native ({session.RuntimeName})"
            : $"VR : simulate ({session.RuntimeName})";
        _logger.Info("VR", _vrStatus);
        return true;
    }

    public async Task DisableVrAsync()
    {
        if (_vrSession == null)
        {
            _vrStatus = "VR : off";
            return;
        }

        await _vrSession.DisposeAsync().ConfigureAwait(false);
        _vrSession = null;
        _vrStatus = "VR : off";
    }

    /// <summary>
    /// Hosts a WAN collaboration room (STUN + rendezvous + AES-GCM).
    /// </summary>
    public async Task StartWanHostAsync(string sessionCode, int port = 0, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionCode);
        InitializeModules();
        await StopWanAsync().ConfigureAwait(false);

        var hub = new WanSimulationPeerHub(_logger, sessionCode, rendezvousPort: 0);
        hub.ScenePatchReceived += OnWanPatchReceived;
        // port <= 0 => ephemeral TCP bind (safe for parallel tests / lab).
        int listenPort = port > 0 ? port : 0;
        await hub.StartHostAsync(listenPort, ct).ConfigureAwait(false);
        _wanHub = hub;
        _wanStatus =
            $"WAN host : {sessionCode} tcp={hub.ListenPort} rdv={hub.RendezvousPort} " +
            $"mapped={hub.MappedEndpoint} loopback={hub.IsLoopbackOnly}";
        _logger.Info("Network", _wanStatus);
    }

    /// <summary>
    /// Joins an existing WAN room. When <paramref name="rendezvousPort"/> is 0, uses the hub's
    /// in-process relay (same-host lab). For remote peers, pass the host's rendezvous port.
    /// </summary>
    public async Task JoinWanAsync(string sessionCode, int rendezvousPort = 0, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionCode);
        InitializeModules();
        await StopWanAsync().ConfigureAwait(false);

        var hub = new WanSimulationPeerHub(
            _logger,
            sessionCode,
            rendezvousAddress: IPAddress.Loopback,
            rendezvousPort: rendezvousPort > 0 ? rendezvousPort : 7778,
            hostRelay: false);
        hub.ScenePatchReceived += OnWanPatchReceived;
        await hub.JoinAsync(rendezvousPort > 0 ? rendezvousPort : null, ct).ConfigureAwait(false);
        _wanHub = hub;
        _wanStatus = $"WAN join : {sessionCode} connected (remote rdv={rendezvousPort})";
        _logger.Info("Network", _wanStatus);
    }

    public async Task StopWanAsync()
    {
        if (_wanHub == null)
        {
            _wanStatus = "WAN : off";
            return;
        }

        _wanHub.ScenePatchReceived -= OnWanPatchReceived;
        await _wanHub.DisposeAsync().ConfigureAwait(false);
        _wanHub = null;
        _wanStatus = "WAN : off";
    }

    /// <summary>Publishes the current scene as a Blazor WASM / WebGPU Studio bundle.</summary>
    public async Task<WasmStudioPublisher.PublishResult> ExportWebStudioAsync(
        string outputDirectory,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        InitializeModules();
        var result = await WasmStudioPublisher.PublishAsync(
            outputDirectory,
            _scene.Name,
            _scene.ActiveLawId,
            _scene.Entities.Count,
            _scene.ToJson(),
            ct: ct).ConfigureAwait(false);
        _webStatus = result.UsedDotnetPublish
            ? $"Web : WASM publié → {result.OutputDirectory}"
            : $"Web : site WebGPU → {result.OutputDirectory}";
        _logger.Info("Web", _webStatus);
        return result;
    }

    /// <summary>OpenXR poll + begin-frame (no-op when VR is off).</summary>
    public async Task TickVrBeginAsync(CancellationToken ct = default)
    {
        if (_vrSession is not { IsAvailable: true })
            return;
        await _vrSession.PollEventsAsync(ct).ConfigureAwait(false);
        await _vrSession.BeginFrameAsync(ct).ConfigureAwait(false);
    }

    /// <summary>OpenXR end-frame / swapchain submit (no-op when VR is off).</summary>
    public async Task TickVrEndAsync(CancellationToken ct = default)
    {
        if (_vrSession is not { IsAvailable: true })
            return;
        await _vrSession.EndFrameAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Periodically broadcasts a scene transform patch to WAN peers (about 10 Hz).
    /// </summary>
    public async Task TickCollaborationAsync(CancellationToken ct = default)
    {
        var hub = _wanHub;
        if (hub == null)
            return;

        // Throttle: every 6 ticks ≈ 10 Hz at 60 IPS.
        if (Interlocked.Increment(ref _wanSyncCounter) % 6 != 0)
            return;

        long seq = Interlocked.Increment(ref _wanPatchSequence);
        byte[] patch;
        lock (_collaborationGate)
            patch = ScenePatchCodec.Encode(_scene, seq);

        await hub.BroadcastScenePatchAsync(patch, ct).ConfigureAwait(false);
        Interlocked.Increment(ref _wanPatchesSent);
    }

    /// <summary>
    /// Applies config-driven optional modules (VR / WAN) after <see cref="InitializeModules"/>.
    /// Safe to call from Studio bootstrap or CLI.
    /// </summary>
    public async Task ApplyOptionalCollaborationFromConfigAsync(CancellationToken ct = default)
    {
        if (_config.EnableVr)
            await EnableVrAsync(_config.Width, _config.Height, ct).ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(_config.WanSessionCode))
            return;

        if (_config.WanHost)
        {
            int listen = _config.WanPort > 0 ? _config.WanPort : 0;
            await StartWanHostAsync(_config.WanSessionCode, listen, ct).ConfigureAwait(false);
            return;
        }

        try
        {
            await JoinWanAsync(
                _config.WanSessionCode,
                rendezvousPort: _config.WanRendezvousPort,
                ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // Lab convenience: if join fails with no peer, auto-host the room.
            _logger.Warn("Network", $"WAN join failed ({ex.Message}) — falling back to host mode");
            int listen = _config.WanPort > 0 ? _config.WanPort : 0;
            await StartWanHostAsync(_config.WanSessionCode, listen, ct).ConfigureAwait(false);
        }
    }

    private void OnWanPatchReceived(string peerId, ReadOnlyMemory<byte> encryptedOrPlain)
    {
        if (string.Equals(peerId, "self", StringComparison.Ordinal))
            return;

        try
        {
            byte[] payload = encryptedOrPlain.ToArray();
            // WanSimulationPeerHub delivers encrypted payloads on the wire; decrypt when possible.
            if (_wanHub != null)
            {
                try
                { payload = _wanHub.DecryptPatch(payload); }
                catch { /* may already be plaintext from local echo */ }
            }

            if (!ScenePatchCodec.TryDecode(payload, out var patch) || patch == null)
                return;

            // Ignore our own echo (sequence already known) — still apply peer patches.
            int touched;
            lock (_collaborationGate)
            {
                touched = ScenePatchCodec.Apply(_scene, patch);
                if (touched > 0)
                {
                    ApplySceneToSimulation(_scene);
                    SyncSceneToPhysics();
                    SyncSceneToRenderer();
                }
            }

            if (touched > 0)
            {
                Interlocked.Increment(ref _wanPatchesReceived);
                CollaborationPatchApplied?.Invoke(peerId);
            }
        }
        catch (Exception ex)
        {
            RecordRuntimeError("WAN", ex.Message);
        }
    }

    private async Task DisposeCollaborationAsync()
    {
        await DisableVrAsync().ConfigureAwait(false);
        await StopWanAsync().ConfigureAwait(false);
    }
}
