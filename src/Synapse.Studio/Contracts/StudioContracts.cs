using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Studio.Contracts
{
    [Flags]
    public enum ViewportMode
    {
        Solid = 0,
        Wireframe = 1 << 0,
        Textured = 1 << 1,
        Lit = 1 << 2,
        Unlit = 1 << 3,
        NeuronVisualization = 1 << 10
    }

    public enum ToolMode
    {
        Select, Translate, Rotate, Scale, FlyCamera, Place
    }

    public enum EditorMode
    {
        Standard, LLM, Debug, GenomeEdit, SimulationDesign
    }

    [Flags]
    public enum OverlayMask
    {
        None = 0,
        Grid = 1 << 0,
        Gizmos = 1 << 1,
        Stats = 1 << 2,
        EntityLabels = 1 << 9,
        CollisionVolumes = 1 << 10,
        FrameTiming = 1 << 15,
        All = ~0
    }

    /// <summary>
    /// Scene entity kinds for the simulation atelier.
    /// <see cref="Agent"/> is a sentient inhabitant — not a game Character/NPC.
    /// </summary>
    public enum EntityType
    {
        Unknown, Mesh, Light, Camera, ParticleSystem, Genome, Empty, Agent, Volume
    }

    public enum ComponentType
    {
        Transform, MeshRenderer, Genome, Material, Light, Camera, Collider,
        Rigidbody, BehaviorTree, ParticleSystem, Animation, LOD
    }

    public enum CompilationStatus
    {
        NotStarted, Queued, Compiling, Success, Failed, Cancelled, Warning
    }

    public enum LLMProvider
    {
        OpenAI, Anthropic, Local, Azure, Ollama, Gemini
    }

    public sealed class ViewportCamera
    {
        public Vector3 Position { get; set; } = new(0, 2, 5);
        public Vector3 Front { get; set; } = new(0, 0, -1);
        public Vector3 Up { get; set; } = Vector3.UnitY;
        public float Fov { get; set; } = 60f;
        public float Yaw { get; set; } = -90f;
        public float Pitch { get; set; }
    }

    public sealed class ViewportStats
    {
        public float Fps { get; set; }
        public float FrameTimeMs { get; set; }
        public float TotalTime { get; set; }
        public string QualityPreset { get; set; } = "High";
        public int EntityCount { get; set; }
        public float PhysicsTimeMs { get; set; }
        public float SimulationTimeMs { get; set; }
        public string ActiveLawId { get; set; } = "";
        public float FieldTemperatureAvg { get; set; }
    }

    public sealed class RaycastResult
    {
        public bool Hit { get; set; }
        public Guid EntityId { get; set; }
        public Vector3 Point { get; set; }
        public float Distance { get; set; }
    }

    public sealed class SceneEntity : INotifyPropertyChanged
    {
        private string _name = "Entity";
        private bool _isVisible = true;
        private EntityType _type = EntityType.Empty;

        public Guid Id { get; init; } = Guid.NewGuid();
        public string Name
        {
            get => _name;
            set { _name = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Name))); }
        }
        public EntityType Type
        {
            get => _type;
            set { _type = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Type))); }
        }
        public bool IsVisible
        {
            get => _isVisible;
            set { _isVisible = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsVisible))); }
        }
        public Vector3 Position { get; set; }
        public Vector3 Scale { get; set; } = Vector3.One;
        public string? BehaviorProfile { get; set; }
        public string? GenomeId { get; set; }
        public string? LawId { get; set; }
        public string? MeshPath { get; set; }
        public bool IsVehicle { get; set; }
        public bool BakeNeuralSdf { get; set; }
        public ObservableCollection<ComponentType> Components { get; } = new();
        public event PropertyChangedEventHandler? PropertyChanged;
    }

    public sealed class ChatMessageRecord
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string Role { get; set; } = "user";
        public string Content { get; set; } = "";
        public string Provider { get; set; } = "";
        public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;
    }

    public sealed class CompilationResultInfo
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public CompilationStatus Status { get; set; }
        public IReadOnlyList<string> Errors { get; set; } = Array.Empty<string>();
    }

    public interface IViewportService
    {
        Task InitializeAsync(IntPtr windowHandle, int width, int height);
        Task RenderFrameAsync(CancellationToken cancellationToken = default);
        void Resize(int width, int height);
        void SetCamera(ViewportCamera camera);
        ViewportCamera GetCamera();
        void SetOverlayFlags(OverlayMask flags);
        void SetViewportMode(ViewportMode mode);
        ViewportStats GetStats();
        event EventHandler? FrameRendered;
    }

    public interface ISceneService
    {
        Task<bool> LoadSceneAsync(string filePath, CancellationToken cancellationToken = default);
        Task<bool> SaveSceneAsync(string filePath, CancellationToken cancellationToken = default);
        Task<Guid> CreateEntityAsync(string name, EntityType type = EntityType.Empty);
        Task<bool> DeleteEntityAsync(Guid entityId);
        IReadOnlyList<SceneEntity> GetEntities();
        SceneEntity? GetEntityById(Guid entityId);
        event EventHandler? SceneChanged;
    }

    public interface ICompilationService
    {
        Task<CompilationResultInfo> CompileGenomeAsync(Guid genomeId, CancellationToken cancellationToken = default);
        CompilationStatus GetCompilationStatus(Guid genomeId);
    }

    public interface ILLMConsoleService
    {
        Task<ChatMessageRecord> SendPromptAsync(string prompt, LLMProvider provider, string model, CancellationToken cancellationToken = default);
        IReadOnlyList<ChatMessageRecord> GetHistory();
        void ClearHistory();
    }

    public interface ILawEditorService
    {
        IReadOnlyList<(string Id, string Name, string Expression)> ListLaws();
        Task<CompilationResultInfo> CompileAsync(string lawId, string expression, CancellationToken cancellationToken = default);
        Task ApplyActiveLawAsync(string lawId, CancellationToken cancellationToken = default);
        string? ActiveLawId { get; }
        float AverageTemperature { get; }
    }

    public interface IEvolutionService
    {
        bool IsRunning { get; }
        int CurrentGeneration { get; }
        double BestFitness { get; }
        Task StartAsync(int population, int generations, CancellationToken cancellationToken = default);
        void Cancel();
    }

    public interface ISimulationControlService
    {
        bool IsPlaying { get; }
        int EntityCount { get; }
        Task PlayAsync();
        void Pause();
        Task SpawnAgentAsync(string profile, Vector3 position);
    }
}
