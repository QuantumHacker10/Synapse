using System;
using System.ComponentModel;
using System.Numerics;

namespace Synapse.Studio.Contracts
{
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

    public sealed class LawCatalogEntry
    {
        public string Id { get; init; } = "";
        public string Name { get; init; } = "";
        public string Category { get; init; } = "";
        public string Description { get; init; } = "";
        public string Expression { get; init; } = "";

        public string DisplayLine => $"{Id} — {Name}";
        public string CategoryLabel => string.IsNullOrWhiteSpace(Category) ? "—" : Category;
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
        public event PropertyChangedEventHandler? PropertyChanged;
    }

    public sealed class LawCatalogEntry
    {
        public string Id { get; init; } = "";
        public string Name { get; init; } = "";
        public string Category { get; init; } = "";
        public string Description { get; init; } = "";
        public string Expression { get; init; } = "";
        public string DisplayLine => string.IsNullOrWhiteSpace(Name) ? Id : $"{Name} ({Id})";
        public string CategoryLabel => string.IsNullOrWhiteSpace(Category) ? "—" : Category;
    }

    public sealed class ChatMessageRecord
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string Role { get; set; } = "user";
        public string Content { get; set; } = "";
        public string Provider { get; set; } = "";
        public DateTimeOffset Timestamp { get; set; } = DateTimeOffset.UtcNow;
    }

    /// <summary>One row in the Studio live inspector panel (NEAT-G, living laws, milestones).</summary>
    public sealed class LiveInspectorEntry
    {
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;
        public string Category { get; init; } = "";
        public string Title { get; init; } = "";
        public string Detail { get; init; } = "";
        public string DisplayTime => Timestamp.LocalDateTime.ToString("HH:mm:ss");
    }

    public sealed class TwinListEntry
    {
        public Guid Id { get; init; }
        public string PhysicalId { get; init; } = "";
        public string DisplayLine => string.IsNullOrWhiteSpace(PhysicalId) ? Id.ToString("N")[..8] : PhysicalId;
        public string Status { get; init; } = "";
    }

    public sealed class PluginListEntry
    {
        public string Id { get; init; } = "";
        public string Name { get; init; } = "";
        public string Version { get; init; } = "";
        public string DisplayLine => $"{Name} ({Id}) v{Version}";
    }
}
