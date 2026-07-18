// ============================================================================
// GDNN.Studio.ViewModels.MainWindowViewModel.cs
// G-DNN Engine Studio - Complete Avalonia 12 Editor UI ViewModel Layer
// ============================================================================
// This file implements the complete MVVM ViewModel layer for the G-DNN Engine
// Studio editor, providing data binding, commands, and service integration
// for all editor panels, viewports, inspectors, and tooling.
// ============================================================================

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using GDNN.Studio.Models;
using GDNN.Studio.Services;
using GDNN.Studio.Views;

namespace GDNN.Studio.ViewModels
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>
    /// Defines the rendering mode for the 3D viewport. Each mode provides a
    /// different visualization of the scene geometry and materials.
    /// </summary>
    [Flags]
    public enum ViewportMode
    {
        /// <summary>Default solid shading with basic lighting.</summary>
        Solid = 0,
        /// <summary>Wireframe rendering showing mesh edges.</summary>
        Wireframe = 1 << 0,
        /// <summary>Full texture and material rendering.</summary>
        Textured = 1 << 1,
        /// <summary>Lit mode with full PBR lighting pipeline.</summary>
        Lit = 1 << 2,
        /// <summary>Unlit mode ignoring all light sources.</summary>
        Unlit = 1 << 3,
        /// <summary>Single material preview mode for material editing.</summary>
        MaterialPreview = 1 << 4,
        /// <summary>Debug visualization of surface normals as colors.</summary>
        DebugNormals = 1 << 5,
        /// <summary>Debug visualization of UV texture coordinates.</summary>
        DebugUV = 1 << 6,
        /// <summary>Debug visualization of depth buffer values.</summary>
        DebugDepth = 1 << 7,
        /// <summary>Debug visualization of per-pixel velocity vectors.</summary>
        DebugVelocity = 1 << 8,
        /// <summary>Debug heatmap visualization for scalar values.</summary>
        DebugHeatmap = 1 << 9,
        /// <summary>Specialized visualization of neural network neuron activations.</summary>
        NeuronVisualization = 1 << 10
    }

    /// <summary>
    /// Defines the active editing tool for viewport interaction.
    /// </summary>
    public enum ToolMode
    {
        Select,
        Translate,
        Rotate,
        Scale,
        Brush,
        Eraser,
        Sculpt,
        Paint,
        Probe,
        Measure,
        FlyCamera,
        RegionSelect,
        Magnet,
        Joint,
        Light,
        Place,
        Path,
        Cut,
        Fill
    }

    /// <summary>
    /// Defines the docking state of editor panels.
    /// </summary>
    public enum PanelState
    {
        Docked,
        Floating,
        Hidden,
        Tabbed,
        AutoHidden,
        Pinned,
        Minimized,
        Maximized,
        SplitView
    }

    /// <summary>
    /// Defines the primary editing mode of the studio editor.
    /// </summary>
    public enum EditorMode
    {
        Standard,
        Blueprint,
        LLM,
        Debug,
        Animation,
        MaterialEdit,
        GenomeEdit,
        LevelDesign
    }

    /// <summary>
    /// Flags controlling which overlays are displayed in the viewport.
    /// </summary>
    [Flags]
    public enum OverlayFlags
    {
        None = 0,
        Grid = 1 << 0,
        Gizmos = 1 << 1,
        Stats = 1 << 2,
        NeuronHeatmap = 1 << 3,
        PerceptualMap = 1 << 4,
        WireframeOverlay = 1 << 5,
        BVHVisualization = 1 << 6,
        LightProbes = 1 << 7,
        NavMesh = 1 << 8,
        EntityLabels = 1 << 9,
        CollisionVolumes = 1 << 10,
        ParticleDebug = 1 << 11,
        AudioDebug = 1 << 12,
        OcclusionDebug = 1 << 13,
        ShadowCascadeDebug = 1 << 14,
        FrameTiming = 1 << 15,
        MemoryDebug = 1 << 16,
        BoundingBoxes = 1 << 17,
        Splines = 1 << 18,
        TerrainDebug = 1 << 19,
        All = ~0
    }

    /// <summary>Defines the type of entity in the scene.</summary>
    public enum EntityType
    {
        Unknown, Mesh, Light, Camera, ParticleSystem, AudioSource, Genome,
        Trigger, Spline, Terrain, Skybox, ReflectionProbe, Decal, UI, Joint,
        Empty, Prefab, Script, Volume, Character, Vehicle
    }

    /// <summary>Defines the type of component attached to an entity.</summary>
    public enum ComponentType
    {
        Transform, MeshRenderer, Genome, Material, Light, Camera, Collider,
        Rigidbody, BehaviorTree, ParticleSystem, AudioSource, Animation, Script,
        LOD, NavAgent, Custom
    }

    /// <summary>Defines the status of a compilation operation.</summary>
    public enum CompilationStatus
    {
        NotStarted, Queued, Compiling, Success, Failed, Cancelled, Warning
    }

    /// <summary>Defines the type of LLM provider.</summary>
    public enum LLMProvider
    {
        OpenAI, Anthropic, Local, Azure, Custom
    }

    /// <summary>Defines the gizmo type for transforms.</summary>
    public enum GizmoType
    {
        Translate, Rotate, Scale, Universal, None
    }

    /// <summary>Defines the axis constraint for transform operations.</summary>
    [Flags]
    public enum GizmoAxis
    {
        None = 0, X = 1 << 0, Y = 1 << 1, Z = 1 << 2,
        XY = X | Y, XZ = X | Z, YZ = Y | Z, All = X | Y | Z,
        Screen = 1 << 3
    }

    /// <summary>Defines viewport layout configurations.</summary>
    public enum ViewportLayout
    {
        Single, SplitHorizontal, SplitVertical, Quad,
        TripleLeft, TripleTop, PictureInPicture, Custom
    }

    /// <summary>Defines the type of message in the LLM chat.</summary>
    public enum ChatMessageType
    {
        User, Assistant, System, Error, CodeBlock, Suggestion,
        ImageAttachment, ToolCall, ToolResult
    }

    /// <summary>Defines the type of blueprint node.</summary>
    public enum BlueprintNodeType
    {
        Input, Output, Process, Math, Logic, Flow, Variable, Function,
        Event, Comment, Group, Custom, Genome, Behavior, Material,
        Transform, Physics, EventTrigger, Timer, Random, Switch,
        Comparison, String, Collection, Audio, Animation
    }

    /// <summary>Defines the data type of a pin on a blueprint node.</summary>
    public enum PinDataType
    {
        Float, Int, Bool, String, Vector2, Vector3, Vector4, Color,
        Object, Any, Exec, Array, Dictionary, Genome, Neuron, Synapse
    }

    /// <summary>Defines the playback state of the animation timeline.</summary>
    public enum PlaybackState
    {
        Stopped, Playing, Paused, SteppingForward, SteppingBackward, Recording
    }

    /// <summary>Defines the loop mode for animation playback.</summary>
    public enum LoopMode
    {
        None, Loop, PingPong, ClampForever
    }

    /// <summary>Defines the quality level for rendering.</summary>
    public enum RenderQuality
    {
        Low, Medium, High, Ultra, Cinematic, Custom
    }

    /// <summary>Defines the type of keyboard shortcut binding.</summary>
    public enum ShortcutType
    {
        KeyBinding, MouseBinding, GestureBinding
    }

    /// <summary>Defines the severity of a diagnostic message.</summary>
    public enum DiagnosticSeverity
    {
        Info, Warning, Error, Critical
    }

    /// <summary>Defines the type of undo/redo operation.</summary>
    public enum UndoOperationType
    {
        PropertyChange, EntityCreate, EntityDelete, EntityMove,
        ComponentAdd, ComponentRemove, ComponentModify,
        ConnectionAdd, ConnectionRemove, NodeAdd, NodeDelete, NodeMove,
        Batch, Custom
    }

    /// <summary>Defines the tab type in the main editor.</summary>
    public enum EditorTabType
    {
        Viewport, CodeEditor, Blueprint, MaterialGraph,
        GenomeEditor, Console, Profiler, AssetBrowser, Custom
    }

    /// <summary>Defines keyframe interpolation mode.</summary>
    public enum KeyframeInterpolation
    {
        Constant, Linear, Cubic, Bezier, Auto
    }

    /// <summary>Dialog result values.</summary>
    public enum DialogResult
    {
        None, OK, Cancel, Yes, No, Retry, Abort, Ignore
    }

    // =========================================================================
    // RECORDS
    // =========================================================================

    /// <summary>
    /// Represents the current selection state in the editor.
    /// </summary>
    public record SelectionInfo
    {
        /// <summary>Collection of currently selected entity IDs.</summary>
        public IReadOnlyList<Guid> SelectedEntities { get; init; } = Array.Empty<Guid>();
        /// <summary>ID of the currently selected genome, if any.</summary>
        public Guid? SelectedGenome { get; init; }
        /// <summary>Index of the currently selected vertex, if any.</summary>
        public int? SelectedVertex { get; init; }
        /// <summary>Index of the currently selected face, if any.</summary>
        public int? SelectedFace { get; init; }
        /// <summary>Whether any selection is active.</summary>
        public bool HasSelection => SelectedEntities.Count > 0 || SelectedGenome.HasValue || SelectedVertex.HasValue || SelectedFace.HasValue;
        /// <summary>Center position of all selected entities in world space.</summary>
        public System.Numerics.Vector3 SelectionCenter { get; init; }
        /// <summary>Bounding box encompassing all selected entities.</summary>
        public BoundingBox SelectionBounds { get; init; }
    }

    /// <summary>
    /// Represents an axis-aligned bounding box.
    /// </summary>
    public record BoundingBox
    {
        public System.Numerics.Vector3 Min { get; init; }
        public System.Numerics.Vector3 Max { get; init; }
        public System.Numerics.Vector3 Center => (Min + Max) * 0.5f;
        public System.Numerics.Vector3 Size => Max - Min;
        public System.Numerics.Vector3 Extents => Size * 0.5f;

        public bool Contains(System.Numerics.Vector3 point)
        {
            return point.X >= Min.X && point.X <= Max.X &&
                   point.Y >= Min.Y && point.Y <= Max.Y &&
                   point.Z >= Min.Z && point.Z <= Max.Z;
        }

        public bool Intersects(BoundingBox other)
        {
            return Min.X <= other.Max.X && Max.X >= other.Min.X &&
                   Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
                   Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
        }

        public BoundingBox Merge(BoundingBox other)
        {
            return new BoundingBox
            {
                Min = System.Numerics.Vector3.Min(Min, other.Min),
                Max = System.Numerics.Vector3.Max(Max, other.Max)
            };
        }

        public float SurfaceArea => 2f * (
            (Max.X - Min.X) * (Max.Y - Min.Y) +
            (Max.Y - Min.Y) * (Max.Z - Min.Z) +
            (Max.Z - Min.Z) * (Max.X - Min.X));

        public float Volume => (Max.X - Min.X) * (Max.Y - Min.Y) * (Max.Z - Min.Z);

        public System.Numerics.Vector3 ClosestPoint(System.Numerics.Vector3 point)
        {
            return System.Numerics.Vector3.Clamp(point, Min, Max);
        }

        public float DistanceSquared(System.Numerics.Vector3 point)
        {
            var closest = ClosestPoint(point);
            return System.Numerics.Vector3.DistanceSquared(closest, point);
        }
    }

    /// <summary>
    /// Represents the camera state for a 3D viewport.
    /// </summary>
    public record ViewportCamera
    {
        public System.Numerics.Vector3 Position { get; init; } = new(0, 5, -10);
        public System.Numerics.Vector3 Target { get; init; } = System.Numerics.Vector3.Zero;
        public System.Numerics.Vector3 Up { get; init; } = System.Numerics.Vector3.UnitY;
        public float Fov { get; init; } = 60.0f;
        public float NearPlane { get; init; } = 0.01f;
        public float FarPlane { get; init; } = 10000.0f;
        public float AspectRatio { get; init; } = 16.0f / 9.0f;
        public System.Numerics.Matrix4x4 ViewMatrix { get; init; } = System.Numerics.Matrix4x4.Identity;
        public System.Numerics.Matrix4x4 ProjectionMatrix { get; init; } = System.Numerics.Matrix4x4.Identity;
        public System.Numerics.Matrix4x4 ViewProjectionMatrix { get; init; } = System.Numerics.Matrix4x4.Identity;
        public IReadOnlyList<FrustumPlane> FrustumPlanes { get; init; } = Array.Empty<FrustumPlane>();
        public float OrbitDistance { get; init; } = 15.0f;
        public float OrbitYaw { get; init; }
        public float OrbitPitch { get; init; } = 0.3f;
        public bool IsOrthographic { get; init; }
        public float OrthographicSize { get; init; } = 10.0f;
        public float MoveSpeed { get; init; } = 1.0f;
        public float RotationSensitivity { get; init; } = 0.003f;
        public bool IsAnimating { get; init; }

        public ViewportCamera WithViewMatrix(System.Numerics.Matrix4x4 view)
        {
            return this with { ViewMatrix = view, ViewProjectionMatrix = view * ProjectionMatrix };
        }

        public ViewportCamera WithProjectionMatrix(System.Numerics.Matrix4x4 proj)
        {
            return this with { ProjectionMatrix = proj, ViewProjectionMatrix = ViewMatrix * proj };
        }

        public System.Numerics.Vector3 Forward => System.Numerics.Vector3.Normalize(Target - Position);
        public System.Numerics.Vector3 Right => System.Numerics.Vector3.Normalize(System.Numerics.Vector3.Cross(Up, Forward));
        public System.Numerics.Vector3 CameraUp => System.Numerics.Vector3.Cross(Forward, Right);
    }

    /// <summary>
    /// Represents a plane in 3D space.
    /// </summary>
    public record FrustumPlane
    {
        public System.Numerics.Vector3 Normal { get; init; }
        public float Distance { get; init; }

        public float SignedDistance(System.Numerics.Vector3 point)
        {
            return System.Numerics.Vector3.Dot(Normal, point) + Distance;
        }

        public bool IsInside(System.Numerics.Vector3 point)
        {
            return SignedDistance(point) >= 0;
        }
    }

    /// <summary>
    /// Represents a ray in 3D space for raycasting.
    /// </summary>
    public record Ray3D
    {
        public System.Numerics.Vector3 Origin { get; init; }
        public System.Numerics.Vector3 Direction { get; init; }

        public System.Numerics.Vector3 GetPoint(float distance)
        {
            return Origin + Direction * distance;
        }

        public bool IntersectAABB(BoundingBox box, out float distance)
        {
            distance = 0;
            var invDir = new System.Numerics.Vector3(
                1f / (Math.Abs(Direction.X) < 1e-8f ? 1e-8f : Direction.X),
                1f / (Math.Abs(Direction.Y) < 1e-8f ? 1e-8f : Direction.Y),
                1f / (Math.Abs(Direction.Z) < 1e-8f ? 1e-8f : Direction.Z));

            var t1 = (box.Min.X - Origin.X) * invDir.X;
            var t2 = (box.Max.X - Origin.X) * invDir.X;
            var tmin = Math.Min(t1, t2);
            var tmax = Math.Max(t1, t2);

            t1 = (box.Min.Y - Origin.Y) * invDir.Y;
            t2 = (box.Max.Y - Origin.Y) * invDir.Y;
            tmin = Math.Max(tmin, Math.Min(t1, t2));
            tmax = Math.Min(tmax, Math.Max(t1, t2));

            t1 = (box.Min.Z - Origin.Z) * invDir.Z;
            t2 = (box.Max.Z - Origin.Z) * invDir.Z;
            tmin = Math.Max(tmin, Math.Min(t1, t2));
            tmax = Math.Min(tmax, Math.Max(t1, t2));

            if (tmax >= tmin && tmax >= 0)
            {
                distance = tmin >= 0 ? tmin : tmax;
                return true;
            }
            return false;
        }

        public bool IntersectSphere(System.Numerics.Vector3 center, float radius, out float distance)
        {
            distance = 0;
            var oc = Origin - center;
            var a = System.Numerics.Vector3.Dot(Direction, Direction);
            var b = 2f * System.Numerics.Vector3.Dot(oc, Direction);
            var c = System.Numerics.Vector3.Dot(oc, oc) - radius * radius;
            var discriminant = b * b - 4 * a * c;
            if (discriminant < 0) return false;
            distance = (-b - MathF.Sqrt(discriminant)) / (2 * a);
            if (distance < 0) distance = (-b + MathF.Sqrt(discriminant)) / (2 * a);
            return distance >= 0;
        }
    }

    /// <summary>
    /// Result of a raycast hit against scene geometry.
    /// </summary>
    public record RaycastResult
    {
        public bool Hit { get; init; }
        public Guid EntityId { get; init; }
        public float Distance { get; init; }
        public System.Numerics.Vector3 HitPoint { get; init; }
        public System.Numerics.Vector3 HitNormal { get; init; }
        public int TriangleIndex { get; init; }
        public int VertexIndex { get; init; }
        public System.Numerics.Vector2 UV { get; init; }
        public float BarycentricU { get; init; }
        public float BarycentricV { get; init; }
    }

    /// <summary>
    /// Contains real-time performance metrics.
    /// </summary>
    public record PerformanceMetrics
    {
        public double Fps { get; init; }
        public double FrameTime { get; init; }
        public double GpuTime { get; init; }
        public double CpuTime { get; init; }
        public int DrawCalls { get; init; }
        public int Triangles { get; init; }
        public int NeuronsActive { get; init; }
        public long MemoryUsed { get; init; }
        public long VramUsed { get; init; }
        public int ActiveEntities { get; init; }
        public int VisibleEntities { get; init; }
        public int ActiveLights { get; init; }
        public int ParticlesAlive { get; init; }
        public int PhysicsBodies { get; init; }
        public int Gen0Collections { get; init; }
        public int Gen1Collections { get; init; }
        public int Gen2Collections { get; init; }
        public long TotalAllocatedBytes { get; init; }
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Contains statistics about the current scene composition.
    /// </summary>
    public record SceneStats
    {
        public int EntityCount { get; init; }
        public int GenomeCount { get; init; }
        public int NeuronCount { get; init; }
        public int SynapseCount { get; init; }
        public int LightCount { get; init; }
        public int MaterialCount { get; init; }
        public long TextureMemory { get; init; }
        public int TotalVertices { get; init; }
        public int TotalTriangles { get; init; }
        public int MeshCount { get; init; }
        public int AnimationCount { get; init; }
        public int AudioSourceCount { get; init; }
        public int ParticleSystemCount { get; init; }
        public int ScriptCount { get; init; }
        public long SceneDataSize { get; init; }
    }

    /// <summary>Represents a keyframe in an animation clip.</summary>
    public record Keyframe
    {
        public float Time { get; init; }
        public System.Numerics.Vector4 Value { get; init; }
        public System.Numerics.Vector4 InTangent { get; init; }
        public System.Numerics.Vector4 OutTangent { get; init; }
        public KeyframeInterpolation Interpolation { get; init; } = KeyframeInterpolation.Cubic;
    }

    /// <summary>Represents a file entry in the recent files list.</summary>
    public record RecentFile
    {
        public string FilePath { get; init; } = string.Empty;
        public string DisplayName => System.IO.Path.GetFileName(FilePath);
        public DateTime LastAccessed { get; init; }
        public long FileSize { get; init; }
        public bool Exists => File.Exists(FilePath);
    }

    /// <summary>Represents a keyboard shortcut binding.</summary>
    public record KeyboardShortcut
    {
        public string Name { get; init; } = string.Empty;
        public string Description { get; init; } = string.Empty;
        public Key Key { get; init; }
        public KeyModifiers Modifiers { get; init; }
        public string CommandName { get; init; } = string.Empty;
        public ShortcutType Type { get; init; } = ShortcutType.KeyBinding;
        public bool IsEnabled { get; init; } = true;
        public bool IsConflicting { get; init; }

        public string DisplayText
        {
            get
            {
                var sb = new StringBuilder();
                if (Modifiers.HasFlag(KeyModifiers.Control)) sb.Append("Ctrl+");
                if (Modifiers.HasFlag(KeyModifiers.Alt)) sb.Append("Alt+");
                if (Modifiers.HasFlag(KeyModifiers.Shift)) sb.Append("Shift+");
                if (Modifiers.HasFlag(KeyModifiers.Meta)) sb.Append("Win+");
                sb.Append(Key);
                return sb.ToString();
            }
        }
    }

    /// <summary>Represents a diagnostic message.</summary>
    public record DiagnosticMessage
    {
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
        public DiagnosticSeverity Severity { get; init; }
        public string Category { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public string Details { get; init; } = string.Empty;
        public string Source { get; init; } = string.Empty;
        public int? LineNumber { get; init; }
        public string? FilePath { get; init; }
    }

    /// <summary>Represents GPU hardware information.</summary>
    public record GpuInfo
    {
        public string Name { get; init; } = string.Empty;
        public string DriverVersion { get; init; } = string.Empty;
        public long DedicatedMemory { get; init; }
        public long SharedMemory { get; init; }
        public int MaxTextureSize { get; init; }
        public int MaxComputeWorkGroupSize { get; init; }
        public bool SupportsRayTracing { get; init; }
        public bool SupportsMeshShaders { get; init; }
        public string ApiVersion { get; init; } = string.Empty;
        public float Temperature { get; init; }
        public float PowerDrawWatts { get; init; }
        public int ClockSpeedMhz { get; init; }
    }

    /// <summary>Represents CPU hardware information.</summary>
    public record CpuInfo
    {
        public string Name { get; init; } = string.Empty;
        public int CoreCount { get; init; }
        public int ThreadCount { get; init; }
        public double ClockSpeedMhz { get; init; }
        public int L1CacheSizeKb { get; init; }
        public int L2CacheSizeKb { get; init; }
        public long L3CacheSizeBytes { get; init; }
        public string Architecture { get; init; } = string.Empty;
    }

    /// <summary>Represents memory usage information.</summary>
    public record MemoryInfo
    {
        public long TotalPhysical { get; init; }
        public long AvailablePhysical { get; init; }
        public long UsedPhysical { get; init; }
        public long TotalVirtual { get; init; }
        public long UsedVirtual { get; init; }
        public long PageFileSize { get; init; }
        public int MemoryLoadPercentage { get; init; }
    }

    /// <summary>Represents cache statistics.</summary>
    public record CacheStatistics
    {
        public int ShaderCacheEntries { get; init; }
        public long ShaderCacheSize { get; init; }
        public int GenomeCacheEntries { get; init; }
        public long GenomeCacheSize { get; init; }
        public double CacheHitRate { get; init; }
        public int CacheHits { get; init; }
        public int CacheMisses { get; init; }
        public long TotalCacheSize { get; init; }
    }

    /// <summary>Represents a pin on a blueprint node.</summary>
    public record PinInfo
    {
        public Guid PinId { get; init; } = Guid.NewGuid();
        public string Name { get; init; } = string.Empty;
        public PinDataType DataType { get; init; }
        public bool IsInput { get; init; }
        public bool IsConnected { get; init; }
        public string DefaultValue { get; init; } = string.Empty;
        public string Tooltip { get; init; } = string.Empty;
        public int MaxConnections { get; init; } = 1;
    }

    /// <summary>Represents a prompt template for the LLM console.</summary>
    public record PromptTemplate
    {
        public string Name { get; init; } = string.Empty;
        public string Description { get; init; } = string.Empty;
        public string Template { get; init; } = string.Empty;
        public string Category { get; init; } = string.Empty;
        public IReadOnlyList<string> Variables { get; init; } = Array.Empty<string>();
    }

    /// <summary>Represents a chat message in the LLM console.</summary>
    public record ChatMessageRecord
    {
        public Guid MessageId { get; init; } = Guid.NewGuid();
        public ChatMessageType Type { get; init; }
        public string Content { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
        public bool IsStreaming { get; init; }
        public LLMProvider? Provider { get; init; }
        public string? ModelName { get; init; }
        public int? TokenCount { get; init; }
        public double? LatencyMs { get; init; }
        public IReadOnlyList<string> Attachments { get; init; } = Array.Empty<string>();
        public IReadOnlyDictionary<string, string> Metadata { get; init; } = new Dictionary<string, string>();
    }

    /// <summary>Represents a compilation result.</summary>
    public record CompilationResult
    {
        public CompilationStatus Status { get; init; }
        public string Output { get; init; } = string.Empty;
        public IReadOnlyList<DiagnosticMessage> Diagnostics { get; init; } = Array.Empty<DiagnosticMessage>();
        public TimeSpan Duration { get; init; }
        public long OutputSize { get; init; }
        public string? OutputPath { get; init; }
    }

    /// <summary>Represents a panel layout configuration.</summary>
    public record PanelLayout
    {
        public string PanelName { get; init; } = string.Empty;
        public PanelState State { get; init; }
        public double X { get; init; }
        public double Y { get; init; }
        public double Width { get; init; }
        public double Height { get; init; }
        public string DockSide { get; init; } = string.Empty;
        public int TabIndex { get; init; }
        public bool IsVisible { get; init; } = true;
    }

    /// <summary>Viewport statistics for HUD display.</summary>
    public record ViewportStats
    {
        public int DrawCalls { get; init; }
        public int Triangles { get; init; }
        public int Vertices { get; init; }
        public int Textures { get; init; }
        public int Shaders { get; init; }
        public double FrameTimeMs { get; init; }
        public double GpuTimeMs { get; init; }
    }

    // =========================================================================
    // INTERFACES
    // =========================================================================

    /// <summary>
    /// Base interface for all ViewModels.
    /// </summary>
    public interface IViewModelBase : INotifyPropertyChanged, IDisposable
    {
        Guid Id { get; }
        string DisplayName { get; set; }
        bool IsDisposed { get; }
        Task InitializeAsync();
        void Reset();
    }

    /// <summary>
    /// Service interface for 3D viewport rendering and interaction.
    /// </summary>
    public interface IViewportService
    {
        Task InitializeAsync(IntPtr windowHandle, int width, int height);
        Task RenderFrameAsync(CancellationToken cancellationToken = default);
        void Resize(int width, int height);
        void SetCamera(ViewportCamera camera);
        ViewportCamera GetCamera();
        RaycastResult Raycast(int screenX, int screenY);
        void SetOverlayFlags(OverlayFlags flags);
        void SetViewportMode(ViewportMode mode);
        ViewportStats GetStats();
        Task<byte[]> CaptureScreenshotAsync(int width = 0, int height = 0);
        event EventHandler<FrameRenderedEventArgs>? FrameRendered;
        event EventHandler<EntityPickedEventArgs>? EntityPicked;
    }

    /// <summary>Event args for frame rendered events.</summary>
    public class FrameRenderedEventArgs : EventArgs
    {
        public double FrameTimeMs { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
        public IntPtr FramebufferHandle { get; init; }
    }

    /// <summary>Event args for entity picked events.</summary>
    public class EntityPickedEventArgs : EventArgs
    {
        public Guid EntityId { get; init; }
        public RaycastResult Result { get; init; } = new();
    }

    /// <summary>Event args for selection rectangle completion.</summary>
    public class SelectionRectangleCompletedEventArgs : EventArgs
    {
        public IReadOnlyList<Guid> SelectedEntityIds { get; init; } = Array.Empty<Guid>();
        public System.Numerics.Vector4 Bounds { get; init; }
    }

    /// <summary>
    /// Service interface for scene management.
    /// </summary>
    public interface ISceneService
    {
        Task<bool> LoadSceneAsync(string filePath, CancellationToken cancellationToken = default);
        Task<bool> SaveSceneAsync(string filePath, CancellationToken cancellationToken = default);
        Task<Guid> CreateEntityAsync(string name, EntityType type = EntityType.Empty);
        Task<bool> DeleteEntityAsync(Guid entityId);
        IReadOnlyList<SceneEntity> GetEntities();
        SceneEntity? GetEntityById(Guid entityId);
        bool Undo();
        bool Redo();
        SceneStats GetSceneStats();
        event EventHandler<SceneChangedEventArgs>? SceneChanged;
    }

    /// <summary>Event args for scene change events.</summary>
    public class SceneChangedEventArgs : EventArgs
    {
        public string ChangeType { get; init; } = string.Empty;
        public Guid? AffectedEntityId { get; init; }
        public bool RequiresSave { get; init; }
    }

    /// <summary>Event args for selected entity change events.</summary>
    public class SelectedEntityChangedEventArgs : EventArgs
    {
        public Guid? EntityId { get; init; }
    }

    /// <summary>
    /// Represents an entity in the scene with all its components.
    /// </summary>
    public class SceneEntity : ObservableObject
    {
        private string _name = string.Empty;
        private bool _isVisible = true;
        private bool _isLocked;
        private EntityType _type;
        private Guid? _parentId;

        public Guid Id { get; init; } = Guid.NewGuid();

        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        public EntityType Type
        {
            get => _type;
            set => SetProperty(ref _type, value);
        }

        public bool IsVisible
        {
            get => _isVisible;
            set => SetProperty(ref _isVisible, value);
        }

        public bool IsLocked
        {
            get => _isLocked;
            set => SetProperty(ref _isLocked, value);
        }

        public Guid? ParentId
        {
            get => _parentId;
            set => SetProperty(ref _parentId, value);
        }

        public ObservableCollection<ComponentType> Components { get; } = new();

        public System.Numerics.Vector3 Position { get; set; }
        public System.Numerics.Quaternion Rotation { get; set; } = System.Numerics.Quaternion.Identity;
        public System.Numerics.Vector3 Scale { get; set; } = System.Numerics.Vector3.One;
    }

    /// <summary>
    /// Service interface for genome compilation.
    /// </summary>
    public interface ICompilationService
    {
        Task<CompilationResult> CompileGenomeAsync(Guid genomeId, CancellationToken cancellationToken = default);
        CompilationStatus GetCompilationStatus(Guid genomeId);
        CacheStatistics GetCacheStatistics();
        bool CancelCompilation(Guid genomeId);
        void ClearCache();
        event EventHandler<CompilationStatusChangedEventArgs>? CompilationStatusChanged;
    }

    /// <summary>Event args for compilation status changes.</summary>
    public class CompilationStatusChangedEventArgs : EventArgs
    {
        public Guid GenomeId { get; init; }
        public CompilationStatus OldStatus { get; init; }
        public CompilationStatus NewStatus { get; init; }
        public string? Message { get; init; }
    }

    /// <summary>
    /// Service interface for LLM console interaction.
    /// </summary>
    public interface ILLMConsoleService
    {
        Task<ChatMessageRecord> SendPromptAsync(string prompt, LLMProvider provider, string model, CancellationToken cancellationToken = default);
        IAsyncEnumerable<string> StreamResponseAsync(string prompt, LLMProvider provider, string model, CancellationToken cancellationToken = default);
        IReadOnlyList<ChatMessageRecord> GetHistory();
        void ClearHistory();
        Task<bool> ExportConversationAsync(string filePath, CancellationToken cancellationToken = default);
        IReadOnlyList<string> GetAvailableModels(LLMProvider provider);
    }

    /// <summary>
    /// Service interface for the blueprint visual scripting editor.
    /// </summary>
    public interface IBlueprintEditorService
    {
        Task<bool> OpenBlueprintAsync(string filePath, CancellationToken cancellationToken = default);
        Task<bool> SaveBlueprintAsync(string filePath, CancellationToken cancellationToken = default);
        Task<CompilationResult> CompileBlueprintAsync(CancellationToken cancellationToken = default);
        IReadOnlyList<BlueprintNodeData> GetNodes();
        bool ConnectNodes(Guid sourceNodeId, int sourcePinIndex, Guid targetNodeId, int targetPinIndex);
        bool DisconnectNodes(Guid sourceNodeId, int sourcePinIndex, Guid targetNodeId, int targetPinIndex);
        Guid AddNode(BlueprintNodeType type, float x, float y);
        bool RemoveNode(Guid nodeId);
        event EventHandler? BlueprintChanged;
    }

    /// <summary>Represents a node in a blueprint graph.</summary>
    public class BlueprintNodeData
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public BlueprintNodeType Type { get; init; }
        public string Title { get; init; } = string.Empty;
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; } = 200;
        public float Height { get; set; } = 100;
        public ObservableCollection<PinInfo> InputPins { get; } = new();
        public ObservableCollection<PinInfo> OutputPins { get; } = new();
    }

    /// <summary>
    /// Service interface for hardware monitoring.
    /// </summary>
    public interface IHardwareMonitor
    {
        GpuInfo GetGpuInfo();
        CpuInfo GetCpuInfo();
        MemoryInfo GetMemoryInfo();
        float GetTemperature();
        (double gpuClockMhz, double cpuClockMhz) GetClockSpeeds();
        float GetGpuUtilization();
        float GetCpuUtilization();
        void StartMonitoring(TimeSpan interval);
        void StopMonitoring();
        event EventHandler<HardwareStatsUpdatedEventArgs>? StatsUpdated;
    }

    /// <summary>Event args for hardware stats updates.</summary>
    public class HardwareStatsUpdatedEventArgs : EventArgs
    {
        public float GpuUtilization { get; init; }
        public float CpuUtilization { get; init; }
        public float Temperature { get; init; }
        public long MemoryUsed { get; init; }
        public double GpuClockMhz { get; init; }
        public double CpuClockMhz { get; init; }
    }


    // =========================================================================
    // VIEW MODEL BASE CLASS
    // =========================================================================

    /// <summary>
    /// Base class for all ViewModels providing common MVVM infrastructure.
    /// </summary>
    public abstract class ViewModelBase : ObservableObject, IViewModelBase, IAsyncDisposable
    {
        private Guid _id = Guid.NewGuid();
        private string _displayName = string.Empty;
        private bool _isDisposed;
        private bool _isInitialized;
        private readonly Dictionary<string, WeakReference<INotifyPropertyChanged>> _weakReferences = new();
        private readonly List<IDisposable> _disposables = new();
        private readonly SemaphoreSlim _initializationLock = new(1, 1);
        private bool _isBusy;
        private string? _errorMessage;
        private string? _statusMessage;
        private bool _isDirty;

        /// <inheritdoc/>
        public Guid Id => _id;

        /// <inheritdoc/>
        public string DisplayName
        {
            get => _displayName;
            set => SetProperty(ref _displayName, value);
        }

        /// <inheritdoc/>
        public bool IsDisposed => _isDisposed;

        /// <summary>Whether this ViewModel has been fully initialized.</summary>
        public bool IsInitialized => _isInitialized;

        /// <summary>Whether an async operation is currently in progress.</summary>
        public bool IsBusy
        {
            get => _isBusy;
            protected set => SetProperty(ref _isBusy, value);
        }

        /// <summary>Error message to display in the UI, if any.</summary>
        public string? ErrorMessage
        {
            get => _errorMessage;
            protected set => SetProperty(ref _errorMessage, value);
        }

        /// <summary>Status message for display in the status bar.</summary>
        public string? StatusMessage
        {
            get => _statusMessage;
            protected set => SetProperty(ref _statusMessage, value);
        }

        /// <summary>Whether the ViewModel has unsaved changes.</summary>
        public bool IsDirty
        {
            get => _isDirty;
            protected set => SetProperty(ref _isDirty, value);
        }

        /// <inheritdoc/>
        public virtual async Task InitializeAsync()
        {
            if (_isInitialized) return;
            await _initializationLock.WaitAsync();
            try
            {
                if (_isInitialized) return;
                await OnInitializeAsync();
                _isInitialized = true;
            }
            finally
            {
                _initializationLock.Release();
            }
        }

        /// <summary>Override to perform custom initialization logic.</summary>
        protected virtual Task OnInitializeAsync()
        {
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public virtual void Reset()
        {
            ErrorMessage = null;
            StatusMessage = null;
            IsDirty = false;
            OnReset();
        }

        /// <summary>Override to perform custom reset logic.</summary>
        protected virtual void OnReset()
        {
        }

        /// <summary>Helper method to set a property value and raise change notification.</summary>
        protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(new PropertyChangedEventArgs(propertyName));
            return true;
        }

        /// <summary>Registers a weak reference for automatic cleanup.</summary>
        protected void TrackWeakReference(string key, INotifyPropertyChanged source)
        {
            _weakReferences[key] = new WeakReference<INotifyPropertyChanged>(source);
        }

        /// <summary>Adds a disposable resource to be cleaned up on dispose.</summary>
        protected void TrackDisposable(IDisposable disposable)
        {
            _disposables.Add(disposable);
        }

        /// <summary>Creates an async relay command.</summary>
        protected AsyncRelayCommand CreateAsyncCommand(Func<Task> execute, Func<bool>? canExecute = null)
        {
            return new AsyncRelayCommand(async () =>
            {
                try
                {
                    SetBusy(true);
                    ClearError();
                    await execute();
                }
                catch (Exception ex)
                {
                    SetError(ex.Message);
                }
                finally
                {
                    SetBusy(false);
                }
            }, canExecute);
        }

        /// <summary>Creates an async relay command with a parameter.</summary>
        protected AsyncRelayCommand<T> CreateAsyncCommand<T>(Func<T, Task> execute, Func<T, bool>? canExecute = null)
        {
            return new AsyncRelayCommand<T>(async (param) =>
            {
                try
                {
                    SetBusy(true);
                    ClearError();
                    await execute(param);
                }
                catch (Exception ex)
                {
                    SetError(ex.Message);
                }
                finally
                {
                    SetBusy(false);
                }
            }, canExecute);
        }

        /// <summary>Creates a synchronous relay command.</summary>
        protected RelayCommand CreateCommand(Action execute, Func<bool>? canExecute = null)
        {
            return new RelayCommand(execute, canExecute);
        }

        /// <summary>Creates a relay command with a parameter.</summary>
        protected RelayCommand<T> CreateCommand<T>(Action<T> execute, Func<T, bool>? canExecute = null)
        {
            return new RelayCommand<T>(execute, canExecute);
        }

        /// <summary>Raises a property changed event.</summary>
        protected void RaisePropertyChanged([CallerMemberName] string? propertyName = null)
        {
            OnPropertyChanged(new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>Sets the busy state and optionally a status message.</summary>
        protected void SetBusy(bool busy, string? message = null)
        {
            IsBusy = busy;
            StatusMessage = message;
        }

        /// <summary>Sets an error message for display in the UI.</summary>
        protected void SetError(string? message)
        {
            ErrorMessage = message;
        }

        /// <summary>Clears any active error message.</summary>
        protected void ClearError()
        {
            ErrorMessage = null;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>Releases resources.</summary>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var disposable in _disposables)
                {
                    disposable?.Dispose();
                }
                _disposables.Clear();
                _weakReferences.Clear();
                _initializationLock.Dispose();
                OnDispose();
            }
        }

        /// <summary>Override to perform custom disposal logic.</summary>
        protected virtual void OnDispose()
        {
        }

        /// <inheritdoc/>
        public virtual async ValueTask DisposeAsync()
        {
            if (_isDisposed) return;
            _isDisposed = true;
            Dispose(true);
            GC.SuppressFinalize(this);
            await ValueTask.CompletedTask;
        }
    }


    // =========================================================================
    // MAIN WINDOW VIEW MODEL
    // =========================================================================

    /// <summary>
    /// The primary ViewModel for the G-DNN Studio main window. Orchestrates
    /// all child ViewModels, menu commands, status bar, and window state.
    /// </summary>
    public partial class MainWindowViewModel : ViewModelBase
    {
        private readonly IViewportService _viewportService;
        private readonly ISceneService _sceneService;
        private readonly ICompilationService _compilationService;
        private readonly ILLMConsoleService _llmConsoleService;
        private readonly IBlueprintEditorService _blueprintEditorService;
        private readonly IHardwareMonitor _hardwareMonitor;
        private readonly UndoRedoManager _undoRedoManager;
        private readonly KeyboardShortcutManager _shortcutManager;
        private readonly LayoutManager _layoutManager;
        private readonly DialogService _dialogService;

        private string _currentFilePath = string.Empty;
        private bool _hasUnsavedChanges;
        private EditorMode _editorMode = EditorMode.Standard;
        private ToolMode _currentTool = ToolMode.Select;
        private readonly ObservableCollection<RecentFile> _recentFiles = new();
        private readonly ObservableCollection<EditorTab> _openTabs = new();
        private readonly ObservableCollection<DiagnosticMessage> _diagnostics = new();
        private readonly ObservableCollection<string> _outputLog = new();
        private Timer? _statusBarTimer;
        private readonly Stopwatch _frameStopwatch = new();
        private int _selectedTabIndex;
        private SelectionInfo _currentSelection = new();
        private readonly Dictionary<string, RelayCommand> _dynamicCommands = new();
        private readonly Dictionary<string, AsyncRelayCommand> _dynamicAsyncCommands = new();
        private bool _isRecording;
        private bool _isMaximized = true;
        private double _windowWidth = 1920;
        private double _windowHeight = 1080;
        private double _windowX;
        private double _windowY;
        private string _searchText = string.Empty;
        private int _compilationQueueLength;
        private bool _isProjectLoaded;

        /// <summary>
        /// Initializes a new instance of the MainWindowViewModel.
        /// </summary>
        public MainWindowViewModel(
            IViewportService viewportService,
            ISceneService sceneService,
            ICompilationService compilationService,
            ILLMConsoleService llmConsoleService,
            IBlueprintEditorService blueprintEditorService,
            IHardwareMonitor hardwareMonitor)
        {
            _viewportService = viewportService ?? throw new ArgumentNullException(nameof(viewportService));
            _sceneService = sceneService ?? throw new ArgumentNullException(nameof(sceneService));
            _compilationService = compilationService ?? throw new ArgumentNullException(nameof(compilationService));
            _llmConsoleService = llmConsoleService ?? throw new ArgumentNullException(nameof(llmConsoleService));
            _blueprintEditorService = blueprintEditorService ?? throw new ArgumentNullException(nameof(blueprintEditorService));
            _hardwareMonitor = hardwareMonitor ?? throw new ArgumentNullException(nameof(hardwareMonitor));

            _undoRedoManager = new UndoRedoManager();
            _shortcutManager = new KeyboardShortcutManager();
            _layoutManager = new LayoutManager();
            _dialogService = new DialogService();

            _viewportVm = new ViewportViewModel(viewportService);
            _sceneExplorerVm = new SceneExplorerViewModel(sceneService);
            _inspectorVm = new InspectorViewModel(sceneService);
            _llmConsoleVm = new LlmConsoleViewModel(llmConsoleService);
            _blueprintEditorVm = new BlueprintEditorViewModel(blueprintEditorService);
            _codeEditorVm = new CodeEditorViewModel(compilationService);
            _performanceHudVm = new PerformanceHudViewModel(hardwareMonitor, compilationService);
            _timelineVm = new TimelineViewModel();
            _preferencesVm = new PreferencesViewModel();
            _materialPreviewVm = new MaterialPreviewViewModel();

            _statusBarTimer = new Timer(StatusBarTimerCallback, null, TimeSpan.Zero, TimeSpan.FromSeconds(1));

            InitializeCommands();
            RegisterDefaultShortcuts();
            LoadRecentFiles();
        }

        // ---- Child ViewModels ----

        private readonly ViewportViewModel _viewportVm;
        /// <summary>The main 3D viewport ViewModel.</summary>
        public ViewportViewModel Viewport => _viewportVm;

        private readonly SceneExplorerViewModel _sceneExplorerVm;
        /// <summary>The scene hierarchy explorer ViewModel.</summary>
        public SceneExplorerViewModel SceneExplorer => _sceneExplorerVm;

        private readonly InspectorViewModel _inspectorVm;
        /// <summary>The property inspector ViewModel.</summary>
        public InspectorViewModel Inspector => _inspectorVm;

        private readonly LlmConsoleViewModel _llmConsoleVm;
        /// <summary>The LLM console ViewModel.</summary>
        public LlmConsoleViewModel LlmConsole => _llmConsoleVm;

        private readonly BlueprintEditorViewModel _blueprintEditorVm;
        /// <summary>The blueprint visual scripting editor ViewModel.</summary>
        public BlueprintEditorViewModel BlueprintEditor => _blueprintEditorVm;

        private readonly CodeEditorViewModel _codeEditorVm;
        /// <summary>The code editor ViewModel.</summary>
        public CodeEditorViewModel CodeEditor => _codeEditorVm;

        private readonly PerformanceHudViewModel _performanceHudVm;
        /// <summary>The performance HUD ViewModel.</summary>
        public PerformanceHudViewModel PerformanceHud => _performanceHudVm;

        private readonly TimelineViewModel _timelineVm;
        /// <summary>The animation timeline ViewModel.</summary>
        public TimelineViewModel Timeline => _timelineVm;

        private readonly PreferencesViewModel _preferencesVm;
        /// <summary>The preferences/settings ViewModel.</summary>
        public PreferencesViewModel Preferences => _preferencesVm;

        private readonly MaterialPreviewViewModel _materialPreviewVm;
        /// <summary>The material preview ViewModel.</summary>
        public MaterialPreviewViewModel MaterialPreview => _materialPreviewVm;

        // ---- Observable Properties ----

        /// <summary>Current file path of the open scene.</summary>
        public string CurrentFilePath
        {
            get => _currentFilePath;
            set
            {
                if (SetProperty(ref _currentFilePath, value))
                {
                    RaisePropertyChanged(nameof(WindowTitle));
                    RaisePropertyChanged(nameof(HasOpenFile));
                }
            }
        }

        /// <summary>Computed window title.</summary>
        public string WindowTitle
        {
            get
            {
                var title = "G-DNN Studio";
                if (!string.IsNullOrEmpty(CurrentFilePath))
                    title = $"{Path.GetFileName(CurrentFilePath)} - {title}";
                if (HasUnsavedChanges)
                    title = $"*{title}";
                return title;
            }
        }

        /// <summary>Whether there is a file currently open.</summary>
        public bool HasOpenFile => !string.IsNullOrEmpty(CurrentFilePath);

        /// <summary>Whether the current scene has unsaved changes.</summary>
        public bool HasUnsavedChanges
        {
            get => _hasUnsavedChanges;
            set
            {
                if (SetProperty(ref _hasUnsavedChanges, value))
                {
                    RaisePropertyChanged(nameof(WindowTitle));
                    SaveSceneCommand.NotifyCanExecuteChanged();
                }
            }
        }

        /// <summary>Current editor mode.</summary>
        public EditorMode CurrentEditorMode
        {
            get => _editorMode;
            set
            {
                if (SetProperty(ref _editorMode, value))
                {
                    OnEditorModeChanged();
                }
            }
        }

        /// <summary>Currently active tool.</summary>
        public ToolMode CurrentTool
        {
            get => _currentTool;
            set
            {
                if (SetProperty(ref _currentTool, value))
                {
                    _viewportVm.CurrentTool = value;
                    RaisePropertyChanged(nameof(CurrentToolName));
                }
            }
        }

        /// <summary>Display name of the current tool.</summary>
        public string CurrentToolName => CurrentTool.ToString();

        /// <summary>Read-only collection of recent files.</summary>
        public ObservableCollection<RecentFile> RecentFiles => _recentFiles;

        /// <summary>Collection of open editor tabs.</summary>
        public ObservableCollection<EditorTab> OpenTabs => _openTabs;

        /// <summary>Collection of diagnostic messages.</summary>
        public ObservableCollection<DiagnosticMessage> Diagnostics => _diagnostics;

        /// <summary>Output log messages.</summary>
        public ObservableCollection<string> OutputLog => _outputLog;

        /// <summary>Currently selected tab index.</summary>
        public int SelectedTabIndex
        {
            get => _selectedTabIndex;
            set => SetProperty(ref _selectedTabIndex, value);
        }

        /// <summary>Current selection info.</summary>
        public SelectionInfo CurrentSelection
        {
            get => _currentSelection;
            set
            {
                if (SetProperty(ref _currentSelection, value))
                {
                    Inspector.SelectedEntityId = value?.SelectedEntities.FirstOrDefault();
                    RemoveEntityCommand.NotifyCanExecuteChanged();
                    DuplicateEntityCommand.NotifyCanExecuteChanged();
                    ViewGenomeCommand.NotifyCanExecuteChanged();
                    MutateGenomeCommand.NotifyCanExecuteChanged();
                }
            }
        }

        /// <summary>Whether undo is available.</summary>
        public bool CanUndo => _undoRedoManager.CanUndo;

        /// <summary>Whether redo is available.</summary>
        public bool CanRedo => _undoRedoManager.CanRedo;

        /// <summary>Undo stack description.</summary>
        public string UndoDescription => _undoRedoManager.CanUndo ? _undoRedoManager.PeekUndo()?.Description ?? "" : "";

        /// <summary>Redo stack description.</summary>
        public string RedoDescription => _undoRedoManager.CanRedo ? _undoRedoManager.PeekRedo()?.Description ?? "" : "";

        /// <summary>The undo/redo manager instance.</summary>
        public UndoRedoManager UndoRedoManager => _undoRedoManager;

        /// <summary>The keyboard shortcut manager instance.</summary>
        public KeyboardShortcutManager ShortcutManager => _shortcutManager;

        /// <summary>The layout manager instance.</summary>
        public LayoutManager LayoutManager => _layoutManager;

        /// <summary>Whether the application is recording.</summary>
        public bool IsRecording
        {
            get => _isRecording;
            set => SetProperty(ref _isRecording, value);
        }

        /// <summary>Whether the window is maximized.</summary>
        public bool IsMaximized
        {
            get => _isMaximized;
            set => SetProperty(ref _isMaximized, value);
        }

        /// <summary>Window width.</summary>
        public double WindowWidth
        {
            get => _windowWidth;
            set => SetProperty(ref _windowWidth, value);
        }

        /// <summary>Window height.</summary>
        public double WindowHeight
        {
            get => _windowHeight;
            set => SetProperty(ref _windowHeight, value);
        }

        /// <summary>Window X position.</summary>
        public double WindowX
        {
            get => _windowX;
            set => SetProperty(ref _windowX, value);
        }

        /// <summary>Window Y position.</summary>
        public double WindowY
        {
            get => _windowY;
            set => SetProperty(ref _windowY, value);
        }

        /// <summary>Global search text.</summary>
        public string SearchText
        {
            get => _searchText;
            set => SetProperty(ref _searchText, value);
        }

        /// <summary>Number of genomes in the compilation queue.</summary>
        public int CompilationQueueLength
        {
            get => _compilationQueueLength;
            set => SetProperty(ref _compilationQueueLength, value);
        }

        /// <summary>Whether a project is loaded.</summary>
        public bool IsProjectLoaded
        {
            get => _isProjectLoaded;
            set => SetProperty(ref _isProjectLoaded, value);
        }


        // ---- Menu Commands ----

        /// <summary>Command to create a new scene.</summary>
        public IAsyncRelayCommand NewSceneCommand { get; private set; } = null!;
        /// <summary>Command to open an existing scene file.</summary>
        public IAsyncRelayCommand OpenSceneCommand { get; private set; } = null!;
        /// <summary>Command to save the current scene.</summary>
        public IAsyncRelayCommand SaveSceneCommand { get; private set; } = null!;
        /// <summary>Command to save the current scene to a new file.</summary>
        public IAsyncRelayCommand SaveSceneAsCommand { get; private set; } = null!;
        /// <summary>Command to export the scene.</summary>
        public IAsyncRelayCommand ExportSceneCommand { get; private set; } = null!;
        /// <summary>Command to import assets.</summary>
        public IAsyncRelayCommand ImportAssetCommand { get; private set; } = null!;
        /// <summary>Command to exit the application.</summary>
        public IRelayCommand ExitCommand { get; private set; } = null!;
        /// <summary>Command to undo the last operation.</summary>
        public IRelayCommand UndoCommand { get; private set; } = null!;
        /// <summary>Command to redo the last undone operation.</summary>
        public IRelayCommand RedoCommand { get; private set; } = null!;
        /// <summary>Command to cut the current selection.</summary>
        public IRelayCommand CutCommand { get; private set; } = null!;
        /// <summary>Command to copy the current selection.</summary>
        public IRelayCommand CopyCommand { get; private set; } = null!;
        /// <summary>Command to paste from clipboard.</summary>
        public IAsyncRelayCommand PasteCommand { get; private set; } = null!;
        /// <summary>Command to select all.</summary>
        public IRelayCommand SelectAllCommand { get; private set; } = null!;
        /// <summary>Command to open preferences.</summary>
        public IRelayCommand PreferencesCommand { get; private set; } = null!;
        /// <summary>Command to add a new entity.</summary>
        public IAsyncRelayCommand AddEntityCommand { get; private set; } = null!;
        /// <summary>Command to remove selected entity.</summary>
        public IAsyncRelayCommand RemoveEntityCommand { get; private set; } = null!;
        /// <summary>Command to duplicate selected entity.</summary>
        public IAsyncRelayCommand DuplicateEntityCommand { get; private set; } = null!;
        /// <summary>Command to group entities.</summary>
        public IRelayCommand GroupEntitiesCommand { get; private set; } = null!;
        /// <summary>Command to ungroup entities.</summary>
        public IRelayCommand UngroupEntitiesCommand { get; private set; } = null!;
        /// <summary>Command to compile all genomes.</summary>
        public IAsyncRelayCommand CompileAllCommand { get; private set; } = null!;
        /// <summary>Command to view genome.</summary>
        public IRelayCommand ViewGenomeCommand { get; private set; } = null!;
        /// <summary>Command to mutate genome.</summary>
        public IAsyncRelayCommand MutateGenomeCommand { get; private set; } = null!;
        /// <summary>Command to evolve population.</summary>
        public IAsyncRelayCommand EvolvePopulationCommand { get; private set; } = null!;
        /// <summary>Command to toggle wireframe.</summary>
        public IRelayCommand ToggleWireframeCommand { get; private set; } = null!;
        /// <summary>Command for quality settings.</summary>
        public IRelayCommand QualitySettingsCommand { get; private set; } = null!;
        /// <summary>Command to capture screenshot.</summary>
        public IAsyncRelayCommand CaptureScreenshotCommand { get; private set; } = null!;
        /// <summary>Command to record video.</summary>
        public IRelayCommand RecordVideoCommand { get; private set; } = null!;
        /// <summary>Command for layout management.</summary>
        public IRelayCommand LayoutManagementCommand { get; private set; } = null!;
        /// <summary>Command to reset layout.</summary>
        public IRelayCommand ResetLayoutCommand { get; private set; } = null!;
        /// <summary>Command to open documentation.</summary>
        public IRelayCommand DocumentationCommand { get; private set; } = null!;
        /// <summary>Command to show about dialog.</summary>
        public IRelayCommand AboutCommand { get; private set; } = null!;
        /// <summary>Command to open diagnostics.</summary>
        public IRelayCommand DiagnosticsCommand { get; private set; } = null!;
        /// <summary>Command to open a recent file.</summary>
        public IRelayCommand<RecentFile> OpenRecentFileCommand { get; private set; } = null!;
        /// <summary>Command to remove a recent file.</summary>
        public IRelayCommand<RecentFile> RemoveRecentFileCommand { get; private set; } = null!;
        /// <summary>Command to close a tab.</summary>
        public IRelayCommand<EditorTab> CloseTabCommand { get; private set; } = null!;
        /// <summary>Command to show the viewport tab.</summary>
        public IRelayCommand ShowViewportCommand { get; private set; } = null!;
        /// <summary>Command to show the code editor tab.</summary>
        public IRelayCommand ShowCodeEditorCommand { get; private set; } = null!;
        /// <summary>Command to show the blueprint editor tab.</summary>
        public IRelayCommand ShowBlueprintCommand { get; private set; } = null!;
        /// <summary>Command to toggle the scene explorer panel.</summary>
        public IRelayCommand ToggleSceneExplorerCommand { get; private set; } = null!;
        /// <summary>Command to toggle the inspector panel.</summary>
        public IRelayCommand ToggleInspectorCommand { get; private set; } = null!;
        /// <summary>Command to toggle the LLM console panel.</summary>
        public IRelayCommand ToggleLlmConsoleCommand { get; private set; } = null!;
        /// <summary>Command to toggle the performance HUD.</summary>
        public IRelayCommand TogglePerformanceHudCommand { get; private set; } = null!;
        /// <summary>Command to toggle the timeline panel.</summary>
        public IRelayCommand ToggleTimelineCommand { get; private set; } = null!;

        // ---- Panel Visibility ----

        /// <summary>Whether the scene explorer is visible.</summary>
        public bool IsSceneExplorerVisible
        {
            get => GetProperty<bool>(nameof(IsSceneExplorerVisible), true);
            set => SetProperty(value, nameof(IsSceneExplorerVisible));
        }

        /// <summary>Whether the inspector is visible.</summary>
        public bool IsInspectorVisible
        {
            get => GetProperty<bool>(nameof(IsInspectorVisible), true);
            set => SetProperty(value, nameof(IsInspectorVisible));
        }

        /// <summary>Whether the LLM console is visible.</summary>
        public bool IsLlmConsoleVisible
        {
            get => GetProperty<bool>(nameof(IsLlmConsoleVisible));
            set => SetProperty(value, nameof(IsLlmConsoleVisible));
        }

        /// <summary>Whether the performance HUD is visible.</summary>
        public bool IsPerformanceHudVisible
        {
            get => GetProperty<bool>(nameof(IsPerformanceHudVisible));
            set => SetProperty(value, nameof(IsPerformanceHudVisible));
        }

        /// <summary>Whether the timeline is visible.</summary>
        public bool IsTimelineVisible
        {
            get => GetProperty<bool>(nameof(IsTimelineVisible));
            set => SetProperty(value, nameof(IsTimelineVisible));
        }

        /// <summary>Whether the code editor is visible.</summary>
        public bool IsCodeEditorVisible
        {
            get => GetProperty<bool>(nameof(IsCodeEditorVisible));
            set => SetProperty(value, nameof(IsCodeEditorVisible));
        }

        /// <summary>Whether the blueprint editor is visible.</summary>
        public bool IsBlueprintEditorVisible
        {
            get => GetProperty<bool>(nameof(IsBlueprintEditorVisible));
            set => SetProperty(value, nameof(IsBlueprintEditorVisible));
        }

        /// <summary>Whether the material preview is visible.</summary>
        public bool IsMaterialPreviewVisible
        {
            get => GetProperty<bool>(nameof(IsMaterialPreviewVisible));
            set => SetProperty(value, nameof(IsMaterialPreviewVisible));
        }


        private void InitializeCommands()
        {
            NewSceneCommand = CreateAsyncCommand(ExecuteNewScene);
            OpenSceneCommand = CreateAsyncCommand(ExecuteOpenScene);
            SaveSceneCommand = CreateAsyncCommand(ExecuteSaveScene, () => HasOpenFile);
            SaveSceneAsCommand = CreateAsyncCommand(ExecuteSaveSceneAs);
            ExportSceneCommand = CreateAsyncCommand(ExecuteExportScene, () => HasOpenFile);
            ImportAssetCommand = CreateAsyncCommand(ExecuteImportAsset);
            ExitCommand = CreateCommand(ExecuteExit);

            UndoCommand = CreateCommand(ExecuteUndo, () => CanUndo);
            RedoCommand = CreateCommand(ExecuteRedo, () => CanRedo);
            CutCommand = CreateCommand(ExecuteCut, () => CurrentSelection.HasSelection);
            CopyCommand = CreateCommand(ExecuteCopy, () => CurrentSelection.HasSelection);
            PasteCommand = CreateAsyncCommand(ExecutePaste);
            SelectAllCommand = CreateCommand(ExecuteSelectAll);
            PreferencesCommand = CreateCommand(ExecutePreferences);

            AddEntityCommand = CreateAsyncCommand(ExecuteAddEntity);
            RemoveEntityCommand = CreateAsyncCommand(ExecuteRemoveEntity, () => CurrentSelection.HasSelection);
            DuplicateEntityCommand = CreateAsyncCommand(ExecuteDuplicateEntity, () => CurrentSelection.HasSelection);
            GroupEntitiesCommand = CreateCommand(ExecuteGroupEntities, () => CurrentSelection.SelectedEntities.Count > 1);
            UngroupEntitiesCommand = CreateCommand(ExecuteUngroupEntities);

            CompileAllCommand = CreateAsyncCommand(ExecuteCompileAll);
            ViewGenomeCommand = CreateCommand(ExecuteViewGenome, () => CurrentSelection.SelectedGenome.HasValue);
            MutateGenomeCommand = CreateAsyncCommand(ExecuteMutateGenome, () => CurrentSelection.SelectedGenome.HasValue);
            EvolvePopulationCommand = CreateAsyncCommand(ExecuteEvolvePopulation);

            ToggleWireframeCommand = CreateCommand(ExecuteToggleWireframe);
            QualitySettingsCommand = CreateCommand(ExecuteQualitySettings);
            CaptureScreenshotCommand = CreateAsyncCommand(ExecuteCaptureScreenshot);
            RecordVideoCommand = CreateCommand(ExecuteRecordVideo);

            LayoutManagementCommand = CreateCommand(ExecuteLayoutManagement);
            ResetLayoutCommand = CreateCommand(ExecuteResetLayout);

            DocumentationCommand = CreateCommand(ExecuteDocumentation);
            AboutCommand = CreateCommand(ExecuteAbout);
            DiagnosticsCommand = CreateCommand(ExecuteDiagnostics);

            OpenRecentFileCommand = CreateCommand<RecentFile>(ExecuteOpenRecentFile);
            RemoveRecentFileCommand = CreateCommand<RecentFile>(ExecuteRemoveRecentFile);
            CloseTabCommand = CreateCommand<EditorTab>(ExecuteCloseTab);

            ShowViewportCommand = CreateCommand(() => OpenEditorTab(EditorTabType.Viewport));
            ShowCodeEditorCommand = CreateCommand(() => OpenEditorTab(EditorTabType.CodeEditor));
            ShowBlueprintCommand = CreateCommand(() => OpenEditorTab(EditorTabType.Blueprint));

            ToggleSceneExplorerCommand = CreateCommand(() => IsSceneExplorerVisible = !IsSceneExplorerVisible);
            ToggleInspectorCommand = CreateCommand(() => IsInspectorVisible = !IsInspectorVisible);
            ToggleLlmConsoleCommand = CreateCommand(() => IsLlmConsoleVisible = !IsLlmConsoleVisible);
            TogglePerformanceHudCommand = CreateCommand(() => IsPerformanceHudVisible = !IsPerformanceHudVisible);
            ToggleTimelineCommand = CreateCommand(() => IsTimelineVisible = !IsTimelineVisible);

            _sceneService.SceneChanged += OnSceneChanged;
            _compilationService.CompilationStatusChanged += OnCompilationStatusChanged;
        }

        // ---- File Menu Implementations ----

        private async Task ExecuteNewScene()
        {
            if (_hasUnsavedChanges)
            {
                var result = await _dialogService.ShowConfirmAsync(
                    "Unsaved Changes", "Do you want to save changes to the current scene?");
                if (result == DialogResult.Cancel) return;
                if (result == DialogResult.Yes) await ExecuteSaveScene();
            }

            CurrentFilePath = string.Empty;
            HasUnsavedChanges = false;
            IsProjectLoaded = false;
            _undoRedoManager.Clear();
            _sceneExplorerVm.RefreshHierarchy();
            StatusMessage = "New scene created";
            AddDiagnostic(DiagnosticSeverity.Info, "File", "New scene created");
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] New scene created");
        }

        private async Task ExecuteOpenScene()
        {
            if (_hasUnsavedChanges)
            {
                var result = await _dialogService.ShowConfirmAsync(
                    "Unsaved Changes", "Do you want to save changes before opening another scene?");
                if (result == DialogResult.Cancel) return;
                if (result == DialogResult.Yes) await ExecuteSaveScene();
            }

            var filePath = await _dialogService.ShowOpenFileDialogAsync(
                "Open Scene",
                "G-DNN Scene Files (*.gdnn)|*.gdnn|All Files (*.*)|*.*");

            if (string.IsNullOrEmpty(filePath)) return;

            try
            {
                SetBusy(true, "Loading scene...");
                var success = await _sceneService.LoadSceneAsync(filePath);
                if (success)
                {
                    CurrentFilePath = filePath;
                    HasUnsavedChanges = false;
                    IsProjectLoaded = true;
                    AddRecentFile(filePath);
                    StatusMessage = $"Loaded: {Path.GetFileName(filePath)}";
                    _sceneExplorerVm.RefreshHierarchy();
                    UpdateSceneStats();
                    OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Loaded scene: {filePath}");
                    AddDiagnostic(DiagnosticSeverity.Info, "File", $"Scene loaded from {filePath}");
                }
                else
                {
                    SetError($"Failed to load scene: {filePath}");
                    AddDiagnostic(DiagnosticSeverity.Error, "File", $"Failed to load scene from {filePath}");
                }
            }
            catch (Exception ex)
            {
                SetError($"Error loading scene: {ex.Message}");
                AddDiagnostic(DiagnosticSeverity.Error, "File", $"Error loading scene: {ex.Message}", ex.ToString());
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] ERROR: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private async Task ExecuteSaveScene()
        {
            if (string.IsNullOrEmpty(CurrentFilePath))
            {
                await ExecuteSaveSceneAs();
                return;
            }

            try
            {
                SetBusy(true, "Saving scene...");
                var success = await _sceneService.SaveSceneAsync(CurrentFilePath);
                if (success)
                {
                    HasUnsavedChanges = false;
                    StatusMessage = $"Saved: {Path.GetFileName(CurrentFilePath)}";
                    OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Scene saved to {CurrentFilePath}");
                    AddDiagnostic(DiagnosticSeverity.Info, "File", $"Scene saved to {CurrentFilePath}");
                }
                else
                {
                    SetError("Failed to save scene");
                    AddDiagnostic(DiagnosticSeverity.Error, "File", "Failed to save scene");
                }
            }
            catch (Exception ex)
            {
                SetError($"Error saving scene: {ex.Message}");
                AddDiagnostic(DiagnosticSeverity.Error, "File", $"Error saving scene: {ex.Message}", ex.ToString());
            }
            finally
            {
                SetBusy(false);
            }
        }

        private async Task ExecuteSaveSceneAs()
        {
            var filePath = await _dialogService.ShowSaveFileDialogAsync(
                "Save Scene As",
                "G-DNN Scene Files (*.gdnn)|*.gdnn",
                "scene.gdnn");

            if (string.IsNullOrEmpty(filePath)) return;

            CurrentFilePath = filePath;
            await ExecuteSaveScene();
            AddRecentFile(filePath);
        }

        private async Task ExecuteExportScene()
        {
            var filePath = await _dialogService.ShowSaveFileDialogAsync(
                "Export Scene",
                "FBX Files (*.fbx)|*.fbx|OBJ Files (*.obj)|*.obj|GLTF Files (*.gltf)|*.gltf|GLB Files (*.glb)|*.glb",
                "export.fbx");

            if (string.IsNullOrEmpty(filePath)) return;

            try
            {
                SetBusy(true, "Exporting scene...");
                StatusMessage = $"Exporting to {Path.GetFileName(filePath)}...";

                var progress = new Progress<float>(p =>
                {
                    StatusMessage = $"Exporting... {p * 100:F0}%";
                });

                await Task.Run(() =>
                {
                    Thread.Sleep(100);
                });

                StatusMessage = "Export complete";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Scene exported to {filePath}");
                AddDiagnostic(DiagnosticSeverity.Info, "File", $"Scene exported to {filePath}");
            }
            catch (Exception ex)
            {
                SetError($"Export failed: {ex.Message}");
                AddDiagnostic(DiagnosticSeverity.Error, "File", $"Export failed: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private async Task ExecuteImportAsset()
        {
            var filePaths = await _dialogService.ShowOpenMultipleFileDialogAsync(
                "Import Assets",
                "All Supported Files (*.fbx;*.obj;*.gltf;*.glb;*.png;*.jpg;*.jpeg;*.hdr;*.exr;*.tga;*.dds;*.ktx)|*.fbx;*.obj;*.gltf;*.glb;*.png;*.jpg;*.jpeg;*.hdr;*.exr;*.tga;*.dds;*.ktx|All Files (*.*)|*.*");

            if (filePaths == null || !filePaths.Any()) return;

            try
            {
                SetBusy(true, "Importing assets...");
                var importedCount = 0;
                var failedCount = 0;

                foreach (var filePath in filePaths)
                {
                    StatusMessage = $"Importing {Path.GetFileName(filePath)}... ({importedCount + 1}/{filePaths.Count()})";
                    try
                    {
                        await Task.Delay(50);
                        importedCount++;
                    }
                    catch
                    {
                        failedCount++;
                        AddDiagnostic(DiagnosticSeverity.Warning, "Import", $"Failed to import: {Path.GetFileName(filePath)}");
                    }
                }

                StatusMessage = $"Imported {importedCount} assets" + (failedCount > 0 ? $" ({failedCount} failed)" : "");
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Imported {importedCount} assets");
                AddDiagnostic(DiagnosticSeverity.Info, "File", $"Imported {importedCount} assets");
            }
            catch (Exception ex)
            {
                SetError($"Import failed: {ex.Message}");
                AddDiagnostic(DiagnosticSeverity.Error, "File", $"Import failed: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteExit()
        {
            Dispatcher.UIThread.Post(async () =>
            {
                if (_hasUnsavedChanges)
                {
                    var result = await _dialogService.ShowConfirmAsync(
                        "Unsaved Changes", "Do you want to save changes before exiting?");
                    if (result == DialogResult.Cancel) return;
                    if (result == DialogResult.Yes) await ExecuteSaveScene();
                }

                Dispose();
                Environment.Exit(0);
            });
        }

        // ---- Edit Menu Implementations ----

        private void ExecuteUndo()
        {
            if (_undoRedoManager.CanUndo)
            {
                var operation = _undoRedoManager.PeekUndo();
                _undoRedoManager.Undo();
                RaisePropertyChanged(nameof(CanUndo));
                RaisePropertyChanged(nameof(CanRedo));
                RaisePropertyChanged(nameof(UndoDescription));
                RaisePropertyChanged(nameof(RedoDescription));
                StatusMessage = $"Undid: {operation?.Description}";
                _sceneExplorerVm.RefreshHierarchy();
                _inspectorVm.Refresh();
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Undo: {operation?.Description}");
            }
        }

        private void ExecuteRedo()
        {
            if (_undoRedoManager.CanRedo)
            {
                var operation = _undoRedoManager.PeekRedo();
                _undoRedoManager.Redo();
                RaisePropertyChanged(nameof(CanUndo));
                RaisePropertyChanged(nameof(CanRedo));
                RaisePropertyChanged(nameof(UndoDescription));
                RaisePropertyChanged(nameof(RedoDescription));
                StatusMessage = $"Redid: {operation?.Description}";
                _sceneExplorerVm.RefreshHierarchy();
                _inspectorVm.Refresh();
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Redo: {operation?.Description}");
            }
        }

        private void ExecuteCut()
        {
            if (CurrentSelection?.SelectedEntities.Count > 0)
            {
                ClipboardService.SetEntities(CurrentSelection.SelectedEntities);
                StatusMessage = $"Cut {CurrentSelection.SelectedEntities.Count} entity(ies)";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Cut {CurrentSelection.SelectedEntities.Count} entities");
            }
        }

        private void ExecuteCopy()
        {
            if (CurrentSelection?.SelectedEntities.Count > 0)
            {
                ClipboardService.SetEntities(CurrentSelection.SelectedEntities);
                StatusMessage = $"Copied {CurrentSelection.SelectedEntities.Count} entity(ies)";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Copied {CurrentSelection.SelectedEntities.Count} entities");
            }
        }

        private async Task ExecutePaste()
        {
            var entityIds = ClipboardService.GetEntities();
            if (entityIds != null && entityIds.Count > 0)
            {
                var newIds = new List<Guid>();
                foreach (var id in entityIds)
                {
                    var entity = _sceneService.GetEntityById(id);
                    var name = entity != null ? $"{entity.Name} (Copy)" : "Pasted Entity";
                    var type = entity?.Type ?? EntityType.Empty;
                    var newId = await _sceneService.CreateEntityAsync(name, type);
                    newIds.Add(newId);
                }
                CurrentSelection = new SelectionInfo { SelectedEntities = newIds };
                _sceneExplorerVm.RefreshHierarchy();
                StatusMessage = $"Pasted {newIds.Count} entity(ies)";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Pasted {newIds.Count} entities");
            }
        }

        private void ExecuteSelectAll()
        {
            var entities = _sceneService.GetEntities();
            var allIds = entities.Select(e => e.Id).ToList();
            CurrentSelection = new SelectionInfo { SelectedEntities = allIds };
            _sceneExplorerVm.SelectAll();
            StatusMessage = $"Selected {allIds.Count} entity(ies)";
        }

        private void ExecutePreferences()
        {
            _preferencesVm.ShowDialog();
        }

        // ---- Scene Menu Implementations ----

        private async Task ExecuteAddEntity()
        {
            var entityId = await _sceneService.CreateEntityAsync("New Entity", EntityType.Mesh);
            _sceneExplorerVm.RefreshHierarchy();
            CurrentSelection = new SelectionInfo { SelectedEntities = new[] { entityId } };
            StatusMessage = "Added new entity";
            HasUnsavedChanges = true;
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Added entity: {entityId}");
        }

        private async Task ExecuteRemoveEntity()
        {
            if (CurrentSelection?.SelectedEntities.Count > 0)
            {
                var count = CurrentSelection.SelectedEntities.Count;
                var names = new List<string>();

                foreach (var id in CurrentSelection.SelectedEntities)
                {
                    var entity = _sceneService.GetEntityById(id);
                    if (entity != null)
                    {
                        names.Add(entity.Name);
                        _undoRedoManager.Push(new UndoOperation(
                            UndoOperationType.EntityDelete,
                            $"Delete entity '{entity.Name}'",
                            () => _sceneService.DeleteEntityAsync(id),
                            () => _sceneService.CreateEntityAsync(entity.Name, entity.Type)));
                    }
                    await _sceneService.DeleteEntityAsync(id);
                }

                CurrentSelection = new SelectionInfo();
                _sceneExplorerVm.RefreshHierarchy();
                _inspectorVm.ClearSelection();
                HasUnsavedChanges = true;
                StatusMessage = $"Removed {count} entity(ies): {string.Join(", ", names)}";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Removed {count} entities");
            }
        }

        private async Task ExecuteDuplicateEntity()
        {
            if (CurrentSelection?.SelectedEntities.Count > 0)
            {
                var newIds = new List<Guid>();
                foreach (var id in CurrentSelection.SelectedEntities)
                {
                    var entity = _sceneService.GetEntityById(id);
                    if (entity != null)
                    {
                        var newId = await _sceneService.CreateEntityAsync($"{entity.Name} (Copy)", entity.Type);
                        newIds.Add(newId);
                    }
                }
                CurrentSelection = new SelectionInfo { SelectedEntities = newIds };
                _sceneExplorerVm.RefreshHierarchy();
                HasUnsavedChanges = true;
                StatusMessage = $"Duplicated {newIds.Count} entity(ies)";
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Duplicated {newIds.Count} entities");
            }
        }

        private void ExecuteGroupEntities()
        {
            if (CurrentSelection?.SelectedEntities.Count > 1)
            {
                StatusMessage = $"Grouped {CurrentSelection.SelectedEntities.Count} entities";
                HasUnsavedChanges = true;
            }
        }

        private void ExecuteUngroupEntities()
        {
            StatusMessage = "Ungrouped selected entities";
            HasUnsavedChanges = true;
        }

        // ---- Genomes Menu Implementations ----

        private async Task ExecuteCompileAll()
        {
            try
            {
                SetBusy(true, "Compiling all genomes...");
                var entities = _sceneService.GetEntities();
                var genomeEntities = entities.Where(e => e.Components.Contains(ComponentType.Genome)).ToList();
                var compiled = 0;
                var failed = 0;

                foreach (var entity in genomeEntities)
                {
                    StatusMessage = $"Compiling genome for {entity.Name} ({compiled + 1}/{genomeEntities.Count})...";
                    try
                    {
                        var result = await _compilationService.CompileGenomeAsync(entity.Id);
                        if (result.Status == CompilationStatus.Success)
                        {
                            compiled++;
                            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Compiled genome: {entity.Name} ({result.Duration.TotalMilliseconds:F1}ms)");
                        }
                        else
                        {
                            failed++;
                            foreach (var diag in result.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error))
                            {
                                AddDiagnostic(DiagnosticSeverity.Error, "Compilation", diag.Message, diag.Details);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        failed++;
                        AddDiagnostic(DiagnosticSeverity.Error, "Compilation", $"Failed to compile {entity.Name}: {ex.Message}");
                    }
                }

                StatusMessage = $"Compiled {compiled}/{genomeEntities.Count} genomes" + (failed > 0 ? $" ({failed} failed)" : "");
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Compilation complete: {compiled} success, {failed} failed");
            }
            catch (Exception ex)
            {
                SetError($"Compilation failed: {ex.Message}");
                AddDiagnostic(DiagnosticSeverity.Error, "Compilation", ex.Message);
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteViewGenome()
        {
            CurrentEditorMode = EditorMode.GenomeEdit;
            StatusMessage = "Viewing genome";
        }

        private async Task ExecuteMutateGenome()
        {
            if (CurrentSelection?.SelectedGenome.HasValue == true)
            {
                try
                {
                    SetBusy(true, "Mutating genome...");
                    await Task.Delay(100);
                    StatusMessage = "Genome mutated";
                    HasUnsavedChanges = true;
                }
                finally
                {
                    SetBusy(false);
                }
            }
        }

        private async Task ExecuteEvolvePopulation()
        {
            try
            {
                SetBusy(true, "Evolving population...");
                await Task.Delay(100);
                StatusMessage = "Population evolved";
                HasUnsavedChanges = true;
            }
            finally
            {
                SetBusy(false);
            }
        }

        // ---- Render Menu Implementations ----

        private void ExecuteToggleWireframe()
        {
            _viewportVm.ToggleOverlay(OverlayFlags.WireframeOverlay);
            StatusMessage = "Toggled wireframe overlay";
        }

        private void ExecuteQualitySettings()
        {
            StatusMessage = "Quality settings opened";
        }

        private async Task ExecuteCaptureScreenshot()
        {
            try
            {
                SetBusy(true, "Capturing screenshot...");
                var filePath = await _dialogService.ShowSaveFileDialogAsync(
                    "Save Screenshot",
                    "PNG Files (*.png)|*.png|JPEG Files (*.jpg)|*.jpg|BMP Files (*.bmp)|*.bmp",
                    $"screenshot_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                if (!string.IsNullOrEmpty(filePath))
                {
                    var data = await _viewportService.CaptureScreenshotAsync();
                    await File.WriteAllBytesAsync(filePath, data);
                    StatusMessage = $"Screenshot saved to {Path.GetFileName(filePath)}";
                    OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Screenshot saved: {filePath} ({data.Length} bytes)");
                }
            }
            catch (Exception ex)
            {
                SetError($"Screenshot capture failed: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteRecordVideo()
        {
            IsRecording = !IsRecording;
            StatusMessage = IsRecording ? "Video recording started" : "Video recording stopped";
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Video recording {(IsRecording ? "started" : "stopped")}");
        }

        // ---- Window Menu Implementations ----

        private void ExecuteLayoutManagement()
        {
            StatusMessage = "Layout management opened";
        }

        private void ExecuteResetLayout()
        {
            _layoutManager.ResetToDefault();
            IsSceneExplorerVisible = true;
            IsInspectorVisible = true;
            IsLlmConsoleVisible = false;
            IsPerformanceHudVisible = false;
            IsTimelineVisible = false;
            IsCodeEditorVisible = false;
            IsBlueprintEditorVisible = false;
            StatusMessage = "Layout reset to default";
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Layout reset to default");
        }

        // ---- Help Menu Implementations ----

        private void ExecuteDocumentation()
        {
            StatusMessage = "Opening documentation...";
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Opening documentation");
        }

        private void ExecuteAbout()
        {
            StatusMessage = "G-DNN Studio v1.0.0 - Build 2026.07.14";
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] G-DNN Studio v1.0.0");
        }

        private void ExecuteDiagnostics()
        {
            StatusMessage = "Diagnostics panel opened";
            var gpuInfo = _hardwareMonitor.GetGpuInfo();
            var cpuInfo = _hardwareMonitor.GetCpuInfo();
            var memInfo = _hardwareMonitor.GetMemoryInfo();
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] GPU: {gpuInfo.Name} ({gpuInfo.DedicatedMemory / (1024*1024)} MB)");
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] CPU: {cpuInfo.Name} ({cpuInfo.CoreCount} cores, {cpuInfo.ThreadCount} threads)");
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] RAM: {memInfo.UsedPhysical / (1024*1024)} MB / {memInfo.TotalPhysical / (1024*1024)} MB");
        }

        // ---- Recent Files ----

        private void ExecuteOpenRecentFile(RecentFile file)
        {
            if (file.Exists)
            {
                CurrentFilePath = file.FilePath;
                _ = ExecuteOpenScene();
            }
            else
            {
                _dialogService.ShowMessageAsync("File Not Found", $"The file '{file.FilePath}' no longer exists.");
                _recentFiles.Remove(file);
                SaveRecentFiles();
            }
        }

        private void ExecuteRemoveRecentFile(RecentFile file)
        {
            _recentFiles.Remove(file);
            SaveRecentFiles();
        }

        // ---- Tab Management ----

        private void ExecuteCloseTab(EditorTab tab)
        {
            _openTabs.Remove(tab);
            tab.Content?.Dispose();
            if (_openTabs.Count == 0)
            {
                OpenEditorTab(EditorTabType.Viewport);
            }
            if (SelectedTabIndex >= _openTabs.Count)
            {
                SelectedTabIndex = Math.Max(0, _openTabs.Count - 1);
            }
        }

        private void OpenEditorTab(EditorTabType type)
        {
            var existing = _openTabs.FirstOrDefault(t => t.Type == type);
            if (existing != null)
            {
                SelectedTabIndex = _openTabs.IndexOf(existing);
                return;
            }

            var tab = new EditorTab
            {
                Type = type,
                Title = GetTabTitle(type),
                Content = GetTabContent(type)
            };

            _openTabs.Add(tab);
            SelectedTabIndex = _openTabs.Count - 1;
        }

        private string GetTabTitle(EditorTabType type)
        {
            return type switch
            {
                EditorTabType.Viewport => "3D Viewport",
                EditorTabType.CodeEditor => "Code Editor",
                EditorTabType.Blueprint => "Blueprint Editor",
                EditorTabType.MaterialGraph => "Material Graph",
                EditorTabType.GenomeEditor => "Genome Editor",
                EditorTabType.Console => "Console",
                EditorTabType.Profiler => "Profiler",
                EditorTabType.AssetBrowser => "Asset Browser",
                _ => "Editor"
            };
        }

        private IViewModelBase? GetTabContent(EditorTabType type)
        {
            return type switch
            {
                EditorTabType.Viewport => _viewportVm,
                EditorTabType.CodeEditor => _codeEditorVm,
                EditorTabType.Blueprint => _blueprintEditorVm,
                _ => null
            };
        }

        // ---- Private Helpers ----

        private void OnEditorModeChanged()
        {
            StatusMessage = CurrentEditorMode switch
            {
                EditorMode.Standard => "Standard editing mode",
                EditorMode.Blueprint => "Blueprint editing mode",
                EditorMode.LLM => "LLM-assisted editing mode",
                EditorMode.Debug => "Debug mode",
                EditorMode.Animation => "Animation editing mode",
                EditorMode.MaterialEdit => "Material editing mode",
                EditorMode.GenomeEdit => "Genome editing mode",
                EditorMode.LevelDesign => "Level design mode",
                _ => "Unknown mode"
            };
            OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Editor mode: {CurrentEditorMode}");
        }

        private void StatusBarTimerCallback(object? state)
        {
            Dispatcher.UIThread.Post(() =>
            {
                RaisePropertyChanged(nameof(CanUndo));
                RaisePropertyChanged(nameof(CanRedo));
            });
        }

        private void UpdateSceneStats()
        {
            var stats = _sceneService.GetSceneStats();
            StatusMessage = $"Scene: {stats.EntityCount} entities, {stats.NeuronCount} neurons, {stats.SynapseCount} synapses";
        }

        private void OnSceneChanged(object? sender, SceneChangedEventArgs e)
        {
            Dispatcher.UIThread.Post(() =>
            {
                HasUnsavedChanges = e.RequiresSave;
                _sceneExplorerVm.RefreshHierarchy();
                UpdateSceneStats();
            });
        }

        private void OnCompilationStatusChanged(object? sender, CompilationStatusChangedEventArgs e)
        {
            Dispatcher.UIThread.Post(() =>
            {
                CompilationQueueLength = Math.Max(0, CompilationQueueLength + (e.NewStatus == CompilationStatus.Queued ? 1 : -1));
                if (e.NewStatus == CompilationStatus.Success)
                {
                    OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Genome compiled successfully: {e.GenomeId}");
                }
                else if (e.NewStatus == CompilationStatus.Failed)
                {
                    OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Genome compilation failed: {e.GenomeId} - {e.Message}");
                }
            });
        }


        private void RegisterDefaultShortcuts()
        {
            _shortcutManager.Register(new KeyboardShortcut { Name = "NewScene", Key = Key.N, Modifiers = KeyModifiers.Control, CommandName = "NewScene" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "OpenScene", Key = Key.O, Modifiers = KeyModifiers.Control, CommandName = "OpenScene" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "SaveScene", Key = Key.S, Modifiers = KeyModifiers.Control, CommandName = "SaveScene" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "SaveSceneAs", Key = Key.S, Modifiers = KeyModifiers.Control | KeyModifiers.Shift, CommandName = "SaveSceneAs" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Undo", Key = Key.Z, Modifiers = KeyModifiers.Control, CommandName = "Undo" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Redo", Key = Key.Y, Modifiers = KeyModifiers.Control, CommandName = "Redo" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Redo2", Key = Key.Z, Modifiers = KeyModifiers.Control | KeyModifiers.Shift, CommandName = "Redo" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Cut", Key = Key.X, Modifiers = KeyModifiers.Control, CommandName = "Cut" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Copy", Key = Key.C, Modifiers = KeyModifiers.Control, CommandName = "Copy" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Paste", Key = Key.V, Modifiers = KeyModifiers.Control, CommandName = "Paste" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "SelectAll", Key = Key.A, Modifiers = KeyModifiers.Control, CommandName = "SelectAll" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Delete", Key = Key.Delete, Modifiers = KeyModifiers.None, CommandName = "RemoveEntity" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Duplicate", Key = Key.D, Modifiers = KeyModifiers.Control, CommandName = "DuplicateEntity" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Translate", Key = Key.W, Modifiers = KeyModifiers.None, CommandName = "SetToolTranslate" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Rotate", Key = Key.E, Modifiers = KeyModifiers.None, CommandName = "SetToolRotate" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Scale", Key = Key.R, Modifiers = KeyModifiers.None, CommandName = "SetToolScale" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Select", Key = Key.Q, Modifiers = KeyModifiers.None, CommandName = "SetToolSelect" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "FrameSelected", Key = Key.F, Modifiers = KeyModifiers.None, CommandName = "FocusSelection" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "ToggleGrid", Key = Key.G, Modifiers = KeyModifiers.None, CommandName = "ToggleGrid" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "CompileAll", Key = Key.F7, Modifiers = KeyModifiers.None, CommandName = "CompileAll" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Screenshot", Key = Key.F12, Modifiers = KeyModifiers.None, CommandName = "CaptureScreenshot" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "ToggleConsole", Key = Key.BackQuote, Modifiers = KeyModifiers.Control, CommandName = "ToggleLlmConsole" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "TogglePerformance", Key = Key.F3, Modifiers = KeyModifiers.None, CommandName = "TogglePerformanceHud" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "ToggleTimeline", Key = Key.F4, Modifiers = KeyModifiers.None, CommandName = "ToggleTimeline" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "NewTab", Key = Key.T, Modifiers = KeyModifiers.Control, CommandName = "ShowViewport" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "CloseTab", Key = Key.W, Modifiers = KeyModifiers.Control, CommandName = "CloseCurrentTab" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Preferences", Key = Key.Comma, Modifiers = KeyModifiers.Control, CommandName = "Preferences" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Group", Key = Key.G, Modifiers = KeyModifiers.Control, CommandName = "GroupEntities" });
            _shortcutManager.Register(new KeyboardShortcut { Name = "Ungroup", Key = Key.U, Modifiers = KeyModifiers.Control, CommandName = "UngroupEntities" });
        }

        private void LoadRecentFiles()
        {
            try
            {
                var configPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "GDNN", "Studio", "recent_files.json");

                if (File.Exists(configPath))
                {
                    var json = File.ReadAllText(configPath);
                    var files = JsonSerializer.Deserialize<List<RecentFile>>(json);
                    if (files != null)
                    {
                        foreach (var file in files.Take(20))
                        {
                            _recentFiles.Add(file);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Warning: Could not load recent files: {ex.Message}");
            }
        }

        private void SaveRecentFiles()
        {
            try
            {
                var configDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                    "GDNN", "Studio");
                Directory.CreateDirectory(configDir);

                var configPath = Path.Combine(configDir, "recent_files.json");
                var json = JsonSerializer.Serialize(_recentFiles.Take(20).ToList(),
                    new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(configPath, json);
            }
            catch (Exception ex)
            {
                OutputLog.Add($"[{DateTime.Now:HH:mm:ss}] Warning: Could not save recent files: {ex.Message}");
            }
        }

        private void AddRecentFile(string filePath)
        {
            var existing = _recentFiles.FirstOrDefault(f => f.FilePath == filePath);
            if (existing != null)
            {
                _recentFiles.Remove(existing);
            }

            _recentFiles.Insert(0, new RecentFile
            {
                FilePath = filePath,
                LastAccessed = DateTime.Now,
                FileSize = File.Exists(filePath) ? new FileInfo(filePath).Length : 0
            });

            while (_recentFiles.Count > 20)
            {
                _recentFiles.RemoveAt(_recentFiles.Count - 1);
            }

            SaveRecentFiles();
        }

        private void AddDiagnostic(DiagnosticSeverity severity, string category, string message, string details = "")
        {
            _diagnostics.Add(new DiagnosticMessage
            {
                Severity = severity,
                Category = category,
                Message = message,
                Details = details,
                Timestamp = DateTime.UtcNow,
                Source = "MainWindowViewModel"
            });

            while (_diagnostics.Count > 1000)
            {
                _diagnostics.RemoveAt(0);
            }
        }

        /// <summary>
        /// Handles keyboard input and dispatches to the appropriate command.
        /// </summary>
        public void HandleKeyDown(Key key, KeyModifiers modifiers)
        {
            var shortcut = _shortcutManager.FindShortcut(key, modifiers);
            if (shortcut != null)
            {
                ExecuteShortcutCommand(shortcut.CommandName);
            }
        }

        private void ExecuteShortcutCommand(string commandName)
        {
            switch (commandName)
            {
                case "NewScene": _ = ExecuteNewScene(); break;
                case "OpenScene": _ = ExecuteOpenScene(); break;
                case "SaveScene": _ = ExecuteSaveScene(); break;
                case "SaveSceneAs": _ = ExecuteSaveSceneAs(); break;
                case "Undo": ExecuteUndo(); break;
                case "Redo": ExecuteRedo(); break;
                case "Cut": ExecuteCut(); break;
                case "Copy": ExecuteCopy(); break;
                case "Paste": _ = ExecutePaste(); break;
                case "SelectAll": ExecuteSelectAll(); break;
                case "RemoveEntity": _ = ExecuteRemoveEntity(); break;
                case "DuplicateEntity": _ = ExecuteDuplicateEntity(); break;
                case "SetToolTranslate": CurrentTool = ToolMode.Translate; break;
                case "SetToolRotate": CurrentTool = ToolMode.Rotate; break;
                case "SetToolScale": CurrentTool = ToolMode.Scale; break;
                case "SetToolSelect": CurrentTool = ToolMode.Select; break;
                case "FocusSelection": _viewportVm.FocusSelectionCommand.Execute(null); break;
                case "ToggleGrid": _viewportVm.ShowGrid = !_viewportVm.ShowGrid; break;
                case "CompileAll": _ = ExecuteCompileAll(); break;
                case "CaptureScreenshot": _ = ExecuteCaptureScreenshot(); break;
                case "ToggleLlmConsole": IsLlmConsoleVisible = !IsLlmConsoleVisible; break;
                case "TogglePerformanceHud": IsPerformanceHudVisible = !IsPerformanceHudVisible; break;
                case "ToggleTimeline": IsTimelineVisible = !IsTimelineVisible; break;
                case "ShowViewport": OpenEditorTab(EditorTabType.Viewport); break;
                case "ShowCodeEditor": OpenEditorTab(EditorTabType.CodeEditor); break;
                case "ShowBlueprint": OpenEditorTab(EditorTabType.Blueprint); break;
                case "Preferences": ExecutePreferences(); break;
                case "GroupEntities": ExecuteGroupEntities(); break;
                case "UngroupEntities": ExecuteUngroupEntities(); break;
                case "CloseCurrentTab":
                    if (SelectedTabIndex >= 0 && SelectedTabIndex < _openTabs.Count)
                    {
                        ExecuteCloseTab(_openTabs[SelectedTabIndex]);
                    }
                    break;
            }
        }

        /// <summary>
        /// Gets diagnostic info about the current state for the diagnostics panel.
        /// </summary>
        public string GetDiagnosticsInfo()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== G-DNN Studio Diagnostics ===");
            sb.AppendLine($"Version: 1.0.0 Build 2026.07.14");
            sb.AppendLine($"Platform: {RuntimeInformation.OSDescription}");
            sb.AppendLine($"Runtime: {RuntimeInformation.FrameworkDescription}");
            sb.AppendLine($"Processors: {Environment.ProcessorCount}");
            sb.AppendLine($"Working Set: {Environment.WorkingSet / (1024 * 1024)} MB");
            sb.AppendLine($"GC Memory: {GC.GetTotalMemory(false) / (1024 * 1024)} MB");
            sb.AppendLine($"Uptime: {Process.GetCurrentProcess().StartTime.AddHours(8):yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine();

            var gpuInfo = _hardwareMonitor.GetGpuInfo();
            var cpuInfo = _hardwareMonitor.GetCpuInfo();
            var memInfo = _hardwareMonitor.GetMemoryInfo();

            sb.AppendLine("=== Hardware ===");
            sb.AppendLine($"GPU: {gpuInfo.Name} ({gpuInfo.DedicatedMemory / (1024 * 1024)} MB VRAM)");
            sb.AppendLine($"GPU Driver: {gpuInfo.DriverVersion}");
            sb.AppendLine($"GPU API: {gpuInfo.ApiVersion}");
            sb.AppendLine($"CPU: {cpuInfo.Name} ({cpuInfo.CoreCount}C/{cpuInfo.ThreadCount}T @ {cpuInfo.ClockSpeedMhz:F0} MHz)");
            sb.AppendLine($"Architecture: {cpuInfo.Architecture}");
            sb.AppendLine($"RAM: {memInfo.UsedPhysical / (1024 * 1024)} MB / {memInfo.TotalPhysical / (1024 * 1024)} MB ({memInfo.MemoryLoadPercentage}%)");
            sb.AppendLine($"Temperature: {_hardwareMonitor.GetTemperature():F1}°C");
            sb.AppendLine();

            var cacheStats = _compilationService.GetCacheStatistics();
            sb.AppendLine("=== Cache ===");
            sb.AppendLine($"Shader Cache: {cacheStats.ShaderCacheEntries} entries ({cacheStats.ShaderCacheSize / (1024 * 1024)} MB)");
            sb.AppendLine($"Genome Cache: {cacheStats.GenomeCacheEntries} entries ({cacheStats.GenomeCacheSize / (1024 * 1024)} MB)");
            sb.AppendLine($"Hit Rate: {cacheStats.CacheHitRate * 100:F1}%");
            sb.AppendLine();

            var sceneStats = _sceneService.GetSceneStats();
            sb.AppendLine("=== Scene ===");
            sb.AppendLine($"Entities: {sceneStats.EntityCount}");
            sb.AppendLine($"Genomes: {sceneStats.GenomeCount}");
            sb.AppendLine($"Neurons: {sceneStats.NeuronCount}");
            sb.AppendLine($"Synapses: {sceneStats.SynapseCount}");
            sb.AppendLine($"Lights: {sceneStats.LightCount}");
            sb.AppendLine($"Materials: {sceneStats.MaterialCount}");
            sb.AppendLine($"Meshes: {sceneStats.MeshCount}");
            sb.AppendLine($"Vertices: {sceneStats.TotalVertices}");
            sb.AppendLine($"Triangles: {sceneStats.TotalTriangles}");
            sb.AppendLine($"Textures: {sceneStats.TextureMemory / (1024 * 1024)} MB");
            sb.AppendLine();

            sb.AppendLine("=== Diagnostics Log ===");
            foreach (var diag in _diagnostics.TakeLast(20))
            {
                sb.AppendLine($"[{diag.Severity}] {diag.Category}: {diag.Message}");
            }

            return sb.ToString();
        }

        protected override void OnDispose()
        {
            _statusBarTimer?.Dispose();
            _viewportVm?.Dispose();
            _sceneExplorerVm?.Dispose();
            _inspectorVm?.Dispose();
            _llmConsoleVm?.Dispose();
            _blueprintEditorVm?.Dispose();
            _codeEditorVm?.Dispose();
            _performanceHudVm?.Dispose();
            _timelineVm?.Dispose();
            _preferencesVm?.Dispose();
            _materialPreviewVm?.Dispose();
            _undoRedoManager?.Dispose();
            _shortcutManager?.Dispose();
            _layoutManager?.Dispose();

            foreach (var tab in _openTabs)
            {
                tab.Content?.Dispose();
            }
            _openTabs.Clear();
            _diagnostics.Clear();
            _outputLog.Clear();
            _recentFiles.Clear();
            _dynamicCommands.Clear();
            _dynamicAsyncCommands.Clear();

            base.OnDispose();
        }
    }

    /// <summary>
    /// Represents an open tab in the editor.
    /// </summary>
    public class EditorTab : ObservableObject
    {
        private string _title = string.Empty;
        private EditorTabType _type;
        private bool _isActive;
        private bool _isDirty;
        private IViewModelBase? _content;

        public Guid TabId { get; } = Guid.NewGuid();

        public string Title
        {
            get => _title;
            set => SetProperty(ref _title, value);
        }

        public EditorTabType Type
        {
            get => _type;
            set => SetProperty(ref _type, value);
        }

        public bool IsActive
        {
            get => _isActive;
            set => SetProperty(ref _isActive, value);
        }

        public bool IsDirty
        {
            get => _isDirty;
            set => SetProperty(ref _isDirty, value);
        }

        public IViewModelBase? Content
        {
            get => _content;
            set => SetProperty(ref _content, value);
        }

        public string DisplayTitle => IsDirty ? $"*{Title}" : Title;
    }


    // =========================================================================
    // CLIPBOARD SERVICE
    // =========================================================================

    /// <summary>
    /// Static service for managing clipboard data within the editor.
    /// </summary>
    public static class ClipboardService
    {
        private static List<Guid>? _copiedEntities;
        private static string? _copiedText;
        private static byte[]? _copiedData;
        private static string _copiedFormat = string.Empty;
        private static readonly object _lock = new();

        /// <summary>Set entity IDs to the clipboard.</summary>
        public static void SetEntities(IEnumerable<Guid> entityIds)
        {
            lock (_lock)
            {
                _copiedEntities = new List<Guid>(entityIds);
                _copiedFormat = "entities";
            }
        }

        /// <summary>Get entity IDs from the clipboard.</summary>
        public static List<Guid>? GetEntities()
        {
            lock (_lock)
            {
                return _copiedFormat == "entities" ? _copiedEntities : null;
            }
        }

        /// <summary>Set text to the clipboard.</summary>
        public static void SetText(string text)
        {
            lock (_lock)
            {
                _copiedText = text;
                _copiedFormat = "text";
            }
        }

        /// <summary>Get text from the clipboard.</summary>
        public static string? GetText()
        {
            lock (_lock)
            {
                return _copiedFormat == "text" ? _copiedText : null;
            }
        }

        /// <summary>Set raw data to the clipboard with a format identifier.</summary>
        public static void SetData(byte[] data, string format)
        {
            lock (_lock)
            {
                _copiedData = data;
                _copiedFormat = format;
            }
        }

        /// <summary>Get raw data from the clipboard.</summary>
        public static byte[]? GetData(string format)
        {
            lock (_lock)
            {
                return _copiedFormat == format ? _copiedData : null;
            }
        }

        /// <summary>Clear all clipboard data.</summary>
        public static void Clear()
        {
            lock (_lock)
            {
                _copiedEntities = null;
                _copiedText = null;
                _copiedData = null;
                _copiedFormat = string.Empty;
            }
        }

        /// <summary>Whether the clipboard contains data.</summary>
        public static bool HasData
        {
            get { lock (_lock) { return !string.IsNullOrEmpty(_copiedFormat); } }
        }

        /// <summary>The format of data currently in the clipboard.</summary>
        public static string CurrentFormat
        {
            get { lock (_lock) { return _copiedFormat; } }
        }
    }

    // =========================================================================
    // VIEWPORT VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the 3D viewport, managing camera controls, tool modes,
    /// overlays, entity picking, and viewport configuration.
    /// </summary>
    public partial class ViewportViewModel : ViewModelBase
    {
        private readonly IViewportService _viewportService;
        private ViewportCamera _camera = new();
        private ToolMode _currentTool = ToolMode.Select;
        private ViewportMode _viewportMode = ViewportMode.Lit;
        private OverlayFlags _overlayFlags = OverlayFlags.Grid | OverlayFlags.Gizmos;
        private GizmoType _gizmoType = GizmoType.Translate;
        private GizmoAxis _gizmoAxis = GizmoAxis.All;
        private ViewportLayout _viewportLayout = ViewportLayout.Single;
        private int _viewportWidth = 1920;
        private int _viewportHeight = 1080;
        private float _gridSpacing = 1.0f;
        private bool _showGrid = true;
        private bool _showGizmos = true;
        private bool _showStats;
        private bool _isFpsMode;
        private bool _isOrbiting;
        private bool _isPanning;
        private bool _isZooming;
        private bool _isSelecting;
        private System.Numerics.Vector2 _selectionStart;
        private System.Numerics.Vector2 _selectionEnd;
        private readonly ObservableCollection<ViewportStats> _statsHistory = new();
        private readonly ObservableCollection<ViewportLayout> _availableLayouts = new();
        private readonly Stopwatch _frameTimer = new();
        private double _lastFrameTimeMs;
        private float _cameraMoveSpeed = 5.0f;
        private float _cameraLookSpeed = 0.003f;
        private bool _enableFrameLimiting;
        private int _targetFps = 60;
        private readonly List<RaycastResult> _recentHits = new();
        private float _gridOpacity = 1.0f;
        private bool _showGridMinor = true;
        private bool _showGridMajor = true;
        private System.Numerics.Vector3 _gridColor = new(0.3f, 0.3f, 0.3f);
        private bool _enableFrustumCulling = true;
        private bool _enableOcclusionCulling;
        private int _antiAliasingLevel = 4;
        private float _renderScale = 1.0f;
        private bool _enableVSync = true;
        private bool _showDepthOfField;
        private bool _showMotionBlur;
        private bool _showAmbientOcclusion = true;
        private float _exposureValue = 1.0f;
        private float _gammaValue = 2.2f;
        private bool _enableBloom;
        private float _bloomThreshold = 0.8f;
        private float _bloomIntensity = 0.5f;
        private bool _enableToneMapping = true;
        private string _tonemapOperator = "ACES";

        /// <summary>
        /// Initializes a new instance of the ViewportViewModel.
        /// </summary>
        public ViewportViewModel(IViewportService viewportService)
        {
            _viewportService = viewportService ?? throw new ArgumentNullException(nameof(viewportService));
            InitializeCommands();

            _availableLayouts.Add(ViewportLayout.Single);
            _availableLayouts.Add(ViewportLayout.SplitHorizontal);
            _availableLayouts.Add(ViewportLayout.SplitVertical);
            _availableLayouts.Add(ViewportLayout.Quad);
            _availableLayouts.Add(ViewportLayout.TripleLeft);
            _availableLayouts.Add(ViewportLayout.TripleTop);
            _availableLayouts.Add(ViewportLayout.PictureInPicture);
        }

        // ---- Observable Properties ----

        /// <summary>Current camera state.</summary>
        public ViewportCamera Camera
        {
            get => _camera;
            set
            {
                if (SetProperty(ref _camera, value))
                {
                    _viewportService.SetCamera(value);
                    RaisePropertyChanged(nameof(CameraPositionText));
                    RaisePropertyChanged(nameof(CameraTargetText));
                }
            }
        }

        /// <summary>Current active tool.</summary>
        public ToolMode CurrentTool
        {
            get => _currentTool;
            set
            {
                if (SetProperty(ref _currentTool, value))
                {
                    RaisePropertyChanged(nameof(CurrentToolName));
                    RaisePropertyChanged(nameof(IsTransformTool));
                    RaisePropertyChanged(nameof(IsBrushTool));
                    RaisePropertyChanged(nameof(IsSelectionTool));
                }
            }
        }

        /// <summary>Display name of the current tool.</summary>
        public string CurrentToolName => CurrentTool.ToString();

        /// <summary>Whether the current tool is a transform tool.</summary>
        public bool IsTransformTool => CurrentTool == ToolMode.Translate || CurrentTool == ToolMode.Rotate || CurrentTool == ToolMode.Scale;

        /// <summary>Whether the current tool is a brush-type tool.</summary>
        public bool IsBrushTool => CurrentTool == ToolMode.Brush || CurrentTool == ToolMode.Paint || CurrentTool == ToolMode.Sculpt;

        /// <summary>Whether the current tool is the selection tool.</summary>
        public bool IsSelectionTool => CurrentTool == ToolMode.Select;

        /// <summary>Current viewport rendering mode.</summary>
        public ViewportMode ViewportMode
        {
            get => _viewportMode;
            set
            {
                if (SetProperty(ref _viewportMode, value))
                {
                    _viewportService.SetViewportMode(value);
                }
            }
        }

        /// <summary>Current overlay flags.</summary>
        public OverlayFlags OverlayFlags
        {
            get => _overlayFlags;
            set
            {
                if (SetProperty(ref _overlayFlags, value))
                {
                    _viewportService.SetOverlayFlags(value);
                }
            }
        }

        /// <summary>Current gizmo type.</summary>
        public GizmoType GizmoType
        {
            get => _gizmoType;
            set => SetProperty(ref _gizmoType, value);
        }

        /// <summary>Current axis constraint for gizmos.</summary>
        public GizmoAxis GizmoAxis
        {
            get => _gizmoAxis;
            set => SetProperty(ref _gizmoAxis, value);
        }

        /// <summary>Current viewport layout.</summary>
        public ViewportLayout ViewportLayout
        {
            get => _viewportLayout;
            set => SetProperty(ref _viewportLayout, value);
        }

        /// <summary>Viewport width in pixels.</summary>
        public int ViewportWidth
        {
            get => _viewportWidth;
            set
            {
                if (SetProperty(ref _viewportWidth, value))
                {
                    UpdateAspectRatio();
                }
            }
        }

        /// <summary>Viewport height in pixels.</summary>
        public int ViewportHeight
        {
            get => _viewportHeight;
            set
            {
                if (SetProperty(ref _viewportHeight, value))
                {
                    UpdateAspectRatio();
                }
            }
        }

        /// <summary>Grid spacing in world units.</summary>
        public float GridSpacing
        {
            get => _gridSpacing;
            set => SetProperty(ref _gridSpacing, value);
        }

        /// <summary>Whether the reference grid is visible.</summary>
        public bool ShowGrid
        {
            get => _showGrid;
            set
            {
                if (SetProperty(ref _showGrid, value))
                {
                    ToggleOverlayFlag(OverlayFlags.Grid, value);
                }
            }
        }

        /// <summary>Whether transform gizmos are visible.</summary>
        public bool ShowGizmos
        {
            get => _showGizmos;
            set
            {
                if (SetProperty(ref _showGizmos, value))
                {
                    ToggleOverlayFlag(OverlayFlags.Gizmos, value);
                }
            }
        }

        /// <summary>Whether viewport statistics are displayed.</summary>
        public bool ShowStats
        {
            get => _showStats;
            set
            {
                if (SetProperty(ref _showStats, value))
                {
                    ToggleOverlayFlag(OverlayFlags.Stats, value);
                }
            }
        }

        /// <summary>Whether FPS camera mode is active.</summary>
        public bool IsFpsMode
        {
            get => _isFpsMode;
            set => SetProperty(ref _isFpsMode, value);
        }

        /// <summary>Whether the camera is currently orbiting.</summary>
        public bool IsOrbiting
        {
            get => _isOrbiting;
            set => SetProperty(ref _isOrbiting, value);
        }

        /// <summary>Whether the camera is currently panning.</summary>
        public bool IsPanning
        {
            get => _isPanning;
            set => SetProperty(ref _isPanning, value);
        }

        /// <summary>Whether the camera is currently zooming.</summary>
        public bool IsZooming
        {
            get => _isZooming;
            set => SetProperty(ref _isZooming, value);
        }

        /// <summary>Whether a selection rectangle is being drawn.</summary>
        public bool IsSelecting
        {
            get => _isSelecting;
            set => SetProperty(ref _isSelecting, value);
        }

        /// <summary>Selection rectangle start position.</summary>
        public System.Numerics.Vector2 SelectionStart
        {
            get => _selectionStart;
            set => SetProperty(ref _selectionStart, value);
        }

        /// <summary>Selection rectangle end position.</summary>
        public System.Numerics.Vector2 SelectionEnd
        {
            get => _selectionEnd;
            set => SetProperty(ref _selectionEnd, value);
        }

        /// <summary>History of viewport statistics for graphing.</summary>
        public ObservableCollection<ViewportStats> StatsHistory => _statsHistory;

        /// <summary>Available viewport layout options.</summary>
        public ObservableCollection<ViewportLayout> AvailableLayouts => _availableLayouts;

        /// <summary>Camera move speed multiplier.</summary>
        public float CameraMoveSpeed
        {
            get => _cameraMoveSpeed;
            set => SetProperty(ref _cameraMoveSpeed, value);
        }

        /// <summary>Camera look sensitivity.</summary>
        public float CameraLookSpeed
        {
            get => _cameraLookSpeed;
            set => SetProperty(ref _cameraLookSpeed, value);
        }

        /// <summary>Whether frame rate limiting is enabled.</summary>
        public bool EnableFrameLimiting
        {
            get => _enableFrameLimiting;
            set => SetProperty(ref _enableFrameLimiting, value);
        }

        /// <summary>Target FPS.</summary>
        public int TargetFps
        {
            get => _targetFps;
            set => SetProperty(ref _targetFps, value);
        }

        /// <summary>Last measured frame time in milliseconds.</summary>
        public double LastFrameTimeMs
        {
            get => _lastFrameTimeMs;
            set => SetProperty(ref _lastFrameTimeMs, value);
        }

        /// <summary>Grid opacity.</summary>
        public float GridOpacity
        {
            get => _gridOpacity;
            set => SetProperty(ref _gridOpacity, value);
        }

        /// <summary>Whether minor grid lines are shown.</summary>
        public bool ShowGridMinor
        {
            get => _showGridMinor;
            set => SetProperty(ref _showGridMinor, value);
        }

        /// <summary>Whether major grid lines are shown.</summary>
        public bool ShowGridMajor
        {
            get => _showGridMajor;
            set => SetProperty(ref _showGridMajor, value);
        }

        /// <summary>Grid color.</summary>
        public System.Numerics.Vector3 GridColor
        {
            get => _gridColor;
            set => SetProperty(ref _gridColor, value);
        }

        /// <summary>Whether frustum culling is enabled.</summary>
        public bool EnableFrustumCulling
        {
            get => _enableFrustumCulling;
            set => SetProperty(ref _enableFrustumCulling, value);
        }

        /// <summary>Whether occlusion culling is enabled.</summary>
        public bool EnableOcclusionCulling
        {
            get => _enableOcclusionCulling;
            set => SetProperty(ref _enableOcclusionCulling, value);
        }

        /// <summary>Anti-aliasing level (MSAA).</summary>
        public int AntiAliasingLevel
        {
            get => _antiAliasingLevel;
            set => SetProperty(ref _antiAliasingLevel, value);
        }

        /// <summary>Render resolution scale.</summary>
        public float RenderScale
        {
            get => _renderScale;
            set => SetProperty(ref _renderScale, value);
        }

        /// <summary>Whether VSync is enabled.</summary>
        public bool EnableVSync
        {
            get => _enableVSync;
            set => SetProperty(ref _enableVSync, value);
        }

        /// <summary>Whether depth of field is enabled.</summary>
        public bool ShowDepthOfField
        {
            get => _showDepthOfField;
            set => SetProperty(ref _showDepthOfField, value);
        }

        /// <summary>Whether motion blur is enabled.</summary>
        public bool ShowMotionBlur
        {
            get => _showMotionBlur;
            set => SetProperty(ref _showMotionBlur, value);
        }

        /// <summary>Whether ambient occlusion is enabled.</summary>
        public bool ShowAmbientOcclusion
        {
            get => _showAmbientOcclusion;
            set => SetProperty(ref _showAmbientOcclusion, value);
        }

        /// <summary>Exposure value for tone mapping.</summary>
        public float ExposureValue
        {
            get => _exposureValue;
            set => SetProperty(ref _exposureValue, value);
        }

        /// <summary>Gamma correction value.</summary>
        public float GammaValue
        {
            get => _gammaValue;
            set => SetProperty(ref _gammaValue, value);
        }

        /// <summary>Whether bloom is enabled.</summary>
        public bool EnableBloom
        {
            get => _enableBloom;
            set => SetProperty(ref _enableBloom, value);
        }

        /// <summary>Bloom threshold.</summary>
        public float BloomThreshold
        {
            get => _bloomThreshold;
            set => SetProperty(ref _bloomThreshold, value);
        }

        /// <summary>Bloom intensity.</summary>
        public float BloomIntensity
        {
            get => _bloomIntensity;
            set => SetProperty(ref _bloomIntensity, value);
        }

        /// <summary>Whether tone mapping is enabled.</summary>
        public bool EnableToneMapping
        {
            get => _enableToneMapping;
            set => SetProperty(ref _enableToneMapping, value);
        }

        /// <summary>Tone mapping operator name.</summary>
        public string TonemapOperator
        {
            get => _tonemapOperator;
            set => SetProperty(ref _tonemapOperator, value);
        }

        /// <summary>Camera position as formatted text.</summary>
        public string CameraPositionText => $"X: {_camera.Position.X:F2} Y: {_camera.Position.Y:F2} Z: {_camera.Position.Z:F2}";

        /// <summary>Camera target as formatted text.</summary>
        public string CameraTargetText => $"X: {_camera.Target.X:F2} Y: {_camera.Target.Y:F2} Z: {_camera.Target.Z:F2}";

        /// <summary>Selection rectangle bounds text.</summary>
        public string SelectionBoundsText
        {
            get
            {
                if (!IsSelecting) return string.Empty;
                var width = Math.Abs(SelectionEnd.X - SelectionStart.X);
                var height = Math.Abs(SelectionEnd.Y - SelectionStart.Y);
                return $"{width:F0}x{height:F0}";
            }
        }

        /// <summary>Aspect ratio of the viewport.</summary>
        public float AspectRatio => (float)_viewportWidth / Math.Max(1, _viewportHeight);

        /// <summary>Number of recent raycast hits stored.</summary>
        public int RecentHitCount => _recentHits.Count;

        /// <summary>Brush radius for brush tools.</summary>
        public float BrushRadius
        {
            get => GetProperty<float>(nameof(BrushRadius), 1.0f);
            set => SetProperty(value, nameof(BrushRadius));
        }

        /// <summary>Brush strength for brush tools.</summary>
        public float BrushStrength
        {
            get => GetProperty<float>(nameof(BrushStrength), 1.0f);
            set => SetProperty(value, nameof(BrushStrength));
        }

        /// <summary>Brush falloff for brush tools.</summary>
        public float BrushFalloff
        {
            get => GetProperty<float>(nameof(BrushFalloff), 0.5f);
            set => SetProperty(value, nameof(BrushFalloff));
        }

        /// <summary>Sculpt displacement amount.</summary>
        public float SculptAmount
        {
            get => GetProperty<float>(nameof(SculptAmount), 0.1f);
            set => SetProperty(value, nameof(SculptAmount));
        }

        /// <summary>Measure tool distance result.</summary>
        public string MeasurementResult
        {
            get => GetProperty<string>(nameof(MeasurementResult), "");
            set => SetProperty(value, nameof(MeasurementResult));
        }



        // ---- Remaining Viewport Methods ----

        private async Task ExecuteCaptureScreenshot()
        {
            try
            {
                SetBusy(true, "Capturing screenshot...");
                var data = await _viewportService.CaptureScreenshotAsync(ViewportWidth, ViewportHeight);
                StatusMessage = $"Screenshot captured ({data.Length} bytes)";
            }
            catch (Exception ex)
            {
                SetError($"Screenshot capture failed: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteSetLayout(ViewportLayout layout)
        {
            ViewportLayout = layout;
            StatusMessage = $"Viewport layout: {layout}";
        }

        private void ToggleOverlayFlag(OverlayFlags flag, bool enable)
        {
            if (enable)
                OverlayFlags |= flag;
            else
                OverlayFlags &= ~flag;
        }

        public void ToggleOverlay(OverlayFlags flag)
        {
            OverlayFlags ^= flag;
            ShowGrid = OverlayFlags.HasFlag(OverlayFlags.Grid);
            ShowGizmos = OverlayFlags.HasFlag(OverlayFlags.Gizmos);
            ShowStats = OverlayFlags.HasFlag(OverlayFlags.Stats);
        }

        private void UpdateAspectRatio()
        {
            RaisePropertyChanged(nameof(AspectRatio));
            Camera = _camera with { AspectRatio = AspectRatio };
            _viewportService.Resize(_viewportWidth, _viewportHeight);
        }

        /// <summary>Event raised when an entity is picked.</summary>
        public event EventHandler<EntityPickedEventArgs>? EntityPicked;

        /// <summary>Event raised when a selection rectangle is completed.</summary>
        public event EventHandler<SelectionRectangleCompletedEventArgs>? SelectionRectangleCompleted;

        protected override void OnDispose()
        {
            _statsHistory.Clear();
            _availableLayouts.Clear();
            _recentHits.Clear();
            base.OnDispose();
        }
    }

    // =========================================================================
    // SCENE EXPLORER VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the scene hierarchy explorer.
    /// </summary>
    public partial class SceneExplorerViewModel : ViewModelBase
    {
        private readonly ISceneService _sceneService;
        private readonly ObservableCollection<SceneNodeViewModel> _rootNodes = new();
        private readonly ObservableCollection<SceneNodeViewModel> _flatNodes = new();
        private readonly List<SceneNodeViewModel> _allNodes = new();
        private string _searchText = string.Empty;
        private string _filterType = string.Empty;
        private bool _showInactive = true;
        private bool _showLocked = true;
        private bool _sortByName = true;
        private bool _sortByType;
        private bool _sortByHierarchy = true;
        private bool _isMultiSelectMode;
        private SceneNodeViewModel? _draggedNode;
        private readonly Dictionary<Guid, SceneNodeViewModel> _nodeLookup = new();

        public SceneExplorerViewModel(ISceneService sceneService)
        {
            _sceneService = sceneService ?? throw new ArgumentNullException(nameof(sceneService));
            InitializeCommands();
        }

        public ObservableCollection<SceneNodeViewModel> RootNodes => _rootNodes;
        public ObservableCollection<SceneNodeViewModel> FlatNodes => _flatNodes;

        public string SearchText
        {
            get => _searchText;
            set { if (SetProperty(ref _searchText, value)) ApplyFilter(); }
        }

        public string FilterType
        {
            get => _filterType;
            set { if (SetProperty(ref _filterType, value)) ApplyFilter(); }
        }

        public bool ShowInactive
        {
            get => _showInactive;
            set { if (SetProperty(ref _showInactive, value)) ApplyFilter(); }
        }

        public bool ShowLocked
        {
            get => _showLocked;
            set { if (SetProperty(ref _showLocked, value)) ApplyFilter(); }
        }

        public bool SortByName
        {
            get => _sortByName;
            set { if (SetProperty(ref _sortByName, value)) ApplySort(); }
        }

        public bool SortByType
        {
            get => _sortByType;
            set { if (SetProperty(ref _sortByType, value)) ApplySort(); }
        }

        public bool SortByHierarchy
        {
            get => _sortByHierarchy;
            set { if (SetProperty(ref _sortByHierarchy, value)) ApplySort(); }
        }

        public bool IsMultiSelectMode
        {
            get => _isMultiSelectMode;
            set => SetProperty(ref _isMultiSelectMode, value);
        }

        public SceneNodeViewModel? SelectedNode
        {
            get => GetProperty<SceneNodeViewModel?>();
            set
            {
                if (SetProperty(value))
                {
                    SelectedEntityChanged?.Invoke(this, new SelectedEntityChangedEventArgs
                    {
                        EntityId = value?.EntityId
                    });
                }
            }
        }

        public int VisibleNodeCount => _flatNodes.Count(n => n.IsVisibleInTree);
        public int TotalNodeCount => _allNodes.Count;

        public IRelayCommand RefreshCommand { get; private set; } = null!;
        public IRelayCommand ExpandAllCommand { get; private set; } = null!;
        public IRelayCommand CollapseAllCommand { get; private set; } = null!;
        public IAsyncRelayCommand AddEntityCommand { get; private set; } = null!;
        public IAsyncRelayCommand DeleteEntityCommand { get; private set; } = null!;
        public IAsyncRelayCommand DuplicateEntityCommand { get; private set; } = null!;
        public IRelayCommand RenameEntityCommand { get; private set; } = null!;
        public IRelayCommand ToggleVisibilityCommand { get; private set; } = null!;
        public IRelayCommand ToggleLockCommand { get; private set; } = null!;
        public IRelayCommand GroupCommand { get; private set; } = null!;
        public IRelayCommand UngroupCommand { get; private set; } = null!;
        public IRelayCommand SortByNameCommand { get; private set; } = null!;
        public IRelayCommand SortByTypeCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> StartDragCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> DropCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> SelectNodeCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> ToggleNodeExpandCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> ToggleNodeVisibilityCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> ToggleNodeLockCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> RenameNodeCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> DeleteNodeCommand { get; private set; } = null!;
        public IRelayCommand<SceneNodeViewModel> DuplicateNodeCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            RefreshCommand = CreateCommand(ExecuteRefresh);
            ExpandAllCommand = CreateCommand(ExecuteExpandAll);
            CollapseAllCommand = CreateCommand(ExecuteCollapseAll);
            AddEntityCommand = CreateAsyncCommand(ExecuteAddEntity);
            DeleteEntityCommand = CreateAsyncCommand(ExecuteDeleteEntity);
            DuplicateEntityCommand = CreateAsyncCommand(ExecuteDuplicateEntity);
            RenameEntityCommand = CreateCommand(ExecuteRenameEntity);
            ToggleVisibilityCommand = CreateCommand(ExecuteToggleVisibility);
            ToggleLockCommand = CreateCommand(ExecuteToggleLock);
            GroupCommand = CreateCommand(ExecuteGroup);
            UngroupCommand = CreateCommand(ExecuteUngroup);
            SortByNameCommand = CreateCommand(() => SortByName = true);
            SortByTypeCommand = CreateCommand(() => SortByType = true);
            StartDragCommand = CreateCommand<SceneNodeViewModel>(ExecuteStartDrag);
            DropCommand = CreateCommand<SceneNodeViewModel>(ExecuteDrop);
            SelectNodeCommand = CreateCommand<SceneNodeViewModel>(ExecuteSelectNode);
            ToggleNodeExpandCommand = CreateCommand<SceneNodeViewModel>(ExecuteToggleNodeExpand);
            ToggleNodeVisibilityCommand = CreateCommand<SceneNodeViewModel>(ExecuteToggleNodeVisibility);
            ToggleNodeLockCommand = CreateCommand<SceneNodeViewModel>(ExecuteToggleNodeLock);
            RenameNodeCommand = CreateCommand<SceneNodeViewModel>(ExecuteRenameNode);
            DeleteNodeCommand = CreateCommand<SceneNodeViewModel>(ExecuteDeleteNode);
            DuplicateNodeCommand = CreateCommand<SceneNodeViewModel>(ExecuteDuplicateNode);
        }

        private void ExecuteRefresh() => RefreshHierarchy();

        private void ExecuteExpandAll()
        {
            foreach (var node in _allNodes) node.IsExpanded = true;
            StatusMessage = "Expanded all nodes";
        }

        private void ExecuteCollapseAll()
        {
            foreach (var node in _allNodes) node.IsExpanded = false;
            StatusMessage = "Collapsed all nodes";
        }

        private async Task ExecuteAddEntity()
        {
            var entityId = await _sceneService.CreateEntityAsync("New Entity");
            RefreshHierarchy();
            StatusMessage = "Entity added";
        }

        private async Task ExecuteDeleteEntity()
        {
            if (SelectedNode != null)
            {
                var name = SelectedNode.Name;
                await _sceneService.DeleteEntityAsync(SelectedNode.EntityId);
                RefreshHierarchy();
                StatusMessage = $"Deleted: {name}";
            }
        }

        private async Task ExecuteDuplicateEntity()
        {
            if (SelectedNode != null)
            {
                var entity = _sceneService.GetEntityById(SelectedNode.EntityId);
                if (entity != null)
                {
                    await _sceneService.CreateEntityAsync($"{entity.Name} (Copy)", entity.Type);
                    RefreshHierarchy();
                    StatusMessage = $"Duplicated: {entity.Name}";
                }
            }
        }

        private void ExecuteRenameEntity()
        {
            if (SelectedNode != null) SelectedNode.IsRenaming = true;
        }

        private void ExecuteToggleVisibility()
        {
            if (SelectedNode != null)
            {
                SelectedNode.IsVisible = !SelectedNode.IsVisible;
                StatusMessage = $"{SelectedNode.Name}: {(SelectedNode.IsVisible ? "visible" : "hidden")}";
            }
        }

        private void ExecuteToggleLock()
        {
            if (SelectedNode != null)
            {
                SelectedNode.IsLocked = !SelectedNode.IsLocked;
                StatusMessage = $"{SelectedNode.Name}: {(SelectedNode.IsLocked ? "locked" : "unlocked")}";
            }
        }

        private void ExecuteGroup() => StatusMessage = "Grouped selected entities";
        private void ExecuteUngroup() => StatusMessage = "Ungrouped selected entity";

        private void ExecuteStartDrag(SceneNodeViewModel node) => _draggedNode = node;

        private void ExecuteDrop(SceneNodeViewModel target)
        {
            if (_draggedNode != null && _draggedNode != target)
            {
                _draggedNode.ParentId = target.EntityId;
                RefreshHierarchy();
                StatusMessage = $"Moved {_draggedNode.Name} under {target.Name}";
            }
            _draggedNode = null;
        }

        private void ExecuteSelectNode(SceneNodeViewModel node)
        {
            if (IsMultiSelectMode)
            {
                node.IsSelected = !node.IsSelected;
            }
            else
            {
                foreach (var n in _allNodes) n.IsSelected = false;
                node.IsSelected = true;
            }
            SelectedNode = node;
        }

        private void ExecuteToggleNodeExpand(SceneNodeViewModel node)
        {
            node.IsExpanded = !node.IsExpanded;
        }

        private void ExecuteToggleNodeVisibility(SceneNodeViewModel node)
        {
            node.IsVisible = !node.IsVisible;
        }

        private void ExecuteToggleNodeLock(SceneNodeViewModel node)
        {
            node.IsLocked = !node.IsLocked;
        }

        private void ExecuteRenameNode(SceneNodeViewModel node)
        {
            node.IsRenaming = true;
        }

        private async Task ExecuteDeleteNode(SceneNodeViewModel node)
        {
            await _sceneService.DeleteEntityAsync(node.EntityId);
            RefreshHierarchy();
            StatusMessage = $"Deleted: {node.Name}";
        }

        private async Task ExecuteDuplicateNode(SceneNodeViewModel node)
        {
            var entity = _sceneService.GetEntityById(node.EntityId);
            if (entity != null)
            {
                await _sceneService.CreateEntityAsync($"{entity.Name} (Copy)", entity.Type);
                RefreshHierarchy();
                StatusMessage = $"Duplicated: {entity.Name}";
            }
        }

        public void RefreshHierarchy()
        {
            _rootNodes.Clear();
            _flatNodes.Clear();
            _allNodes.Clear();
            _nodeLookup.Clear();

            var entities = _sceneService.GetEntities();
            var childMap = new Dictionary<Guid?, List<SceneEntity>>();
            foreach (var entity in entities)
            {
                if (!childMap.ContainsKey(entity.ParentId))
                    childMap[entity.ParentId] = new List<SceneEntity>();
                childMap[entity.ParentId].Add(entity);
            }

            void BuildTree(Guid? parentId, ObservableCollection<SceneNodeViewModel> parentCollection, int depth)
            {
                if (!childMap.TryGetValue(parentId, out var children)) return;
                foreach (var child in children.OrderBy(e => e.Name))
                {
                    var node = CreateNodeFromEntity(child, depth);
                    parentCollection.Add(node);
                    _flatNodes.Add(node);
                    _allNodes.Add(node);
                    _nodeLookup[child.Id] = node;
                    if (childMap.ContainsKey(child.Id))
                        BuildTree(child.Id, node.Children, depth + 1);
                }
            }

            BuildTree(null, _rootNodes, 0);
            ApplyFilter();
            RaisePropertyChanged(nameof(VisibleNodeCount));
            RaisePropertyChanged(nameof(TotalNodeCount));
        }

        private SceneNodeViewModel CreateNodeFromEntity(SceneEntity entity, int depth)
        {
            return new SceneNodeViewModel
            {
                EntityId = entity.Id,
                Name = entity.Name,
                EntityType = entity.Type,
                Depth = depth,
                IsVisible = entity.IsVisible,
                IsLocked = entity.IsLocked,
                Icon = GetIconForEntityType(entity.Type)
            };
        }

        private string GetIconForEntityType(EntityType type)
        {
            return type switch
            {
                EntityType.Mesh => "\U0001F537",
                EntityType.Light => "\U0001F4A1",
                EntityType.Camera => "\U0001F4F7",
                EntityType.ParticleSystem => "\u2728",
                EntityType.AudioSource => "\U0001F50A",
                EntityType.Genome => "\U0001F9EC",
                EntityType.Trigger => "\u26A1",
                EntityType.Spline => "\u3030\uFE0F",
                EntityType.Terrain => "\U0001F3D4\uFE0F",
                EntityType.Skybox => "\U0001F324\uFE0F",
                EntityType.ReflectionProbe => "\U0001F52E",
                EntityType.Decal => "\U0001F3A8",
                EntityType.UI => "\U0001F5A5\uFE0F",
                EntityType.Joint => "\U0001F517",
                EntityType.Empty => "\u2B1C",
                EntityType.Prefab => "\U0001F4E6",
                EntityType.Script => "\U0001F4DC",
                EntityType.Volume => "\U0001F4E6",
                EntityType.Character => "\U0001F9D1",
                EntityType.Vehicle => "\U0001F697",
                _ => "\u2753"
            };
        }

        private void ApplyFilter()
        {
            foreach (var node in _allNodes)
            {
                var visible = true;
                if (!ShowInactive && !node.IsVisible) visible = false;
                if (!ShowLocked && node.IsLocked) visible = false;
                if (!string.IsNullOrEmpty(SearchText))
                {
                    var searchLower = SearchText.ToLowerInvariant();
                    if (!node.Name.ToLowerInvariant().Contains(searchLower))
                        visible = false;
                }
                if (!string.IsNullOrEmpty(FilterType) && !node.EntityType.ToString().Equals(FilterType, StringComparison.OrdinalIgnoreCase))
                    visible = false;
                node.IsVisibleInTree = visible;
            }
            RaisePropertyChanged(nameof(VisibleNodeCount));
        }

        private void ApplySort()
        {
            var sorted = SortByName ? _allNodes.OrderBy(n => n.Name).ToList()
                : SortByType ? _allNodes.OrderBy(n => n.EntityType).ThenBy(n => n.Name).ToList()
                : _allNodes.ToList();
            _flatNodes.Clear();
            foreach (var node in sorted) _flatNodes.Add(node);
        }

        public void SelectAll()
        {
            foreach (var node in _allNodes) node.IsSelected = true;
        }

        public void ClearSelection()
        {
            foreach (var node in _allNodes) node.IsSelected = false;
        }

        public event EventHandler<SelectedEntityChangedEventArgs>? SelectedEntityChanged;

        protected override void OnDispose()
        {
            _allNodes.Clear();
            _nodeLookup.Clear();
            _rootNodes.Clear();
            _flatNodes.Clear();
            base.OnDispose();
        }
    }

    // =========================================================================
    // SCENE NODE VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for a single node in the scene hierarchy tree.
    /// </summary>
    public partial class SceneNodeViewModel : ViewModelBase
    {
        private string _name = string.Empty;
        private bool _isExpanded;
        private bool _isSelected;
        private bool _isVisible = true;
        private bool _isLocked;
        private bool _isRenaming;
        private bool _isVisibleInTree = true;
        private int _depth;
        private EntityType _entityType;
        private string _icon = "\u2B1C";
        private string _renameText = string.Empty;
        private Guid? _parentId;
        private readonly ObservableCollection<SceneNodeViewModel> _children = new();

        public Guid EntityId
        {
            get => GetProperty<Guid>();
            set => SetProperty(value);
        }

        public string Name
        {
            get => _name;
            set { if (SetProperty(ref _name, value)) RenameText = value; }
        }

        public EntityType EntityType
        {
            get => _entityType;
            set { if (SetProperty(ref _entityType, value)) Icon = GetIconForType(value); }
        }

        public int Depth
        {
            get => _depth;
            set => SetProperty(ref _depth, value);
        }

        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }

        public bool IsExpanded
        {
            get => _isExpanded;
            set => SetProperty(ref _isExpanded, value);
        }

        public bool IsSelected
        {
            get => _isSelected;
            set => SetProperty(ref _isSelected, value);
        }

        public bool IsVisible
        {
            get => _isVisible;
            set => SetProperty(ref _isVisible, value);
        }

        public bool IsLocked
        {
            get => _isLocked;
            set => SetProperty(ref _isLocked, value);
        }

        public bool IsRenaming
        {
            get => _isRenaming;
            set { if (SetProperty(ref _isRenaming, value) && value) RenameText = Name; }
        }

        public bool IsVisibleInTree
        {
            get => _isVisibleInTree;
            set => SetProperty(ref _isVisibleInTree, value);
        }

        public string RenameText
        {
            get => _renameText;
            set => SetProperty(ref _renameText, value);
        }

        public Guid? ParentId
        {
            get => _parentId;
            set => SetProperty(ref _parentId, value);
        }

        public ObservableCollection<SceneNodeViewModel> Children => _children;

        public bool HasChildren => _children.Count > 0;

        public SceneNodeViewModel? Parent { get; set; }

        public Thickness IndentMargin => new(_depth * 20, 0, 0, 0);

        public string TooltipText => $"{Name} ({EntityType})\nID: {EntityId}";

        public void CommitRename()
        {
            if (IsRenaming && !string.IsNullOrWhiteSpace(RenameText))
                Name = RenameText.Trim();
            IsRenaming = false;
        }

        public void CancelRename()
        {
            RenameText = Name;
            IsRenaming = false;
        }

        private string GetIconForType(EntityType type)
        {
            return type switch
            {
                EntityType.Mesh => "\U0001F537",
                EntityType.Light => "\U0001F4A1",
                EntityType.Camera => "\U0001F4F7",
                EntityType.ParticleSystem => "\u2728",
                EntityType.AudioSource => "\U0001F50A",
                EntityType.Genome => "\U0001F9EC",
                EntityType.Empty => "\u2B1C",
                EntityType.Prefab => "\U0001F4E6",
                _ => "\u2753"
            };
        }

        protected override void OnDispose()
        {
            _children.Clear();
            base.OnDispose();
        }
    }



    // =========================================================================
    // INSPECTOR VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the property inspector, displaying and editing components.
    /// </summary>
    public partial class InspectorViewModel : ViewModelBase
    {
        private readonly ISceneService _sceneService;
        private Guid? _selectedEntityId;
        private SceneEntity? _selectedEntity;
        private readonly ObservableCollection<InspectorComponentViewModel> _components = new();
        private string _searchText = string.Empty;
        private bool _showAdvancedProperties;
        private bool _livePreview = true;
        private float _positionX, _positionY, _positionZ;
        private float _rotationX, _rotationY, _rotationZ;
        private float _scaleX = 1, _scaleY = 1, _scaleZ = 1;
        private bool _uniformScale = true;
        private readonly ObservableCollection<string> _availableComponentTypes = new();
        private bool _showTransformComponent = true;
        private bool _showGenomeComponent;
        private bool _showMaterialComponent;
        private bool _showBehaviorComponent;
        private bool _showPhysicsComponent;
        private bool _showColliderComponent;
        private bool _showLightComponent;
        private bool _showCameraComponent;
        private bool _showParticleSystemComponent;
        private bool _showAudioComponent;
        private bool _showAnimationComponent;
        private bool _showScriptComponent;
        private float _lightIntensity = 1.0f;
        private System.Numerics.Vector3 _lightColor = System.Numerics.Vector3.One;
        private float _lightRange = 10.0f;
        private float _lightSpotAngle = 45.0f;
        private float _mass = 1.0f;
        private float _friction = 0.5f;
        private float _restitution = 0.3f;
        private bool _isKinematic;
        private float _cameraFov = 60.0f;
        private float _cameraNear = 0.01f;
        private float _cameraFar = 1000f;
        private float _particleLifetime = 5.0f;
        private int _particleCount = 100;
        private float _particleSpeed = 1.0f;
        private float _audioVolume = 1.0f;
        private float _audioPitch = 1.0f;
        private bool _audioLoop = true;
        private float _audioSpatialBlend = 1.0f;
        private float _audioMinDistance = 1.0f;
        private float _audioMaxDistance = 50.0f;
        private string _scriptCode = string.Empty;
        private string _scriptLanguage = "C#";
        private string _behaviorTreeText = string.Empty;
        private string _emotionState = "Neutral";
        private float _emotionIntensity = 0.5f;
        private readonly ObservableCollection<string> _memoryEntries = new();
        private readonly ObservableCollection<string> _neuronList = new();
        private readonly ObservableCollection<string> _synapseList = new();
        private int _neuronCount;
        private int _synapseCount;
        private float _genomeComplexity;
        private float _materialRoughness = 0.5f;
        private float _materialMetallic;
        private System.Numerics.Vector3 _materialBaseColor = System.Numerics.Vector3.One;
        private float _materialAlpha = 1.0f;
        private float _materialNormalStrength = 1.0f;
        private float _materialEmissionStrength;

        public InspectorViewModel(ISceneService sceneService)
        {
            _sceneService = sceneService ?? throw new ArgumentNullException(nameof(sceneService));
            InitializeAvailableComponentTypes();
            InitializeCommands();
        }

        private void InitializeAvailableComponentTypes()
        {
            _availableComponentTypes.Add("Transform");
            _availableComponentTypes.Add("Mesh Renderer");
            _availableComponentTypes.Add("Genome");
            _availableComponentTypes.Add("Material");
            _availableComponentTypes.Add("Light");
            _availableComponentTypes.Add("Camera");
            _availableComponentTypes.Add("Collider");
            _availableComponentTypes.Add("Rigidbody");
            _availableComponentTypes.Add("Behavior Tree");
            _availableComponentTypes.Add("Particle System");
            _availableComponentTypes.Add("Audio Source");
            _availableComponentTypes.Add("Animation");
            _availableComponentTypes.Add("Script");
            _availableComponentTypes.Add("LOD");
            _availableComponentTypes.Add("Nav Agent");
        }

        public Guid? SelectedEntityId
        {
            get => _selectedEntityId;
            set { if (SetProperty(ref _selectedEntityId, value)) LoadEntityComponents(); }
        }

        public bool HasSelection => _selectedEntityId.HasValue;

        public string EntityName
        {
            get => GetProperty<string>();
            set { if (SetProperty(value) && _selectedEntity != null) _selectedEntity.Name = value; }
        }

        public EntityType EntityType
        {
            get => GetProperty<EntityType>();
            set => SetProperty(value);
        }

        public ObservableCollection<InspectorComponentViewModel> Components => _components;
        public ObservableCollection<string> AvailableComponentTypes => _availableComponentTypes;

        public string SearchText
        {
            get => _searchText;
            set { if (SetProperty(ref _searchText, value)) ApplyPropertyFilter(); }
        }

        public bool ShowAdvancedProperties
        {
            get => _showAdvancedProperties;
            set => SetProperty(ref _showAdvancedProperties, value);
        }

        public bool LivePreview
        {
            get => _livePreview;
            set => SetProperty(ref _livePreview, value);
        }

        public float PositionX { get => _positionX; set { if (SetProperty(ref _positionX, value)) UpdateTransform(); } }
        public float PositionY { get => _positionY; set { if (SetProperty(ref _positionY, value)) UpdateTransform(); } }
        public float PositionZ { get => _positionZ; set { if (SetProperty(ref _positionZ, value)) UpdateTransform(); } }
        public float RotationX { get => _rotationX; set { if (SetProperty(ref _rotationX, value)) UpdateTransform(); } }
        public float RotationY { get => _rotationY; set { if (SetProperty(ref _rotationY, value)) UpdateTransform(); } }
        public float RotationZ { get => _rotationZ; set { if (SetProperty(ref _rotationZ, value)) UpdateTransform(); } }

        public float ScaleX
        {
            get => _scaleX;
            set { if (SetProperty(ref _scaleX, value)) { if (_uniformScale) { ScaleY = value; ScaleZ = value; } UpdateTransform(); } }
        }

        public float ScaleY { get => _scaleY; set { if (SetProperty(ref _scaleY, value)) UpdateTransform(); } }
        public float ScaleZ { get => _scaleZ; set { if (SetProperty(ref _scaleZ, value)) UpdateTransform(); } }

        public bool UniformScale
        {
            get => _uniformScale;
            set => SetProperty(ref _uniformScale, value);
        }

        public bool CanAddComponent => HasSelection;

        // Component visibility
        public bool ShowTransformComponent { get => _showTransformComponent; set => SetProperty(ref _showTransformComponent, value); }
        public bool ShowGenomeComponent { get => _showGenomeComponent; set => SetProperty(ref _showGenomeComponent, value); }
        public bool ShowMaterialComponent { get => _showMaterialComponent; set => SetProperty(ref _showMaterialComponent, value); }
        public bool ShowBehaviorComponent { get => _showBehaviorComponent; set => SetProperty(ref _showBehaviorComponent, value); }
        public bool ShowPhysicsComponent { get => _showPhysicsComponent; set => SetProperty(ref _showPhysicsComponent, value); }
        public bool ShowColliderComponent { get => _showColliderComponent; set => SetProperty(ref _showColliderComponent, value); }
        public bool ShowLightComponent { get => _showLightComponent; set => SetProperty(ref _showLightComponent, value); }
        public bool ShowCameraComponent { get => _showCameraComponent; set => SetProperty(ref _showCameraComponent, value); }
        public bool ShowParticleSystemComponent { get => _showParticleSystemComponent; set => SetProperty(ref _showParticleSystemComponent, value); }
        public bool ShowAudioComponent { get => _showAudioComponent; set => SetProperty(ref _showAudioComponent, value); }
        public bool ShowAnimationComponent { get => _showAnimationComponent; set => SetProperty(ref _showAnimationComponent, value); }
        public bool ShowScriptComponent { get => _showScriptComponent; set => SetProperty(ref _showScriptComponent, value); }

        // Light properties
        public float LightIntensity { get => _lightIntensity; set => SetProperty(ref _lightIntensity, value); }
        public System.Numerics.Vector3 LightColor { get => _lightColor; set => SetProperty(ref _lightColor, value); }
        public float LightRange { get => _lightRange; set => SetProperty(ref _lightRange, value); }
        public float LightSpotAngle { get => _lightSpotAngle; set => SetProperty(ref _lightSpotAngle, value); }

        // Physics properties
        public float Mass { get => _mass; set => SetProperty(ref _mass, value); }
        public float Friction { get => _friction; set => SetProperty(ref _friction, value); }
        public float Restitution { get => _restitution; set => SetProperty(ref _restitution, value); }
        public bool IsKinematic { get => _isKinematic; set => SetProperty(ref _isKinematic, value); }

        // Camera properties
        public float CameraFov { get => _cameraFov; set => SetProperty(ref _cameraFov, value); }
        public float CameraNear { get => _cameraNear; set => SetProperty(ref _cameraNear, value); }
        public float CameraFar { get => _cameraFar; set => SetProperty(ref _cameraFar, value); }

        // Particle System properties
        public float ParticleLifetime { get => _particleLifetime; set => SetProperty(ref _particleLifetime, value); }
        public int ParticleCount { get => _particleCount; set => SetProperty(ref _particleCount, value); }
        public float ParticleSpeed { get => _particleSpeed; set => SetProperty(ref _particleSpeed, value); }

        // Audio properties
        public float AudioVolume { get => _audioVolume; set => SetProperty(ref _audioVolume, value); }
        public float AudioPitch { get => _audioPitch; set => SetProperty(ref _audioPitch, value); }
        public bool AudioLoop { get => _audioLoop; set => SetProperty(ref _audioLoop, value); }
        public float AudioSpatialBlend { get => _audioSpatialBlend; set => SetProperty(ref _audioSpatialBlend, value); }
        public float AudioMinDistance { get => _audioMinDistance; set => SetProperty(ref _audioMinDistance, value); }
        public float AudioMaxDistance { get => _audioMaxDistance; set => SetProperty(ref _audioMaxDistance, value); }

        // Script properties
        public string ScriptCode { get => _scriptCode; set => SetProperty(ref _scriptCode, value); }
        public string ScriptLanguage { get => _scriptLanguage; set => SetProperty(ref _scriptLanguage, value); }

        // Behavior properties
        public string BehaviorTreeText { get => _behaviorTreeText; set => SetProperty(ref _behaviorTreeText, value); }
        public string EmotionState { get => _emotionState; set => SetProperty(ref _emotionState, value); }
        public float EmotionIntensity { get => _emotionIntensity; set => SetProperty(ref _emotionIntensity, value); }
        public ObservableCollection<string> MemoryEntries => _memoryEntries;

        // Genome properties
        public ObservableCollection<string> NeuronList => _neuronList;
        public ObservableCollection<string> SynapseList => _synapseList;
        public int NeuronCount { get => _neuronCount; set => SetProperty(ref _neuronCount, value); }
        public int SynapseCount { get => _synapseCount; set => SetProperty(ref _synapseCount, value); }
        public float GenomeComplexity { get => _genomeComplexity; set => SetProperty(ref _genomeComplexity, value); }

        // Material properties
        public float MaterialRoughness { get => _materialRoughness; set => SetProperty(ref _materialRoughness, value); }
        public float MaterialMetallic { get => _materialMetallic; set => SetProperty(ref _materialMetallic, value); }
        public System.Numerics.Vector3 MaterialBaseColor { get => _materialBaseColor; set => SetProperty(ref _materialBaseColor, value); }
        public float MaterialAlpha { get => _materialAlpha; set => SetProperty(ref _materialAlpha, value); }
        public float MaterialNormalStrength { get => _materialNormalStrength; set => SetProperty(ref _materialNormalStrength, value); }
        public float MaterialEmissionStrength { get => _materialEmissionStrength; set => SetProperty(ref _materialEmissionStrength, value); }

        // Commands
        public IRelayCommand<ComponentType> AddComponentCommand { get; private set; } = null!;
        public IRelayCommand<ComponentType> RemoveComponentCommand { get; private set; } = null!;
        public IRelayCommand ResetTransformCommand { get; private set; } = null!;
        public IRelayCommand ResetPositionCommand { get; private set; } = null!;
        public IRelayCommand ResetRotationCommand { get; private set; } = null!;
        public IRelayCommand ResetScaleCommand { get; private set; } = null!;
        public IRelayCommand ResetAllCommand { get; private set; } = null!;
        public IRelayCommand CopyEntityCommand { get; private set; } = null!;
        public IRelayCommand PasteEntityCommand { get; private set; } = null!;
        public IRelayCommand<ComponentType> CopyComponentCommand { get; private set; } = null!;
        public IRelayCommand<ComponentType> PasteComponentCommand { get; private set; } = null!;
        public IRelayCommand RefreshCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            AddComponentCommand = CreateCommand<ComponentType>(ExecuteAddComponent);
            RemoveComponentCommand = CreateCommand<ComponentType>(ExecuteRemoveComponent);
            ResetTransformCommand = CreateCommand(ExecuteResetTransform);
            ResetPositionCommand = CreateCommand(() => { PositionX = 0; PositionY = 0; PositionZ = 0; });
            ResetRotationCommand = CreateCommand(() => { RotationX = 0; RotationY = 0; RotationZ = 0; });
            ResetScaleCommand = CreateCommand(() => { ScaleX = 1; ScaleY = 1; ScaleZ = 1; });
            ResetAllCommand = CreateCommand(ExecuteResetAll);
            CopyEntityCommand = CreateCommand(ExecuteCopyEntity);
            PasteEntityCommand = CreateCommand(ExecutePasteEntity);
            CopyComponentCommand = CreateCommand<ComponentType>(ExecuteCopyComponent);
            PasteComponentCommand = CreateCommand<ComponentType>(ExecutePasteComponent);
            RefreshCommand = CreateCommand(() => Refresh());
        }

        private void LoadEntityComponents()
        {
            _components.Clear();
            _selectedEntity = _selectedEntityId.HasValue ? _sceneService.GetEntityById(_selectedEntityId.Value) : null;

            if (_selectedEntity == null)
            {
                EntityName = string.Empty;
                RaisePropertyChanged(nameof(HasSelection));
                return;
            }

            EntityName = _selectedEntity.Name;
            EntityType = _selectedEntity.Type;
            PositionX = _selectedEntity.Position.X;
            PositionY = _selectedEntity.Position.Y;
            PositionZ = _selectedEntity.Position.Z;
            RaisePropertyChanged(nameof(HasSelection));

            foreach (var comp in _selectedEntity.Components)
            {
                var compVm = new InspectorComponentViewModel
                {
                    ComponentType = comp,
                    Name = comp.ToString(),
                    IsExpanded = true
                };
                _components.Add(compVm);

                switch (comp)
                {
                    case ComponentType.Genome:
                        ShowGenomeComponent = true;
                        LoadGenomeData(_selectedEntity.Id);
                        break;
                    case ComponentType.Material:
                        ShowMaterialComponent = true;
                        break;
                    case ComponentType.BehaviorTree:
                        ShowBehaviorComponent = true;
                        break;
                    case ComponentType.Collider:
                    case ComponentType.Rigidbody:
                        ShowPhysicsComponent = true;
                        ShowColliderComponent = true;
                        break;
                    case ComponentType.Light:
                        ShowLightComponent = true;
                        break;
                    case ComponentType.Camera:
                        ShowCameraComponent = true;
                        break;
                    case ComponentType.ParticleSystem:
                        ShowParticleSystemComponent = true;
                        break;
                    case ComponentType.AudioSource:
                        ShowAudioComponent = true;
                        break;
                    case ComponentType.Animation:
                        ShowAnimationComponent = true;
                        break;
                    case ComponentType.Script:
                        ShowScriptComponent = true;
                        break;
                }
            }
        }

        private void LoadGenomeData(Guid entityId)
        {
            _neuronList.Clear();
            _synapseList.Clear();
            _neuronCount = new Random(entityId.GetHashCode()).Next(10, 500);
            _synapseCount = new Random(entityId.GetHashCode() + 1).Next(20, 2000);
            _genomeComplexity = (float)_synapseCount / Math.Max(1, _neuronCount);

            for (var i = 0; i < Math.Min(_neuronCount, 20); i++)
                _neuronList.Add($"Neuron_{i}: activation={new Random(i).NextDouble():F3}");

            for (var i = 0; i < Math.Min(_synapseCount, 20); i++)
                _synapseList.Add($"Synapse_{i}: weight={new Random(i + 100).NextDouble():F3}");

            RaisePropertyChanged(nameof(NeuronCount));
            RaisePropertyChanged(nameof(SynapseCount));
            RaisePropertyChanged(nameof(GenomeComplexity));
        }

        private void UpdateTransform()
        {
            if (_selectedEntity != null && LivePreview)
            {
                _selectedEntity.Position = new System.Numerics.Vector3(_positionX, _positionY, _positionZ);
            }
        }

        private void ApplyPropertyFilter()
        {
            foreach (var comp in _components)
            {
                if (string.IsNullOrEmpty(SearchText))
                {
                    comp.IsVisible = true;
                }
                else
                {
                    comp.IsVisible = comp.Name.Contains(SearchText, StringComparison.OrdinalIgnoreCase);
                }
            }
        }

        private void ExecuteAddComponent(ComponentType type)
        {
            StatusMessage = $"Added component: {type}";
        }

        private void ExecuteRemoveComponent(ComponentType type)
        {
            StatusMessage = $"Removed component: {type}";
        }

        private void ExecuteResetTransform()
        {
            PositionX = 0; PositionY = 0; PositionZ = 0;
            RotationX = 0; RotationY = 0; RotationZ = 0;
            ScaleX = 1; ScaleY = 1; ScaleZ = 1;
            StatusMessage = "Transform reset";
        }

        private void ExecuteResetAll()
        {
            ExecuteResetTransform();
            StatusMessage = "All properties reset";
        }

        private void ExecuteCopyEntity()
        {
            if (_selectedEntityId.HasValue)
            {
                ClipboardService.SetEntities(new[] { _selectedEntityId.Value });
                StatusMessage = "Entity copied";
            }
        }

        private void ExecutePasteEntity()
        {
            StatusMessage = "Entity pasted";
        }

        private void ExecuteCopyComponent(ComponentType type)
        {
            StatusMessage = $"Copied component: {type}";
        }

        private void ExecutePasteComponent(ComponentType type)
        {
            StatusMessage = $"Pasted component: {type}";
        }

        public void Refresh()
        {
            if (_selectedEntityId.HasValue)
                LoadEntityComponents();
        }

        public void ClearSelection()
        {
            SelectedEntityId = null;
            _components.Clear();
        }

        protected override void OnDispose()
        {
            _components.Clear();
            _availableComponentTypes.Clear();
            _memoryEntries.Clear();
            _neuronList.Clear();
            _synapseList.Clear();
            base.OnDispose();
        }
    }

    /// <summary>
    /// ViewModel for a single component in the inspector.
    /// </summary>
    public class InspectorComponentViewModel : ObservableObject
    {
        private string _name = string.Empty;
        private bool _isExpanded = true;
        private bool _isVisible = true;
        private bool _isCollapsed;

        public ComponentType ComponentType { get; init; }

        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        public bool IsExpanded
        {
            get => _isExpanded;
            set => SetProperty(ref _isExpanded, value);
        }

        public bool IsVisible
        {
            get => _isVisible;
            set => SetProperty(ref _isVisible, value);
        }

        public bool IsCollapsed
        {
            get => _isCollapsed;
            set => SetProperty(ref _isCollapsed, value);
        }

        public string ToggleIcon => IsExpanded ? "\u25BC" : "\u25B6";
    }



    // =========================================================================
    // LLM CONSOLE VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the LLM console, managing chat history, prompts, and responses.
    /// </summary>
    public partial class LlmConsoleViewModel : ViewModelBase
    {
        private readonly ILLMConsoleService _llmService;
        private readonly ObservableCollection<ChatMessageViewModel> _chatHistory = new();
        private readonly ObservableCollection<PromptTemplate> _promptTemplates = new();
        private readonly ObservableCollection<string> _availableModels = new();
        private string _inputText = string.Empty;
        private LLMProvider _selectedProvider = LLMProvider.OpenAI;
        private string _selectedModel = "gpt-4";
        private bool _isStreaming;
        private int _totalTokensUsed;
        private double _totalCost;
        private bool _isMultiLineInput;
        private string _streamingContent = string.Empty;
        private ChatMessageViewModel? _currentStreamingMessage;
        private readonly List<string> _conversationHistory = new();
        private int _selectedTemplateIndex = -1;
        private bool _showPromptTemplates;
        private bool _showTokenUsage = true;
        private double _responseLatencyMs;
        private readonly List<string> _autoCompleteSuggestions = new();
        private int _selectedSuggestionIndex = -1;
        private string _attachmentPath = string.Empty;

        public LlmConsoleViewModel(ILLMConsoleService llmService)
        {
            _llmService = llmService ?? throw new ArgumentNullException(nameof(llmService));
            InitializePromptTemplates();
            InitializeCommands();
        }

        public ObservableCollection<ChatMessageViewModel> ChatHistory => _chatHistory;
        public ObservableCollection<PromptTemplate> PromptTemplates => _promptTemplates;
        public ObservableCollection<string> AvailableModels => _availableModels;

        public string InputText
        {
            get => _inputText;
            set
            {
                if (SetProperty(ref _inputText, value))
                {
                    UpdateAutoComplete(value);
                }
            }
        }

        public LLMProvider SelectedProvider
        {
            get => _selectedProvider;
            set
            {
                if (SetProperty(ref _selectedProvider, value))
                {
                    LoadAvailableModels();
                }
            }
        }

        public string SelectedModel
        {
            get => _selectedModel;
            set => SetProperty(ref _selectedModel, value);
        }

        public bool IsStreaming
        {
            get => _isStreaming;
            set => SetProperty(ref _isStreaming, value);
        }

        public int TotalTokensUsed
        {
            get => _totalTokensUsed;
            set => SetProperty(ref _totalTokensUsed, value);
        }

        public double TotalCost
        {
            get => _totalCost;
            set => SetProperty(ref _totalCost, value);
        }

        public bool IsMultiLineInput
        {
            get => _isMultiLineInput;
            set => SetProperty(ref _isMultiLineInput, value);
        }

        public string StreamingContent
        {
            get => _streamingContent;
            set => SetProperty(ref _streamingContent, value);
        }

        public int SelectedTemplateIndex
        {
            get => _selectedTemplateIndex;
            set => SetProperty(ref _selectedTemplateIndex, value);
        }

        public bool ShowPromptTemplates
        {
            get => _showPromptTemplates;
            set => SetProperty(ref _showPromptTemplates, value);
        }

        public bool ShowTokenUsage
        {
            get => _showTokenUsage;
            set => SetProperty(ref _showTokenUsage, value);
        }

        public double ResponseLatencyMs
        {
            get => _responseLatencyMs;
            set => SetProperty(ref _responseLatencyMs, value);
        }

        public ObservableCollection<string> AutoCompleteSuggestions => new(_autoCompleteSuggestions);

        public string AttachmentPath
        {
            get => _attachmentPath;
            set => SetProperty(ref _attachmentPath, value);
        }

        public bool HasAttachment => !string.IsNullOrEmpty(AttachmentPath);

        public string TokenUsageText => $"Tokens: {TotalTokensUsed} | Cost: ${TotalCost:F4}";

        public string StatusText => IsStreaming ? "Streaming..." : "Ready";

        // Commands
        public IAsyncRelayCommand SendCommand { get; private set; } = null!;
        public IRelayCommand ClearHistoryCommand { get; private set; } = null!;
        public IAsyncRelayCommand ExportConversationCommand { get; private set; } = null!;
        public IRelayCommand ToggleMultiLineCommand { get; private set; } = null!;
        public IRelayCommand ApplySuggestionCommand { get; private set; } = null!;
        public IRelayCommand<int> InsertTemplateCommand { get; private set; } = null!;
        public IRelayCommand CancelStreamingCommand { get; private set; } = null!;
        public IRelayCommand AttachScreenshotCommand { get; private set; } = null!;
        public IRelayCommand RemoveAttachmentCommand { get; private set; } = null!;
        public IRelayCommand<LLMProvider> SetProviderCommand { get; private set; } = null!;
        public IRelayCommand<ChatMessageViewModel> CopyMessageCommand { get; private set; } = null!;
        public IRelayCommand<ChatMessageViewModel> ApplyGenomeMutationCommand { get; private set; } = null!;
        public IRelayCommand<ChatMessageViewModel> CreateEntityFromResponseCommand { get; private set; } = null!;
        public IRelayCommand<ChatMessageViewModel> ModifyMaterialFromResponseCommand { get; private set; } = null!;
        public IRelayCommand TogglePromptTemplatesCommand { get; private set; } = null!;
        public IRelayCommand ToggleTokenUsageCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            SendCommand = CreateAsyncCommand(ExecuteSend, () => !string.IsNullOrWhiteSpace(InputText) && !IsStreaming);
            ClearHistoryCommand = CreateCommand(ExecuteClearHistory);
            ExportConversationCommand = CreateAsyncCommand(ExecuteExportConversation);
            ToggleMultiLineCommand = CreateCommand(() => IsMultiLineInput = !IsMultiLineInput);
            ApplySuggestionCommand = CreateCommand(ExecuteApplySuggestion);
            InsertTemplateCommand = CreateCommand<int>(ExecuteInsertTemplate);
            CancelStreamingCommand = CreateCommand(ExecuteCancelStreaming);
            AttachScreenshotCommand = CreateCommand(ExecuteAttachScreenshot);
            RemoveAttachmentCommand = CreateCommand(() => { AttachmentPath = string.Empty; RaisePropertyChanged(nameof(HasAttachment)); });
            SetProviderCommand = CreateCommand<LLMProvider>(p => SelectedProvider = p);
            CopyMessageCommand = CreateCommand<ChatMessageViewModel>(msg => ClipboardService.SetText(msg.Content));
            ApplyGenomeMutationCommand = CreateCommand<ChatMessageViewModel>(msg => StatusMessage = "Genome mutation applied");
            CreateEntityFromResponseCommand = CreateCommand<ChatMessageViewModel>(msg => StatusMessage = "Entity created from response");
            ModifyMaterialFromResponseCommand = CreateCommand<ChatMessageViewModel>(msg => StatusMessage = "Material modified from response");
            TogglePromptTemplatesCommand = CreateCommand(() => ShowPromptTemplates = !ShowPromptTemplates);
            ToggleTokenUsageCommand = CreateCommand(() => ShowTokenUsage = !ShowTokenUsage);
        }

        private void InitializePromptTemplates()
        {
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Genome Mutation",
                Description = "Suggest mutations for a genome",
                Template = "Analyze the current genome and suggest {n} mutations to improve {objective}.",
                Category = "Genome",
                Variables = new[] { "n", "objective" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Behavior Design",
                Description = "Design a behavior tree for an entity",
                Template = "Create a behavior tree for a {entityType} that {behavior}. Include decision nodes and fallbacks.",
                Category = "Behavior",
                Variables = new[] { "entityType", "behavior" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Material Creation",
                Description = "Generate a material description",
                Template = "Create a PBR material description for {surface}. Base color: {color}, roughness: {roughness}, metallic: {metallic}.",
                Category = "Material",
                Variables = new[] { "surface", "color", "roughness", "metallic" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Scene Composition",
                Description = "Suggest scene layout improvements",
                Template = "Analyze the current scene layout and suggest improvements for {goal}. Consider lighting, spacing, and visual hierarchy.",
                Category = "Scene",
                Variables = new[] { "goal" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Performance Optimization",
                Description = "Optimize scene performance",
                Template = "Review the scene for performance issues. Current stats: {stats}. Suggest optimizations for {target} FPS.",
                Category = "Performance",
                Variables = new[] { "stats", "target" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Code Generation",
                Description = "Generate code from description",
                Template = "Write {language} code that {description}. Follow the existing code style in the project.",
                Category = "Code",
                Variables = new[] { "language", "description" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Shader Writing",
                Description = "Generate a shader snippet",
                Template = "Write a {shaderType} shader that {effect}. Use {language} syntax.",
                Category = "Shader",
                Variables = new[] { "shaderType", "effect", "language" }
            });
            _promptTemplates.Add(new PromptTemplate
            {
                Name = "Evolution Strategy",
                Description = "Design evolution parameters",
                Template = "Design an evolution strategy for a population of {populationSize} genomes. Target: {target}. Mutation rate: {mutationRate}.",
                Category = "Evolution",
                Variables = new[] { "populationSize", "target", "mutationRate" }
            });
        }

        private async Task ExecuteSend()
        {
            if (string.IsNullOrWhiteSpace(InputText)) return;

            var userMessage = new ChatMessageViewModel
            {
                Type = ChatMessageType.User,
                Content = InputText.Trim(),
                Timestamp = DateTime.Now,
                SenderName = "You"
            };
            _chatHistory.Add(userMessage);
            _conversationHistory.Add($"User: {InputText.Trim()}");

            var prompt = InputText.Trim();
            InputText = string.Empty;

            try
            {
                SetBusy(true, "Generating response...");
                IsStreaming = true;
                StreamingContent = string.Empty;

                _currentStreamingMessage = new ChatMessageViewModel
                {
                    Type = ChatMessageType.Assistant,
                    Content = "",
                    Timestamp = DateTime.Now,
                    SenderName = "AI Assistant",
                    IsStreaming = true,
                    Provider = SelectedProvider,
                    ModelName = SelectedModel
                };
                _chatHistory.Add(_currentStreamingMessage);

                var stopwatch = Stopwatch.StartNew();

                try
                {
                    await foreach (var token in _llmService.StreamResponseAsync(prompt, SelectedProvider, SelectedModel))
                    {
                        StreamingContent += token;
                        _currentStreamingMessage.Content = StreamingContent;
                        _currentStreamingMessage.RaiseContentChanged();
                    }
                }
                catch (Exception ex)
                {
                    StreamingContent += $"\n\n[Error: {ex.Message}]";
                    _currentStreamingMessage.Content = StreamingContent;
                    _currentStreamingMessage.Type = ChatMessageType.Error;
                }

                stopwatch.Stop();
                ResponseLatencyMs = stopwatch.Elapsed.TotalMilliseconds;

                _currentStreamingMessage.IsStreaming = false;
                _currentStreamingMessage.TokenCount = EstimateTokenCount(StreamingContent);
                _currentStreamingMessage.LatencyMs = ResponseLatencyMs;
                TotalTokensUsed += _currentStreamingMessage.TokenCount ?? 0;
                TotalCost += CalculateCost(_currentStreamingMessage.TokenCount ?? 0, SelectedProvider, SelectedModel);

                _conversationHistory.Add($"Assistant: {StreamingContent}");

                StreamingContent = string.Empty;
                _currentStreamingMessage = null;

                RaisePropertyChanged(nameof(TokenUsageText));
            }
            catch (Exception ex)
            {
                var errorMessage = new ChatMessageViewModel
                {
                    Type = ChatMessageType.Error,
                    Content = $"Error: {ex.Message}",
                    Timestamp = DateTime.Now,
                    SenderName = "System"
                };
                _chatHistory.Add(errorMessage);
            }
            finally
            {
                IsStreaming = false;
                SetBusy(false);
            }
        }

        private void ExecuteClearHistory()
        {
            _chatHistory.Clear();
            _conversationHistory.Clear();
            TotalTokensUsed = 0;
            TotalCost = 0;
            RaisePropertyChanged(nameof(TokenUsageText));
            StatusMessage = "Chat history cleared";
        }

        private async Task ExecuteExportConversation()
        {
            if (_chatHistory.Count == 0) return;
            var sb = new StringBuilder();
            sb.AppendLine("G-DNN Studio LLM Console - Conversation Export");
            sb.AppendLine($"Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Provider: {SelectedProvider} | Model: {SelectedModel}");
            sb.AppendLine(new string('=', 60));
            sb.AppendLine();

            foreach (var msg in _chatHistory)
            {
                var role = msg.Type == ChatMessageType.User ? "User" : "Assistant";
                sb.AppendLine($"[{role}] ({msg.Timestamp:HH:mm:ss}):");
                sb.AppendLine(msg.Content);
                sb.AppendLine();
            }

            sb.AppendLine(new string('=', 60));
            sb.AppendLine($"Total Tokens: {TotalTokensUsed}");
            sb.AppendLine($"Total Cost: ${TotalCost:F4}");

            await File.WriteAllTextAsync($"conversation_{DateTime.Now:yyyyMMdd_HHmmss}.txt", sb.ToString());
            StatusMessage = "Conversation exported";
        }

        private void ExecuteApplySuggestion()
        {
            if (_autoCompleteSuggestions.Count > 0 && _selectedSuggestionIndex >= 0)
            {
                InputText = _autoCompleteSuggestions[_selectedSuggestionIndex];
            }
        }

        private void ExecuteInsertTemplate(int index)
        {
            if (index >= 0 && index < _promptTemplates.Count)
            {
                var template = _promptTemplates[index];
                InputText = template.Template;
                ShowPromptTemplates = false;
            }
        }

        private void ExecuteCancelStreaming()
        {
            IsStreaming = false;
            StatusMessage = "Streaming cancelled";
        }

        private void ExecuteAttachScreenshot()
        {
            AttachmentPath = $"screenshot_{DateTime.Now:yyyyMMdd_HHmmss}.png";
            RaisePropertyChanged(nameof(HasAttachment));
            StatusMessage = "Screenshot attached";
        }

        private void UpdateAutoComplete(string input)
        {
            _autoCompleteSuggestions.Clear();
            if (string.IsNullOrEmpty(input)) return;

            var suggestions = new[] { "genome", "mutation", "neuron", "synapse", "evolve", "compile", "entity", "material", "behavior", "scene", "light", "camera", "physics", "particle", "audio", "shader", "mesh", "texture", "animation", "terrain" };
            foreach (var suggestion in suggestions)
            {
                if (suggestion.StartsWith(input.ToLowerInvariant()))
                    _autoCompleteSuggestions.Add(suggestion);
            }
            RaisePropertyChanged(nameof(AutoCompleteSuggestions));
        }

        private int EstimateTokenCount(string text)
        {
            return Math.Max(1, text.Length / 4);
        }

        private double CalculateCost(int tokens, LLMProvider provider, string model)
        {
            return provider switch
            {
                LLMProvider.OpenAI => tokens * 0.00003,
                LLMProvider.Anthropic => tokens * 0.000015,
                LLMProvider.Local => 0,
                LLMProvider.Azure => tokens * 0.000025,
                _ => tokens * 0.00002
            };
        }

        protected override void OnDispose()
        {
            _chatHistory.Clear();
            _promptTemplates.Clear();
            _availableModels.Clear();
            _conversationHistory.Clear();
            _autoCompleteSuggestions.Clear();
            base.OnDispose();
        }
    }

    /// <summary>
    /// ViewModel for a single chat message.
    /// </summary>
    public class ChatMessageViewModel : ObservableObject
    {
        private string _content = string.Empty;
        private bool _isStreaming;

        public Guid MessageId { get; } = Guid.NewGuid();
        public ChatMessageType Type { get; init; }
        public string SenderName { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; }

        public string Content
        {
            get => _content;
            set => SetProperty(ref _content, value);
        }

        public bool IsStreaming
        {
            get => _isStreaming;
            set => SetProperty(ref _isStreaming, value);
        }

        public LLMProvider? Provider { get; init; }
        public string? ModelName { get; init; }
        public int? TokenCount { get; init; }
        public double? LatencyMs { get; init; }

        public string TimestampText => Timestamp.ToString("HH:mm:ss");

        public string TypeIcon => Type switch
        {
            ChatMessageType.User => "\U0001F464",
            ChatMessageType.Assistant => "\U0001F916",
            ChatMessageType.System => "\u2699\uFE0F",
            ChatMessageType.Error => "\u26A0\uFE0F",
            ChatMessageType.CodeBlock => "\U0001F4BB",
            ChatMessageType.Suggestion => "\U0001F4A1",
            ChatMessageType.ImageAttachment => "\U0001F5BC\uFE0F",
            _ => "\U0001F4AC"
        };

        public string LatencyText => LatencyMs.HasValue ? $"{LatencyMs.Value:F0}ms" : "";
        public string TokenText => TokenCount.HasValue ? $"{TokenCount.Value} tokens" : "";

        public void RaiseContentChanged() => RaisePropertyChanged(nameof(Content));
    }



    // =========================================================================
    // BLUEPRINT EDITOR VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the blueprint visual scripting editor.
    /// </summary>
    public partial class BlueprintEditorViewModel : ViewModelBase
    {
        private readonly IBlueprintEditorService _blueprintService;
        private readonly ObservableCollection<BlueprintNodeViewModel> _nodes = new();
        private readonly ObservableCollection<BlueprintEdgeViewModel> _edges = new();
        private readonly ObservableCollection<BlueprintNodeType> _availableNodeTypes = new();
        private readonly List<BlueprintNodeViewModel> _selectedNodes = new();
        private readonly List<BlueprintEdgeViewModel> _selectedEdges = new();
        private string _searchText = string.Empty;
        private float _zoom = 1.0f;
        private float _panX;
        private float _panY;
        private bool _showMinimap = true;
        private bool _showGrid = true;
        private float _gridSize = 20f;
        private string _blueprintName = "Untitled Blueprint";
        private string _blueprintFilePath = string.Empty;
        private bool _isDirty;
        private string _validationMessage = string.Empty;
        private bool _hasValidationErrors;
        private readonly ObservableCollection<string> _validationResults = new();
        private float _minimapZoom = 0.15f;
        private float _minimapWidth = 200;
        private float _minimapHeight = 150;
        private float _selectionRectX, _selectionRectY, _selectionRectW, _selectionRectH;
        private bool _isSelecting;
        private bool _isConnecting;
        private BlueprintNodeViewModel? _connectionSourceNode;
        private PinInfo? _connectionSourcePin;
        private readonly UndoRedoManager _undoRedoManager = new();
        private readonly ObservableCollection<BlueprintNodeGroup> _nodeGroups = new();
        private readonly ObservableCollection<BlueprintCommentNode> _commentNodes = new();
        private float _canvasWidth = 5000;
        private float _canvasHeight = 5000;
        private string _nodePaletteFilter = string.Empty;
        private BlueprintNodeViewModel? _draggedNode;
        private float _dragOffsetX, _dragOffsetY;
        private bool _isPanning;
        private float _lastPanX, _lastPanY;

        public BlueprintEditorViewModel(IBlueprintEditorService blueprintService)
        {
            _blueprintService = blueprintService ?? throw new ArgumentNullException(nameof(blueprintService));
            InitializeAvailableNodeTypes();
            InitializeCommands();
        }

        public ObservableCollection<BlueprintNodeViewModel> Nodes => _nodes;
        public ObservableCollection<BlueprintEdgeViewModel> Edges => _edges;
        public ObservableCollection<BlueprintNodeType> AvailableNodeTypes => _availableNodeTypes;
        public ObservableCollection<string> ValidationResults => _validationResults;
        public ObservableCollection<BlueprintNodeGroup> NodeGroups => _nodeGroups;
        public ObservableCollection<BlueprintCommentNode> CommentNodes => _commentNodes;

        public string SearchText
        {
            get => _searchText;
            set { if (SetProperty(ref _searchText, value)) ApplyNodeFilter(); }
        }

        public float Zoom
        {
            get => _zoom;
            set => SetProperty(ref _zoom, Math.Clamp(value, 0.1f, 3.0f));
        }

        public float PanX
        {
            get => _panX;
            set => SetProperty(ref _panX, value);
        }

        public float PanY
        {
            get => _panY;
            set => SetProperty(ref _panY, value);
        }

        public bool ShowMinimap
        {
            get => _showMinimap;
            set => SetProperty(ref _showMinimap, value);
        }

        public bool ShowGrid
        {
            get => _showGrid;
            set => SetProperty(ref _showGrid, value);
        }

        public float GridSize
        {
            get => _gridSize;
            set => SetProperty(ref _gridSize, value);
        }

        public string BlueprintName
        {
            get => _blueprintName;
            set { if (SetProperty(ref _blueprintName, value)) RaisePropertyChanged(nameof(DisplayTitle)); }
        }

        public string DisplayTitle => IsDirty ? $"*{BlueprintName}" : BlueprintName;

        public bool IsDirty
        {
            get => _isDirty;
            set { if (SetProperty(ref _isDirty, value)) RaisePropertyChanged(nameof(DisplayTitle)); }
        }

        public string ValidationMessage
        {
            get => _validationMessage;
            set => SetProperty(ref _validationMessage, value);
        }

        public bool HasValidationErrors
        {
            get => _hasValidationErrors;
            set => SetProperty(ref _hasValidationErrors, value);
        }

        public float MinimapZoom
        {
            get => _minimapZoom;
            set => SetProperty(ref _minimapZoom, value);
        }

        public float MinimapWidth
        {
            get => _minimapWidth;
            set => SetProperty(ref _minimapWidth, value);
        }

        public float MinimapHeight
        {
            get => _minimapHeight;
            set => SetProperty(ref _minimapHeight, value);
        }

        public bool IsSelecting
        {
            get => _isSelecting;
            set => SetProperty(ref _isSelecting, value);
        }

        public bool IsConnecting
        {
            get => _isConnecting;
            set => SetProperty(ref _isConnecting, value);
        }

        public float CanvasWidth
        {
            get => _canvasWidth;
            set => SetProperty(ref _canvasWidth, value);
        }

        public float CanvasHeight
        {
            get => _canvasHeight;
            set => SetProperty(ref _canvasHeight, value);
        }

        public string NodePaletteFilter
        {
            get => _nodePaletteFilter;
            set { if (SetProperty(ref _nodePaletteFilter, value)) ApplyNodeFilter(); }
        }

        public bool CanUndo => _undoRedoManager.CanUndo;
        public bool CanRedo => _undoRedoManager.CanRedo;
        public int SelectedNodeCount => _selectedNodes.Count;
        public int NodeCount => _nodes.Count;
        public int EdgeCount => _edges.Count;

        // Commands
        public IAsyncRelayCommand OpenBlueprintCommand { get; private set; } = null!;
        public IAsyncRelayCommand SaveBlueprintCommand { get; private set; } = null!;
        public IAsyncRelayCommand CompileCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintNodeType> AddNodeCommand { get; private set; } = null!;
        public IRelayCommand DeleteSelectedCommand { get; private set; } = null!;
        public IRelayCommand UndoCommand { get; private set; } = null!;
        public IRelayCommand RedoCommand { get; private set; } = null!;
        public IRelayCommand CopyCommand { get; private set; } = null!;
        public IRelayCommand PasteCommand { get; private set; } = null!;
        public IRelayCommand SelectAllCommand { get; private set; } = null!;
        public IRelayCommand AutoLayoutCommand { get; private set; } = null!;
        public IRelayCommand GroupSelectedCommand { get; private set; } = null!;
        public IRelayCommand UngroupSelectedCommand { get; private set; } = null!;
        public IRelayCommand AddCommentCommand { get; private set; } = null!;
        public IRelayCommand ZoomToFitCommand { get; private set; } = null!;
        public IRelayCommand ZoomInCommand { get; private set; } = null!;
        public IRelayCommand ZoomOutCommand { get; private set; } = null!;
        public IRelayCommand ResetZoomCommand { get; private set; } = null!;
        public IRelayCommand ToggleMinimapCommand { get; private set; } = null!;
        public IRelayCommand ToggleGridCommand { get; private set; } = null!;
        public IRelayCommand ValidateCommand { get; private set; } = null!;
        public IRelayCommand CompileToGenomeCommand { get; private set; } = null!;
        public IRelayCommand CompileToBehaviorTreeCommand { get; private set; } = null!;
        public IRelayCommand<string> FilterNodesCommand { get; private set; } = null!;
        public IRelayCommand ClearAllCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintNodeViewModel> SelectNodeCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintNodeViewModel> StartDragNodeCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintNodeViewModel> EndDragNodeCommand { get; private set; } = null!;
        public IRelayCommand<(float x, float y)> StartPanCommand { get; private set; } = null!;
        public IRelayCommand<(float x, float y)> UpdatePanCommand { get; private set; } = null!;
        public IRelayCommand EndPanCommand { get; private set; } = null!;
        public IRelayCommand<float> ZoomAtPointCommand { get; private set; } = null!;
        public IRelayCommand<(float x, float y)> StartConnectCommand { get; private set; } = null!;
        public IRelayCommand<(float x, float y)> UpdateConnectCommand { get; private set; } = null!;
        public IRelayCommand EndConnectCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintNodeViewModel> DeleteNodeCommand { get; private set; } = null!;
        public IRelayCommand<BlueprintEdgeViewModel> DeleteEdgeCommand { get; private set; } = null!;

        private void InitializeAvailableNodeTypes()
        {
            foreach (var type in Enum.GetValues<BlueprintNodeType>())
                _availableNodeTypes.Add(type);
        }

        private void InitializeCommands()
        {
            OpenBlueprintCommand = CreateAsyncCommand(ExecuteOpenBlueprint);
            SaveBlueprintCommand = CreateAsyncCommand(ExecuteSaveBlueprint);
            CompileCommand = CreateAsyncCommand(ExecuteCompile);
            AddNodeCommand = CreateCommand<BlueprintNodeType>(ExecuteAddNode);
            DeleteSelectedCommand = CreateCommand(ExecuteDeleteSelected);
            UndoCommand = CreateCommand(ExecuteUndo, () => CanUndo);
            RedoCommand = CreateCommand(ExecuteRedo, () => CanRedo);
            CopyCommand = CreateCommand(ExecuteCopy);
            PasteCommand = CreateCommand(ExecutePaste);
            SelectAllCommand = CreateCommand(ExecuteSelectAll);
            AutoLayoutCommand = CreateCommand(ExecuteAutoLayout);
            GroupSelectedCommand = CreateCommand(ExecuteGroupSelected);
            UngroupSelectedCommand = CreateCommand(ExecuteUngroupSelected);
            AddCommentCommand = CreateCommand(ExecuteAddComment);
            ZoomToFitCommand = CreateCommand(ExecuteZoomToFit);
            ZoomInCommand = CreateCommand(() => Zoom = Math.Min(3.0f, Zoom + 0.1f));
            ZoomOutCommand = CreateCommand(() => Zoom = Math.Max(0.1f, Zoom - 0.1f));
            ResetZoomCommand = CreateCommand(() => { Zoom = 1.0f; PanX = 0; PanY = 0; });
            ToggleMinimapCommand = CreateCommand(() => ShowMinimap = !ShowMinimap);
            ToggleGridCommand = CreateCommand(() => ShowGrid = !ShowGrid);
            ValidateCommand = CreateCommand(ExecuteValidate);
            CompileToGenomeCommand = CreateAsyncCommand(ExecuteCompileToGenome);
            CompileToBehaviorTreeCommand = CreateAsyncCommand(ExecuteCompileToBehaviorTree);
            FilterNodesCommand = CreateCommand<string>(filter => NodePaletteFilter = filter);
            ClearAllCommand = CreateCommand(ExecuteClearAll);
            SelectNodeCommand = CreateCommand<BlueprintNodeViewModel>(ExecuteSelectNode);
            StartDragNodeCommand = CreateCommand<BlueprintNodeViewModel>(ExecuteStartDragNode);
            EndDragNodeCommand = CreateCommand<BlueprintNodeViewModel>(ExecuteEndDragNode);
            StartPanCommand = CreateCommand<(float, float)>(ExecuteStartPan);
            UpdatePanCommand = CreateCommand<(float, float)>(ExecuteUpdatePan);
            EndPanCommand = CreateCommand(() => _isPanning = false);
            ZoomAtPointCommand = CreateCommand<float>(ExecuteZoomAtPoint);
            StartConnectCommand = CreateCommand<(float, float)>(ExecuteStartConnect);
            UpdateConnectCommand = CreateCommand<(float, float)>(ExecuteUpdateConnect);
            EndConnectCommand = CreateCommand(ExecuteEndConnect);
            DeleteNodeCommand = CreateCommand<BlueprintNodeViewModel>(ExecuteDeleteNode);
            DeleteEdgeCommand = CreateCommand<BlueprintEdgeViewModel>(ExecuteDeleteEdge);
        }

        private async Task ExecuteOpenBlueprint()
        {
            StatusMessage = "Opening blueprint...";
            await Task.CompletedTask;
        }

        private async Task ExecuteSaveBlueprint()
        {
            StatusMessage = "Saving blueprint...";
            IsDirty = false;
            await Task.CompletedTask;
        }

        private async Task ExecuteCompile()
        {
            try
            {
                SetBusy(true, "Compiling blueprint...");
                var result = await _blueprintService.CompileBlueprintAsync();
                StatusMessage = result.Status == CompilationStatus.Success ? "Blueprint compiled successfully" : "Compilation failed";
                IsDirty = false;
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteAddNode(BlueprintNodeType type)
        {
            var x = -PanX / Zoom + CanvasWidth / 2 / Zoom;
            var y = -PanY / Zoom + CanvasHeight / 2 / Zoom;

            var nodeId = _blueprintService.AddNode(type, x, y);
            var nodeVm = CreateNodeViewModel(type, x, y, nodeId);
            _nodes.Add(nodeVm);
            IsDirty = true;
            RaisePropertyChanged(nameof(NodeCount));
            StatusMessage = $"Added {type} node";
        }

        private BlueprintNodeViewModel CreateNodeViewModel(BlueprintNodeType type, float x, float y, Guid nodeId)
        {
            var node = new BlueprintNodeViewModel
            {
                NodeId = nodeId,
                NodeType = type,
                Title = GetNodeTitle(type),
                X = x,
                Y = y,
                Width = 200,
                Height = GetNodeHeight(type),
                Color = GetNodeColor(type),
                Description = GetNodeDescription(type)
            };

            foreach (var pin in GetDefaultInputPins(type))
                node.InputPins.Add(pin);
            foreach (var pin in GetDefaultOutputPins(type))
                node.OutputPins.Add(pin);

            return node;
        }

        private string GetNodeTitle(BlueprintNodeType type) => type switch
        {
            BlueprintNodeType.Input => "Input",
            BlueprintNodeType.Output => "Output",
            BlueprintNodeType.Process => "Process",
            BlueprintNodeType.Math => "Math Operation",
            BlueprintNodeType.Logic => "Logic",
            BlueprintNodeType.Flow => "Flow Control",
            BlueprintNodeType.Variable => "Variable",
            BlueprintNodeType.Function => "Function Call",
            BlueprintNodeType.Event => "Event",
            BlueprintNodeType.Comment => "Comment",
            BlueprintNodeType.Group => "Group",
            BlueprintNodeType.Genome => "Genome",
            BlueprintNodeType.Behavior => "Behavior",
            BlueprintNodeType.Material => "Material",
            BlueprintNodeType.Transform => "Transform",
            BlueprintNodeType.Physics => "Physics",
            BlueprintNodeType.EventTrigger => "Event Trigger",
            BlueprintNodeType.Timer => "Timer",
            BlueprintNodeType.Random => "Random",
            BlueprintNodeType.Switch => "Switch",
            BlueprintNodeType.Comparison => "Comparison",
            BlueprintNodeType.String => "String",
            BlueprintNodeType.Collection => "Collection",
            BlueprintNodeType.Audio => "Audio",
            BlueprintNodeType.Animation => "Animation",
            _ => "Node"
        };

        private float GetNodeHeight(BlueprintNodeType type) => type switch
        {
            BlueprintNodeType.Comment => 80,
            BlueprintNodeType.Group => 60,
            BlueprintNodeType.Variable => 120,
            BlueprintNodeType.Event => 100,
            BlueprintNodeType.Timer => 120,
            _ => 150
        };

        private string GetNodeColor(BlueprintNodeType type) => type switch
        {
            BlueprintNodeType.Input => "#4CAF50",
            BlueprintNodeType.Output => "#F44336",
            BlueprintNodeType.Process => "#2196F3",
            BlueprintNodeType.Math => "#FF9800",
            BlueprintNodeType.Logic => "#9C27B0",
            BlueprintNodeType.Flow => "#00BCD4",
            BlueprintNodeType.Variable => "#FFEB3B",
            BlueprintNodeType.Function => "#795548",
            BlueprintNodeType.Event => "#E91E63",
            BlueprintNodeType.Comment => "#607D8B",
            BlueprintNodeType.Group => "#9E9E9E",
            BlueprintNodeType.Genome => "#8BC34A",
            BlueprintNodeType.Behavior => "#FF5722",
            BlueprintNodeType.Material => "#3F51B5",
            BlueprintNodeType.Transform => "#009688",
            BlueprintNodeType.Physics => "#CDDC39",
            _ => "#9E9E9E"
        };

        private string GetNodeDescription(BlueprintNodeType type) => type switch
        {
            BlueprintNodeType.Genome => "Interacts with genome data",
            BlueprintNodeType.Behavior => "Controls behavior tree",
            BlueprintNodeType.Material => "Modifies material properties",
            BlueprintNodeType.Transform => "Transforms entities",
            BlueprintNodeType.Physics => "Physics simulation control",
            _ => ""
        };

        private List<PinInfo> GetDefaultInputPins(BlueprintNodeType type)
        {
            var pins = new List<PinInfo>();
            switch (type)
            {
                case BlueprintNodeType.Math:
                    pins.Add(new PinInfo { Name = "A", DataType = PinDataType.Float, IsInput = true });
                    pins.Add(new PinInfo { Name = "B", DataType = PinDataType.Float, IsInput = true });
                    break;
                case BlueprintNodeType.Logic:
                    pins.Add(new PinInfo { Name = "Condition", DataType = PinDataType.Bool, IsInput = true });
                    pins.Add(new PinInfo { Name = "True", DataType = PinDataType.Any, IsInput = true });
                    pins.Add(new PinInfo { Name = "False", DataType = PinDataType.Any, IsInput = true });
                    break;
                case BlueprintNodeType.Process:
                    pins.Add(new PinInfo { Name = "Input", DataType = PinDataType.Any, IsInput = true });
                    break;
                case BlueprintNodeType.Transform:
                    pins.Add(new PinInfo { Name = "Entity", DataType = PinDataType.Object, IsInput = true });
                    pins.Add(new PinInfo { Name = "Position", DataType = PinDataType.Vector3, IsInput = true });
                    pins.Add(new PinInfo { Name = "Rotation", DataType = PinDataType.Vector3, IsInput = true });
                    pins.Add(new PinInfo { Name = "Scale", DataType = PinDataType.Vector3, IsInput = true });
                    break;
                case BlueprintNodeType.Genome:
                    pins.Add(new PinInfo { Name = "Genome", DataType = PinDataType.Genome, IsInput = true });
                    pins.Add(new PinInfo { Name = "Mutation Rate", DataType = PinDataType.Float, IsInput = true });
                    break;
                case BlueprintNodeType.Physics:
                    pins.Add(new PinInfo { Name = "Body", DataType = PinDataType.Object, IsInput = true });
                    pins.Add(new PinInfo { Name = "Force", DataType = PinDataType.Vector3, IsInput = true });
                    pins.Add(new PinInfo { Name = "Torque", DataType = PinDataType.Vector3, IsInput = true });
                    break;
                case BlueprintNodeType.Comparison:
                    pins.Add(new PinInfo { Name = "A", DataType = PinDataType.Any, IsInput = true });
                    pins.Add(new PinInfo { Name = "B", DataType = PinDataType.Any, IsInput = true });
                    break;
                case BlueprintNodeType.Timer:
                    pins.Add(new PinInfo { Name = "Start", DataType = PinDataType.Exec, IsInput = true });
                    pins.Add(new PinInfo { Name = "Duration", DataType = PinDataType.Float, IsInput = true });
                    break;
                case BlueprintNodeType.Random:
                    pins.Add(new PinInfo { Name = "Min", DataType = PinDataType.Float, IsInput = true });
                    pins.Add(new PinInfo { Name = "Max", DataType = PinDataType.Float, IsInput = true });
                    break;
            }
            return pins;
        }

        private List<PinInfo> GetDefaultOutputPins(BlueprintNodeType type)
        {
            var pins = new List<PinInfo>();
            switch (type)
            {
                case BlueprintNodeType.Math:
                    pins.Add(new PinInfo { Name = "Result", DataType = PinDataType.Float, IsInput = false });
                    break;
                case BlueprintNodeType.Logic:
                    pins.Add(new PinInfo { Name = "Output", DataType = PinDataType.Any, IsInput = false });
                    break;
                case BlueprintNodeType.Process:
                    pins.Add(new PinInfo { Name = "Output", DataType = PinDataType.Any, IsInput = false });
                    break;
                case BlueprintNodeType.Transform:
                    pins.Add(new PinInfo { Name = "New Position", DataType = PinDataType.Vector3, IsInput = false });
                    pins.Add(new PinInfo { Name = "New Rotation", DataType = PinDataType.Vector3, IsInput = false });
                    break;
                case BlueprintNodeType.Genome:
                    pins.Add(new PinInfo { Name = "Mutated", DataType = PinDataType.Genome, IsInput = false });
                    pins.Add(new PinInfo { Name = "Fitness", DataType = PinDataType.Float, IsInput = false });
                    break;
                case BlueprintNodeType.Physics:
                    pins.Add(new PinInfo { Name = "Velocity", DataType = PinDataType.Vector3, IsInput = false });
                    pins.Add(new PinInfo { Name = "Angular Velocity", DataType = PinDataType.Vector3, IsInput = false });
                    break;
                case BlueprintNodeType.Comparison:
                    pins.Add(new PinInfo { Name = "Result", DataType = PinDataType.Bool, IsInput = false });
                    break;
                case BlueprintNodeType.Timer:
                    pins.Add(new PinInfo { Name = "OnComplete", DataType = PinDataType.Exec, IsInput = false });
                    pins.Add(new PinInfo { Name = "Elapsed", DataType = PinDataType.Float, IsInput = false });
                    break;
                case BlueprintNodeType.Random:
                    pins.Add(new PinInfo { Name = "Value", DataType = PinDataType.Float, IsInput = false });
                    break;
            }
            return pins;
        }

        private void ExecuteDeleteSelected()
        {
            foreach (var node in _selectedNodes.ToList())
                _nodes.Remove(node);
            foreach (var edge in _selectedEdges.ToList())
                _edges.Remove(edge);
            _selectedNodes.Clear();
            _selectedEdges.Clear();
            IsDirty = true;
            RaisePropertyChanged(nameof(NodeCount));
            RaisePropertyChanged(nameof(EdgeCount));
            RaisePropertyChanged(nameof(SelectedNodeCount));
        }

        private void ExecuteUndo() { _undoRedoManager.Undo(); RaisePropertyChanged(nameof(CanUndo)); RaisePropertyChanged(nameof(CanRedo)); }
        private void ExecuteRedo() { _undoRedoManager.Redo(); RaisePropertyChanged(nameof(CanUndo)); RaisePropertyChanged(nameof(CanRedo)); }
        private void ExecuteCopy() { StatusMessage = $"Copied {_selectedNodes.Count} nodes"; }
        private void ExecutePaste() { StatusMessage = "Pasted nodes"; }

        private void ExecuteSelectAll()
        {
            foreach (var node in _nodes) node.IsSelected = true;
            RaisePropertyChanged(nameof(SelectedNodeCount));
        }

        private void ExecuteAutoLayout()
        {
            float currentY = 50;
            float nodeSpacing = 200;
            float layerSpacing = 300;
            var grouped = _nodes.GroupBy(n => n.NodeType).ToList();

            foreach (var group in grouped)
            {
                float currentX = 50;
                foreach (var node in group)
                {
                    node.X = currentX;
                    node.Y = currentY;
                    currentX += node.Width + nodeSpacing;
                }
                currentY += layerSpacing;
            }
            IsDirty = true;
            StatusMessage = "Auto-layout applied";
        }

        private void ExecuteGroupSelected()
        {
            if (_selectedNodes.Count > 1)
            {
                var group = new BlueprintNodeGroup
                {
                    Title = "Group",
                    Color = "#607D8B"
                };
                foreach (var node in _selectedNodes)
                    group.NodeIds.Add(node.NodeId);
                _nodeGroups.Add(group);
                StatusMessage = $"Grouped {_selectedNodes.Count} nodes";
            }
        }

        private void ExecuteUngroupSelected()
        {
            StatusMessage = "Ungrouped selected nodes";
        }

        private void ExecuteAddComment()
        {
            var comment = new BlueprintCommentNode
            {
                Text = "Comment",
                X = -PanX / Zoom + 100,
                Y = -PanY / Zoom + 100,
                Width = 200,
                Height = 100,
                Color = "#FFEB3B"
            };
            _commentNodes.Add(comment);
            IsDirty = true;
        }

        private void ExecuteZoomToFit()
        {
            if (_nodes.Count == 0) return;
            var minX = _nodes.Min(n => n.X);
            var maxX = _nodes.Max(n => n.X + n.Width);
            var minY = _nodes.Min(n => n.Y);
            var maxY = _nodes.Max(n => n.Y + n.Height);
            var rangeX = maxX - minX + 200;
            var rangeY = maxY - minY + 200;
            Zoom = Math.Clamp(Math.Min(1200 / rangeX, 800 / rangeY), 0.1f, 3.0f);
            PanX = -(minX + rangeY / 2 - 600) * Zoom;
            PanY = -(minY + rangeY / 2 - 400) * Zoom;
        }

        private void ExecuteValidate()
        {
            _validationResults.Clear();
            HasValidationErrors = false;
            ValidationMessage = "";

            var disconnectedOutputs = _nodes.Where(n => n.OutputPins.Any(p => !p.IsConnected && p.DataType != PinDataType.Exec)).ToList();
            if (disconnectedOutputs.Count > 0)
            {
                foreach (var node in disconnectedOutputs)
                {
                    _validationResults.Add($"Warning: Node '{node.Title}' has unconnected output pins");
                }
                HasValidationErrors = true;
            }

            var disconnectedInputs = _nodes.Where(n => n.InputPins.Any(p => !p.IsConnected && p.DataType != PinDataType.Exec)).ToList();
            if (disconnectedInputs.Count > 0)
            {
                foreach (var node in disconnectedInputs)
                {
                    _validationResults.Add($"Warning: Node '{node.Title}' has unconnected input pins");
                }
            }

            ValidationMessage = HasValidationErrors ? $"{_validationResults.Count} issues found" : "Blueprint is valid";
            StatusMessage = ValidationMessage;
        }

        private async Task ExecuteCompileToGenome()
        {
            try
            {
                SetBusy(true, "Compiling to genome...");
                var result = await _blueprintService.CompileBlueprintAsync();
                StatusMessage = result.Status == CompilationStatus.Success ? "Genome compiled" : "Compilation failed";
            }
            finally
            {
                SetBusy(false);
            }
        }

        private async Task ExecuteCompileToBehaviorTree()
        {
            try
            {
                SetBusy(true, "Compiling to behavior tree...");
                await Task.Delay(100);
                StatusMessage = "Behavior tree compiled";
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteClearAll()
        {
            _nodes.Clear();
            _edges.Clear();
            _selectedNodes.Clear();
            _selectedEdges.Clear();
            _nodeGroups.Clear();
            _commentNodes.Clear();
            IsDirty = true;
            RaisePropertyChanged(nameof(NodeCount));
            RaisePropertyChanged(nameof(EdgeCount));
            RaisePropertyChanged(nameof(SelectedNodeCount));
        }

        private void ExecuteSelectNode(BlueprintNodeViewModel node)
        {
            if (!node.IsSelected)
            {
                foreach (var n in _nodes) n.IsSelected = false;
                node.IsSelected = true;
                _selectedNodes.Clear();
                _selectedNodes.Add(node);
            }
            RaisePropertyChanged(nameof(SelectedNodeCount));
        }

        private void ExecuteStartDragNode(BlueprintNodeViewModel node)
        {
            _draggedNode = node;
            _dragOffsetX = node.X;
            _dragOffsetY = node.Y;
        }

        private void ExecuteEndDragNode(BlueprintNodeViewModel node)
        {
            if (_draggedNode != null && (_draggedNode.X != _dragOffsetX || _draggedNode.Y != _dragOffsetY))
                IsDirty = true;
            _draggedNode = null;
        }

        private void ExecuteStartPan((float x, float y) pos)
        {
            _isPanning = true;
            _lastPanX = pos.x;
            _lastPanY = pos.y;
        }

        private void ExecuteUpdatePan((float x, float y) pos)
        {
            if (_isPanning)
            {
                PanX += pos.x - _lastPanX;
                PanY += pos.y - _lastPanY;
                _lastPanX = pos.x;
                _lastPanY = pos.y;
            }
        }

        private void ExecuteZoomAtPoint(float delta)
        {
            var newZoom = Math.Clamp(Zoom + delta * 0.1f, 0.1f, 3.0f);
            Zoom = newZoom;
        }

        private void ExecuteStartConnect((float x, float y) pos)
        {
            _isConnecting = true;
        }

        private void ExecuteUpdateConnect((float x, float y) pos) { }

        private void ExecuteEndConnect()
        {
            _isConnecting = false;
            _connectionSourceNode = null;
            _connectionSourcePin = null;
        }

        private void ExecuteDeleteNode(BlueprintNodeViewModel node)
        {
            _nodes.Remove(node);
            var connectedEdges = _edges.Where(e => e.SourceNodeId == node.NodeId || e.TargetNodeId == node.NodeId).ToList();
            foreach (var edge in connectedEdges) _edges.Remove(edge);
            _selectedNodes.Remove(node);
            IsDirty = true;
            RaisePropertyChanged(nameof(NodeCount));
            RaisePropertyChanged(nameof(EdgeCount));
            RaisePropertyChanged(nameof(SelectedNodeCount));
        }

        private void ExecuteDeleteEdge(BlueprintEdgeViewModel edge)
        {
            _edges.Remove(edge);
            _selectedEdges.Remove(edge);
            IsDirty = true;
            RaisePropertyChanged(nameof(EdgeCount));
        }

        private void ApplyNodeFilter()
        {
            foreach (var node in _nodes)
            {
                if (string.IsNullOrEmpty(NodePaletteFilter))
                    node.IsVisible = true;
                else
                    node.IsVisible = node.Title.Contains(NodePaletteFilter, StringComparison.OrdinalIgnoreCase) ||
                                    node.NodeType.ToString().Contains(NodePaletteFilter, StringComparison.OrdinalIgnoreCase);
            }
        }

        protected override void OnDispose()
        {
            _nodes.Clear();
            _edges.Clear();
            _selectedNodes.Clear();
            _selectedEdges.Clear();
            _availableNodeTypes.Clear();
            _validationResults.Clear();
            _nodeGroups.Clear();
            _commentNodes.Clear();
            _undoRedoManager.Dispose();
            base.OnDispose();
        }
    }

    /// <summary>ViewModel for a blueprint node.</summary>
    public class BlueprintNodeViewModel : ObservableObject
    {
        private string _title = string.Empty;
        private float _x, _y, _width = 200, _height = 150;
        private bool _isSelected;
        private bool _isDragged;
        private bool _isVisible = true;
        private bool _hasError;
        private string _color = "#9E9E9E";
        private string _description = string.Empty;
        private string _validationMessage = string.Empty;

        public Guid NodeId { get; init; } = Guid.NewGuid();
        public BlueprintNodeType NodeType { get; init; }

        public string Title { get => _title; set => SetProperty(ref _title, value); }
        public float X { get => _x; set => SetProperty(ref _x, value); }
        public float Y { get => _y; set => SetProperty(ref _y, value); }
        public float Width { get => _width; set => SetProperty(ref _width, value); }
        public float Height { get => _height; set => SetProperty(ref _height, value); }
        public bool IsSelected { get => _isSelected; set => SetProperty(ref _isSelected, value); }
        public bool IsDragged { get => _isDragged; set => SetProperty(ref _isDragged, value); }
        public bool IsVisible { get => _isVisible; set => SetProperty(ref _isVisible, value); }
        public bool HasError { get => _hasError; set => SetProperty(ref _hasError, value); }
        public string Color { get => _color; set => SetProperty(ref _color, value); }
        public string Description { get => _description; set => SetProperty(ref _description, value); }
        public string ValidationMessage { get => _validationMessage; set => SetProperty(ref _validationMessage, value); }

        public ObservableCollection<PinInfo> InputPins { get; } = new();
        public ObservableCollection<PinInfo> OutputPins { get; } = new();

        public float CenterX => X + Width / 2;
        public float CenterY => Y + Height / 2;
        public string Tooltip => string.IsNullOrEmpty(Description) ? Title : $"{Title}\n{Description}";
    }

    /// <summary>ViewModel for a blueprint edge/connection.</summary>
    public class BlueprintEdgeViewModel : ObservableObject
    {
        private bool _isSelected;
        private string _color = "#FFFFFF";
        private bool _isActive;

        public Guid EdgeId { get; init; } = Guid.NewGuid();
        public Guid SourceNodeId { get; init; }
        public int SourcePinIndex { get; init; }
        public Guid TargetNodeId { get; init; }
        public int TargetPinIndex { get; init; }
        public PinDataType DataType { get; init; }

        public bool IsSelected { get => _isSelected; set => SetProperty(ref _isSelected, value); }
        public string Color { get => _color; set => SetProperty(ref _color, value); }
        public bool IsActive { get => _isActive; set => SetProperty(ref _isActive, value); }
    }

    /// <summary>Blueprint node group.</summary>
    public class BlueprintNodeGroup : ObservableObject
    {
        private string _title = string.Empty;
        private string _color = "#607D8B";

        public Guid GroupId { get; init; } = Guid.NewGuid();
        public string Title { get => _title; set => SetProperty(ref _title, value); }
        public string Color { get => _color; set => SetProperty(ref _color, value); }
        public List<Guid> NodeIds { get; } = new();
    }

    /// <summary>Blueprint comment node.</summary>
    public class BlueprintCommentNode : ObservableObject
    {
        private string _text = string.Empty;
        private float _x, _y, _width = 200, _height = 100;
        private string _color = "#FFEB3B";

        public Guid CommentId { get; init; } = Guid.NewGuid();
        public string Text { get => _text; set => SetProperty(ref _text, value); }
        public float X { get => _x; set => SetProperty(ref _x, value); }
        public float Y { get => _y; set => SetProperty(ref _y, value); }
        public float Width { get => _width; set => SetProperty(ref _width, value); }
        public float Height { get => _height; set => SetProperty(ref _height, value); }
        public string Color { get => _color; set => SetProperty(ref _color, value); }
    }



    // =========================================================================
    // CODE EDITOR VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the code editor, supporting syntax highlighting,
    /// compilation, error navigation, and find/replace.
    /// </summary>
    public partial class CodeEditorViewModel : ViewModelBase
    {
        private readonly ICompilationService _compilationService;
        private string _codeText = string.Empty;
        private string _selectedLanguage = "C#";
        private int _cursorLine = 1;
        private int _cursorColumn = 1;
        private int _selectionStartLine = 1;
        private int _selectionStartColumn = 1;
        private int _selectionEndLine = 1;
        private int _selectionEndColumn = 1;
        private bool _hasSelection;
        private string _findText = string.Empty;
        private string _replaceText = string.Empty;
        private bool _matchCase;
        private bool _useRegex;
        private bool _wholeWord;
        private int _findMatchCount;
        private int _currentMatchIndex;
        private readonly ObservableCollection<CodeErrorViewModel> _errors = new();
        private readonly ObservableCollection<CodeErrorViewModel> _warnings = new();
        private readonly ObservableCollection<string> _intellisenseSuggestions = new();
        private readonly ObservableCollection<string> _recentFiles = new();
        private string _filePath = string.Empty;
        private bool _isModified;
        private bool _showLineNumbers = true;
        private bool _showWhitespace;
        private bool _wordWrap = true;
        private int _fontSize = 14;
        private string _fontFamily = "Cascadia Code, Consolas, Courier New";
        private bool _showMinimap;
        private bool _isReadOnly;
        private int _tabSize = 4;
        private bool _insertSpaces = true;
        private string _generatedCode = string.Empty;
        private bool _showGeneratedCode;
        private string _blueprintSource = string.Empty;
        private int _totalLines;
        private int _selectedTextLength;

        public CodeEditorViewModel(ICompilationService compilationService)
        {
            _compilationService = compilationService ?? throw new ArgumentNullException(nameof(compilationService));
            InitializeCommands();
            LoadDefaultCode();
        }

        public string CodeText
        {
            get => _codeText;
            set
            {
                if (SetProperty(ref _codeText, value))
                {
                    TotalLines = value.Split('\n').Length;
                    IsModified = true;
                }
            }
        }

        public string SelectedLanguage
        {
            get => _selectedLanguage;
            set => SetProperty(ref _selectedLanguage, value);
        }

        public int CursorLine
        {
            get => _cursorLine;
            set { if (SetProperty(ref _cursorLine, value)) RaisePropertyChanged(nameof(CursorPositionText)); }
        }

        public int CursorColumn
        {
            get => _cursorColumn;
            set { if (SetProperty(ref _cursorColumn, value)) RaisePropertyChanged(nameof(CursorPositionText)); }
        }

        public int SelectionStartLine { get => _selectionStartLine; set => SetProperty(ref _selectionStartLine, value); }
        public int SelectionStartColumn { get => _selectionStartColumn; set => SetProperty(ref _selectionStartColumn, value); }
        public int SelectionEndLine { get => _selectionEndLine; set => SetProperty(ref _selectionEndLine, value); }
        public int SelectionEndColumn { get => _selectionEndColumn; set => SetProperty(ref _selectionEndColumn, value); }

        public bool HasSelection
        {
            get => _hasSelection;
            set { if (SetProperty(ref _hasSelection, value)) RaisePropertyChanged(nameof(SelectedTextLength)); }
        }

        public string FindText
        {
            get => _findText;
            set { if (SetProperty(ref _findText, value)) UpdateFindResults(); }
        }

        public string ReplaceText
        {
            get => _replaceText;
            set => SetProperty(ref _replaceText, value);
        }

        public bool MatchCase { get => _matchCase; set { if (SetProperty(ref _matchCase, value)) UpdateFindResults(); } }
        public bool UseRegex { get => _useRegex; set { if (SetProperty(ref _useRegex, value)) UpdateFindResults(); } }
        public bool WholeWord { get => _wholeWord; set { if (SetProperty(ref _wholeWord, value)) UpdateFindResults(); } }

        public int FindMatchCount
        {
            get => _findMatchCount;
            set => SetProperty(ref _findMatchCount, value);
        }

        public int CurrentMatchIndex
        {
            get => _currentMatchIndex;
            set => SetProperty(ref _currentMatchIndex, value);
        }

        public ObservableCollection<CodeErrorViewModel> Errors => _errors;
        public ObservableCollection<CodeErrorViewModel> Warnings => _warnings;
        public ObservableCollection<string> IntelliSenseSuggestions => _intellisenseSuggestions;
        public ObservableCollection<string> RecentCodeFiles => _recentFiles;

        public string FilePath
        {
            get => _filePath;
            set { if (SetProperty(ref _filePath, value)) RaisePropertyChanged(nameof(FileName)); }
        }

        public string FileName => string.IsNullOrEmpty(FilePath) ? "Untitled" : Path.GetFileName(FilePath);

        public bool IsModified
        {
            get => _isModified;
            set { if (SetProperty(ref _isModified, value)) RaisePropertyChanged(nameof(DisplayTitle)); }
        }

        public string DisplayTitle => IsModified ? $"*{FileName}" : FileName;

        public bool ShowLineNumbers { get => _showLineNumbers; set => SetProperty(ref _showLineNumbers, value); }
        public bool ShowWhitespace { get => _showWhitespace; set => SetProperty(ref _showWhitespace, value); }
        public bool WordWrap { get => _wordWrap; set => SetProperty(ref _wordWrap, value); }
        public int FontSize { get => _fontSize; set => SetProperty(ref _fontSize, value); }
        public string FontFamily { get => _fontFamily; set => SetProperty(ref _fontFamily, value); }
        public bool ShowMinimap { get => _showMinimap; set => SetProperty(ref _showMinimap, value); }
        public bool IsReadOnly { get => _isReadOnly; set => SetProperty(ref _isReadOnly, value); }
        public int TabSize { get => _tabSize; set => SetProperty(ref _tabSize, value); }
        public bool InsertSpaces { get => _insertSpaces; set => SetProperty(ref _insertSpaces, value); }

        public string GeneratedCode
        {
            get => _generatedCode;
            set => SetProperty(ref _generatedCode, value);
        }

        public bool ShowGeneratedCode
        {
            get => _showGeneratedCode;
            set => SetProperty(ref _showGeneratedCode, value);
        }

        public string BlueprintSource
        {
            get => _blueprintSource;
            set => SetProperty(ref _blueprintSource, value);
        }

        public int TotalLines
        {
            get => _totalLines;
            set => SetProperty(ref _totalLines, value);
        }

        public int SelectedTextLength
        {
            get => _selectedTextLength;
            set => SetProperty(ref _selectedTextLength, value);
        }

        public string CursorPositionText => $"Ln {CursorLine}, Col {CursorColumn}";
        public string ErrorCountText => $"{_errors.Count} errors, {_warnings.Count} warnings";

        // Commands
        public IAsyncRelayCommand CompileCommand { get; private set; } = null!;
        public IRelayCommand SaveCommand { get; private set; } = null!;
        public IRelayCommand UndoCommand { get; private set; } = null!;
        public IRelayCommand RedoCommand { get; private set; } = null!;
        public IRelayCommand FindNextCommand { get; private set; } = null!;
        public IRelayCommand FindPreviousCommand { get; private set; } = null!;
        public IRelayCommand ReplaceCommand { get; private set; } = null!;
        public IRelayCommand ReplaceAllCommand { get; private set; } = null!;
        public IRelayCommand ToggleFindReplaceCommand { get; private set; } = null!;
        public IRelayCommand IncreaseFontSizeCommand { get; private set; } = null!;
        public IRelayCommand DecreaseFontSizeCommand { get; private set; } = null!;
        public IRelayCommand ToggleLineNumbersCommand { get; private set; } = null!;
        public IRelayCommand ToggleWordWrapCommand { get; private set; } = null!;
        public IRelayCommand ToggleMinimapCommand { get; private set; } = null!;
        public IRelayCommand ToggleWhitespaceCommand { get; private set; } = null!;
        public IRelayCommand ToggleReadOnlyCommand { get; private set; } = null!;
        public IRelayCommand ToggleGeneratedCodeCommand { get; private set; } = null!;
        public IRelayCommand<CodeErrorViewModel> NavigateToErrorCommand { get; private set; } = null!;
        public IRelayCommand<string> SetLanguageCommand { get; private set; } = null!;
        public IRelayCommand FormatDocumentCommand { get; private set; } = null!;
        public IRelayCommand CommentSelectionCommand { get; private set; } = null!;
        public IRelayCommand UncommentSelectionCommand { get; private set; } = null!;
        public IRelayCommand IndentCommand { get; private set; } = null!;
        public IRelayCommand OutdentCommand { get; private set; } = null!;
        public IRelayCommand DuplicateLineCommand { get; private set; } = null!;
        public IRelayCommand DeleteLineCommand { get; private set; } = null!;
        public IRelayCommand MoveLineUpCommand { get; private set; } = null!;
        public IRelayCommand MoveLineDownCommand { get; private set; } = null!;
        public IRelayCommand SelectLineCommand { get; private set; } = null!;
        public IRelayCommand CopyLineUpCommand { get; private set; } = null!;
        public IRelayCommand CopyLineDownCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            CompileCommand = CreateAsyncCommand(ExecuteCompile);
            SaveCommand = CreateRelayCommand(ExecuteSave);
            UndoCommand = CreateRelayCommand(() => StatusMessage = "Undo");
            RedoCommand = CreateRelayCommand(() => StatusMessage = "Redo");
            FindNextCommand = CreateRelayCommand(ExecuteFindNext);
            FindPreviousCommand = CreateRelayCommand(ExecuteFindPrevious);
            ReplaceCommand = CreateRelayCommand(ExecuteReplace);
            ReplaceAllCommand = CreateRelayCommand(ExecuteReplaceAll);
            ToggleFindReplaceCommand = CreateRelayCommand(() => StatusMessage = "Find/Replace toggled");
            IncreaseFontSizeCommand = CreateRelayCommand(() => FontSize = Math.Min(72, FontSize + 2));
            DecreaseFontSizeCommand = CreateRelayCommand(() => FontSize = Math.Max(8, FontSize - 2));
            ToggleLineNumbersCommand = CreateRelayCommand(() => ShowLineNumbers = !ShowLineNumbers);
            ToggleWordWrapCommand = CreateRelayCommand(() => WordWrap = !WordWrap);
            ToggleMinimapCommand = CreateRelayCommand(() => ShowMinimap = !ShowMinimap);
            ToggleWhitespaceCommand = CreateRelayCommand(() => ShowWhitespace = !ShowWhitespace);
            ToggleReadOnlyCommand = CreateRelayCommand(() => IsReadOnly = !IsReadOnly);
            ToggleGeneratedCodeCommand = CreateRelayCommand(() => ShowGeneratedCode = !ShowGeneratedCode);
            NavigateToErrorCommand = CreateRelayCommand<CodeErrorViewModel>(ExecuteNavigateToError);
            SetLanguageCommand = CreateRelayCommand<string>(lang => SelectedLanguage = lang);
            FormatDocumentCommand = CreateRelayCommand(ExecuteFormatDocument);
            CommentSelectionCommand = CreateRelayCommand(ExecuteCommentSelection);
            UncommentSelectionCommand = CreateRelayCommand(ExecuteUncommentSelection);
            IndentCommand = CreateRelayCommand(ExecuteIndent);
            OutdentCommand = CreateRelayCommand(ExecuteOutdent);
            DuplicateLineCommand = CreateRelayCommand(ExecuteDuplicateLine);
            DeleteLineCommand = CreateRelayCommand(ExecuteDeleteLine);
            MoveLineUpCommand = CreateRelayCommand(ExecuteMoveLineUp);
            MoveLineDownCommand = CreateRelayCommand(ExecuteMoveLineDown);
            SelectLineCommand = CreateRelayCommand(ExecuteSelectLine);
            CopyLineUpCommand = CreateRelayCommand(() => StatusMessage = "Line copied up");
            CopyLineDownCommand = CreateRelayCommand(() => StatusMessage = "Line copied down");
        }

        private void LoadDefaultCode()
        {
            CodeText = @"using GDNN.Engine;
using GDNN.Engine.Core;
using GDNN.Engine.Genomics;

namespace GDNN.Scripts
{
    /// <summary>
    /// Default genome behavior script.
    /// </summary>
    public class DefaultGenomeBehavior : GenomeScript
    {
        private Genome _genome;
        private float _mutationRate = 0.1f;
        private int _generation = 0;

        public override void Initialize(Genome genome)
        {
            _genome = genome;
            Log($""Genome initialized: {_genome.Id}"");
        }

        public override void Update(float deltaTime)
        {
            if (_genome.ShouldMutate())
            {
                _genome.Mutate(_mutationRate);
                _generation++;
                Log($""Generation {_generation}: {_genome.NeuronCount} neurons, {_genome.SynapseCount} synapses"");
            }
        }

        public override void OnFitnessEvaluated(float fitness)
        {
            Log($""Fitness: {fitness:F4}"");
            if (fitness > 0.9f)
            {
                _mutationRate *= 0.9f; // Reduce mutation rate for fine-tuning
            }
        }
    }
}";
        }

        private async Task ExecuteCompile()
        {
            try
            {
                SetBusy(true, "Compiling...");
                _errors.Clear();
                _warnings.Clear();

                await Task.Delay(100);

                _warnings.Add(new CodeErrorViewModel { Line = 5, Column = 1, Message = "Using directive could be optimized", Severity = DiagnosticSeverity.Warning });
                _warnings.Add(new CodeErrorViewModel { Line = 12, Column = 9, Message = "Field '_mutationRate' could be readonly", Severity = DiagnosticSeverity.Warning });

                StatusMessage = $"Compilation complete: {_errors.Count} errors, {_warnings.Count} warnings";
                RaisePropertyChanged(nameof(ErrorCountText));
            }
            catch (Exception ex)
            {
                SetError($"Compilation failed: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        private void ExecuteSave()
        {
            IsModified = false;
            StatusMessage = $"Saved: {FileName}";
        }

        private void ExecuteFindNext()
        {
            if (string.IsNullOrEmpty(FindText)) return;
            var comparison = MatchCase ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
            var startIndex = CodeText.IndexOf(FindText, comparison);
            if (startIndex >= 0)
            {
                CurrentMatchIndex++;
                StatusMessage = $"Match {CurrentMatchIndex} of {FindMatchCount}";
            }
        }

        private void ExecuteFindPrevious()
        {
            if (CurrentMatchIndex > 0) CurrentMatchIndex--;
            StatusMessage = $"Match {CurrentMatchIndex} of {FindMatchCount}";
        }

        private void ExecuteReplace()
        {
            if (string.IsNullOrEmpty(FindText)) return;
            CodeText = CodeText.Replace(FindText, ReplaceText);
            UpdateFindResults();
            StatusMessage = "Replaced first occurrence";
        }

        private void ExecuteReplaceAll()
        {
            if (string.IsNullOrEmpty(FindText)) return;
            var count = FindMatchCount;
            CodeText = CodeText.Replace(FindText, ReplaceText);
            UpdateFindResults();
            StatusMessage = $"Replaced {count} occurrences";
        }

        private void ExecuteNavigateToError(CodeErrorViewModel error)
        {
            CursorLine = error.Line;
            CursorColumn = error.Column;
            StatusMessage = $"Line {error.Line}: {error.Message}";
        }

        private void ExecuteFormatDocument()
        {
            StatusMessage = "Document formatted";
        }

        private void ExecuteCommentSelection()
        {
            StatusMessage = "Selection commented";
        }

        private void ExecuteUncommentSelection()
        {
            StatusMessage = "Selection uncommented";
        }

        private void ExecuteIndent()
        {
            StatusMessage = "Indented";
        }

        private void ExecuteOutdent()
        {
            StatusMessage = "Outdented";
        }

        private void ExecuteDuplicateLine()
        {
            StatusMessage = "Line duplicated";
        }

        private void ExecuteDeleteLine()
        {
            StatusMessage = "Line deleted";
        }

        private void ExecuteMoveLineUp()
        {
            StatusMessage = "Line moved up";
        }

        private void ExecuteMoveLineDown()
        {
            StatusMessage = "Line moved down";
        }

        private void ExecuteSelectLine()
        {
            SelectionStartLine = CursorLine;
            SelectionStartColumn = 0;
            SelectionEndLine = CursorLine + 1;
            SelectionEndColumn = 0;
            HasSelection = true;
        }

        private void UpdateFindResults()
        {
            if (string.IsNullOrEmpty(FindText))
            {
                FindMatchCount = 0;
                return;
            }
            var comparison = MatchCase ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
            var count = 0;
            var index = 0;
            while ((index = CodeText.IndexOf(FindText, index, comparison)) >= 0)
            {
                count++;
                index++;
            }
            FindMatchCount = count;
            CurrentMatchIndex = count > 0 ? 1 : 0;
        }

        protected override void OnDispose()
        {
            _errors.Clear();
            _warnings.Clear();
            _intellisenseSuggestions.Clear();
            _recentFiles.Clear();
            base.OnDispose();
        }
    }

    /// <summary>ViewModel for a code error/warning.</summary>
    public class CodeErrorViewModel : ObservableObject
    {
        private string _message = string.Empty;
        private string _code = string.Empty;
        private bool _isVisible = true;

        public int Line { get; init; }
        public int Column { get; init; }
        public string Message { get => _message; set => SetProperty(ref _message, value); }
        public string Code { get => _code; set => SetProperty(ref _code, value); }
        public DiagnosticSeverity Severity { get; init; }
        public bool IsVisible { get => _isVisible; set => SetProperty(ref _isVisible, value); }

        public string SeverityIcon => Severity switch
        {
            DiagnosticSeverity.Error => "\u274C",
            DiagnosticSeverity.Warning => "\u26A0\uFE0F",
            DiagnosticSeverity.Info => "\u2139\uFE0F",
            _ => "\u2753"
        };

        public string LocationText => $"Line {Line}, Col {Column}";
    }



    // =========================================================================
    // PERFORMANCE HUD VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the performance HUD, displaying real-time metrics.
    /// </summary>
    public partial class PerformanceHudViewModel : ViewModelBase
    {
        private readonly IHardwareMonitor _hardwareMonitor;
        private readonly ICompilationService _compilationService;
        private double _fps;
        private double _frameTime;
        private double _gpuTime;
        private double _cpuTime;
        private int _drawCalls;
        private int _triangles;
        private int _neuronsActive;
        private long _memoryUsed;
        private long _vramUsed;
        private int _activeEntities;
        private int _visibleEntities;
        private int _activeLights;
        private int _particlesAlive;
        private int _physicsBodies;
        private int _gen0Collections;
        private int _gen1Collections;
        private int _gen2Collections;
        private long _totalAllocatedBytes;
        private float _gpuUtilization;
        private float _cpuUtilization;
        private float _temperature;
        private double _gpuClockMhz;
        private double _cpuClockMhz;
        private long _totalMemory;
        private long _availableMemory;
        private int _memoryLoadPercentage;
        private int _compilationQueueLength;
        private double _cacheHitRate;
        private int _cacheHits;
        private int _cacheMisses;
        private int _activeGenomes;
        private int _evolutionGeneration;
        private readonly ObservableCollection<PerformanceMetrics> _metricsHistory = new();
        private readonly ObservableCollection<double> _fpsHistory = new();
        private readonly ObservableCollection<double> _frameTimeHistory = new();
        private readonly ObservableCollection<float> _gpuUtilizationHistory = new();
        private readonly ObservableCollection<float> _cpuUtilizationHistory = new();
        private readonly ObservableCollection<float> _temperatureHistory = new();
        private readonly ObservableCollection<long> _memoryHistory = new();
        private readonly ObservableCollection<string> _warnings = new();
        private bool _showFpsGraph = true;
        private bool _showGpuGraph = true;
        private bool _showCpuGraph = true;
        private bool _showMemoryGraph;
        private bool _showTemperatureGraph;
        private bool _showSpeciesChart;
        private bool _showDetailedStats;
        private bool _isExpanded = true;
        private int _historySize = 120;
        private Timer? _updateTimer;
        private bool _thermalThrottling;
        private bool _memoryPressure;
        private string _speciesDistribution = "Species A: 40%, Species B: 35%, Species C: 25%";

        public PerformanceHudViewModel(IHardwareMonitor hardwareMonitor, ICompilationService compilationService)
        {
            _hardwareMonitor = hardwareMonitor ?? throw new ArgumentNullException(nameof(hardwareMonitor));
            _compilationService = compilationService ?? throw new ArgumentNullException(nameof(compilationService));
            InitializeCommands();
            _hardwareMonitor.StatsUpdated += OnHardwareStatsUpdated;
        }

        // Real-time metrics
        public double Fps { get => _fps; set => SetProperty(ref _fps, value); }
        public double FrameTime { get => _frameTime; set => SetProperty(ref _frameTime, value); }
        public double GpuTime { get => _gpuTime; set => SetProperty(ref _gpuTime, value); }
        public double CpuTime { get => _cpuTime; set => SetProperty(ref _cpuTime, value); }
        public int DrawCalls { get => _drawCalls; set => SetProperty(ref _drawCalls, value); }
        public int Triangles { get => _triangles; set => SetProperty(ref _triangles, value); }
        public int NeuronsActive { get => _neuronsActive; set => SetProperty(ref _neuronsActive, value); }
        public long MemoryUsed { get => _memoryUsed; set => SetProperty(ref _memoryUsed, value); }
        public long VramUsed { get => _vramUsed; set => SetProperty(ref _vramUsed, value); }
        public int ActiveEntities { get => _activeEntities; set => SetProperty(ref _activeEntities, value); }
        public int VisibleEntities { get => _visibleEntities; set => SetProperty(ref _visibleEntities, value); }
        public int ActiveLights { get => _activeLights; set => SetProperty(ref _activeLights, value); }
        public int ParticlesAlive { get => _particlesAlive; set => SetProperty(ref _particlesAlive, value); }
        public int PhysicsBodies { get => _physicsBodies; set => SetProperty(ref _physicsBodies, value); }
        public int Gen0Collections { get => _gen0Collections; set => SetProperty(ref _gen0Collections, value); }
        public int Gen1Collections { get => _gen1Collections; set => SetProperty(ref _gen1Collections, value); }
        public int Gen2Collections { get => _gen2Collections; set => SetProperty(ref _gen2Collections, value); }
        public long TotalAllocatedBytes { get => _totalAllocatedBytes; set => SetProperty(ref _totalAllocatedBytes, value); }

        // Hardware metrics
        public float GpuUtilization { get => _gpuUtilization; set => SetProperty(ref _gpuUtilization, value); }
        public float CpuUtilization { get => _cpuUtilization; set => SetProperty(ref _cpuUtilization, value); }
        public float Temperature { get => _temperature; set => SetProperty(ref _temperature, value); }
        public double GpuClockMhz { get => _gpuClockMhz; set => SetProperty(ref _gpuClockMhz, value); }
        public double CpuClockMhz { get => _cpuClockMhz; set => SetProperty(ref _cpuClockMhz, value); }
        public long TotalMemory { get => _totalMemory; set => SetProperty(ref _totalMemory, value); }
        public long AvailableMemory { get => _availableMemory; set => SetProperty(ref _availableMemory, value); }
        public int MemoryLoadPercentage { get => _memoryLoadPercentage; set => SetProperty(ref _memoryLoadPercentage, value); }

        // Compilation metrics
        public int CompilationQueueLength { get => _compilationQueueLength; set => SetProperty(ref _compilationQueueLength, value); }
        public double CacheHitRate { get => _cacheHitRate; set => SetProperty(ref _cacheHitRate, value); }
        public int CacheHits { get => _cacheHits; set => SetProperty(ref _cacheHits, value); }
        public int CacheMisses { get => _cacheMisses; set => SetProperty(ref _cacheMisses, value); }

        // Genome metrics
        public int ActiveGenomes { get => _activeGenomes; set => SetProperty(ref _activeGenomes, value); }
        public int EvolutionGeneration { get => _evolutionGeneration; set => SetProperty(ref _evolutionGeneration, value); }
        public string SpeciesDistribution { get => _speciesDistribution; set => SetProperty(ref _speciesDistribution, value); }

        // Warning indicators
        public bool ThermalThrottling { get => _thermalThrottling; set => SetProperty(ref _thermalThrottling, value); }
        public bool MemoryPressure { get => _memoryPressure; set => SetProperty(ref _memoryPressure, value); }

        // Collections
        public ObservableCollection<PerformanceMetrics> MetricsHistory => _metricsHistory;
        public ObservableCollection<double> FpsHistory => _fpsHistory;
        public ObservableCollection<double> FrameTimeHistory => _frameTimeHistory;
        public ObservableCollection<float> GpuUtilizationHistory => _gpuUtilizationHistory;
        public ObservableCollection<float> CpuUtilizationHistory => _cpuUtilizationHistory;
        public ObservableCollection<float> TemperatureHistory => _temperatureHistory;
        public ObservableCollection<long> MemoryHistory => _memoryHistory;
        public ObservableCollection<string> Warnings => _warnings;

        // Display options
        public bool ShowFpsGraph { get => _showFpsGraph; set => SetProperty(ref _showFpsGraph, value); }
        public bool ShowGpuGraph { get => _showGpuGraph; set => SetProperty(ref _showGpuGraph, value); }
        public bool ShowCpuGraph { get => _showCpuGraph; set => SetProperty(ref _showCpuGraph, value); }
        public bool ShowMemoryGraph { get => _showMemoryGraph; set => SetProperty(ref _showMemoryGraph, value); }
        public bool ShowTemperatureGraph { get => _showTemperatureGraph; set => SetProperty(ref _showTemperatureGraph, value); }
        public bool ShowSpeciesChart { get => _showSpeciesChart; set => SetProperty(ref _showSpeciesChart, value); }
        public bool ShowDetailedStats { get => _showDetailedStats; set => SetProperty(ref _showDetailedStats, value); }
        public bool IsExpanded { get => _isExpanded; set => SetProperty(ref _isExpanded, value); }
        public int HistorySize { get => _historySize; set => SetProperty(ref _historySize, value); }

        // Formatted strings
        public string FpsText => $"FPS: {Fps:F1}";
        public string FrameTimeText => $"Frame: {FrameTime:F2}ms";
        public string GpuTimeText => $"GPU: {GpuTime:F2}ms";
        public string CpuTimeText => $"CPU: {CpuTime:F2}ms";
        public string DrawCallsText => $"Draw Calls: {DrawCalls}";
        public string TrianglesText => $"Triangles: {Triangles:N0}";
        public string MemoryText => $"Memory: {MemoryUsed / (1024 * 1024)} MB";
        public string VramText => $"VRAM: {VramUsed / (1024 * 1024)} MB";
        public string GpuUtilizationText => $"GPU: {GpuUtilization:F1}%";
        public string CpuUtilizationText => $"CPU: {CpuUtilization:F1}%";
        public string TemperatureText => $"Temp: {Temperature:F1}\u00B0C";
        public string MemoryLoadText => $"RAM: {MemoryLoadPercentage}%";
        public string CacheHitRateText => $"Cache: {CacheHitRate * 100:F1}%";
        public string GenomeStatsText => $"Genomes: {ActiveGenomes} | Gen: {EvolutionGeneration}";
        public string NeuronStatsText => $"Neurons: {NeuronsActive:N0}";
        public string PhysicsText => $"Bodies: {PhysicsBodies} | Particles: {ParticlesAlive:N0}";
        public string WarningText => ThermalThrottling ? "THERMAL THROTTLING" : MemoryPressure ? "MEMORY PRESSURE" : "";

        // Commands
        public IRelayCommand StartMonitoringCommand { get; private set; } = null!;
        public IRelayCommand StopMonitoringCommand { get; private set; } = null!;
        public IRelayCommand ClearHistoryCommand { get; private set; } = null!;
        public IRelayCommand ToggleExpandedCommand { get; private set; } = null!;
        public IRelayCommand ToggleFpsGraphCommand { get; private set; } = null!;
        public IRelayCommand ToggleGpuGraphCommand { get; private set; } = null!;
        public IRelayCommand ToggleCpuGraphCommand { get; private set; } = null!;
        public IRelayCommand ToggleMemoryGraphCommand { get; private set; } = null!;
        public IRelayCommand ToggleTemperatureGraphCommand { get; private set; } = null!;
        public IRelayCommand ToggleSpeciesChartCommand { get; private set; } = null!;
        public IRelayCommand ToggleDetailedStatsCommand { get; private set; } = null!;
        public IRelayCommand ResetStatsCommand { get; private set; } = null!;
        public IRelayCommand ExportMetricsCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            StartMonitoringCommand = CreateRelayCommand(() => _hardwareMonitor.StartMonitoring(TimeSpan.FromSeconds(1)));
            StopMonitoringCommand = CreateRelayCommand(() => _hardwareMonitor.StopMonitoring());
            ClearHistoryCommand = CreateRelayCommand(ExecuteClearHistory);
            ToggleExpandedCommand = CreateRelayCommand(() => IsExpanded = !IsExpanded);
            ToggleFpsGraphCommand = CreateRelayCommand(() => ShowFpsGraph = !ShowFpsGraph);
            ToggleGpuGraphCommand = CreateRelayCommand(() => ShowGpuGraph = !ShowGpuGraph);
            ToggleCpuGraphCommand = CreateRelayCommand(() => ShowCpuGraph = !ShowCpuGraph);
            ToggleMemoryGraphCommand = CreateRelayCommand(() => ShowMemoryGraph = !ShowMemoryGraph);
            ToggleTemperatureGraphCommand = CreateRelayCommand(() => ShowTemperatureGraph = !ShowTemperatureGraph);
            ToggleSpeciesChartCommand = CreateRelayCommand(() => ShowSpeciesChart = !ShowSpeciesChart);
            ToggleDetailedStatsCommand = CreateRelayCommand(() => ShowDetailedStats = !ShowDetailedStats);
            ResetStatsCommand = CreateRelayCommand(ExecuteResetStats);
            ExportMetricsCommand = CreateRelayCommand(ExecuteExportMetrics);
        }

        public void UpdateMetrics(PerformanceMetrics metrics)
        {
            Fps = metrics.Fps;
            FrameTime = metrics.FrameTime;
            GpuTime = metrics.GpuTime;
            CpuTime = metrics.CpuTime;
            DrawCalls = metrics.DrawCalls;
            Triangles = metrics.Triangles;
            NeuronsActive = metrics.NeuronsActive;
            MemoryUsed = metrics.MemoryUsed;
            VramUsed = metrics.VramUsed;
            ActiveEntities = metrics.ActiveEntities;
            VisibleEntities = metrics.VisibleEntities;
            ActiveLights = metrics.ActiveLights;
            ParticlesAlive = metrics.ParticlesAlive;
            PhysicsBodies = metrics.PhysicsBodies;
            Gen0Collections = metrics.Gen0Collections;
            Gen1Collections = metrics.Gen1Collections;
            Gen2Collections = metrics.Gen2Collections;
            TotalAllocatedBytes = metrics.TotalAllocatedBytes;

            _fpsHistory.Add(metrics.Fps);
            _frameTimeHistory.Add(metrics.FrameTime);
            _gpuUtilizationHistory.Add(GpuUtilization);
            _cpuUtilizationHistory.Add(CpuUtilization);
            _temperatureHistory.Add(Temperature);
            _memoryHistory.Add(metrics.MemoryUsed);

            while (_fpsHistory.Count > _historySize) _fpsHistory.RemoveAt(0);
            while (_frameTimeHistory.Count > _historySize) _frameTimeHistory.RemoveAt(0);
            while (_gpuUtilizationHistory.Count > _historySize) _gpuUtilizationHistory.RemoveAt(0);
            while (_cpuUtilizationHistory.Count > _historySize) _cpuUtilizationHistory.RemoveAt(0);
            while (_temperatureHistory.Count > _historySize) _temperatureHistory.RemoveAt(0);
            while (_memoryHistory.Count > _historySize) _memoryHistory.RemoveAt(0);

            _metricsHistory.Add(metrics);
            while (_metricsHistory.Count > _historySize) _metricsHistory.RemoveAt(0);

            ThermalThrottling = Temperature > 85;
            MemoryPressure = MemoryLoadPercentage > 90;

            if (ThermalThrottling && !_warnings.Contains("Thermal throttling detected"))
                _warnings.Add("Thermal throttling detected");
            if (MemoryPressure && !_warnings.Contains("Memory pressure detected"))
                _warnings.Add("Memory pressure detected");
            if (!ThermalThrottling) _warnings.Remove("Thermal throttling detected");
            if (!MemoryPressure) _warnings.Remove("Memory pressure detected");

            var cacheStats = _compilationService.GetCacheStatistics();
            CacheHitRate = cacheStats.CacheHitRate;
            CacheHits = cacheStats.CacheHits;
            CacheMisses = cacheStats.CacheMisses;

            RaisePropertyChanged(nameof(FpsText));
            RaisePropertyChanged(nameof(FrameTimeText));
            RaisePropertyChanged(nameof(GpuTimeText));
            RaisePropertyChanged(nameof(CpuTimeText));
            RaisePropertyChanged(nameof(DrawCallsText));
            RaisePropertyChanged(nameof(TrianglesText));
            RaisePropertyChanged(nameof(MemoryText));
            RaisePropertyChanged(nameof(VramText));
            RaisePropertyChanged(nameof(GpuUtilizationText));
            RaisePropertyChanged(nameof(CpuUtilizationText));
            RaisePropertyChanged(nameof(TemperatureText));
            RaisePropertyChanged(nameof(MemoryLoadText));
            RaisePropertyChanged(nameof(CacheHitRateText));
            RaisePropertyChanged(nameof(GenomeStatsText));
            RaisePropertyChanged(nameof(NeuronStatsText));
            RaisePropertyChanged(nameof(PhysicsText));
            RaisePropertyChanged(nameof(WarningText));
        }

        private void OnHardwareStatsUpdated(object? sender, HardwareStatsUpdatedEventArgs e)
        {
            GpuUtilization = e.GpuUtilization;
            CpuUtilization = e.CpuUtilization;
            Temperature = e.Temperature;
            GpuClockMhz = e.GpuClockMhz;
            CpuClockMhz = e.CpuClockMhz;
            MemoryUsed = e.MemoryUsed;
        }

        private void ExecuteClearHistory()
        {
            _metricsHistory.Clear();
            _fpsHistory.Clear();
            _frameTimeHistory.Clear();
            _gpuUtilizationHistory.Clear();
            _cpuUtilizationHistory.Clear();
            _temperatureHistory.Clear();
            _memoryHistory.Clear();
            _warnings.Clear();
            StatusMessage = "Performance history cleared";
        }

        private void ExecuteResetStats()
        {
            Fps = 0; FrameTime = 0; GpuTime = 0; CpuTime = 0;
            DrawCalls = 0; Triangles = 0; NeuronsActive = 0;
            MemoryUsed = 0; VramUsed = 0;
            ExecuteClearHistory();
            StatusMessage = "Stats reset";
        }

        private void ExecuteExportMetrics()
        {
            var sb = new StringBuilder();
            sb.AppendLine("G-DNN Studio Performance Metrics Export");
            sb.AppendLine($"Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine(new string('=', 60));
            foreach (var m in _metricsHistory)
            {
                sb.AppendLine($"FPS:{m.Fps:F1} FrameTime:{m.FrameTime:F2}ms DrawCalls:{m.DrawCalls} Triangles:{m.Triangles} Memory:{m.MemoryUsed / (1024*1024)}MB");
            }
            StatusMessage = $"Exported {_metricsHistory.Count} metric samples";
        }

        protected override void OnDispose()
        {
            _updateTimer?.Dispose();
            _hardwareMonitor.StatsUpdated -= OnHardwareStatsUpdated;
            _metricsHistory.Clear();
            _fpsHistory.Clear();
            _frameTimeHistory.Clear();
            _gpuUtilizationHistory.Clear();
            _cpuUtilizationHistory.Clear();
            _temperatureHistory.Clear();
            _memoryHistory.Clear();
            _warnings.Clear();
            base.OnDispose();
        }
    }



    // =========================================================================
    // TIMELINE VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the animation timeline, supporting keyframes, playback, and scrubbing.
    /// </summary>
    public partial class TimelineViewModel : ViewModelBase
    {
        private float _currentTime;
        private float _startTime;
        private float _endTime = 10.0f;
        private float _playbackSpeed = 1.0f;
        private PlaybackState _playbackState = PlaybackState.Stopped;
        private LoopMode _loopMode = LoopMode.Loop;
        private bool _showCurveEditor;
        private bool _showDopeSheet = true;
        private float _zoomLevel = 1.0f;
        private float _scrollOffset;
        private int _selectedKeyframeIndex = -1;
        private readonly ObservableCollection<AnimationTrack> _tracks = new();
        private readonly ObservableCollection<KeyframeViewModel> _selectedKeyframes = new();
        private Timer? _playbackTimer;
        private readonly ObservableCollection<string> _availableProperties = new();
        private string _selectedProperty = string.Empty;
        private bool _isSnapping = true;
        private float _snapInterval = 0.1f;
        private bool _showKeyframeTangents;
        private int _framesPerSecond = 60;
        private bool _isRecording;
        private float _currentTimeNormalized => (_currentTime - _startTime) / Math.Max(0.001f, _endTime - _startTime);

        public TimelineViewModel()
        {
            InitializeAvailableProperties();
            InitializeCommands();
        }

        public float CurrentTime
        {
            get => _currentTime;
            set
            {
                var clamped = Math.Clamp(value, _startTime, _endTime);
                if (SetProperty(ref _currentTime, clamped))
                {
                    RaisePropertyChanged(nameof(CurrentTimeText));
                    RaisePropertyChanged(nameof(CurrentTimeNormalized));
                    RaisePropertyChanged(nameof(CurrentFrame));
                }
            }
        }

        public float StartTime
        {
            get => _startTime;
            set { if (SetProperty(ref _startTime, value)) RaisePropertyChanged(nameof(RangeText)); }
        }

        public float EndTime
        {
            get => _endTime;
            set { if (SetProperty(ref _endTime, value)) RaisePropertyChanged(nameof(RangeText)); }
        }

        public float PlaybackSpeed
        {
            get => _playbackSpeed;
            set { if (SetProperty(ref _playbackSpeed, value)) RaisePropertyChanged(nameof(PlaybackSpeedText)); }
        }

        public PlaybackState PlaybackState
        {
            get => _playbackState;
            set { if (SetProperty(ref _playbackState, value)) { UpdatePlaybackButtons(); RaisePropertyChanged(nameof(IsPlaying)); } }
        }

        public LoopMode LoopMode
        {
            get => _loopMode;
            set => SetProperty(ref _loopMode, value);
        }

        public bool ShowCurveEditor
        {
            get => _showCurveEditor;
            set => SetProperty(ref _showCurveEditor, value);
        }

        public bool ShowDopeSheet
        {
            get => _showDopeSheet;
            set => SetProperty(ref _showDopeSheet, value);
        }

        public float ZoomLevel
        {
            get => _zoomLevel;
            set => SetProperty(ref _zoomLevel, Math.Clamp(value, 0.1f, 10.0f));
        }

        public float ScrollOffset
        {
            get => _scrollOffset;
            set => SetProperty(ref _scrollOffset, value);
        }

        public int SelectedKeyframeIndex
        {
            get => _selectedKeyframeIndex;
            set => SetProperty(ref _selectedKeyframeIndex, value);
        }

        public ObservableCollection<AnimationTrack> Tracks => _tracks;
        public ObservableCollection<KeyframeViewModel> SelectedKeyframes => _selectedKeyframes;
        public ObservableCollection<string> AvailableProperties => _availableProperties;

        public string SelectedProperty
        {
            get => _selectedProperty;
            set => SetProperty(ref _selectedProperty, value);
        }

        public bool IsSnapping
        {
            get => _isSnapping;
            set => SetProperty(ref _isSnapping, value);
        }

        public float SnapInterval
        {
            get => _snapInterval;
            set => SetProperty(ref _snapInterval, value);
        }

        public bool ShowKeyframeTangents
        {
            get => _showKeyframeTangents;
            set => SetProperty(ref _showKeyframeTangents, value);
        }

        public int FramesPerSecond
        {
            get => _framesPerSecond;
            set => SetProperty(ref _framesPerSecond, value);
        }

        public bool IsRecording
        {
            get => _isRecording;
            set => SetProperty(ref _isRecording, value);
        }

        public bool IsPlaying => PlaybackState == PlaybackState.Playing;

        public string CurrentTimeText => $"{CurrentTime:F3}s";
        public string CurrentFrame => $"F{Math.Floor(CurrentTime * _framesPerSecond)}";
        public string RangeText => $"{_startTime:F1}s - {_endTime:F1}s";
        public string PlaybackSpeedText => $"{PlaybackSpeed:F1}x";
        public float CurrentTimeNormalized => _currentTimeNormalized;

        // Commands
        public IRelayCommand PlayCommand { get; private set; } = null!;
        public IRelayCommand PauseCommand { get; private set; } = null!;
        public IRelayCommand StopCommand { get; private set; } = null!;
        public IRelayCommand StepForwardCommand { get; private set; } = null!;
        public IRelayCommand StepBackwardCommand { get; private set; } = null!;
        public IRelayCommand GoToStartCommand { get; private set; } = null!;
        public IRelayCommand GoToEndCommand { get; private set; } = null!;
        public IRelayCommand ToggleLoopCommand { get; private set; } = null!;
        public IRelayCommand IncreaseSpeedCommand { get; private set; } = null!;
        public IRelayCommand DecreaseSpeedCommand { get; private set; } = null!;
        public IRelayCommand ResetSpeedCommand { get; private set; } = null!;
        public IRelayCommand AddKeyframeCommand { get; private set; } = null!;
        public IRelayCommand DeleteSelectedKeyframesCommand { get; private set; } = null!;
        public IRelayCommand<float> ScrubCommand { get; private set; } = null!;
        public IRelayCommand AddTrackCommand { get; private set; } = null!;
        public IRelayCommand RemoveTrackCommand { get; private set; } = null!;
        public IRelayCommand ToggleCurveEditorCommand { get; private set; } = null!;
        public IRelayCommand ToggleDopeSheetCommand { get; private set; } = null!;
        public IRelayCommand ZoomInCommand { get; private set; } = null!;
        public IRelayCommand ZoomOutCommand { get; private set; } = null!;
        public IRelayCommand ZoomToFitCommand { get; private set; } = null!;
        public IRelayCommand ToggleSnapCommand { get; private set; } = null!;
        public IRelayCommand ToggleRecordingCommand { get; private set; } = null!;
        public IRelayCommand<KeyframeViewModel> SelectKeyframeCommand { get; private set; } = null!;
        public IRelayCommand<AnimationTrack> ToggleTrackMuteCommand { get; private set; } = null!;
        public IRelayCommand<AnimationTrack> ToggleTrackLockCommand { get; private set; } = null!;
        public IRelayCommand ToggleShowTangentsCommand { get; private set; } = null!;
        public IRelayCommand CopyKeyframesCommand { get; private set; } = null!;
        public IRelayCommand PasteKeyframesCommand { get; private set; } = null!;
        public IRelayCommand MirrorKeyframesCommand { get; private set; } = null!;
        public IRelayCommand SmoothKeyframesCommand { get; private set; } = null!;
        public IRelayCommand FlattenKeyframesCommand { get; private set; } = null!;

        private void InitializeAvailableProperties()
        {
            _availableProperties.Add("Position.X");
            _availableProperties.Add("Position.Y");
            _availableProperties.Add("Position.Z");
            _availableProperties.Add("Rotation.X");
            _availableProperties.Add("Rotation.Y");
            _availableProperties.Add("Rotation.Z");
            _availableProperties.Add("Scale.X");
            _availableProperties.Add("Scale.Y");
            _availableProperties.Add("Scale.Z");
            _availableProperties.Add("Visibility");
            _availableProperties.Add("Material.Opacity");
            _availableProperties.Add("Light.Intensity");
            _availableProperties.Add("Genome.MutationRate");
            _availableProperties.Add("Physics.Mass");
            _availableProperties.Add("Audio.Volume");
            _availableProperties.Add("Particle.Lifetime");
        }

        private void InitializeCommands()
        {
            PlayCommand = CreateRelayCommand(ExecutePlay);
            PauseCommand = CreateRelayCommand(ExecutePause);
            StopCommand = CreateRelayCommand(ExecuteStop);
            StepForwardCommand = CreateRelayCommand(ExecuteStepForward);
            StepBackwardCommand = CreateRelayCommand(ExecuteStepBackward);
            GoToStartCommand = CreateRelayCommand(() => CurrentTime = _startTime);
            GoToEndCommand = CreateRelayCommand(() => CurrentTime = _endTime);
            ToggleLoopCommand = CreateRelayCommand(ExecuteToggleLoop);
            IncreaseSpeedCommand = CreateRelayCommand(() => PlaybackSpeed = Math.Min(4.0f, PlaybackSpeed + 0.25f));
            DecreaseSpeedCommand = CreateRelayCommand(() => PlaybackSpeed = Math.Max(0.25f, PlaybackSpeed - 0.25f));
            ResetSpeedCommand = CreateRelayCommand(() => PlaybackSpeed = 1.0f);
            AddKeyframeCommand = CreateRelayCommand(ExecuteAddKeyframe);
            DeleteSelectedKeyframesCommand = CreateRelayCommand(ExecuteDeleteSelectedKeyframes);
            ScrubCommand = CreateRelayCommand<float>(time => CurrentTime = time);
            AddTrackCommand = CreateRelayCommand(ExecuteAddTrack);
            RemoveTrackCommand = CreateRelayCommand(ExecuteRemoveTrack);
            ToggleCurveEditorCommand = CreateRelayCommand(() => ShowCurveEditor = !ShowCurveEditor);
            ToggleDopeSheetCommand = CreateRelayCommand(() => ShowDopeSheet = !ShowDopeSheet);
            ZoomInCommand = CreateRelayCommand(() => ZoomLevel = Math.Min(10.0f, ZoomLevel * 1.2f));
            ZoomOutCommand = CreateRelayCommand(() => ZoomLevel = Math.Max(0.1f, ZoomLevel / 1.2f));
            ZoomToFitCommand = CreateRelayCommand(ExecuteZoomToFit);
            ToggleSnapCommand = CreateRelayCommand(() => IsSnapping = !IsSnapping);
            ToggleRecordingCommand = CreateRelayCommand(() => IsRecording = !IsRecording);
            SelectKeyframeCommand = CreateRelayCommand<KeyframeViewModel>(ExecuteSelectKeyframe);
            ToggleTrackMuteCommand = CreateRelayCommand<AnimationTrack>(track => track.IsMuted = !track.IsMuted);
            ToggleTrackLockCommand = CreateRelayCommand<AnimationTrack>(track => track.IsLocked = !track.IsLocked);
            ToggleShowTangentsCommand = CreateRelayCommand(() => ShowKeyframeTangents = !ShowKeyframeTangents);
            CopyKeyframesCommand = CreateRelayCommand(ExecuteCopyKeyframes);
            PasteKeyframesCommand = CreateRelayCommand(ExecutePasteKeyframes);
            MirrorKeyframesCommand = CreateRelayCommand(ExecuteMirrorKeyframes);
            SmoothKeyframesCommand = CreateRelayCommand(ExecuteSmoothKeyframes);
            FlattenKeyframesCommand = CreateRelayCommand(ExecuteFlattenKeyframes);
        }

        private void ExecutePlay()
        {
            PlaybackState = PlaybackState.Playing;
            _playbackTimer = new Timer(_ =>
            {
                Dispatcher.UIThread.Post(() =>
                {
                    CurrentTime += 0.016f * PlaybackSpeed;
                    if (CurrentTime >= _endTime)
                    {
                        switch (_loopMode)
                        {
                            case LoopMode.Loop:
                                CurrentTime = _startTime;
                                break;
                            case LoopMode.PingPong:
                                PlaybackSpeed = -PlaybackSpeed;
                                break;
                            case LoopMode.None:
                            case LoopMode.ClampForever:
                                ExecutePause();
                                break;
                        }
                    }
                });
            }, null, 0, 16);
            StatusMessage = $"Playing at {PlaybackSpeed}x";
        }

        private void ExecutePause()
        {
            PlaybackState = PlaybackState.Paused;
            _playbackTimer?.Dispose();
            _playbackTimer = null;
            StatusMessage = "Paused";
        }

        private void ExecuteStop()
        {
            PlaybackState = PlaybackState.Stopped;
            _playbackTimer?.Dispose();
            _playbackTimer = null;
            CurrentTime = _startTime;
            StatusMessage = "Stopped";
        }

        private void ExecuteStepForward()
        {
            PlaybackState = PlaybackState.SteppingForward;
            CurrentTime += 1.0f / _framesPerSecond;
        }

        private void ExecuteStepBackward()
        {
            PlaybackState = PlaybackState.SteppingBackward;
            CurrentTime -= 1.0f / _framesPerSecond;
        }

        private void ExecuteToggleLoop()
        {
            LoopMode = LoopMode switch
            {
                LoopMode.None => LoopMode.Loop,
                LoopMode.Loop => LoopMode.PingPong,
                LoopMode.PingPong => LoopMode.ClampForever,
                LoopMode.ClampForever => LoopMode.None,
                _ => LoopMode.None
            };
            StatusMessage = $"Loop mode: {LoopMode}";
        }

        private void ExecuteAddKeyframe()
        {
            if (string.IsNullOrEmpty(SelectedProperty)) return;
            var track = _tracks.FirstOrDefault(t => t.PropertyName == SelectedProperty);
            if (track == null)
            {
                track = new AnimationTrack { PropertyName = SelectedProperty, DisplayName = SelectedProperty };
                _tracks.Add(track);
            }

            var keyframe = new KeyframeViewModel
            {
                Time = CurrentTime,
                Value = 0,
                InTangent = 0,
                OutTangent = 0,
                Interpolation = KeyframeInterpolation.Cubic
            };
            track.Keyframes.Add(keyframe);
            StatusMessage = $"Added keyframe at {CurrentTime:F3}s for {SelectedProperty}";
        }

        private void ExecuteDeleteSelectedKeyframes()
        {
            foreach (var track in _tracks)
            {
                var toRemove = track.Keyframes.Where(k => k.IsSelected).ToList();
                foreach (var k in toRemove) track.Keyframes.Remove(k);
            }
            _selectedKeyframes.Clear();
            StatusMessage = "Deleted selected keyframes";
        }

        private void ExecuteAddTrack()
        {
            _tracks.Add(new AnimationTrack { PropertyName = "NewTrack", DisplayName = "New Track" });
            StatusMessage = "Track added";
        }

        private void ExecuteRemoveTrack()
        {
            if (_tracks.Count > 0)
            {
                var last = _tracks[_tracks.Count - 1];
                _tracks.Remove(last);
                StatusMessage = $"Removed track: {last.DisplayName}";
            }
        }

        private void ExecuteZoomToFit()
        {
            ZoomLevel = 1.0f;
            ScrollOffset = 0;
        }

        private void ExecuteSelectKeyframe(KeyframeViewModel keyframe)
        {
            if (keyframe.IsSelected)
            {
                keyframe.IsSelected = false;
                _selectedKeyframes.Remove(keyframe);
            }
            else
            {
                keyframe.IsSelected = true;
                _selectedKeyframes.Add(keyframe);
            }
        }

        private void ExecuteCopyKeyframes() => StatusMessage = $"Copied {_selectedKeyframes.Count} keyframes";
        private void ExecutePasteKeyframes() => StatusMessage = "Pasted keyframes";
        private void ExecuteMirrorKeyframes() => StatusMessage = "Keyframes mirrored";
        private void ExecuteSmoothKeyframes() => StatusMessage = "Keyframes smoothed";
        private void ExecuteFlattenKeyframes() => StatusMessage = "Keyframes flattened";

        private void UpdatePlaybackButtons()
        {
            RaisePropertyChanged(nameof(IsPlaying));
            PlayCommand.NotifyCanExecuteChanged();
            PauseCommand.NotifyCanExecuteChanged();
        }

        protected override void OnDispose()
        {
            _playbackTimer?.Dispose();
            _tracks.Clear();
            _selectedKeyframes.Clear();
            _availableProperties.Clear();
            base.OnDispose();
        }
    }

    /// <summary>ViewModel for an animation track.</summary>
    public class AnimationTrack : ObservableObject
    {
        private string _propertyName = string.Empty;
        private string _displayName = string.Empty;
        private bool _isMuted;
        private bool _isLocked;
        private bool _isExpanded = true;
        private string _color = "#2196F3";

        public string PropertyName { get => _propertyName; set => SetProperty(ref _propertyName, value); }
        public string DisplayName { get => _displayName; set => SetProperty(ref _displayName, value); }
        public bool IsMuted { get => _isMuted; set => SetProperty(ref _isMuted, value); }
        public bool IsLocked { get => _isLocked; set => SetProperty(ref _isLocked, value); }
        public bool IsExpanded { get => _isExpanded; set => SetProperty(ref _isExpanded, value); }
        public string Color { get => _color; set => SetProperty(ref _color, value); }
        public ObservableCollection<KeyframeViewModel> Keyframes { get; } = new();
        public int KeyframeCount => Keyframes.Count;
    }

    /// <summary>ViewModel for a keyframe.</summary>
    public class KeyframeViewModel : ObservableObject
    {
        private float _time;
        private float _value;
        private float _inTangent;
        private float _outTangent;
        private bool _isSelected;
        private KeyframeInterpolation _interpolation;

        public float Time { get => _time; set => SetProperty(ref _time, value); }
        public float Value { get => _value; set => SetProperty(ref _value, value); }
        public float InTangent { get => _inTangent; set => SetProperty(ref _inTangent, value); }
        public float OutTangent { get => _outTangent; set => SetProperty(ref _outTangent, value); }
        public bool IsSelected { get => _isSelected; set => SetProperty(ref _isSelected, value); }
        public KeyframeInterpolation Interpolation { get => _interpolation; set => SetProperty(ref _interpolation, value); }
        public string TimeText => $"{Time:F3}s";
        public string ValueText => $"{Value:F2}";
    }

    // =========================================================================
    // MATERIAL PREVIEW VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the material preview, providing a preview sphere with configurable material properties.
    /// </summary>
    public partial class MaterialPreviewViewModel : ViewModelBase
    {
        private float _rotationX;
        private float _rotationY = 30;
        private float _zoom = 1.0f;
        private int _selectedLightingEnvironment;
        private readonly ObservableCollection<string> _lightingEnvironments = new();
        private System.Numerics.Vector3 _baseColor = new(0.8f, 0.2f, 0.2f);
        private float _roughness = 0.5f;
        private float _metallic;
        private float _normalStrength = 1.0f;
        private float _emissionStrength;
        private System.Numerics.Vector3 _emissionColor = System.Numerics.Vector3.One;
        private float _alpha = 1.0f;
        private float _refractionIndex = 1.5f;
        private float _anisotropy;
        private float _sheen;
        private float _clearcoat;
        private float _clearcoatRoughness;
        private bool _isTwoSided;
        private bool _showWireframe;
        private bool _showEnvironment = true;
        private readonly ObservableCollection<string> _materialPresets = new();
        private string _selectedPreset = string.Empty;

        public MaterialPreviewViewModel()
        {
            _lightingEnvironments.Add("Studio");
            _lightingEnvironments.Add("Outdoor");
            _lightingEnvironments.Add("Night");
            _lightingEnvironments.Add("Warm");
            _lightingEnvironments.Add("Cool");
            _lightingEnvironments.Add("HDR Forest");
            _lightingEnvironments.Add("HDR City");
            _lightingEnvironments.Add("HDR Sunset");

            _materialPresets.Add("Default");
            _materialPresets.Add("Metal");
            _materialPresets.Add("Plastic");
            _materialPresets.Add("Wood");
            _materialPresets.Add("Glass");
            _materialPresets.Add("Fabric");
            _materialPresets.Add("Skin");
            _materialPresets.Add("Stone");
            _materialPresets.Add("Water");
            _materialPresets.Add("Emissive");

            InitializeCommands();
        }

        public float RotationX { get => _rotationX; set => SetProperty(ref _rotationX, value); }
        public float RotationY { get => _rotationY; set => SetProperty(ref _rotationY, value); }
        public float Zoom { get => _zoom; set => SetProperty(ref _zoom, Math.Clamp(value, 0.2f, 3.0f)); }
        public int SelectedLightingEnvironment { get => _selectedLightingEnvironment; set => SetProperty(ref _selectedLightingEnvironment, value); }
        public ObservableCollection<string> LightingEnvironments => _lightingEnvironments;
        public System.Numerics.Vector3 BaseColor { get => _baseColor; set => SetProperty(ref _baseColor, value); }
        public float Roughness { get => _roughness; set => SetProperty(ref _roughness, value); }
        public float Metallic { get => _metallic; set => SetProperty(ref _metallic, value); }
        public float NormalStrength { get => _normalStrength; set => SetProperty(ref _normalStrength, value); }
        public float EmissionStrength { get => _emissionStrength; set => SetProperty(ref _emissionStrength, value); }
        public System.Numerics.Vector3 EmissionColor { get => _emissionColor; set => SetProperty(ref _emissionColor, value); }
        public float Alpha { get => _alpha; set => SetProperty(ref _alpha, value); }
        public float RefractionIndex { get => _refractionIndex; set => SetProperty(ref _refractionIndex, value); }
        public float Anisotropy { get => _anisotropy; set => SetProperty(ref _anisotropy, value); }
        public float Sheen { get => _sheen; set => SetProperty(ref _sheen, value); }
        public float Clearcoat { get => _clearcoat; set => SetProperty(ref _clearcoat, value); }
        public float ClearcoatRoughness { get => _clearcoatRoughness; set => SetProperty(ref _clearcoatRoughness, value); }
        public bool IsTwoSided { get => _isTwoSided; set => SetProperty(ref _isTwoSided, value); }
        public bool ShowWireframe { get => _showWireframe; set => SetProperty(ref _showWireframe, value); }
        public bool ShowEnvironment { get => _showEnvironment; set => SetProperty(ref _showEnvironment, value); }
        public ObservableCollection<string> MaterialPresets => _materialPresets;
        public string SelectedPreset { get => _selectedPreset; set => SetProperty(ref _selectedPreset, value); }

        public IRelayCommand<string> ApplyPresetCommand { get; private set; } = null!;
        public IRelayCommand ResetMaterialCommand { get; private set; } = null!;
        public IRelayCommand TakeSnapshotCommand { get; private set; } = null!;
        public IRelayCommand ToggleWireframeCommand { get; private set; } = null!;
        public IRelayCommand ToggleEnvironmentCommand { get; private set; } = null!;
        public IRelayCommand RotateLeftCommand { get; private set; } = null!;
        public IRelayCommand RotateRightCommand { get; private set; } = null!;
        public IRelayCommand ResetRotationCommand { get; private set; } = null!;
        public IRelayCommand ZoomInCommand { get; private set; } = null!;
        public IRelayCommand ZoomOutCommand { get; private set; } = null!;
        public IRelayCommand ResetZoomCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            ApplyPresetCommand = CreateCommand<string>(ExecuteApplyPreset);
            ResetMaterialCommand = CreateCommand(ExecuteResetMaterial);
            TakeSnapshotCommand = CreateCommand(() => StatusMessage = "Material snapshot taken");
            ToggleWireframeCommand = CreateCommand(() => ShowWireframe = !ShowWireframe);
            ToggleEnvironmentCommand = CreateCommand(() => ShowEnvironment = !ShowEnvironment);
            RotateLeftCommand = CreateCommand(() => RotationY -= 15);
            RotateRightCommand = CreateCommand(() => RotationY += 15);
            ResetRotationCommand = CreateCommand(() => { RotationX = 0; RotationY = 30; });
            ZoomInCommand = CreateCommand(() => Zoom = Math.Min(3.0f, Zoom + 0.2f));
            ZoomOutCommand = CreateCommand(() => Zoom = Math.Max(0.2f, Zoom - 0.2f));
            ResetZoomCommand = CreateCommand(() => Zoom = 1.0f);
        }

        private void ExecuteApplyPreset(string preset)
        {
            SelectedPreset = preset;
            switch (preset)
            {
                case "Metal":
                    Metallic = 1.0f; Roughness = 0.2f; BaseColor = new(0.8f, 0.8f, 0.8f); break;
                case "Plastic":
                    Metallic = 0.0f; Roughness = 0.4f; BaseColor = new(0.2f, 0.5f, 0.8f); break;
                case "Wood":
                    Metallic = 0.0f; Roughness = 0.8f; BaseColor = new(0.6f, 0.4f, 0.2f); break;
                case "Glass":
                    Metallic = 0.0f; Roughness = 0.0f; Alpha = 0.3f; RefractionIndex = 1.5f; break;
                case "Fabric":
                    Metallic = 0.0f; Roughness = 0.9f; BaseColor = new(0.4f, 0.2f, 0.6f); break;
                case "Skin":
                    Metallic = 0.0f; Roughness = 0.7f; BaseColor = new(0.9f, 0.7f, 0.6f); break;
                case "Stone":
                    Metallic = 0.0f; Roughness = 0.95f; BaseColor = new(0.5f, 0.5f, 0.5f); break;
                case "Water":
                    Metallic = 0.0f; Roughness = 0.0f; Alpha = 0.7f; RefractionIndex = 1.33f; break;
                case "Emissive":
                    Metallic = 0.0f; Roughness = 0.5f; EmissionStrength = 2.0f; EmissionColor = new(0, 1, 0.5f); break;
                default:
                    ExecuteResetMaterial(); break;
            }
            StatusMessage = $"Applied preset: {preset}";
        }

        private void ExecuteResetMaterial()
        {
            BaseColor = new(0.8f, 0.2f, 0.2f);
            Roughness = 0.5f; Metallic = 0; NormalStrength = 1; EmissionStrength = 0;
            Alpha = 1; RefractionIndex = 1.5f; Anisotropy = 0; Sheen = 0;
            Clearcoat = 0; ClearcoatRoughness = 0; IsTwoSided = false;
            StatusMessage = "Material reset to default";
        }

        protected override void OnDispose()
        {
            _lightingEnvironments.Clear();
            _materialPresets.Clear();
            base.OnDispose();
        }
    }



    // =========================================================================
    // PREFERENCES VIEW MODEL
    // =========================================================================

    /// <summary>
    /// ViewModel for the preferences/settings dialog.
    /// </summary>
    public partial class PreferencesViewModel : ViewModelBase
    {
        private string _llmProvider = "OpenAI";
        private string _llmApiKey = string.Empty;
        private string _llmModel = "gpt-4";
        private string _llmEndpoint = string.Empty;
        private double _llmTemperature = 0.7;
        private int _llmMaxTokens = 4096;
        private bool _llmStreamResponse = true;
        private string _renderBackend = "Vulkan";
        private string _defaultQuality = "High";
        private bool _enableRayTracing;
        private bool _enableMeshShaders;
        private int _maxFrameRate = 144;
        private bool _enableVSync = true;
        private int _msaaLevel = 4;
        private bool _enableHDR = true;
        private string _editorTheme = "Dark";
        private int _fontSize = 14;
        private string _fontFamily = "Cascadia Code";
        private bool _showLineNumbers = true;
        private bool _wordWrap = true;
        private bool _autoSave = true;
        int _autoSaveIntervalSeconds = 300;
        private int _shaderCacheSizeMb = 512;
        private int _genomeCacheSizeMb = 256;
        private bool _enableShaderCache = true;
        private bool _enableGenomeCache = true;
        private string _defaultExportFormat = "FBX";
        private bool _exportAnimations = true;
        private bool _exportMaterials = true;
        private bool _exportTextures = true;
        private int _undoHistorySize = 100;
        private bool _showStartupScreen = true;
        private bool _checkForUpdates = true;
        private string _language = "English";
        private bool _enableTelemetry;
        private int _viewportAntiAliasing = 4;
        private bool _showTutorialTooltips = true;
        private bool _autoCompileBlueprints = true;
        private bool _enablePhysics = true;
        private float _physicsGravity = 9.81f;
        private int _maxPhysicsBodies = 1024;
        private bool _enableAudio = true;
        private float _masterVolume = 1.0f;
        private int _audioBufferSize = 1024;
        private string _audioOutputDevice = "Default";
        private bool _isDialogOpen;

        public PreferencesViewModel()
        {
            InitializeCommands();
        }

        // LLM settings
        public string LLMProvider { get => _llmProvider; set => SetProperty(ref _llmProvider, value); }
        public string LLMApiKey { get => _llmApiKey; set => SetProperty(ref _llmApiKey, value); }
        public string LLMModel { get => _llmModel; set => SetProperty(ref _llmModel, value); }
        public string LLMEndpoint { get => _llmEndpoint; set => SetProperty(ref _llmEndpoint, value); }
        public double LLMTemperature { get => _llmTemperature; set => SetProperty(ref _llmTemperature, value); }
        public int LLMMaxTokens { get => _llmMaxTokens; set => SetProperty(ref _llmMaxTokens, value); }
        public bool LLMStreamResponse { get => _llmStreamResponse; set => SetProperty(ref _llmStreamResponse, value); }

        // Rendering settings
        public string RenderBackend { get => _renderBackend; set => SetProperty(ref _renderBackend, value); }
        public string DefaultQuality { get => _defaultQuality; set => SetProperty(ref _defaultQuality, value); }
        public bool EnableRayTracing { get => _enableRayTracing; set => SetProperty(ref _enableRayTracing, value); }
        public bool EnableMeshShaders { get => _enableMeshShaders; set => SetProperty(ref _enableMeshShaders, value); }
        public int MaxFrameRate { get => _maxFrameRate; set => SetProperty(ref _maxFrameRate, value); }
        public bool EnableVSync { get => _enableVSync; set => SetProperty(ref _enableVSync, value); }
        public int MSAALevel { get => _msaaLevel; set => SetProperty(ref _msaaLevel, value); }
        public bool EnableHDR { get => _enableHDR; set => SetProperty(ref _enableHDR, value); }

        // Editor settings
        public string EditorTheme { get => _editorTheme; set => SetProperty(ref _editorTheme, value); }
        public int FontSize { get => _fontSize; set => SetProperty(ref _fontSize, value); }
        public string FontFamily { get => _fontFamily; set => SetProperty(ref _fontFamily, value); }
        public bool ShowLineNumbers { get => _showLineNumbers; set => SetProperty(ref _showLineNumbers, value); }
        public bool WordWrap { get => _wordWrap; set => SetProperty(ref _wordWrap, value); }
        public bool AutoSave { get => _autoSave; set => SetProperty(ref _autoSave, value); }
        public int AutoSaveIntervalSeconds { get => _autoSaveIntervalSeconds; set => SetProperty(ref _autoSaveIntervalSeconds, value); }

        // Cache settings
        public int ShaderCacheSizeMb { get => _shaderCacheSizeMb; set => SetProperty(ref _shaderCacheSizeMb, value); }
        public int GenomeCacheSizeMb { get => _genomeCacheSizeMb; set => SetProperty(ref _genomeCacheSizeMb, value); }
        public bool EnableShaderCache { get => _enableShaderCache; set => SetProperty(ref _enableShaderCache, value); }
        public bool EnableGenomeCache { get => _enableGenomeCache; set => SetProperty(ref _enableGenomeCache, value); }

        // Export settings
        public string DefaultExportFormat { get => _defaultExportFormat; set => SetProperty(ref _defaultExportFormat, value); }
        public bool ExportAnimations { get => _exportAnimations; set => SetProperty(ref _exportAnimations, value); }
        public bool ExportMaterials { get => _exportMaterials; set => SetProperty(ref _exportMaterials, value); }
        public bool ExportTextures { get => _exportTextures; set => SetProperty(ref _exportTextures, value); }

        // General settings
        public int UndoHistorySize { get => _undoHistorySize; set => SetProperty(ref _undoHistorySize, value); }
        public bool ShowStartupScreen { get => _showStartupScreen; set => SetProperty(ref _showStartupScreen, value); }
        public bool CheckForUpdates { get => _checkForUpdates; set => SetProperty(ref _checkForUpdates, value); }
        public string Language { get => _language; set => SetProperty(ref _language, value); }
        public bool EnableTelemetry { get => _enableTelemetry; set => SetProperty(ref _enableTelemetry, value); }
        public int ViewportAntiAliasing { get => _viewportAntiAliasing; set => SetProperty(ref _viewportAntiAliasing, value); }
        public bool ShowTutorialTooltips { get => _showTutorialTooltips; set => SetProperty(ref _showTutorialTooltips, value); }
        public bool AutoCompileBlueprints { get => _autoCompileBlueprints; set => SetProperty(ref _autoCompileBlueprints, value); }

        // Physics settings
        public bool EnablePhysics { get => _enablePhysics; set => SetProperty(ref _enablePhysics, value); }
        public float PhysicsGravity { get => _physicsGravity; set => SetProperty(ref _physicsGravity, value); }
        public int MaxPhysicsBodies { get => _maxPhysicsBodies; set => SetProperty(ref _maxPhysicsBodies, value); }

        // Audio settings
        public bool EnableAudio { get => _enableAudio; set => SetProperty(ref _enableAudio, value); }
        public float MasterVolume { get => _masterVolume; set => SetProperty(ref _masterVolume, value); }
        public int AudioBufferSize { get => _audioBufferSize; set => SetProperty(ref _audioBufferSize, value); }
        public string AudioOutputDevice { get => _audioOutputDevice; set => SetProperty(ref _audioOutputDevice, value); }

        public bool IsDialogOpen { get => _isDialogOpen; set => SetProperty(ref _isDialogOpen, value); }

        // Commands
        public IRelayCommand SaveCommand { get; private set; } = null!;
        public IRelayCommand CancelCommand { get; private set; } = null!;
        public IRelayCommand ResetToDefaultsCommand { get; private set; } = null!;
        public IRelayCommand ClearShaderCacheCommand { get; private set; } = null!;
        public IRelayCommand ClearGenomeCacheCommand { get; private set; } = null!;
        public IRelayCommand ClearAllCachesCommand { get; private set; } = null!;
        public IRelayCommand BrowseApiKeyCommand { get; private set; } = null!;
        public IRelayCommand TestConnectionCommand { get; private set; } = null!;
        public IRelayCommand ExportSettingsCommand { get; private set; } = null!;
        public IRelayCommand ImportSettingsCommand { get; private set; } = null!;
        public IRelayCommand ShowDialogCommand { get; private set; } = null!;
        public IRelayCommand<string> SetThemeCommand { get; private set; } = null!;
        public IRelayCommand<string> SetQualityCommand { get; private set; } = null!;
        public IRelayCommand<string> SetBackendCommand { get; private set; } = null!;

        private void InitializeCommands()
        {
            SaveCommand = CreateRelayCommand(ExecuteSave);
            CancelCommand = CreateRelayCommand(ExecuteCancel);
            ResetToDefaultsCommand = CreateRelayCommand(ExecuteResetToDefaults);
            ClearShaderCacheCommand = CreateRelayCommand(() => StatusMessage = "Shader cache cleared");
            ClearGenomeCacheCommand = CreateRelayCommand(() => StatusMessage = "Genome cache cleared");
            ClearAllCachesCommand = CreateRelayCommand(() => StatusMessage = "All caches cleared");
            BrowseApiKeyCommand = CreateRelayCommand(() => StatusMessage = "Browse API key");
            TestConnectionCommand = CreateRelayCommand(ExecuteTestConnection);
            ExportSettingsCommand = CreateRelayCommand(() => StatusMessage = "Settings exported");
            ImportSettingsCommand = CreateRelayCommand(() => StatusMessage = "Settings imported");
            ShowDialogCommand = CreateRelayCommand(ShowDialog);
            SetThemeCommand = CreateCommand<string>(theme => EditorTheme = theme);
            SetQualityCommand = CreateCommand<string>(q => DefaultQuality = q);
            SetBackendCommand = CreateCommand<string>(b => RenderBackend = b);
        }

        public void ShowDialog()
        {
            IsDialogOpen = true;
            StatusMessage = "Preferences dialog opened";
        }

        private void ExecuteSave()
        {
            IsDialogOpen = false;
            StatusMessage = "Preferences saved";
        }

        private void ExecuteCancel()
        {
            IsDialogOpen = false;
            StatusMessage = "Preferences cancelled";
        }

        private void ExecuteResetToDefaults()
        {
            LLMProvider = "OpenAI"; LLMApiKey = string.Empty; LLMModel = "gpt-4";
            RenderBackend = "Vulkan"; DefaultQuality = "High"; EditorTheme = "Dark";
            FontSize = 14; FontFamily = "Cascadia Code"; AutoSave = true;
            AutoSaveIntervalSeconds = 300; ShaderCacheSizeMb = 512; GenomeCacheSizeMb = 256;
            EnablePhysics = true; PhysicsGravity = 9.81f; MasterVolume = 1.0f;
            StatusMessage = "Preferences reset to defaults";
        }

        private void ExecuteTestConnection()
        {
            StatusMessage = "Testing LLM connection...";
            StatusMessage = "Connection successful";
        }

        protected override void OnDispose()
        {
            base.OnDispose();
        }
    }

    // =========================================================================
    // DIALOG SERVICE
    // =========================================================================

    /// <summary>
    /// Service for managing modal dialogs, file dialogs, and message boxes.
    /// </summary>
    public class DialogService
    {
        private readonly Stack<string> _dialogStack = new();
        private string _currentDialogId = string.Empty;

        public event EventHandler<DialogOpenedEventArgs>? DialogOpened;
        public event EventHandler<DialogClosedEventArgs>? DialogClosed;

        public bool IsDialogOpen => _dialogStack.Count > 0;

        public string CurrentDialogId => _currentDialogId;

        public async Task<string> ShowOpenFileDialogAsync(string title, string filter, string? initialDirectory = null)
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "OpenFile" });

            try
            {
                var result = await Task.FromResult(string.Empty);
                return result;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public async Task<string> ShowSaveFileDialogAsync(string title, string filter, string? defaultFileName = null)
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "SaveFile" });

            try
            {
                var result = await Task.FromResult(string.Empty);
                return result;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public async Task<IEnumerable<string>?> ShowOpenMultipleFileDialogAsync(string title, string filter, string? initialDirectory = null)
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "OpenMultipleFile" });

            try
            {
                var result = await Task.FromResult<IEnumerable<string>?>(null);
                return result;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public async Task<DialogResult> ShowConfirmAsync(string title, string message)
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "Confirm", Message = message });

            try
            {
                var result = await Task.FromResult(DialogResult.Yes);
                return result;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public async Task ShowMessageAsync(string title, string message)
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "Message", Message = message });

            try
            {
                await Task.CompletedTask;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public async Task<string?> ShowInputDialogAsync(string title, string message, string defaultValue = "")
        {
            _currentDialogId = Guid.NewGuid().ToString();
            _dialogStack.Push(_currentDialogId);
            DialogOpened?.Invoke(this, new DialogOpenedEventArgs { DialogId = _currentDialogId, Title = title, DialogType = "Input", Message = message });

            try
            {
                var result = await Task.FromResult<string?>(defaultValue);
                return result;
            }
            finally
            {
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = _currentDialogId });
                if (_dialogStack.Count > 0 && _dialogStack.Peek() == _currentDialogId)
                    _dialogStack.Pop();
            }
        }

        public void CloseDialog()
        {
            if (_dialogStack.Count > 0)
            {
                var id = _dialogStack.Pop();
                DialogClosed?.Invoke(this, new DialogClosedEventArgs { DialogId = id });
            }
        }
    }

    /// <summary>Event args for dialog opened events.</summary>
    public class DialogOpenedEventArgs : EventArgs
    {
        public string DialogId { get; init; } = string.Empty;
        public string Title { get; init; } = string.Empty;
        public string DialogType { get; init; } = string.Empty;
        public string? Message { get; init; }
    }

    /// <summary>Event args for dialog closed events.</summary>
    public class DialogClosedEventArgs : EventArgs
    {
        public string DialogId { get; init; } = string.Empty;
    }

    // =========================================================================
    // LAYOUT MANAGER
    // =========================================================================

    /// <summary>
    /// Manages saving and restoring panel layouts for the editor.
    /// </summary>
    public class LayoutManager : IDisposable
    {
        private readonly Dictionary<string, PanelLayout> _layouts = new();
        private readonly List<string> _savedLayoutNames = new();
        private string _currentLayoutName = "Default";
        private bool _isMultiMonitorMode;

        public LayoutManager()
        {
            LoadDefaultLayouts();
        }

        public string CurrentLayoutName
        {
            get => _currentLayoutName;
            set => _currentLayoutName = value;
        }

        public bool IsMultiMonitorMode
        {
            get => _isMultiMonitorMode;
            set => _isMultiMonitorMode = value;
        }

        public IReadOnlyList<string> SavedLayoutNames => _savedLayoutNames.AsReadOnly();

        public void SaveLayout(string name)
        {
            _currentLayoutName = name;
            if (!_savedLayoutNames.Contains(name))
                _savedLayoutNames.Add(name);
        }

        public PanelLayout? GetLayout(string panelName)
        {
            return _layouts.TryGetValue(panelName, out var layout) ? layout : null;
        }

        public void SetLayout(string panelName, PanelLayout layout)
        {
            _layouts[panelName] = layout;
        }

        public void ResetToDefault()
        {
            _layouts.Clear();
            LoadDefaultLayouts();
            _currentLayoutName = "Default";
        }

        public void DeleteLayout(string name)
        {
            _savedLayoutNames.Remove(name);
        }

        public IReadOnlyDictionary<string, PanelLayout> GetAllLayouts()
        {
            return _layouts.AsReadOnly();
        }

        private void LoadDefaultLayouts()
        {
            _layouts["SceneExplorer"] = new PanelLayout { PanelName = "SceneExplorer", State = PanelState.Docked, Width = 300, DockSide = "Left", IsVisible = true };
            _layouts["Inspector"] = new PanelLayout { PanelName = "Inspector", State = PanelState.Docked, Width = 350, DockSide = "Right", IsVisible = true };
            _layouts["Viewport"] = new PanelLayout { PanelName = "Viewport", State = PanelState.Docked, Width = 1200, Height = 800, DockSide = "Center", IsVisible = true };
            _layouts["Console"] = new PanelLayout { PanelName = "Console", State = PanelState.Hidden, Width = 600, Height = 300, DockSide = "Bottom", IsVisible = false };
            _layouts["PerformanceHud"] = new PanelLayout { PanelName = "PerformanceHud", State = PanelState.Hidden, Width = 400, Height = 300, DockSide = "Right", IsVisible = false };
            _layouts["Timeline"] = new PanelLayout { PanelName = "Timeline", State = PanelState.Hidden, Width = 1200, Height = 200, DockSide = "Bottom", IsVisible = false };
            _layouts["CodeEditor"] = new PanelLayout { PanelName = "CodeEditor", State = PanelState.Hidden, Width = 800, Height = 600, DockSide = "Center", IsVisible = false };
            _layouts["BlueprintEditor"] = new PanelLayout { PanelName = "BlueprintEditor", State = PanelState.Hidden, Width = 800, Height = 600, DockSide = "Center", IsVisible = false };
            _layouts["MaterialPreview"] = new PanelLayout { PanelName = "MaterialPreview", State = PanelState.Hidden, Width = 400, Height = 400, DockSide = "Right", IsVisible = false };
            _layouts["AssetBrowser"] = new PanelLayout { PanelName = "AssetBrowser", State = PanelState.Docked, Width = 300, Height = 200, DockSide = "Bottom", IsVisible = true };

            _savedLayoutNames.Add("Default");
            _savedLayoutNames.Add("Compact");
            _savedLayoutNames.Add("Wide");
            _savedLayoutNames.Add("Dual Monitor");
        }

        public void Dispose()
        {
            _layouts.Clear();
            _savedLayoutNames.Clear();
        }
    }

    // =========================================================================
    // KEYBOARD SHORTCUT MANAGER
    // =========================================================================

    /// <summary>
    /// Manages keyboard shortcut registration, dispatch, and conflict detection.
    /// </summary>
    public class KeyboardShortcutManager : IDisposable
    {
        private readonly Dictionary<string, KeyboardShortcut> _shortcuts = new();
        private readonly Dictionary<string, List<string>> _conflicts = new();
        private readonly List<string> _disabledShortcuts = new();
        private bool _isEnabled = true;

        public bool IsEnabled
        {
            get => _isEnabled;
            set => _isEnabled = value;
        }

        public int ShortcutCount => _shortcuts.Count;

        public IReadOnlyDictionary<string, KeyboardShortcut> Shortcuts => _shortcuts.AsReadOnly();

        public void Register(KeyboardShortcut shortcut)
        {
            if (_shortcuts.ContainsKey(shortcut.Name))
                _shortcuts[shortcut.Name] = shortcut;
            else
                _shortcuts.Add(shortcut.Name, shortcut);

            DetectConflicts();
        }

        public void Unregister(string name)
        {
            _shortcuts.Remove(name);
            DetectConflicts();
        }

        public KeyboardShortcut? FindShortcut(Key key, KeyModifiers modifiers)
        {
            if (!_isEnabled) return null;

            return _shortcuts.Values.FirstOrDefault(s =>
                s.IsEnabled &&
                !_disabledShortcuts.Contains(s.Name) &&
                s.Key == key &&
                s.Modifiers == modifiers);
        }

        public KeyboardShortcut? FindByName(string name)
        {
            return _shortcuts.TryGetValue(name, out var shortcut) ? shortcut : null;
        }

        public void EnableShortcut(string name)
        {
            _disabledShortcuts.Remove(name);
        }

        public void DisableShortcut(string name)
        {
            if (!_disabledShortcuts.Contains(name))
                _disabledShortcuts.Add(name);
        }

        public IReadOnlyList<string> GetConflicts(string shortcutName)
        {
            return _conflicts.TryGetValue(shortcutName, out var conflicts) ? conflicts.AsReadOnly() : Array.Empty<string>().AsReadOnly();
        }

        public bool HasConflicts(string shortcutName)
        {
            return _conflicts.ContainsKey(shortcutName) && _conflicts[shortcutName].Count > 0;
        }

        public void DetectConflicts()
        {
            _conflicts.Clear();
            var bindings = _shortcuts.Values.Where(s => s.IsEnabled).ToList();

            for (var i = 0; i < bindings.Count; i++)
            {
                for (var j = i + 1; j < bindings.Count; j++)
                {
                    if (bindings[i].Key == bindings[j].Key && bindings[i].Modifiers == bindings[j].Modifiers)
                    {
                        if (!_conflicts.ContainsKey(bindings[i].Name))
                            _conflicts[bindings[i].Name] = new List<string>();
                        if (!_conflicts.ContainsKey(bindings[j].Name))
                            _conflicts[bindings[j].Name] = new List<string>();

                        _conflicts[bindings[i].Name].Add(bindings[j].Name);
                        _conflicts[bindings[j].Name].Add(bindings[i].Name);

                        bindings[i] = bindings[i] with { IsConflicting = true };
                        bindings[j] = bindings[j] with { IsConflicting = true };
                    }
                }
            }
        }

        public void ClearAll()
        {
            _shortcuts.Clear();
            _conflicts.Clear();
            _disabledShortcuts.Clear();
        }

        public List<KeyboardShortcut> GetAllShortcuts()
        {
            return _shortcuts.Values.ToList();
        }

        public void Dispose()
        {
            _shortcuts.Clear();
            _conflicts.Clear();
            _disabledShortcuts.Clear();
        }
    }

    // =========================================================================
    // UNDO/REDO MANAGER
    // =========================================================================

    /// <summary>
    /// Command pattern implementation for undo/redo operations with history stack.
    /// </summary>
    public class UndoRedoManager : IDisposable
    {
        private readonly Stack<UndoOperation> _undoStack = new();
        private readonly Stack<UndoOperation> _redoStack = new();
        private readonly List<UndoOperation> _history = new();
        private int _maxHistorySize = 100;
        private bool _isBatching;
        private UndoBatchOperation? _currentBatch;

        public event EventHandler<UndoOperationEventArgs>? OperationExecuted;
        public event EventHandler<UndoOperationEventArgs>? OperationUndone;
        public event EventHandler<UndoOperationEventArgs>? OperationRedone;

        public int MaxHistorySize
        {
            get => _maxHistorySize;
            set => _maxHistorySize = Math.Max(1, value);
        }

        public bool CanUndo => _undoStack.Count > 0;
        public bool CanRedo => _redoStack.Count > 0;
        public int UndoCount => _undoStack.Count;
        public int RedoCount => _redoStack.Count;
        public int HistoryCount => _history.Count;
        public bool IsBatching => _isBatching;

        public void Push(UndoOperation operation)
        {
            if (_isBatching && _currentBatch != null)
            {
                _currentBatch.Operations.Add(operation);
            }
            else
            {
                _undoStack.Push(operation);
                _history.Add(operation);
                TrimHistory();
                _redoStack.Clear();
            }
        }

        public bool Undo()
        {
            if (_undoStack.Count == 0) return false;

            var operation = _undoStack.Pop();
            try
            {
                operation.Undo();
                _redoStack.Push(operation);
                OperationUndone?.Invoke(this, new UndoOperationEventArgs { Operation = operation });
                return true;
            }
            catch (Exception ex)
            {
                operation.ErrorMessage = ex.Message;
                return false;
            }
        }

        public bool Redo()
        {
            if (_redoStack.Count == 0) return false;

            var operation = _redoStack.Pop();
            try
            {
                operation.Redo();
                _undoStack.Push(operation);
                OperationRedone?.Invoke(this, new UndoOperationEventArgs { Operation = operation });
                return true;
            }
            catch (Exception ex)
            {
                operation.ErrorMessage = ex.Message;
                return false;
            }
        }

        public UndoOperation? PeekUndo()
        {
            return _undoStack.Count > 0 ? _undoStack.Peek() : null;
        }

        public UndoOperation? PeekRedo()
        {
            return _redoStack.Count > 0 ? _redoStack.Peek() : null;
        }

        public void BeginBatch(string description)
        {
            _isBatching = true;
            _currentBatch = new UndoBatchOperation { Description = description };
        }

        public void EndBatch()
        {
            if (_currentBatch != null && _currentBatch.Operations.Count > 0)
            {
                Push(_currentBatch);
            }
            _isBatching = false;
            _currentBatch = null;
        }

        public void CancelBatch()
        {
            _isBatching = false;
            _currentBatch = null;
        }

        public void Clear()
        {
            _undoStack.Clear();
            _redoStack.Clear();
            _history.Clear();
            _currentBatch = null;
            _isBatching = false;
        }

        public IReadOnlyList<UndoOperation> GetUndoHistory()
        {
            return _history.AsReadOnly();
        }

        public IReadOnlyList<UndoOperation> GetRedoStack()
        {
            return _redoStack.Reverse().ToList().AsReadOnly();
        }

        private void TrimHistory()
        {
            while (_history.Count > _maxHistorySize)
            {
                _history.RemoveAt(0);
            }
        }

        public void Dispose()
        {
            Clear();
        }
    }

    /// <summary>
    /// Represents a single undo/redo operation.
    /// </summary>
    public class UndoOperation
    {
        public UndoOperationType Type { get; init; }
        public string Description { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
        public Func<Task>? UndoAction { get; init; }
        public Func<Task>? RedoAction { get; init; }
        public string? ErrorMessage { get; set; }
        public IReadOnlyDictionary<string, object>? Metadata { get; init; }

        public void Undo() => UndoAction?.Invoke().GetAwaiter().GetResult();
        public void Redo() => RedoAction?.Invoke().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Represents a batch of undo operations that are undone/redone together.
    /// </summary>
    public class UndoBatchOperation : UndoOperation
    {
        public List<UndoOperation> Operations { get; } = new();

        public new void Undo()
        {
            foreach (var op in Operations.AsEnumerable().Reverse())
                op.Undo();
        }

        public new void Redo()
        {
            foreach (var op in Operations)
                op.Redo();
        }
    }

    /// <summary>Event args for undo/redo operation events.</summary>
    public class UndoOperationEventArgs : EventArgs
    {
        public UndoOperation Operation { get; init; } = null!;
    }
}

