// =============================================================================
// SceneGraph.cs - G-DNN Engine: Scene Graph & Entity Component System
// GDNN.Engine - GDNN.Scene
// Complete runtime scene graph with transform hierarchy, components, and systems
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using GDNN.Rendering.Compat;
using Synapse.Core;

namespace GDNN.Scene
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Types of scene nodes.</summary>
    public enum SceneNodeType
    {
        Empty,
        Mesh,
        Light,
        Camera,
        ParticleSystem,
        AudioSource,
        Terrain,
        Skybox,
        ReflectionProbe,
        Decal,
        Spline,
        Joint,
        Trigger,
        Volume,
        Character,
        Vehicle,
        Custom
    }

    /// <summary>Component types available in the ECS.</summary>
    public enum SceneComponentType
    {
        Transform,
        MeshRenderer,
        MeshFilter,
        Material,
        Light,
        Camera,
        Collider,
        Rigidbody,
        Animation,
        ParticleSystem,
        AudioSource,
        LODGroup,
        Script,
        Custom
    }

    /// <summary>Light types for the scene lighting system.</summary>
    public enum SceneLightType
    {
        Directional,
        Point,
        Spot,
        AreaRect,
        AreaDisc
    }

    /// <summary>Camera projection modes.</summary>
    public enum CameraProjection
    {
        Perspective,
        Orthographic
    }

    /// <summary>State of a scene node.</summary>
    [Flags]
    public enum NodeState : byte
    {
        Active = 1 << 0,
        Visible = 1 << 1,
        Static = 1 << 2,
        Dirty = 1 << 3,
        Culled = 1 << 4
    }

    // =========================================================================
    // TRANSFORM COMPONENT
    // =========================================================================

    /// <summary>
    /// Transform component providing position, rotation, and scale.
    /// Supports hierarchical parent-child relationships with dirty flag propagation.
    /// </summary>
    public class TransformComponent
    {
        private Vector3 _position = Vector3.Zero;
        private Quaternion _rotation = Quaternion.Identity;
        private Vector3 _scale = Vector3.One;
        private Matrix4x4 _localToWorld = Matrix4x4.Identity;
        private Matrix4x4 _worldToLocal = Matrix4x4.Identity;
        private bool _isDirty = true;

        public int EntityId { get; set; }
        public TransformComponent? Parent { get; private set; }
        public List<TransformComponent> Children { get; } = new();

        public Vector3 Position
        {
            get => _position;
            set { _position = value; MarkDirty(); }
        }

        public Quaternion Rotation
        {
            get => _rotation;
            set { _rotation = value; MarkDirty(); }
        }

        public Vector3 Scale
        {
            get => _scale;
            set { _scale = value; MarkDirty(); }
        }

        public Vector3 LocalPosition
        {
            get => Parent != null ? Vector3.Transform(_position, Matrix4x4.CreateFromQuaternion(Parent.Rotation)) + Parent.Position : _position;
            set
            {
                if (Parent != null)
                {
                    var invRot = Quaternion.Conjugate(Parent.Rotation);
                    _position = Vector3.Transform(value - Parent.Position, Matrix4x4.CreateFromQuaternion(invRot));
                }
                else
                    _position = value;
                MarkDirty();
            }
        }

        public Vector3 Forward => Vector3.Transform(Vector3.UnitZ, _rotation);
        public Vector3 Right => Vector3.Transform(Vector3.UnitX, _rotation);
        public Vector3 Up => Vector3.Transform(Vector3.UnitY, _rotation);

        public Matrix4x4 LocalToWorld
        {
            get
            {
                if (_isDirty)
                    UpdateMatrices();
                return _localToWorld;
            }
        }

        public Matrix4x4 WorldToLocal
        {
            get
            {
                if (_isDirty)
                    UpdateMatrices();
                return _worldToLocal;
            }
        }

        public void SetParent(TransformComponent? parent)
        {
            Parent?.Children.Remove(this);
            Parent = parent;
            Parent?.Children.Add(this);
            MarkDirty();
        }

        public void LookAt(Vector3 target, Vector3 up = default)
        {
            if (up == default)
                up = Vector3.UnitY;
            var direction = Vector3.Normalize(target - _position);
            if (direction.LengthSquared() > 0.0001f)
                _rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateLookAt(Vector3.Zero, direction, up));
            MarkDirty();
        }

        public void Translate(Vector3 translation, Space space = Space.Self)
        {
            if (space == Space.Self)
                _position += Vector3.Transform(translation, Matrix4x4.CreateFromQuaternion(_rotation));
            else
                _position += translation;
            MarkDirty();
        }

        public void Rotate(Quaternion rotation, Space space = Space.Self)
        {
            _rotation = space == Space.Self ? _rotation * rotation : rotation * _rotation;
            MarkDirty();
        }

        public void RotateAround(Vector3 point, Quaternion rotation)
        {
            var offset = _position - point;
            _position = point + Vector3.Transform(offset, Matrix4x4.CreateFromQuaternion(rotation));
            MarkDirty();
        }

        private void MarkDirty()
        {
            _isDirty = true;
            foreach (var child in Children)
                child.MarkDirty();
        }

        private void UpdateMatrices()
        {
            _localToWorld = Matrix4x4.CreateScale(_scale) *
                           Matrix4x4.CreateFromQuaternion(_rotation) *
                           Matrix4x4.CreateTranslation(_position);

            if (Parent != null)
                _localToWorld *= Parent.LocalToWorld;

            Matrix4x4.Invert(_localToWorld, out _worldToLocal);
            _isDirty = false;
        }
    }

    public enum Space { World, Self }

    // =========================================================================
    // MESH RENDERER COMPONENT
    // =========================================================================

    /// <summary>
    /// Mesh renderer component linking a mesh asset and material to a scene entity.
    /// Manages LOD selection, visibility, and rendering state.
    /// </summary>
    public class MeshRendererComponent
    {
        public int EntityId { get; set; }
        public int MeshAssetId { get; set; }
        public int MaterialId { get; set; }
        public bool IsVisible { get; set; } = true;
        public bool CastShadows { get; set; } = true;
        public bool ReceiveShadows { get; set; } = true;
        public int LODLevel { get; set; }
        public float LodBias { get; set; } = 1.0f;
        public Vector3 LocalBoundsCenter { get; set; }
        public Vector3 LocalBoundsExtents { get; set; }
        public int Priority { get; set; }
        public string SortingLayer { get; set; } = "Default";
    }

    // =========================================================================
    // LIGHT COMPONENT
    // =========================================================================

    /// <summary>
    /// Light component defining a light source in the scene.
    /// </summary>
    public class LightComponent
    {
        public int EntityId { get; set; }
        public SceneLightType LightType { get; set; } = SceneLightType.Point;
        public Vector3 Color { get; set; } = Vector3.One;
        public float Intensity { get; set; } = 1.0f;
        public float Range { get; set; } = 10.0f;
        public float InnerConeAngle { get; set; } = 30.0f;
        public float OuterConeAngle { get; set; } = 45.0f;
        public bool CastShadows { get; set; } = true;
        public int ShadowResolution { get; set; } = 1024;
        public float ShadowBias { get; set; } = 0.005f;
        public bool IsActive { get; set; } = true;
    }

    // =========================================================================
    // CAMERA COMPONENT
    // =========================================================================

    /// <summary>
    /// Camera component defining view and projection parameters.
    /// </summary>
    public class CameraComponent
    {
        public int EntityId { get; set; }
        public CameraProjection Projection { get; set; } = CameraProjection.Perspective;
        public float FieldOfView { get; set; } = 60.0f;
        public float NearPlane { get; set; } = 0.1f;
        public float FarPlane { get; set; } = 1000.0f;
        public float OrthographicSize { get; set; } = 5.0f;
        public float AspectRatio { get; set; } = 16.0f / 9.0f;
        public Vector3 BackgroundColor { get; set; } = new(0.1f, 0.1f, 0.2f);
        public int Depth { get; set; }
        public bool IsMain { get; set; }
        public int RenderTextureId { get; set; } = -1;
    }

    // =========================================================================
    // SCENE NODE
    // =========================================================================

    /// <summary>
    /// Represents a node in the scene graph hierarchy.
    /// Each node has a transform and can optionally hold components.
    /// </summary>
    [DebuggerDisplay("SceneNode({Name}, Id={Id}, Children={Children.Count})")]
    public class SceneNode
    {
        private static int _nextId;

        public int Id { get; } = Interlocked.Increment(ref _nextId);
        public string Name { get; set; } = "";
        public string Tag { get; set; } = "";
        public SceneNodeType NodeType { get; set; } = SceneNodeType.Empty;
        public NodeState State { get; set; } = NodeState.Active | NodeState.Visible;
        public TransformComponent Transform { get; } = new();
        public SceneNode? Parent { get; private set; }
        public List<SceneNode> Children { get; } = new();
        public Dictionary<SceneComponentType, object> Components { get; } = new();
        public int LayerMask { get; set; } = -1;
        public object? UserData { get; set; }

        public bool IsActive => State.HasFlag(NodeState.Active);
        public bool IsVisible => State.HasFlag(NodeState.Visible) && !State.HasFlag(NodeState.Culled);
        public bool IsStatic => State.HasFlag(NodeState.Static);
        public bool IsDirty => State.HasFlag(NodeState.Dirty);

        public SceneNode(string name = "")
        {
            Name = name;
            Transform.EntityId = Id;
        }

        public T AddComponent<T>() where T : new()
        {
            var comp = new T();
            var compType = GetComponentType<T>();
            Components[compType] = comp;

            if (comp is TransformComponent tc)
                tc.EntityId = Id;
            else if (comp is MeshRendererComponent mr)
                mr.EntityId = Id;
            else if (comp is LightComponent lc)
                lc.EntityId = Id;
            else if (comp is CameraComponent cc)
                cc.EntityId = Id;

            return comp;
        }

        public T? GetComponent<T>() where T : class
        {
            var compType = GetComponentType<T>();
            return Components.TryGetValue(compType, out var comp) ? comp as T : null;
        }

        public bool HasComponent<T>() where T : class
        {
            return Components.ContainsKey(GetComponentType<T>());
        }

        public void RemoveComponent<T>() where T : class
        {
            Components.Remove(GetComponentType<T>());
        }

        public void AddChild(SceneNode child)
        {
            if (child.Parent != null)
                child.Parent.RemoveChild(child);

            child.Parent = this;
            child.Transform.SetParent(Transform);
            Children.Add(child);
        }

        public void RemoveChild(SceneNode child)
        {
            if (child.Parent == this)
            {
                child.Parent = null;
                child.Transform.SetParent(null);
                Children.Remove(child);
            }
        }

        public SceneNode? FindChild(string name)
        {
            return Children.FirstOrDefault(c => c.Name == name);
        }

        public SceneNode? FindChildRecursive(string name)
        {
            var found = FindChild(name);
            if (found != null)
                return found;
            foreach (var child in Children)
            {
                found = child.FindChildRecursive(name);
                if (found != null)
                    return found;
            }
            return null;
        }

        public IEnumerable<SceneNode> GetDescendants()
        {
            foreach (var child in Children)
            {
                yield return child;
                foreach (var desc in child.GetDescendants())
                    yield return desc;
            }
        }

        public void SetDirty()
        {
            State |= NodeState.Dirty;
        }

        private static SceneComponentType GetComponentType<T>()
        {
            if (typeof(T) == typeof(TransformComponent))
                return SceneComponentType.Transform;
            if (typeof(T) == typeof(MeshRendererComponent))
                return SceneComponentType.MeshRenderer;
            if (typeof(T) == typeof(LightComponent))
                return SceneComponentType.Light;
            if (typeof(T) == typeof(CameraComponent))
                return SceneComponentType.Camera;
            return SceneComponentType.Custom;
        }
    }

    // =========================================================================
    // SCENE GRAPH
    // =========================================================================

    /// <summary>
    /// Complete scene graph managing all nodes, components, and spatial queries.
    /// Provides hierarchical transforms, dirty propagation, and entity lookup.
    /// </summary>
    public class SceneGraph
    {
        private readonly Dictionary<int, SceneNode> _nodesById = new();
        private readonly Dictionary<string, List<SceneNode>> _nodesByName = new();
        private SceneNode _root;
        private readonly object _lock = new();

        public SceneNode Root => _root;
        public int NodeCount => _nodesById.Count;
        public string Name { get; set; } = "UntitledScene";

        public SceneGraph()
        {
            _root = new SceneNode("__root__") { NodeType = SceneNodeType.Empty };
            _nodesById[_root.Id] = _root;
        }

        public SceneNode CreateNode(string name = "", SceneNodeType type = SceneNodeType.Empty)
        {
            var node = new SceneNode(name) { NodeType = type };
            lock (_lock)
            {
                _nodesById[node.Id] = node;
                if (!_nodesByName.ContainsKey(name))
                    _nodesByName[name] = new List<SceneNode>();
                _nodesByName[name].Add(node);
            }
            return node;
        }

        public SceneNode CreateChildNode(SceneNode parent, string name = "", SceneNodeType type = SceneNodeType.Empty)
        {
            var node = CreateNode(name, type);
            parent.AddChild(node);
            return node;
        }

        public void RemoveNode(SceneNode node)
        {
            if (node == _root)
                return;
            node.Parent?.RemoveChild(node);
            lock (_lock)
            {
                _nodesById.Remove(node.Id);
                if (_nodesByName.TryGetValue(node.Name, out var list))
                {
                    list.Remove(node);
                    if (list.Count == 0)
                        _nodesByName.Remove(node.Name);
                }
            }
        }

        public SceneNode? FindById(int id)
        {
            lock (_lock)
            { return _nodesById.TryGetValue(id, out var node) ? node : null; }
        }

        public List<SceneNode> FindByName(string name)
        {
            lock (_lock)
            { return _nodesByName.TryGetValue(name, out var list) ? new List<SceneNode>(list) : new(); }
        }

        public SceneNode? FindByTag(string tag)
        {
            lock (_lock)
            { return _nodesById.Values.FirstOrDefault(n => n.Tag == tag); }
        }

        public List<SceneNode> GetAllNodes()
        {
            lock (_lock)
            { return _nodesById.Values.ToList(); }
        }

        public List<SceneNode> GetNodesOfType(SceneNodeType type)
        {
            lock (_lock)
            { return _nodesById.Values.Where(n => n.NodeType == type).ToList(); }
        }

        public List<SceneNode> GetNodesWithComponent<T>() where T : class
        {
            lock (_lock)
            { return _nodesById.Values.Where(n => n.HasComponent<T>()).ToList(); }
        }

        public List<SceneNode> GetVisibleNodes()
        {
            lock (_lock)
            { return _nodesById.Values.Where(n => n.IsVisible).ToList(); }
        }

        public List<SceneNode> GetActiveNodes()
        {
            lock (_lock)
            { return _nodesById.Values.Where(n => n.IsActive).ToList(); }
        }

        public void CullByFrustum(Matrix4x4 viewProjection)
        {
            var frustum = new FrustumPlanes(viewProjection);
            lock (_lock)
            {
                foreach (var node in _nodesById.Values)
                {
                    if (!node.IsActive)
                        continue;
                    var bounds = GetWorldBounds(node);
                    node.State &= ~NodeState.Culled;
                    if (!frustum.Intersects(bounds))
                        node.State |= NodeState.Culled;
                }
            }
        }

        public void UpdateDirtyTransforms()
        {
            _root.Transform.LocalToWorld.ToString();
        }

        public void Clear()
        {
            lock (_lock)
            {
                _nodesById.Clear();
                _nodesByName.Clear();
                _root = new SceneNode("__root__");
                _nodesById[_root.Id] = _root;
            }
        }

        private BoundingBox3D GetWorldBounds(SceneNode node)
        {
            var meshRenderer = node.GetComponent<MeshRendererComponent>();
            if (meshRenderer != null)
            {
                var center = Vector3.Transform(meshRenderer.LocalBoundsCenter, node.Transform.LocalToWorld);
                var extents = Vector3.Abs(Vector3.TransformNormal(meshRenderer.LocalBoundsExtents, node.Transform.LocalToWorld));
                return new BoundingBox3D(RenderingMath.ToVector3D(center - extents), RenderingMath.ToVector3D(center + extents));
            }

            var pos = node.Transform.Position;
            return new BoundingBox3D(
                RenderingMath.ToVector3D(pos - Vector3.One * 0.5f),
                RenderingMath.ToVector3D(pos + Vector3.One * 0.5f));
        }
    }

    // =========================================================================
    // FRUSTUM CULLING
    // =========================================================================

    /// <summary>Extracts frustum planes from a view-projection matrix for culling.</summary>
    public readonly struct FrustumPlanes
    {
        public readonly Vector4 Left, Right, Top, Bottom, Near, Far;

        public FrustumPlanes(Matrix4x4 vp)
        {
            Left = Normalize(new Vector4(
                vp.M14 + vp.M11,
                vp.M24 + vp.M21,
                vp.M34 + vp.M31,
                vp.M44 + vp.M41));

            Right = Normalize(new Vector4(
                vp.M14 - vp.M11,
                vp.M24 - vp.M21,
                vp.M34 - vp.M31,
                vp.M44 - vp.M41));

            Top = Normalize(new Vector4(
                vp.M14 - vp.M12,
                vp.M24 - vp.M22,
                vp.M34 - vp.M32,
                vp.M44 - vp.M42));

            Bottom = Normalize(new Vector4(
                vp.M14 + vp.M12,
                vp.M24 + vp.M22,
                vp.M34 + vp.M32,
                vp.M44 + vp.M42));

            Near = Normalize(new Vector4(
                vp.M14 + vp.M13,
                vp.M24 + vp.M23,
                vp.M34 + vp.M33,
                vp.M44 + vp.M43));

            Far = Normalize(new Vector4(
                vp.M14 - vp.M13,
                vp.M24 - vp.M23,
                vp.M34 - vp.M33,
                vp.M44 - vp.M43));
        }

        public bool Intersects(BoundingBox3D bounds)
        {
            var planes = new[] { Left, Right, Top, Bottom, Near, Far };
            var center = bounds.Center;
            var extents = bounds.HalfSize;

            foreach (var plane in planes)
            {
                var normal = new Vector3((float)plane.X, (float)plane.Y, (float)plane.Z);
                float r = (float)(Math.Abs(normal.X * extents.X) + Math.Abs(normal.Y * extents.Y) + Math.Abs(normal.Z * extents.Z));
                float d = (float)(normal.X * center.X + normal.Y * center.Y + normal.Z * center.Z) + plane.W;
                if (d + r < 0)
                    return false;
            }

            return true;
        }

        private static Vector4 Normalize(Vector4 plane)
        {
            float len = MathF.Sqrt(plane.X * plane.X + plane.Y * plane.Y + plane.Z * plane.Z);
            return len > 0 ? plane / len : plane;
        }
    }

    // =========================================================================
    // SCENE MANAGER
    // =========================================================================

    /// <summary>
    /// High-level scene manager coordinating loading, saving, and lifecycle of scenes.
    /// </summary>
    public class SceneManager
    {
        private readonly Dictionary<string, SceneGraph> _scenes = new();
        private SceneGraph? _activeScene;

        public SceneGraph? ActiveScene => _activeScene;
        public int SceneCount => _scenes.Count;
        public IReadOnlyDictionary<string, SceneGraph> Scenes => _scenes;

        public SceneGraph CreateScene(string name)
        {
            var scene = new SceneGraph { Name = name };
            _scenes[name] = scene;
            _activeScene = scene;
            return scene;
        }

        public void LoadScene(string name)
        {
            if (_scenes.TryGetValue(name, out var scene))
                _activeScene = scene;
        }

        public void UnloadScene(string name)
        {
            if (_scenes.TryGetValue(name, out var scene))
            {
                scene.Clear();
                _scenes.Remove(name);
                if (_activeScene == scene)
                    _activeScene = _scenes.Values.FirstOrDefault();
            }
        }

        public void SetActiveScene(SceneGraph scene)
        {
            _activeScene = scene;
        }

        public void Update()
        {
            _activeScene?.UpdateDirtyTransforms();
        }
    }
}
