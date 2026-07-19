using System;
using System.Numerics;

namespace Synapse.Runtime
{
    public enum ViewportToolMode { Select, Translate, Rotate, Scale, FlyCamera, Place }

    [Flags]
    public enum ViewportOverlayMask
    {
        None = 0,
        Grid = 1 << 0,
        Gizmos = 1 << 1,
        Stats = 1 << 2,
        EntityLabels = 1 << 9,
        All = ~0
    }

    /// <summary>Runtime editor state shared between Studio viewport and the renderer.</summary>
    public sealed class ViewportEditorState
    {
        public Guid SelectedEntityId { get; set; }
        public ViewportToolMode ToolMode { get; set; } = ViewportToolMode.Select;
        public ViewportOverlayMask OverlayMask { get; set; } = ViewportOverlayMask.Grid | ViewportOverlayMask.Gizmos;
        public bool ShowGrid
        {
            get => OverlayMask.HasFlag(ViewportOverlayMask.Grid);
            set => OverlayMask = value ? OverlayMask | ViewportOverlayMask.Grid : OverlayMask & ~ViewportOverlayMask.Grid;
        }
        public bool ShowGizmos
        {
            get => OverlayMask.HasFlag(ViewportOverlayMask.Gizmos);
            set => OverlayMask = value ? OverlayMask | ViewportOverlayMask.Gizmos : OverlayMask & ~ViewportOverlayMask.Gizmos;
        }

        public GizmoAxis ActiveGizmoAxis { get; set; } = GizmoAxis.None;
        public bool IsDragging { get; set; }
        public bool IsOrbitingCamera { get; set; }
        public Vector3 DragStartPosition { get; set; }
        public Vector3 DragStartRotation { get; set; }
        public float DragStartMouseX { get; set; }
        public float DragStartMouseY { get; set; }
        public float OrbitStartYaw { get; set; }
        public float OrbitStartPitch { get; set; }
    }
}
