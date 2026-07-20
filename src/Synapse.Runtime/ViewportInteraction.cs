using System;
using System.Numerics;
using GDNN.Rendering.Engine;

namespace Synapse.Runtime
{
    /// <summary>Viewport picking, gizmo drag, and camera orbit for Studio.</summary>
    public static class ViewportInteraction
    {
        public static Guid? PickEntity(
            SceneDocument scene,
            SceneRenderer renderer,
            ViewportRay ray)
        {
            ArgumentNullException.ThrowIfNull(scene);
            ArgumentNullException.ThrowIfNull(renderer);
            Guid? bestId = null;
            float bestDist = float.MaxValue;

            foreach (var entity in scene.Entities)
            {
                if (!entity.Visible)
                    continue;
                if (entity.Type.Equals("Light", StringComparison.OrdinalIgnoreCase) ||
                    entity.Type.Equals("Camera", StringComparison.OrdinalIgnoreCase) ||
                    entity.Type.Equals("Empty", StringComparison.OrdinalIgnoreCase))
                    continue;

                if (!renderer.TryGetEntityBounds(entity.Id, out var center, out var halfExtents))
                    continue;

                if (ViewportPickUtility.IntersectAabb(ray, center, halfExtents, out float dist) && dist < bestDist)
                {
                    bestDist = dist;
                    bestId = entity.Id;
                }
            }

            return bestId;
        }

        public static GizmoAxis PickGizmoAxis(ViewportRay ray, Vector3 origin, Vector3 scale)
        {
            float axisLen = MathF.Max(scale.X, MathF.Max(scale.Y, scale.Z)) * 0.75f + 0.5f;
            const float radius = 0.12f;

            GizmoAxis best = GizmoAxis.None;
            float bestDist = float.MaxValue;

            TryAxis(GizmoAxis.X, Vector3.UnitX, origin + Vector3.UnitX * axisLen * 0.5f, axisLen * 0.5f);
            TryAxis(GizmoAxis.Y, Vector3.UnitY, origin + Vector3.UnitY * axisLen * 0.5f, axisLen * 0.5f);
            TryAxis(GizmoAxis.Z, Vector3.UnitZ, origin + Vector3.UnitZ * axisLen * 0.5f, axisLen * 0.5f);

            return best;

            void TryAxis(GizmoAxis axis, Vector3 dir, Vector3 handleCenter, float halfLen)
            {
                if (ViewportPickUtility.HitAxisHandle(ray, origin, dir, halfLen * 2f, radius, out float dist) && dist < bestDist)
                {
                    bestDist = dist;
                    best = axis;
                }
                else if (ViewportPickUtility.IntersectAabb(ray, handleCenter, new Vector3(0.08f, 0.08f, 0.08f), out dist) && dist < bestDist)
                {
                    bestDist = dist;
                    best = axis;
                }
            }
        }

        public static void ApplyTranslateDrag(
            ViewportEditorState editor,
            SceneEntityData entity,
            float mouseX,
            float mouseY,
            Matrix4x4 view,
            Matrix4x4 proj,
            int width,
            int height)
        {
            ArgumentNullException.ThrowIfNull(editor);
            ArgumentNullException.ThrowIfNull(entity);
            var axis = editor.ActiveGizmoAxis switch
            {
                GizmoAxis.X => Vector3.UnitX,
                GizmoAxis.Y => Vector3.UnitY,
                GizmoAxis.Z => Vector3.UnitZ,
                _ => Vector3.Zero
            };
            if (axis == Vector3.Zero)
                return;

            var start = editor.DragStartPosition;
            var p0 = ViewportPickUtility.ProjectWorldToScreen(start, view, proj, width, height);
            var p1 = ViewportPickUtility.ProjectWorldToScreen(start + axis, view, proj, width, height);
            var screenAxis = new Vector2(p1.X - p0.X, p1.Y - p0.Y);
            if (screenAxis.LengthSquared() < 1e-4f)
                return;
            screenAxis = Vector2.Normalize(screenAxis);

            float deltaX = mouseX - editor.DragStartMouseX;
            float deltaY = mouseY - editor.DragStartMouseY;
            float along = Vector2.Dot(new Vector2(deltaX, deltaY), screenAxis) * 0.02f;

            var pos = editor.DragStartPosition + axis * along;
            entity.Position = Vec3.From(pos);
        }

        public static void ApplyRotateDrag(
            ViewportEditorState editor,
            SceneEntityData entity,
            float mouseX,
            float mouseY)
        {
            ArgumentNullException.ThrowIfNull(editor);
            ArgumentNullException.ThrowIfNull(entity);
            float deltaX = mouseX - editor.DragStartMouseX;
            float deltaY = mouseY - editor.DragStartMouseY;
            var rot = editor.DragStartRotation;

            switch (editor.ActiveGizmoAxis)
            {
                case GizmoAxis.X:
                    rot.X = editor.DragStartRotation.X + deltaY * 0.5f;
                    break;
                case GizmoAxis.Y:
                    rot.Y = editor.DragStartRotation.Y + deltaX * 0.5f;
                    break;
                case GizmoAxis.Z:
                    rot.Z = editor.DragStartRotation.Z + deltaX * 0.5f;
                    break;
                default:
                    rot.Y = editor.DragStartRotation.Y + deltaX * 0.5f;
                    rot.X = editor.DragStartRotation.X + deltaY * 0.5f;
                    break;
            }

            entity.Rotation = Vec3.From(rot);
        }
    }
}
