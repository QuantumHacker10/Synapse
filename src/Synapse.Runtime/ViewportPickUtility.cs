using System;
using System.Numerics;

namespace Synapse.Runtime
{
    public enum GizmoAxis { None, X, Y, Z }

    public readonly struct ViewportRay
    {
        public Vector3 Origin { get; init; }
        public Vector3 Direction { get; init; }

        public ViewportRay(Vector3 origin, Vector3 direction)
        {
            Origin = origin;
            Direction = Vector3.Normalize(direction);
        }

        public Vector3 PointAt(float t) => Origin + Direction * t;
    }

    /// <summary>Screen-space picking helpers for the embedded viewport.</summary>
    public static class ViewportPickUtility
    {
        public static ViewportRay CreateRayFromScreen(
            Vector3 cameraPos,
            Vector3 cameraFront,
            Vector3 cameraUp,
            float fovDeg,
            float aspect,
            float screenX,
            float screenY,
            int viewportWidth,
            int viewportHeight)
        {
            float ndcX = (2f * screenX / Math.Max(1, viewportWidth)) - 1f;
            float ndcY = 1f - (2f * screenY / Math.Max(1, viewportHeight));

            var right = Vector3.Normalize(Vector3.Cross(cameraFront, cameraUp));
            var up = Vector3.Normalize(Vector3.Cross(right, cameraFront));
            float tan = MathF.Tan(MathF.PI / 180f * fovDeg * 0.5f);
            var dir = Vector3.Normalize(cameraFront + right * (ndcX * tan * aspect) + up * (ndcY * tan));
            return new ViewportRay(cameraPos, dir);
        }

        public static bool IntersectAabb(ViewportRay ray, Vector3 center, Vector3 halfExtents, out float distance)
        {
            distance = float.MaxValue;
            var min = center - halfExtents;
            var max = center + halfExtents;
            float tMin = 0.001f;
            float tMax = 1000f;

            for (int axis = 0; axis < 3; axis++)
            {
                float origin = axis switch { 0 => ray.Origin.X, 1 => ray.Origin.Y, _ => ray.Origin.Z };
                float dir = axis switch { 0 => ray.Direction.X, 1 => ray.Direction.Y, _ => ray.Direction.Z };
                float bMin = axis switch { 0 => min.X, 1 => min.Y, _ => min.Z };
                float bMax = axis switch { 0 => max.X, 1 => max.Y, _ => max.Z };

                if (MathF.Abs(dir) < 1e-6f)
                {
                    if (origin < bMin || origin > bMax)
                        return false;
                    continue;
                }

                float t1 = (bMin - origin) / dir;
                float t2 = (bMax - origin) / dir;
                if (t1 > t2)
                    (t1, t2) = (t2, t1);
                tMin = MathF.Max(tMin, t1);
                tMax = MathF.Min(tMax, t2);
                if (tMin > tMax)
                    return false;
            }

            distance = tMin;
            return true;
        }

        /// <summary>Tests ray against a capsule aligned with <paramref name="axis"/> from origin.</summary>
        public static bool HitAxisHandle(
            ViewportRay ray,
            Vector3 origin,
            Vector3 axis,
            float length,
            float radius,
            out float distance)
        {
            distance = float.MaxValue;
            axis = Vector3.Normalize(axis);
            var end = origin + axis * length;
            var ab = end - origin;
            float abLen2 = ab.LengthSquared();
            if (abLen2 < 1e-8f)
                return false;

            var ao = ray.Origin - origin;
            float t = Math.Clamp(Vector3.Dot(ao, ab) / abLen2, 0f, 1f);
            var closest = origin + ab * t;
            var oc = ray.Origin - closest;
            float b = Vector3.Dot(oc, ray.Direction);
            float c = oc.LengthSquared() - radius * radius;
            float disc = b * b - c;
            if (disc < 0f)
                return false;
            distance = -b - MathF.Sqrt(disc);
            return distance > 0.001f;
        }

        public static Vector3 ProjectWorldToScreen(
            Vector3 world,
            Matrix4x4 view,
            Matrix4x4 proj,
            int width,
            int height)
        {
            var clip = Vector4.Transform(new Vector4(world, 1f), view * proj);
            if (MathF.Abs(clip.W) < 1e-6f)
                return Vector3.Zero;
            float ndcX = clip.X / clip.W;
            float ndcY = clip.Y / clip.W;
            return new Vector3(
                (ndcX * 0.5f + 0.5f) * width,
                (1f - (ndcY * 0.5f + 0.5f)) * height,
                clip.W);
        }
    }
}
