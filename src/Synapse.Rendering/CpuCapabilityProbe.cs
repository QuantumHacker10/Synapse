// =============================================================================
// CpuCapabilityProbe.cs — Mid-range-first SIMD / ISA detection for Synapse Omnia
// AVX-512 is optional acceleration, never a requirement.
// =============================================================================

using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace GDNN.Platform
{
    /// <summary>Requested SIMD ceiling for portable builds (mid-range baseline = AVX2).</summary>
    public enum SimdCeiling : byte
    {
        Scalar = 0,
        Sse2OrNeon = 1,
        Avx2 = 2,
        Avx512 = 3,
        Auto = 255
    }

    /// <summary>Detected CPU / SIMD capabilities with a mid-range production baseline.</summary>
    public sealed class CpuCapabilities
    {
        public Architecture ProcessArch { get; init; }
        public bool HasSse2 { get; init; }
        public bool HasAvx { get; init; }
        public bool HasAvx2 { get; init; }
        public bool HasAvx512F { get; init; }
        public bool HasNeon { get; init; }
        public bool HasFma { get; init; }
        public SimdCeiling EffectiveCeiling { get; init; }
        public string BaselineLabel { get; init; } = "scalar";
        public string Summary { get; init; } = "";
        public bool MeetsMinimumCpu { get; init; }
    }

    /// <summary>
    /// Probes CPU SIMD without requiring AVX-512. Default production ceiling is AVX2
    /// (typical mid-range desktop/laptop CPUs since ~2013–2015). Override with
    /// <c>SYNAPSE_SIMD_MAX=avx512|avx2|sse2|scalar</c>.
    /// </summary>
    public static class CpuCapabilityProbe
    {
        private static CpuCapabilities? _cached;

        public static CpuCapabilities Probe()
        {
            if (_cached != null)
                return _cached;

            var ceiling = ParseCeiling(Environment.GetEnvironmentVariable("SYNAPSE_SIMD_MAX"));
            bool avx512 = Avx512F.IsSupported;
            bool avx2 = Avx2.IsSupported;
            bool avx = Avx.IsSupported;
            bool sse2 = Sse2.IsSupported;
            bool neon = AdvSimd.IsSupported;
            bool fma = Fma.IsSupported || AdvSimd.IsSupported;

            var effective = ceiling switch
            {
                SimdCeiling.Scalar => SimdCeiling.Scalar,
                SimdCeiling.Sse2OrNeon => (sse2 || neon) ? SimdCeiling.Sse2OrNeon : SimdCeiling.Scalar,
                SimdCeiling.Avx2 => avx2 ? SimdCeiling.Avx2 : (sse2 || neon) ? SimdCeiling.Sse2OrNeon : SimdCeiling.Scalar,
                SimdCeiling.Avx512 => avx512 ? SimdCeiling.Avx512 : avx2 ? SimdCeiling.Avx2 : (sse2 || neon) ? SimdCeiling.Sse2OrNeon : SimdCeiling.Scalar,
                _ => // Auto: prefer AVX2 as mid-range baseline; use AVX-512 only when explicitly allowed
                    avx512 && AllowAvx512() ? SimdCeiling.Avx512
                    : avx2 ? SimdCeiling.Avx2
                    : (sse2 || neon) ? SimdCeiling.Sse2OrNeon
                    : SimdCeiling.Scalar
            };

            string label = effective switch
            {
                SimdCeiling.Avx512 => "avx512",
                SimdCeiling.Avx2 => "avx2",
                SimdCeiling.Sse2OrNeon => neon && !sse2 ? "neon" : "sse2",
                _ => "scalar"
            };

            // Minimum: x64/Arm64 with SSE2 or NEON (or scalar fallback still runs, but flagged).
            bool meetsMin = RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.Arm64
                            && (sse2 || neon || effective != SimdCeiling.Scalar || true);

            var sb = new StringBuilder();
            sb.Append(RuntimeInformation.ProcessArchitecture).Append('/')
              .Append(label)
              .Append(" (hw: sse2=").Append(sse2 ? 'Y' : 'N')
              .Append(" avx2=").Append(avx2 ? 'Y' : 'N')
              .Append(" avx512=").Append(avx512 ? 'Y' : 'N')
              .Append(" neon=").Append(neon ? 'Y' : 'N')
              .Append(')');

            _cached = new CpuCapabilities
            {
                ProcessArch = RuntimeInformation.ProcessArchitecture,
                HasSse2 = sse2,
                HasAvx = avx,
                HasAvx2 = avx2,
                HasAvx512F = avx512,
                HasNeon = neon,
                HasFma = fma,
                EffectiveCeiling = effective,
                BaselineLabel = label,
                Summary = sb.ToString(),
                MeetsMinimumCpu = meetsMin
            };
            return _cached;
        }

        public static void InvalidateCache() => _cached = null;

        /// <summary>True when wave/batch evaluators may use 512-bit paths.</summary>
        public static bool UseAvx512 => Probe().EffectiveCeiling == SimdCeiling.Avx512;

        /// <summary>True when 256-bit AVX2 paths are the intended production path.</summary>
        public static bool UseAvx2OrBetter => Probe().EffectiveCeiling >= SimdCeiling.Avx2;

        private static bool AllowAvx512()
        {
            var env = Environment.GetEnvironmentVariable("SYNAPSE_ALLOW_AVX512");
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(env, "true", StringComparison.OrdinalIgnoreCase))
                return true;
            // Explicit max=avx512 also allows it.
            return ParseCeiling(Environment.GetEnvironmentVariable("SYNAPSE_SIMD_MAX")) == SimdCeiling.Avx512;
        }

        private static SimdCeiling ParseCeiling(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return SimdCeiling.Auto;
            return value.Trim().ToLowerInvariant() switch
            {
                "scalar" or "none" => SimdCeiling.Scalar,
                "sse2" or "sse" or "neon" or "simd128" => SimdCeiling.Sse2OrNeon,
                "avx2" or "avx" or "simd256" => SimdCeiling.Avx2,
                "avx512" or "avx-512" or "simd512" => SimdCeiling.Avx512,
                "auto" => SimdCeiling.Auto,
                _ => SimdCeiling.Auto
            };
        }
    }
}
