using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace GDNN.Rendering;

/// <summary>Deterministic hash of L-DNN irradiance output for golden-image regression in CI.</summary>
public static class GiGoldenSnapshot
{
    /// <summary>SHA-256 hex digest of irradiance samples (fixed grid, 3 decimals).</summary>
    public static string ComputeHash(Vector3[,] irradiance)
    {
        ArgumentNullException.ThrowIfNull(irradiance);
        int w = irradiance.GetLength(0);
        int h = irradiance.GetLength(1);
        var sb = new StringBuilder(w * h * 24);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var c = irradiance[x, y];
                sb.Append(c.X.ToString("F3", CultureInfo.InvariantCulture));
                sb.Append(',');
                sb.Append(c.Y.ToString("F3", CultureInfo.InvariantCulture));
                sb.Append(',');
                sb.Append(c.Z.ToString("F3", CultureInfo.InvariantCulture));
                sb.Append(';');
            }
        }

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
