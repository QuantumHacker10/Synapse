using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace GDNN.GPU;

/// <summary>
/// Compile HLSL/GLSL → SPIR-V via DXC ou glslangValidator s'ils sont disponibles sur la machine.
/// </summary>
public static class SpirvToolchain
{
    private static readonly Lazy<(string? Dxc, string? Glslang)> Paths = new(Discover);

    public static bool IsDxcAvailable => Paths.Value.Dxc != null;
    public static bool IsGlslangAvailable => Paths.Value.Glslang != null;
    public static bool IsAvailable => IsDxcAvailable || IsGlslangAvailable;

    public static string? DxcPath => Paths.Value.Dxc;
    public static string? GlslangPath => Paths.Value.Glslang;

    /// <summary>
    /// Compile du HLSL compute en SPIR-V (DXC -spirv).
    /// </summary>
    public static bool TryCompileHlsl(string hlslSource, string entryPoint, out byte[] spirv, out string log)
    {
        spirv = Array.Empty<byte>();
        log = string.Empty;
        if (Paths.Value.Dxc == null)
        {
            log = "dxc.exe not found";
            return false;
        }

        string safeEntry;
        try
        {
            safeEntry = Synapse.Core.Security.PathSecurity.RequireSafeIdentifier(entryPoint, nameof(entryPoint));
        }
        catch (ArgumentException ex)
        {
            log = ex.Message;
            return false;
        }

        if (hlslSource.Length > 4 * 1024 * 1024)
        {
            log = "HLSL source exceeds size limit";
            return false;
        }

        string tempDir = Path.Combine(Path.GetTempPath(), "gdnn_spirv_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string hlslPath = Path.Combine(tempDir, "shader.hlsl");
            string spvPath = Path.Combine(tempDir, "shader.spv");
            File.WriteAllText(hlslPath, hlslSource, Encoding.UTF8);

            var psi = new ProcessStartInfo
            {
                FileName = Paths.Value.Dxc,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            psi.ArgumentList.Add("-T");
            psi.ArgumentList.Add("cs_6_0");
            psi.ArgumentList.Add("-E");
            psi.ArgumentList.Add(safeEntry);
            psi.ArgumentList.Add("-spirv");
            psi.ArgumentList.Add("-fspv-target-env=vulkan1.1");
            psi.ArgumentList.Add(hlslPath);
            psi.ArgumentList.Add("-Fo");
            psi.ArgumentList.Add(spvPath);

            using var proc = Process.Start(psi);
            if (proc == null)
            {
                log = "Failed to start dxc";
                return false;
            }

            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(60_000);
            log = (stdout + "\n" + stderr).Trim();

            if (proc.ExitCode != 0 || !File.Exists(spvPath))
                return false;

            spirv = File.ReadAllBytes(spvPath);
            return spirv.Length >= 20 && BitConverter.ToUInt32(spirv, 0) == 0x07230203u;
        }
        catch (Exception ex)
        {
            log = ex.Message;
            return false;
        }
        finally
        {
            try
            { Directory.Delete(tempDir, recursive: true); }
            catch { /* ignore */ }
        }
    }

    /// <summary>
    /// Compile du GLSL compute en SPIR-V (glslangValidator).
    /// </summary>
    public static bool TryCompileGlsl(string glslSource, out byte[] spirv, out string log)
    {
        spirv = Array.Empty<byte>();
        log = string.Empty;
        if (Paths.Value.Glslang == null)
        {
            log = "glslangValidator not found";
            return false;
        }

        if (glslSource.Length > 4 * 1024 * 1024)
        {
            log = "GLSL source exceeds size limit";
            return false;
        }

        string tempDir = Path.Combine(Path.GetTempPath(), "gdnn_glsl_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string glslPath = Path.Combine(tempDir, "shader.comp");
            string spvPath = Path.Combine(tempDir, "shader.spv");
            File.WriteAllText(glslPath, glslSource, Encoding.UTF8);

            var psi = new ProcessStartInfo
            {
                FileName = Paths.Value.Glslang,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            psi.ArgumentList.Add("-V");
            psi.ArgumentList.Add(glslPath);
            psi.ArgumentList.Add("-o");
            psi.ArgumentList.Add(spvPath);

            using var proc = Process.Start(psi);
            if (proc == null)
            {
                log = "Failed to start glslangValidator";
                return false;
            }

            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(60_000);
            log = (stdout + "\n" + stderr).Trim();

            if (proc.ExitCode != 0 || !File.Exists(spvPath))
                return false;

            spirv = File.ReadAllBytes(spvPath);
            return spirv.Length >= 20 && BitConverter.ToUInt32(spirv, 0) == 0x07230203u;
        }
        catch (Exception ex)
        {
            log = ex.Message;
            return false;
        }
        finally
        {
            try
            { Directory.Delete(tempDir, recursive: true); }
            catch { /* ignore */ }
        }
    }

    private static (string? Dxc, string? Glslang) Discover()
    {
        string? dxc = FindOnPath("dxc.exe") ?? FindOnPath("dxc");
        string? glslang = FindOnPath("glslangValidator.exe") ?? FindOnPath("glslangValidator");

        // Vulkan SDK common locations
        string? sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
        if (!string.IsNullOrEmpty(sdk))
        {
            dxc ??= FirstExisting(
                Path.Combine(sdk, "Bin", "dxc.exe"),
                Path.Combine(sdk, "bin", "dxc.exe"));
            glslang ??= FirstExisting(
                Path.Combine(sdk, "Bin", "glslangValidator.exe"),
                Path.Combine(sdk, "bin", "glslangValidator.exe"));
        }

        string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        if (Directory.Exists(Path.Combine(programFiles, "Vulkan SDK")))
        {
            foreach (var dir in Directory.GetDirectories(Path.Combine(programFiles, "Vulkan SDK")))
            {
                dxc ??= FirstExisting(Path.Combine(dir, "Bin", "dxc.exe"));
                glslang ??= FirstExisting(Path.Combine(dir, "Bin", "glslangValidator.exe"));
            }
        }

        return (dxc, glslang);
    }

    private static string? FindOnPath(string fileName)
    {
        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(pathEnv))
            return null;
        foreach (string dir in pathEnv.Split(Path.PathSeparator))
        {
            try
            {
                string candidate = Path.Combine(dir.Trim(), fileName);
                if (File.Exists(candidate))
                    return candidate;
            }
            catch { /* ignore bad PATH entries */ }
        }
        return null;
    }

    private static string? FirstExisting(params string[] paths)
    {
        foreach (string p in paths)
            if (File.Exists(p))
                return p;
        return null;
    }
}
