// ============================================================================
// Synapse Omnia — Physics Solvers
// Complete implementations of electromagnetic, acoustic, thermodynamic,
// chemical, gravitational, lattice-Boltzmann, quantum, elastic, turbulent,
// and multiphysics solvers.
//
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;

public static class VTKExporter
{
    /// <summary>
    /// Write a 3D scalar field to VTK legacy format (.vtk).
    /// </summary>
    public static void WriteScalarField(string filename, string fieldName,
        double[,,] field, double dx, double dy, double dz)
    {
        int nx = field.GetLength(2);
        int ny = field.GetLength(1);
        int nz = field.GetLength(0);

        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("Synapse Omnia Physics Export");
        writer.WriteLine("ASCII");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {nx} {ny} {nz}");
        writer.WriteLine($"SPACING {dx} {dy} {dz}");
        writer.WriteLine($"ORIGIN 0 0 0");
        writer.WriteLine($"POINT_DATA {nx * ny * nz}");
        writer.WriteLine($"SCALARS {fieldName} double 1");
        writer.WriteLine("LOOKUP_TABLE default");

        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++)
                    writer.WriteLine(field[z, y, x].ToString("G15"));
    }

    /// <summary>
    /// Write a 3D vector field to VTK legacy format.
    /// </summary>
    public static void WriteVectorField(string filename, string fieldName,
        double[,,] vx, double[,,] vy, double[,,] vz,
        double dx, double dy, double dz)
    {
        int nx = vx.GetLength(2);
        int ny = vx.GetLength(1);
        int nz = vx.GetLength(0);

        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("Synapse Omnia Physics Export");
        writer.WriteLine("ASCII");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {nx} {ny} {nz}");
        writer.WriteLine($"SPACING {dx} {dy} {dz}");
        writer.WriteLine($"ORIGIN 0 0 0");
        writer.WriteLine($"POINT_DATA {nx * ny * nz}");
        writer.WriteLine($"VECTORS {fieldName} double");

        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++)
                    writer.WriteLine($"{vx[z, y, x]:G15} {vy[z, y, x]:G15} {vz[z, y, x]:G15}");
    }

    /// <summary>
    /// Write a 1D profile (time series or line cut) to CSV.
    /// </summary>
    public static void WriteProfile1D(string filename, string[] headers, double[][] columns)
    {
        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine(string.Join(",", headers));
        int n = columns[0].Length;
        for (int i = 0; i < n; i++)
        {
            var parts = new string[columns.Length];
            for (int c = 0; c < columns.Length; c++)
                parts[c] = columns[c][i].ToString("G15");
            writer.WriteLine(string.Join(",", parts));
        }
    }
}

// ============================================================================
// End of Synapse.Physics numerical solvers — Synapse OMNIA
// ============================================================================
// ============================================================================
//  Additional Utility: Root Finding and Nonlinear Solvers
// ============================================================================

/// <summary>
/// Root-finding algorithms for nonlinear physics equations.
/// </summary>
