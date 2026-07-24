namespace Synapse.Physics;

/// <summary>Maps law categories to specialized grid applicators in <see cref="LivingLawCompiler"/>.</summary>
public static class LawApplicatorMapper
{
    /// <summary>Resolves the applicator dictionary key for a law category string or enum.</summary>
    public static string Resolve(string? category, LawCategory? enumCategory = null)
    {
        string cat = (category ?? enumCategory?.ToString() ?? string.Empty).ToLowerInvariant();
        return cat switch
        {
            "thermodynamics" or "thermaldynamics" or "thermal" => "heat",
            "acoustics" or "wavedynamics" or "wave" or "optics" => "wave",
            "mechanics" or "elasticity" or "solid" or "materialscience" => "elasticity",
            "fluiddynamics" or "fluid" or "navier_stokes" or "plasma" => "incompressible_ns",
            "electromagnetism" or "electrodynamics" or "em" or "electromagnetic" => "electromagnetic",
            "gravitation" or "gravity" or "astrophysics" => "gravity",
            "chemistry" or "climate" or "epidemiology" or "biology" or "finance"
                or "neuroscience" or "nuclear" or "quantummechanics" => "diffusion",
            _ => "generic"
        };
    }

    /// <summary>Fallback expression when a catalog law cannot be compiled verbatim.</summary>
    public static string FallbackExpression(LawCategory category) =>
        Resolve(null, category) switch
        {
            "heat" => "alpha * laplacian(T)",
            "wave" => "c * c * laplacian(u)",
            "elasticity" => "k * laplacian(u)",
            "incompressible_ns" => "mu * laplacian(v)",
            "electromagnetic" => "laplacian(V)",
            "gravity" => "G * laplacian(T)",
            "diffusion" => "D * laplacian(T)",
            _ => "T"
        };
}
