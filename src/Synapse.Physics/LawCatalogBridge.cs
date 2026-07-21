namespace Synapse.Physics;

/// <summary>Bridges the static <see cref="LawLibraryRegistry"/> catalog into runtime <see cref="LawLibrary"/> entries.</summary>
public static class LawCatalogBridge
{
    /// <summary>All catalog sources merged in registration order.</summary>
    public static IEnumerable<LawDefinition> EnumerateCatalog()
    {
        foreach (var law in LawLibraryRegistry.GetAll())
            yield return law;
        foreach (var law in LawLibraryRegistryExtended.GetAdditionalLaws())
            yield return law;
        foreach (var law in LawLibraryRegistryExtended2.GetAllAdditionalLaws())
            yield return law;
    }

    /// <summary>Converts a registry definition into a compiler <see cref="LawEntry"/>.</summary>
    public static LawEntry ToLawEntry(LawDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);

        var constants = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        foreach (var param in definition.Parameters)
            constants[param.Name] = (float)param.Value;

        return new LawEntry
        {
            Id = definition.Id,
            Name = definition.Name,
            Category = definition.Category.ToString(),
            Expression = definition.Expression,
            Description = definition.Description,
            Constants = constants,
            ResultDimension = Dimension.Scalar
        };
    }

    /// <summary>
    /// Loads legacy simulation laws plus the full reference catalog into one library.
    /// </summary>
    public static LawLibrary LoadFullCatalog()
    {
        var library = new LawLibrary();
        LawLibrary.RegisterCoreSimulationLaws(library);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in library.AllEntries)
            seen.Add(entry.Id);

        foreach (var definition in EnumerateCatalog())
        {
            if (!seen.Add(definition.Id))
                continue;
            library.Register(ToLawEntry(definition));
        }

        return library;
    }
}
