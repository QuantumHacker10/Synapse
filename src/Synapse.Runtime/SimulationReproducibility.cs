using System;

namespace Synapse.Runtime;

/// <summary>Deterministic simulation seed for reproducible benchmarks and research runs.</summary>
public static class SimulationReproducibility
{
    private static int _seed = Environment.TickCount;
    private static Random _random = new(_seed);

    public static int Seed => _seed;

    public static Random Random => _random;

    public static void SetSeed(int seed)
    {
        _seed = seed;
        _random = new Random(seed);
    }

    public static void ResetFromEnvironment()
    {
        var env = Environment.GetEnvironmentVariable("SYNAPSE_SEED");
        if (int.TryParse(env, out var seed))
            SetSeed(seed);
    }

    /// <summary>Hash a string into a stable 32-bit seed.</summary>
    public static int SeedFromString(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        unchecked
        {
            int hash = 17;
            foreach (char c in value)
                hash = hash * 31 + c;
            return hash;
        }
    }
}
