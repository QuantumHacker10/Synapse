// ============================================================
// LawLibraryRegistry.cs - Synapse Omnia Reference Physics Law Library
// The canonical registry of physical laws consumed by LivingLawCompiler.
// C# 14, unsafe code, NativeAOT compatible.
// ============================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;

namespace Synapse.Physics;


//  SECTION 4 — LawRegistry (thread-safe, versioned, importable)

/// <summary>
/// Thread-safe, versioned registry for runtime law management.
/// Supports registration, unregistration, versioning, search, and import/export.
/// </summary>
public sealed class LawRegistry
{
    private readonly ConcurrentDictionary<string, LawDefinition> _registry = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, int> _versionMap = new(StringComparer.OrdinalIgnoreCase);
    private int _nextVersion;
    private readonly object _lock = new();

    /// <summary>Total laws in this registry instance.</summary>
    public int Count => _registry.Count;

    /// <summary>Raised when a law is registered or unregistered.</summary>
    public event Action<string, LawChangeType>? LawChanged;

    /// <summary>Type of change event.</summary>
    public enum LawChangeType { Registered, Unregistered, Updated }

    public LawRegistry()
    {
        _nextVersion = 1;
    }

    /// <summary>Loads all built-in laws from the static library into this registry.</summary>
    public void LoadDefaults()
    {
        foreach (var law in LawLibraryRegistry.GetAll())
        {
            _registry[law.Id] = law;
            _versionMap[law.Id] = _nextVersion++;
        }
    }

    /// <summary>Registers a new law. Throws if id already exists.</summary>
    public LawDefinition Register(LawDefinition law)
    {
        ArgumentNullException.ThrowIfNull(law);
        ArgumentException.ThrowIfNullOrWhiteSpace(law.Id);

        var registered = law with { Version = $"1.{Interlocked.Increment(ref _nextVersion)}.0" };
        if (!_registry.TryAdd(law.Id, registered))
            throw new InvalidOperationException($"Law '{law.Id}' is already registered. Use Update instead.");

        _versionMap[law.Id] = _nextVersion;
        LawChanged?.Invoke(law.Id, LawChangeType.Registered);
        return registered;
    }

    /// <summary>Updates an existing law, incrementing its version.</summary>
    public LawDefinition Update(LawDefinition law)
    {
        ArgumentNullException.ThrowIfNull(law);
        ArgumentException.ThrowIfNullOrWhiteSpace(law.Id);

        var existing = _registry.GetValueOrDefault(law.Id)
            ?? throw new KeyNotFoundException($"Law '{law.Id}' not found.");

        var parts = existing.Version.Split('.');
        var major = int.TryParse(parts[0], out var m) ? m : 1;
        var updated = law with { Version = $"{major}.{Interlocked.Increment(ref _nextVersion)}.0" };

        _registry[law.Id] = updated;
        _versionMap[law.Id] = _nextVersion;
        LawChanged?.Invoke(law.Id, LawChangeType.Updated);
        return updated;
    }

    /// <summary>Unregisters a law by id. Returns true if it existed.</summary>
    public bool Unregister(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        if (_registry.TryRemove(id, out _))
        {
            _versionMap.TryRemove(id, out _);
            LawChanged?.Invoke(id, LawChangeType.Unregistered);
            return true;
        }
        return false;
    }

    /// <summary>Gets a law by id, or null.</summary>
    public LawDefinition? Get(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _registry.GetValueOrDefault(id);
    }

    /// <summary>Returns all registered laws.</summary>
    public IReadOnlyCollection<LawDefinition> GetAll() => _registry.Values.ToList().AsReadOnly();

    /// <summary>Searches by name, id, expression, or description.</summary>
    public IEnumerable<LawDefinition> Search(string query)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _registry.Values.Where(l =>
            l.Id.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Expression.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Description.Contains(query, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Returns laws matching the given category.</summary>
    public IEnumerable<LawDefinition> ByCategory(LawCategory category) =>
        _registry.Values.Where(l => l.Category == category);

    /// <summary>Returns laws containing the given tag.</summary>
    public IEnumerable<LawDefinition> ByTag(string tag) =>
        _registry.Values.Where(l => l.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase));

    /// <summary>Returns the version number for a registered law.</summary>
    public int GetVersion(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _versionMap.GetValueOrDefault(id);
    }

    /// <summary>Checks if a law id is registered.</summary>
    public bool Contains(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _registry.ContainsKey(id);
    }

    /// <summary>Exports all laws as a JSON string.</summary>
    public string ExportJson()
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
        var laws = _registry.Values.OrderBy(l => l.Id).ToList();
        return JsonSerializer.Serialize(laws, options);
    }

    /// <summary>Imports laws from a JSON string. Returns the count of laws imported.</summary>
    public int ImportJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var laws = JsonSerializer.Deserialize<List<LawDefinition>>(json);
        if (laws == null)
            return 0;

        int count = 0;
        foreach (var law in laws)
        {
            if (string.IsNullOrWhiteSpace(law.Id))
                continue;
            if (_registry.TryAdd(law.Id, law))
            {
                _versionMap[law.Id] = _nextVersion;
                count++;
                LawChanged?.Invoke(law.Id, LawChangeType.Registered);
            }
        }
        return count;
    }

    /// <summary>Merges laws from another registry, skipping duplicates.</summary>
    public int MergeFrom(LawRegistry other)
    {
        ArgumentNullException.ThrowIfNull(other);
        int count = 0;
        foreach (var kvp in other._registry)
        {
            if (_registry.TryAdd(kvp.Key, kvp.Value))
            {
                _versionMap[kvp.Key] = _nextVersion;
                count++;
                LawChanged?.Invoke(kvp.Key, LawChangeType.Registered);
            }
        }
        return count;
    }

    /// <summary>Removes all laws from the registry.</summary>
    public void Clear()
    {
        var ids = _registry.Keys.ToList();
        _registry.Clear();
        _versionMap.Clear();
        foreach (var id in ids)
            LawChanged?.Invoke(id, LawChangeType.Unregistered);
    }

    /// <summary>Returns all registered law ids.</summary>
    public IEnumerable<string> AllIds() => _registry.Keys;
}
