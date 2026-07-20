using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Sentience;
using Synapse.Simulation.DigitalTwins;

namespace Synapse.Runtime
{
    public sealed class InMemoryDigitalTwin : IDigitalTwin
    {
        private readonly ConcurrentDictionary<string, object> _properties = new(StringComparer.OrdinalIgnoreCase);

        public Guid Id { get; } = Guid.NewGuid();
        public string PhysicalId { get; set; } = "";
        public DateTime LastSynchronized { get; private set; } = DateTime.UtcNow;
        public TwinSynchronizationStatus SynchronizationStatus { get; private set; } = TwinSynchronizationStatus.Synced;
        public IReadOnlyDictionary<string, object> Properties => _properties;

        public void Synchronize(IReadOnlyDictionary<string, object> physicalState)
        {
            ArgumentNullException.ThrowIfNull(physicalState);
            foreach (var kv in physicalState)
                _properties[kv.Key] = kv.Value;
            LastSynchronized = DateTime.UtcNow;
            SynchronizationStatus = TwinSynchronizationStatus.Synced;
        }

        public T? GetProperty<T>(string key) =>
            _properties.TryGetValue(key, out var value) && value is T typed ? typed : default;

        public void SetProperty(string key, object value) => _properties[key] = value;

        public IDigitalTwinSnapshot TakeSnapshot() => new TwinSnapshot(Id, DateTime.UtcNow, new Dictionary<string, object>(_properties));

        public void RestoreSnapshot(IDigitalTwinSnapshot snapshot)
        {
            ArgumentNullException.ThrowIfNull(snapshot);
            _properties.Clear();
            foreach (var kv in snapshot.Properties)
                _properties[kv.Key] = kv.Value;
            LastSynchronized = snapshot.Timestamp;
            SynchronizationStatus = TwinSynchronizationStatus.Synced;
        }
    }

    public sealed class TwinSnapshot : IDigitalTwinSnapshot
    {
        public TwinSnapshot(Guid twinId, DateTime timestamp, IReadOnlyDictionary<string, object> properties)
        {
            TwinId = twinId;
            Timestamp = timestamp;
            Properties = properties;
        }

        public Guid TwinId { get; }
        public DateTime Timestamp { get; }
        public IReadOnlyDictionary<string, object> Properties { get; }
    }

    public sealed class InMemoryDigitalTwinRegistry : IDigitalTwinRegistry, IDigitalTwinSynchronizer
    {
        private readonly ConcurrentDictionary<Guid, IDigitalTwin> _twins = new();

        public IReadOnlyDictionary<Guid, IDigitalTwin> Twins => _twins;

        public IDigitalTwin Register(IDigitalTwin twin)
        {
            ArgumentNullException.ThrowIfNull(twin);
            _twins[twin.Id] = twin;
            return twin;
        }

        public void Unregister(Guid twinId) => _twins.TryRemove(twinId, out _);

        public IDigitalTwin? GetById(Guid twinId) => _twins.TryGetValue(twinId, out var t) ? t : null;

        public IReadOnlyList<IDigitalTwin> GetByType(EntityType type) =>
            _twins.Values.Where(t => t.GetProperty<string>("EntityType") == type.ToString()).ToList();

        public IReadOnlyList<IDigitalTwin> Search(string query) =>
            _twins.Values.Where(t =>
                t.PhysicalId.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                (t.GetProperty<string>("Name")?.Contains(query, StringComparison.OrdinalIgnoreCase) ?? false)).ToList();

        public Task SynchronizeAllAsync(CancellationToken cancellationToken = default)
        {
            foreach (var twin in _twins.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                twin.Synchronize(twin.Properties);
            }
            return Task.CompletedTask;
        }

        public Task SynchronizeAsync(Guid twinId, CancellationToken cancellationToken = default)
        {
            if (_twins.TryGetValue(twinId, out var twin))
                twin.Synchronize(twin.Properties);
            return Task.CompletedTask;
        }

        public DateTime GetLastSyncTime(Guid twinId) =>
            _twins.TryGetValue(twinId, out var twin) ? twin.LastSynchronized : DateTime.MinValue;
    }
}
