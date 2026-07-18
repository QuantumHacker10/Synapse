using GDNN.Sentience;

namespace Synapse.Simulation.DigitalTwins;

public interface IEntity
{
    Guid Id { get; }
}

public interface IDigitalTwin : IEntity
{
    string PhysicalId { get; set; }
    DateTime LastSynchronized { get; }
    TwinSynchronizationStatus SynchronizationStatus { get; }
    IReadOnlyDictionary<string, object> Properties { get; }
    void Synchronize(IReadOnlyDictionary<string, object> physicalState);
    T? GetProperty<T>(string key);
    void SetProperty(string key, object value);
    IDigitalTwinSnapshot TakeSnapshot();
    void RestoreSnapshot(IDigitalTwinSnapshot snapshot);
}

public interface IDigitalTwinSnapshot
{
    Guid TwinId { get; }
    DateTime Timestamp { get; }
    IReadOnlyDictionary<string, object> Properties { get; }
}

public interface IDigitalTwinRegistry
{
    IReadOnlyDictionary<Guid, IDigitalTwin> Twins { get; }
    IDigitalTwin Register(IDigitalTwin twin);
    void Unregister(Guid twinId);
    IDigitalTwin? GetById(Guid twinId);
    IReadOnlyList<IDigitalTwin> GetByType(EntityType type);
    IReadOnlyList<IDigitalTwin> Search(string query);
}

public interface IDigitalTwinSynchronizer
{
    Task SynchronizeAllAsync(CancellationToken cancellationToken = default);
    Task SynchronizeAsync(Guid twinId, CancellationToken cancellationToken = default);
    DateTime GetLastSyncTime(Guid twinId);
}

public enum TwinSynchronizationStatus
{
    Unknown = 0,
    Synced,
    Stale,
    Synchronizing,
    Error,
    Disconnected
}
