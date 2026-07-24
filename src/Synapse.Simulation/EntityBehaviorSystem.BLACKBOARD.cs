// =============================================================================
// EntityBehaviorSystem.cs
// GDNN.Sentience - Complete Entity Behavior System for G-DNN Engine
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{

    public class Blackboard
    {
        private readonly Dictionary<string, object> _data = new();
        private readonly ReaderWriterLockSlim _lock = new();
        private readonly Dictionary<string, DateTime> _lastModified = new();

        public void Set(string key, object value) { _lock.EnterWriteLock(); try { _data[key] = value; _lastModified[key] = DateTime.UtcNow; } finally { _lock.ExitWriteLock(); } }

        public T? Get<T>(string key) { _lock.EnterReadLock(); try { return _data.TryGetValue(key, out var v) && v is T t ? t : default; } finally { _lock.ExitReadLock(); } }

        public T GetOrSet<T>(string key, Func<T> factory)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_data.TryGetValue(key, out var v) && v is T t)
                    return t;
                _lock.EnterWriteLock();
                try
                { var nv = factory(); _data[key] = nv; _lastModified[key] = DateTime.UtcNow; return nv; }
                finally { _lock.ExitWriteLock(); }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        public bool Has(string key) { _lock.EnterReadLock(); try { return _data.ContainsKey(key); } finally { _lock.ExitReadLock(); } }
        public bool Remove(string key) { _lock.EnterWriteLock(); try { _lastModified.Remove(key); return _data.Remove(key); } finally { _lock.ExitWriteLock(); } }
        public IReadOnlyCollection<string> Keys { get { _lock.EnterReadLock(); try { return _data.Keys.ToList().AsReadOnly(); } finally { _lock.ExitReadLock(); } } }
        public int Count { get { _lock.EnterReadLock(); try { return _data.Count; } finally { _lock.ExitReadLock(); } } }

        public void Clear() { _lock.EnterWriteLock(); try { _data.Clear(); _lastModified.Clear(); } finally { _lock.ExitWriteLock(); } }

        public DateTime? GetLastModified(string key) { _lock.EnterReadLock(); try { return _lastModified.TryGetValue(key, out var t) ? t : null; } finally { _lock.ExitReadLock(); } }
    }

}
