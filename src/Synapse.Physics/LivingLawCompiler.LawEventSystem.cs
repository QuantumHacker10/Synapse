// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Event handler for law compiler events.</summary>
    public delegate void LawEventHandler(object sender, LawEventArgs args);

    /// <summary>Event system for the Living Law Compiler.</summary>
    public sealed class LawEventSystem
    {
        private readonly Dictionary<LawEventType, List<LawEventHandler>> _handlers = new();
        private readonly List<LawEventArgs> _eventLog = new();
        private readonly object _logLock = new();
        private int _maxLogSize;

        public IReadOnlyList<LawEventArgs> EventLog
        {
            get { lock (_logLock) return _eventLog.ToList(); }
        }

        public LawEventSystem(int maxLogSize = 10000)
        {
            _maxLogSize = maxLogSize;
        }

        /// <summary>Subscribe to an event type.</summary>
        public void Subscribe(LawEventType eventType, LawEventHandler handler)
        {
            if (!_handlers.TryGetValue(eventType, out var list))
            {
                list = new List<LawEventHandler>();
                _handlers[eventType] = list;
            }
            list.Add(handler);
        }

        /// <summary>Unsubscribe from an event type.</summary>
        public void Unsubscribe(LawEventType eventType, LawEventHandler handler)
        {
            if (_handlers.TryGetValue(eventType, out var list))
                list.Remove(handler);
        }

        /// <summary>Raise an event.</summary>
        public void Raise(LawEventType eventType, string? lawId = null, string? expression = null, string? message = null, Dictionary<string, object>? metadata = null)
        {
            var args = new LawEventArgs
            {
                EventType = eventType,
                LawId = lawId,
                Expression = expression,
                Message = message,
                Metadata = metadata
            };

            lock (_logLock)
            {
                _eventLog.Add(args);
                if (_eventLog.Count > _maxLogSize)
                    _eventLog.RemoveRange(0, _eventLog.Count - _maxLogSize);
            }

            if (_handlers.TryGetValue(eventType, out var list))
            {
                foreach (var handler in list)
                {
                    try
                    { handler(this, args); }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("LivingLawCompiler", $"Law event handler for '{eventType}' threw an exception.", ex);
                    }
                }
            }
        }

        /// <summary>Get events of a specific type.</summary>
        public IReadOnlyList<LawEventArgs> GetEvents(LawEventType eventType, int maxCount = 100)
        {
            lock (_logLock)
            {
                return _eventLog.Where(e => e.EventType == eventType)
                    .TakeLast(maxCount).ToList();
            }
        }

        /// <summary>Get events for a specific law.</summary>
        public IReadOnlyList<LawEventArgs> GetEventsForLaw(string lawId, int maxCount = 100)
        {
            lock (_logLock)
            {
                return _eventLog.Where(e => e.LawId == lawId)
                    .TakeLast(maxCount).ToList();
            }
        }

        /// <summary>Clear the event log.</summary>
        public void ClearLog()
        {
            lock (_logLock)
            { _eventLog.Clear(); }
        }

        /// <summary>Get event statistics.</summary>
        public Dictionary<LawEventType, int> GetStatistics()
        {
            lock (_logLock)
            {
                return _eventLog.GroupBy(e => e.EventType)
                    .ToDictionary(g => g.Key, g => g.Count());
            }
        }
    }
}
