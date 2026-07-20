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

    public class CommunicationSystem
    {
        private readonly Dictionary<Guid, List<Message>> _inbox = new();
        private readonly List<BroadcastMessage> _broadcasts = new();
        private readonly Dictionary<string, List<Guid>> _channels = new();
        private readonly object _lock = new();

        public void SendMessage(Guid from, Guid to, string content, MessageType type = MessageType.Normal, float urgency = 0.5f)
        {
            lock (_lock)
            {
                if (!_inbox.TryGetValue(to, out var box))
                { box = new List<Message>(); _inbox[to] = box; }
                box.Add(new Message
                {
                    SenderId = from,
                    Content = content,
                    Type = type,
                    Urgency = urgency,
                    Timestamp = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
                });
                if (box.Count > 50)
                    box.RemoveAt(0);
            }
        }

        public void Broadcast(Guid from, string channel, string content, float urgency = 0.5f)
        {
            lock (_lock)
            {
                if (!_channels.TryGetValue(channel, out var subscribers))
                    return;
                foreach (var subId in subscribers)
                {
                    if (subId == from)
                        continue;
                    SendMessage(from, subId, content, MessageType.Broadcast, urgency);
                }
            }
        }

        public List<Message> ReceiveMessages(Guid entityId)
        {
            lock (_lock)
            {
                if (!_inbox.TryGetValue(entityId, out var box))
                    return new List<Message>();
                var messages = box.OrderByDescending(m => m.Urgency).ThenByDescending(m => m.Timestamp).ToList();
                box.Clear();
                return messages;
            }
        }

        public void Subscribe(Guid entityId, string channel)
        {
            lock (_lock)
            {
                if (!_channels.TryGetValue(channel, out var subs))
                { subs = new List<Guid>(); _channels[channel] = subs; }
                if (!subs.Contains(entityId))
                    subs.Add(entityId);
            }
        }

        public void Unsubscribe(Guid entityId, string channel)
        {
            lock (_lock)
            { if (_channels.TryGetValue(channel, out var subs)) subs.Remove(entityId); }
        }

        public int GetInboxCount(Guid entityId) { lock (_lock) { return _inbox.TryGetValue(entityId, out var box) ? box.Count : 0; } }
        public int GetChannelSubscriberCount(string channel) { lock (_lock) { return _channels.TryGetValue(channel, out var s) ? s.Count : 0; } }
    }

    public class Message
    {
        public Guid SenderId { get; set; }
        public string Content { get; set; } = string.Empty;
        public MessageType Type { get; set; }
        public float Urgency { get; set; }
        public double Timestamp { get; set; }
        public bool IsRead { get; set; }
    }

    public class BroadcastMessage
    {
        public Guid SenderId { get; set; }
        public string Channel { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public float Urgency { get; set; }
        public double Timestamp { get; set; }
    }

    public enum MessageType { Normal, Urgent, Broadcast, System, Response }

}
