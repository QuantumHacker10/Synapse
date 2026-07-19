// Multi-provider LLM pipeline for Synapse (split from HybridLlmRouter.cs).

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Timers;
using GDNN.Scene;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Manages multi-turn conversations with LLM providers, including history,
    /// truncation, summarization, persistence, and search.
    /// </summary>
    public sealed class LlmConversationManager
    {
        private readonly ConcurrentDictionary<string, ConversationSession> _sessions;
        private readonly string _storagePath;
        private readonly int _maxTurnsBeforeTruncation;

        /// <summary>Number of active sessions.</summary>
        public int ActiveSessionCount => _sessions.Count;

        /// <summary>
        /// Initializes a new conversation manager.
        /// </summary>
        /// <param name="storagePath">Directory for persisting conversations.</param>
        /// <param name="maxTurnsBeforeTruncation">Max turns before auto-truncation.</param>
        public LlmConversationManager(
            string? storagePath = null,
            int maxTurnsBeforeTruncation = 50)
        {
            _sessions = new ConcurrentDictionary<string, ConversationSession>(StringComparer.OrdinalIgnoreCase);
            _storagePath = storagePath ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "GDNN", "Conversations");
            _maxTurnsBeforeTruncation = maxTurnsBeforeTruncation;

            if (!Directory.Exists(_storagePath))
                Directory.CreateDirectory(_storagePath);
        }

        /// <summary>
        /// Creates a new conversation session.
        /// </summary>
        /// <param name="systemPrompt">Optional system prompt.</param>
        /// <param name="sessionId">Optional custom session ID.</param>
        /// <returns>The session ID.</returns>
        public string CreateSession(string? systemPrompt = null, string? sessionId = null)
        {
            var id = sessionId ?? Guid.NewGuid().ToString("N");
            var session = new ConversationSession
            {
                Id = id,
                CreatedAt = DateTimeOffset.UtcNow,
                SystemPrompt = systemPrompt,
                Turns = new List<ConversationTurn>()
            };

            if (!string.IsNullOrEmpty(systemPrompt))
            {
                session.Turns.Add(new ConversationTurn
                {
                    UserMessage = ChatMessage.System(systemPrompt)
                });
            }

            _sessions[id] = session;
            return id;
        }

        /// <summary>
        /// Adds a user message and assistant response to the conversation.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <param name="userMessage">User's message.</param>
        /// <param name="assistantResponse">Assistant's response.</param>
        /// <param name="usage">Token usage for this turn.</param>
        /// <param name="provider">Provider that handled the request.</param>
        public void AddTurn(
            string sessionId,
            string userMessage,
            string assistantResponse,
            TokenUsage? usage = null,
            string? provider = null)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                throw new ArgumentException($"Session '{sessionId}' not found.");

            session.Turns.Add(new ConversationTurn
            {
                UserMessage = ChatMessage.User(userMessage),
                AssistantMessage = ChatMessage.Assistant(assistantResponse),
                Usage = usage,
                Provider = provider,
                Timestamp = DateTimeOffset.UtcNow
            });

            if (session.Turns.Count > _maxTurnsBeforeTruncation)
            {
                TruncateSession(sessionId, session.Turns.Count - _maxTurnsBeforeTruncation);
            }
        }

        /// <summary>
        /// Gets the message history for a session.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <returns>List of chat messages.</returns>
        public IReadOnlyList<ChatMessage> GetHistory(string sessionId)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return Array.Empty<ChatMessage>();

            var messages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(session.SystemPrompt))
                messages.Add(ChatMessage.System(session.SystemPrompt));

            foreach (var turn in session.Turns)
            {
                messages.Add(turn.UserMessage);
                if (turn.AssistantMessage != null)
                    messages.Add(turn.AssistantMessage);
            }

            return messages;
        }

        /// <summary>
        /// Truncates old turns from the beginning of a session.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <param name="turnCount">Number of turns to remove.</param>
        public void TruncateSession(string sessionId, int turnCount)
        {
            if (!_sessions.TryGetValue(sessionId, out var session)) return;
            if (turnCount <= 0 || turnCount >= session.Turns.Count) return;

            session.Turns.RemoveRange(0, turnCount);
            session.IsTruncated = true;
        }

        /// <summary>
        /// Exports a conversation as JSON.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <returns>JSON string.</returns>
        public string ExportJson(string sessionId)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return "{}";

            return JsonSerializer.Serialize(session, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }

        /// <summary>
        /// Exports a conversation as Markdown.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <returns>Markdown string.</returns>
        public string ExportMarkdown(string sessionId)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return "";

            var sb = new StringBuilder();
            sb.AppendLine($"# Conversation {sessionId}");
            sb.AppendLine($"Created: {session.CreatedAt:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine();

            if (!string.IsNullOrEmpty(session.SystemPrompt))
            {
                sb.AppendLine($"**System:** {session.SystemPrompt}");
                sb.AppendLine();
            }

            foreach (var turn in session.Turns)
            {
                sb.AppendLine($"**User:** {turn.UserMessage.Content}");
                sb.AppendLine();
                if (turn.AssistantMessage != null)
                {
                    sb.AppendLine($"**Assistant:** {turn.AssistantMessage.Content}");
                    sb.AppendLine();
                }
                sb.AppendLine("---");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Saves a conversation to disk.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public async Task SaveSessionAsync(string sessionId, CancellationToken cancellationToken = default)
        {
            if (!_sessions.TryGetValue(sessionId, out var session)) return;

            var filePath = Path.Combine(_storagePath, $"{sessionId}.json");
            var json = ExportJson(sessionId);
            await File.WriteAllTextAsync(filePath, json, cancellationToken);
        }

        /// <summary>
        /// Loads a conversation from disk.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>True if loaded successfully.</returns>
        public async Task<bool> LoadSessionAsync(string sessionId, CancellationToken cancellationToken = default)
        {
            var filePath = Path.Combine(_storagePath, $"{sessionId}.json");
            if (!File.Exists(filePath)) return false;

            var json = await File.ReadAllTextAsync(filePath, cancellationToken);
            using var doc = JsonDocument.Parse(json);

            var session = new ConversationSession
            {
                Id = sessionId,
                Turns = new List<ConversationTurn>()
            };

            if (doc.RootElement.TryGetProperty("createdAt", out var catProp))
                session.CreatedAt = DateTimeOffset.Parse(catProp.GetString() ?? "");
            if (doc.RootElement.TryGetProperty("systemPrompt", out var spProp))
                session.SystemPrompt = spProp.GetString();

            _sessions[sessionId] = session;
            return true;
        }

        /// <summary>
        /// Deletes a session.
        /// </summary>
        /// <param name="sessionId">Session ID.</param>
        /// <returns>True if the session was found and deleted.</returns>
        public bool DeleteSession(string sessionId)
        {
            if (_sessions.TryRemove(sessionId, out _))
            {
                var filePath = Path.Combine(_storagePath, $"{sessionId}.json");
                if (File.Exists(filePath)) File.Delete(filePath);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Gets all active session IDs.
        /// </summary>
        public IReadOnlyList<string> GetSessionIds()
        {
            return _sessions.Keys.ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Represents a conversation session.
    /// </summary>
    internal sealed class ConversationSession
    {
        public string Id { get; set; } = string.Empty;
        public DateTimeOffset CreatedAt { get; set; }
        public string? SystemPrompt { get; set; }
        public List<ConversationTurn> Turns { get; set; } = new();
        public bool IsTruncated { get; set; }
    }
}
