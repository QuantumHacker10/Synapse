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
    /// BPE-based tokenizer supporting encoding, decoding, special tokens,
    /// truncation strategies, and batch operations. Used by the ONNX provider
    /// and for general token counting.
    /// </summary>
    public sealed class LlmTokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<int, string> _reverseVocab;
        private readonly List<(string, string)> _bpeRanks;
        private readonly HashSet<string> _specialTokens;
        private readonly string _bosToken;
        private readonly string _eosToken;
        private readonly string _padToken;
        private readonly string _unkToken;
        private readonly int _vocabSize;
        private readonly object _encodeLock = new();

        /// <summary>Size of the vocabulary.</summary>
        public int VocabSize => _vocabSize;

        /// <summary>BOS token string.</summary>
        public string BosToken => _bosToken;

        /// <summary>EOS token string.</summary>
        public string EosToken => _eosToken;

        /// <summary>PAD token string.</summary>
        public string PadToken => _padToken;

        /// <summary>UNK token string.</summary>
        public string UnkToken => _unkToken;

        /// <summary>Number of BPE merge rules loaded.</summary>
        public int MergeCount => _bpeRanks.Count;

        /// <summary>
        /// Initializes a new tokenizer with a vocabulary and BPE merge rules.
        /// </summary>
        /// <param name="vocab">Vocabulary mapping token string to ID.</param>
        /// <param name="bpeRanks">BPE merge rules in priority order.</param>
        /// <param name="bosToken">Beginning-of-sequence token.</param>
        /// <param name="eosToken">End-of-sequence token.</param>
        /// <param name="padToken">Padding token.</param>
        /// <param name="unkToken">Unknown token.</param>
        public LlmTokenizer(
            Dictionary<string, int> vocab,
            List<(string, string)>? bpeRanks = null,
            string bosToken = "<s>",
            string eosToken = "</s>",
            string padToken = "<pad>",
            string unkToken = "<unk>")
        {
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _reverseVocab = new Dictionary<int, string>();
            foreach (var (key, value) in _vocab)
                _reverseVocab[value] = key;

            _bpeRanks = bpeRanks ?? new List<(string, string)>();
            _bosToken = bosToken;
            _eosToken = eosToken;
            _padToken = padToken;
            _unkToken = unkToken;
            _vocabSize = _vocab.Count;

            _specialTokens = new HashSet<string>
            {
                _bosToken, _eosToken, _padToken, _unkToken,
                "<|system|>", "<|user|>", "<|assistant|>",
                "<|function|>", "<|tool|>"
            };
        }

        /// <summary>
        /// Loads a tokenizer from a vocabulary file (one token per line, tab-separated token and ID).
        /// </summary>
        /// <param name="vocabFilePath">Path to the vocabulary file.</param>
        /// <param name="bpeFilePath">Optional path to the BPE merges file.</param>
        /// <returns>A configured LlmTokenizer instance.</returns>
        public static LlmTokenizer LoadFromFile(string vocabFilePath, string? bpeFilePath = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(vocabFilePath);
            if (!File.Exists(vocabFilePath))
                throw new FileNotFoundException($"Vocabulary file not found: {vocabFilePath}");

            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var line in File.ReadLines(vocabFilePath))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                var parts = line.Split('\t', 2);
                if (parts.Length == 2 && int.TryParse(parts[1], out var id))
                    vocab[parts[0]] = id;
                else if (parts.Length == 1)
                    vocab[parts[0]] = vocab.Count;
            }

            var bpeRanks = new List<(string, string)>();
            if (!string.IsNullOrEmpty(bpeFilePath) && File.Exists(bpeFilePath))
            {
                foreach (var line in File.ReadLines(bpeFilePath))
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    var parts = line.Split(' ', 2);
                    if (parts.Length == 2)
                        bpeRanks.Add((parts[0], parts[1]));
                }
            }

            return new LlmTokenizer(vocab, bpeRanks);
        }

        /// <summary>
        /// Encodes a string into token IDs.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <param name="addBos">Whether to prepend the BOS token.</param>
        /// <param name="addEos">Whether to append the EOS token.</param>
        /// <returns>Array of token IDs.</returns>
        public int[] Encode(string text, bool addBos = false, bool addEos = false)
        {
            if (string.IsNullOrEmpty(text))
            {
                var empty = new List<int>();
                if (addBos)
                    empty.Add(_vocab.GetValueOrDefault(_bosToken, 0));
                if (addEos)
                    empty.Add(_vocab.GetValueOrDefault(_eosToken, 1));
                return empty.ToArray();
            }

            var tokenIds = new List<int>();
            if (addBos)
                tokenIds.Add(_vocab.GetValueOrDefault(_bosToken, 0));

            var segments = text.Split(_specialTokens.ToArray(), StringSplitOptions.None);
            foreach (var segment in segments)
            {
                if (_specialTokens.Contains(segment))
                {
                    tokenIds.Add(_vocab.GetValueOrDefault(segment, _vocab.GetValueOrDefault(_unkToken, 3)));
                    continue;
                }

                var words = SegmentIntoWords(segment);
                foreach (var word in words)
                {
                    var bpeTokens = ApplyBpe(word);
                    foreach (var bpeToken in bpeTokens)
                    {
                        if (_vocab.TryGetValue(bpeToken, out var id))
                            tokenIds.Add(id);
                        else
                            tokenIds.Add(_vocab.GetValueOrDefault(_unkToken, 3));
                    }
                }
            }

            if (addEos)
                tokenIds.Add(_vocab.GetValueOrDefault(_eosToken, 1));
            return tokenIds.ToArray();
        }

        /// <summary>
        /// Decodes token IDs back to a string.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs.</param>
        /// <param name="skipSpecial">Whether to skip special tokens in output.</param>
        /// <returns>Decoded string.</returns>
        public string Decode(int[] tokenIds, bool skipSpecial = true)
        {
            if (tokenIds == null || tokenIds.Length == 0)
                return string.Empty;

            var sb = new StringBuilder();
            foreach (var id in tokenIds)
            {
                if (_reverseVocab.TryGetValue(id, out var token))
                {
                    if (skipSpecial && _specialTokens.Contains(token))
                        continue;
                    sb.Append(token.Replace("Ġ", " ").Replace("Ċ", "\n"));
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Counts the number of tokens in the given text.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <returns>Approximate token count.</returns>
        public int CountTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            return Encode(text).Length;
        }

        /// <summary>
        /// Encodes multiple strings in a batch.
        /// </summary>
        /// <param name="texts">Input texts.</param>
        /// <param name="addBos">Whether to prepend BOS.</param>
        /// <param name="addEos">Whether to append EOS.</param>
        /// <param name="padToMaxLength">Whether to pad all sequences to the longest.</param>
        /// <returns>Batch of token ID arrays.</returns>
        public IReadOnlyList<int[]> BatchEncode(
            IReadOnlyList<string> texts,
            bool addBos = false,
            bool addEos = false,
            bool padToMaxLength = false)
        {
            var results = texts.Select(t => Encode(t, addBos, addEos)).ToList();

            if (padToMaxLength && results.Count > 0)
            {
                int maxLen = results.Max(r => r.Length);
                int padId = _vocab.GetValueOrDefault(_padToken, 0);
                results = results.Select(r =>
                {
                    if (r.Length < maxLen)
                    {
                        var padded = new int[maxLen];
                        Array.Copy(r, padded, r.Length);
                        for (int i = r.Length; i < maxLen; i++)
                            padded[i] = padId;
                        return padded;
                    }
                    return r;
                }).ToList();
            }

            return results;
        }

        /// <summary>
        /// Truncates token IDs to fit within a maximum length using the specified strategy.
        /// </summary>
        /// <param name="tokenIds">Input token IDs.</param>
        /// <param name="maxLength">Maximum allowed length.</param>
        /// <param name="strategy">Truncation strategy.</param>
        /// <returns>Truncated token IDs.</returns>
        public int[] Truncate(int[] tokenIds, int maxLength, TruncationStrategy strategy = TruncationStrategy.Tail)
        {
            if (tokenIds == null || tokenIds.Length <= maxLength)
                return tokenIds ?? Array.Empty<int>();

            return strategy switch
            {
                TruncationStrategy.Head => tokenIds.Skip(tokenIds.Length - maxLength).ToArray(),
                TruncationStrategy.Tail => tokenIds.Take(maxLength).ToArray(),
                TruncationStrategy.Middle => TruncateMiddle(tokenIds, maxLength),
                TruncationStrategy.Balanced => TruncateBalanced(tokenIds, maxLength),
                _ => tokenIds.Take(maxLength).ToArray()
            };
        }

        /// <summary>
        /// Adds a special token to the vocabulary.
        /// </summary>
        /// <param name="token">Token string to add.</param>
        /// <returns>The assigned token ID.</returns>
        public int AddSpecialToken(string token)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(token);
            if (_vocab.TryGetValue(token, out var existingId))
                return existingId;

            var newId = _vocab.Count;
            _vocab[token] = newId;
            _reverseVocab[newId] = token;
            _specialTokens.Add(token);
            return newId;
        }

        /// <summary>
        /// Gets the token ID for a specific token string.
        /// </summary>
        /// <param name="token">Token string.</param>
        /// <returns>Token ID, or -1 if not found.</returns>
        public int GetTokenId(string token)
        {
            return _vocab.TryGetValue(token, out var id) ? id : -1;
        }

        /// <summary>
        /// Gets the token string for a specific token ID.
        /// </summary>
        /// <param name="tokenId">Token ID.</param>
        /// <returns>Token string, or null if not found.</returns>
        public string? GetToken(int tokenId)
        {
            return _reverseVocab.TryGetValue(tokenId, out var token) ? token : null;
        }

        /// <summary>
        /// Analyzes token frequency in the given text.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <returns>Dictionary of token to frequency count.</returns>
        public IReadOnlyDictionary<string, int> AnalyzeFrequency(string text)
        {
            var tokenIds = Encode(text);
            var freq = new Dictionary<string, int>();
            foreach (var id in tokenIds)
            {
                var token = GetToken(id) ?? "<unk>";
                freq.TryGetValue(token, out var count);
                freq[token] = count + 1;
            }
            return freq;
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private static string[] SegmentIntoWords(string text)
        {
            var words = new List<string>();
            var current = new StringBuilder();

            foreach (var c in text)
            {
                if (char.IsLetterOrDigit(c))
                {
                    current.Append(c);
                }
                else
                {
                    if (current.Length > 0)
                    {
                        words.Add(current.ToString());
                        current.Clear();
                    }
                    words.Add(c.ToString());
                }
            }

            if (current.Length > 0)
                words.Add(current.ToString());

            return words.ToArray();
        }

        private List<string> ApplyBpe(string word)
        {
            if (string.IsNullOrEmpty(word))
                return new List<string>();

            var chars = word.Select(c => c.ToString()).ToList();

            if (chars.Count == 1)
                return chars;

            for (int i = 0; i < chars.Count - 1; i++)
            {
                var pair = (chars[i], chars[i + 1]);
                int rank = _bpeRanks.IndexOf(pair);
                if (rank >= 0)
                {
                    var merged = pair.Item1 + pair.Item2;
                    chars[i] = merged;
                    chars.RemoveAt(i + 1);
                    i = Math.Max(-1, i - 2);
                }
            }

            return chars;
        }

        private static int[] TruncateMiddle(int[] tokenIds, int maxLength)
        {
            int half = maxLength / 2;
            var result = new int[maxLength];
            Array.Copy(tokenIds, 0, result, 0, half);
            Array.Copy(tokenIds, tokenIds.Length - (maxLength - half), result, half, maxLength - half);
            return result;
        }

        private static int[] TruncateBalanced(int[] tokenIds, int maxLength)
        {
            int headLen = (int)(maxLength * 0.6);
            int tailLen = maxLength - headLen;
            var result = new int[maxLength];
            Array.Copy(tokenIds, 0, result, 0, headLen);
            Array.Copy(tokenIds, tokenIds.Length - tailLen, result, headLen, tailLen);
            return result;
        }
    }
}
