using System;
using System.Security.Cryptography;
using System.Text;

namespace Synapse.Network;

/// <summary>
/// AES-GCM helpers for experimental P2P payloads (v2.2).
/// Crypto primitives are fine; the WAN transport around them is still lab-only.
/// </summary>
/// <summary>AES-GCM encryption for P2P simulation payloads with session-bound AAD and auth tokens.</summary>
public sealed class PeerEncryption : IDisposable
{
    public const int MaxPacketBytes = 4 * 1024 * 1024;
    private readonly byte[] _key;
    private readonly byte[] _aad;
    private readonly AesGcm _aes;

    public PeerEncryption(byte[] key, string? sessionCode = null)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (key.Length != 32)
            throw new ArgumentException("Peer encryption key must be 32 bytes.", nameof(key));
        _key = key;
        _aad = Encoding.UTF8.GetBytes(string.IsNullOrEmpty(sessionCode) ? "Synapse.P2P" : $"Synapse.P2P|{sessionCode}");
        _aes = new AesGcm(_key, 16);
    }

    public static PeerEncryption FromSessionCode(string sessionCode)
    {
        ArgumentException.ThrowIfNullOrEmpty(sessionCode);
        if (sessionCode.Length < 4)
            throw new ArgumentException("Session code must be at least 4 characters.", nameof(sessionCode));

        // Per-session salt (not a global constant alone).
        var salt = SHA256.HashData(Encoding.UTF8.GetBytes("Synapse.P2P.v2|" + sessionCode));
        var key = Rfc2898DeriveBytes.Pbkdf2(sessionCode, salt, 100_000, HashAlgorithmName.SHA256, 32);
        return new PeerEncryption(key, sessionCode);
    }

    /// <summary>HMAC auth token for TCP handshake (nonce || role).</summary>
    public byte[] ComputeAuthToken(ReadOnlySpan<byte> nonce, string role)
    {
        Span<byte> payload = stackalloc byte[nonce.Length + 16];
        nonce.CopyTo(payload);
        var roleBytes = Encoding.UTF8.GetBytes(role);
        roleBytes.AsSpan(0, Math.Min(roleBytes.Length, 16)).CopyTo(payload[nonce.Length..]);
        return HMACSHA256.HashData(_key, payload[..(nonce.Length + Math.Min(roleBytes.Length, 16))]);
    }

    public bool VerifyAuthToken(ReadOnlySpan<byte> nonce, string role, ReadOnlySpan<byte> token)
    {
        var expected = ComputeAuthToken(nonce, role);
        return CryptographicOperations.FixedTimeEquals(expected, token);
    }

    public byte[] Encrypt(ReadOnlySpan<byte> plaintext)
    {
        if (plaintext.Length > MaxPacketBytes - 28)
            throw new CryptographicException("Plaintext exceeds max packet size.");

        var nonce = new byte[12];
        RandomNumberGenerator.Fill(nonce);
        var ciphertext = new byte[plaintext.Length];
        var tag = new byte[16];
        _aes.Encrypt(nonce, plaintext, ciphertext, tag, _aad);

        var packet = new byte[12 + 16 + ciphertext.Length];
        nonce.CopyTo(packet, 0);
        tag.CopyTo(packet, 12);
        ciphertext.CopyTo(packet, 28);
        return packet;
    }

    public byte[] Decrypt(ReadOnlySpan<byte> packet)
    {
        if (packet.Length < 28)
            throw new CryptographicException("Encrypted packet too short.");
        if (packet.Length > MaxPacketBytes)
            throw new CryptographicException("Encrypted packet too large.");

        var nonce = packet[..12];
        var tag = packet.Slice(12, 16);
        var ciphertext = packet[28..];
        var plaintext = new byte[ciphertext.Length];
        _aes.Decrypt(nonce, ciphertext, tag, plaintext, _aad);
        return plaintext;
    }

    public void Dispose() => _aes.Dispose();
}
