using System;
using System.Security.Cryptography;
using System.Text;

namespace Synapse.Network;

/// <summary>
/// AES-GCM helpers for experimental P2P payloads (v2.2).
/// Crypto primitives are fine; the WAN transport around them is still lab-only.
/// </summary>
public sealed class PeerEncryption : IDisposable
{
    private readonly byte[] _key;
    private readonly AesGcm _aes;

    public PeerEncryption(byte[] key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (key.Length != 32)
            throw new ArgumentException("Peer encryption key must be 32 bytes.", nameof(key));
        _key = key;
        _aes = new AesGcm(_key, 16);
    }

    public static PeerEncryption FromSessionCode(string sessionCode)
    {
        ArgumentException.ThrowIfNullOrEmpty(sessionCode);
        var salt = Encoding.UTF8.GetBytes("Synapse.P2P.v2");
        var key = Rfc2898DeriveBytes.Pbkdf2(sessionCode, salt, 100_000, HashAlgorithmName.SHA256, 32);
        return new PeerEncryption(key);
    }

    public byte[] Encrypt(ReadOnlySpan<byte> plaintext)
    {
        var nonce = new byte[12];
        RandomNumberGenerator.Fill(nonce);
        var ciphertext = new byte[plaintext.Length];
        var tag = new byte[16];
        _aes.Encrypt(nonce, plaintext, ciphertext, tag);

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

        var nonce = packet[..12];
        var tag = packet.Slice(12, 16);
        var ciphertext = packet[28..];
        var plaintext = new byte[ciphertext.Length];
        _aes.Decrypt(nonce, ciphertext, tag, plaintext);
        return plaintext;
    }

    public void Dispose() => _aes.Dispose();
}
