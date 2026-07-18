using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;

namespace Synapse.Core.Security
{
    /// <summary>
    /// Validates outbound URLs for LLM / asset downloads (SSRF mitigation).
    /// </summary>
    public static class UrlSecurity
    {
        private static readonly HashSet<string> PublicApiHosts = new(StringComparer.OrdinalIgnoreCase)
        {
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            "api.github.com"
        };

        public static Uri ValidateOutboundUri(
            string url,
            bool allowLoopbackHttp = true,
            IReadOnlyCollection<string>? extraAllowedHosts = null)
        {
            if (string.IsNullOrWhiteSpace(url))
                throw new ArgumentException("URL is empty.", nameof(url));

            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
                throw new ArgumentException("URL is not absolute.", nameof(url));

            if (uri.Scheme != Uri.UriSchemeHttps &&
                !(allowLoopbackHttp && uri.Scheme == Uri.UriSchemeHttp && IsLoopback(uri)))
            {
                throw new ArgumentException("Only HTTPS is allowed (HTTP limited to loopback).", nameof(url));
            }

            if (string.IsNullOrWhiteSpace(uri.Host))
                throw new ArgumentException("URL host is missing.", nameof(url));

            if (IsBlockedHost(uri.Host))
                throw new ArgumentException("URL host is not allowed.", nameof(url));

            bool allowed =
                IsLoopback(uri) ||
                PublicApiHosts.Contains(uri.Host) ||
                (extraAllowedHosts != null && HostInList(uri.Host, extraAllowedHosts));

            if (!allowed && uri.Host.EndsWith(".openai.azure.com", StringComparison.OrdinalIgnoreCase))
                allowed = true;

            if (!allowed)
                throw new ArgumentException($"Host '{uri.Host}' is not on the allowlist.", nameof(url));

            return uri;
        }

        public static HttpClient CreateSafeHttpClient(TimeSpan? timeout = null)
        {
            var handler = new SocketsHttpHandler
            {
                AllowAutoRedirect = false,
                AutomaticDecompression = DecompressionMethods.All,
                ConnectTimeout = TimeSpan.FromSeconds(10)
            };

            return new HttpClient(handler)
            {
                Timeout = timeout ?? TimeSpan.FromSeconds(60)
            };
        }

        public static bool IsLoopback(Uri uri)
        {
            if (uri.IsLoopback) return true;
            if (IPAddress.TryParse(uri.Host, out var ip))
                return IPAddress.IsLoopback(ip);
            return string.Equals(uri.Host, "localhost", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsBlockedHost(string host)
        {
            if (IPAddress.TryParse(host, out var ip))
            {
                if (IPAddress.IsLoopback(ip)) return false;
                if (ip.AddressFamily == AddressFamily.InterNetwork)
                {
                    var bytes = ip.GetAddressBytes();
                    if (bytes[0] == 10) return true;
                    if (bytes[0] == 172 && bytes[1] >= 16 && bytes[1] <= 31) return true;
                    if (bytes[0] == 192 && bytes[1] == 168) return true;
                    if (bytes[0] == 169 && bytes[1] == 254) return true;
                    if (bytes[0] == 0) return true;
                }

                if (ip.AddressFamily == AddressFamily.InterNetworkV6 && !IPAddress.IsLoopback(ip))
                    return true;
            }

            return false;
        }

        private static bool HostInList(string host, IReadOnlyCollection<string> list)
        {
            foreach (var item in list)
            {
                if (string.Equals(host, item, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }
    }
}
