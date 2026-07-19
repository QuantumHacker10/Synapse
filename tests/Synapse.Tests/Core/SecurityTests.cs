using System;
using System.IO;
using FluentAssertions;
using Synapse.Core.Security;
using Xunit;

namespace Synapse.Tests.Core
{
    public sealed class SecurityTests
    {
        [Theory]
        [InlineData("../secret")]
        [InlineData("..\\secret")]
        [InlineData("/etc/passwd")]
        [InlineData("C:\\Windows\\system32")]
        [InlineData("foo/../../bar")]
        [InlineData("")]
        public void RequireSafeAssetId_RejectsTraversal(string id)
        {
            Action act = () => PathSecurity.RequireSafeAssetId(id);
            act.Should().Throw<ArgumentException>();
        }

        [Fact]
        public void CombineUnderRoot_BlocksEscape()
        {
            var root = Path.Combine(Path.GetTempPath(), "synapse-sec-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                Action act = () => PathSecurity.CombineUnderRoot(root, "..", "escape.txt");
                act.Should().Throw<ArgumentException>();

                var ok = PathSecurity.CombineUnderRoot(root, "asset.bin");
                PathSecurity.IsUnderRoot(root, ok).Should().BeTrue();
            }
            finally
            {
                try
                { Directory.Delete(root, recursive: true); }
                catch { /* ignore */ }
            }
        }

        [Fact]
        public void EnsureUnderRoot_RejectsSiblingEscape()
        {
            var root = Path.Combine(Path.GetTempPath(), "synapse-sec-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                var outside = Path.Combine(Path.GetTempPath(), "outside-" + Guid.NewGuid().ToString("N") + ".txt");
                Action act = () => PathSecurity.EnsureUnderRoot(root, outside);
                act.Should().Throw<UnauthorizedAccessException>();
            }
            finally
            {
                try
                { Directory.Delete(root, recursive: true); }
                catch { /* ignore */ }
            }
        }

        [Theory]
        [InlineData("http://169.254.169.254/latest/meta-data/")]
        [InlineData("http://10.0.0.1/internal")]
        [InlineData("http://192.168.1.1/admin")]
        [InlineData("https://evil.example.com/steal")]
        public void ValidateOutboundUri_BlocksSsrfTargets(string url)
        {
            Action act = () => UrlSecurity.ValidateOutboundUri(url, allowLoopbackHttp: true);
            act.Should().Throw<ArgumentException>();
        }

        [Theory]
        [InlineData("http://127.0.0.1:11434")]
        [InlineData("http://localhost:11434")]
        [InlineData("https://api.openai.com/v1")]
        [InlineData("https://api.anthropic.com")]
        [InlineData("https://generativelanguage.googleapis.com/v1beta")]
        public void ValidateOutboundUri_AllowsTrustedHosts(string url)
        {
            var uri = UrlSecurity.ValidateOutboundUri(url, allowLoopbackHttp: true);
            uri.Should().NotBeNull();
        }

        [Fact]
        public void SecretRedactor_MasksKeysAndBearers()
        {
            var raw = "Authorization: Bearer sk-abcdefghijklmnop api_key=supersecret key=ABCDEFGHIJKLMNOP";
            var redacted = SecretRedactor.Redact(raw);
            redacted.Should().NotContain("supersecret");
            redacted.Should().NotContain("sk-abcdefghijklmnop");
            redacted.Should().Contain("***");
        }

        [Fact]
        public void RequireSafeIdentifier_AcceptsShaderEntryPoints()
        {
            PathSecurity.RequireSafeIdentifier("main").Should().Be("main");
            PathSecurity.RequireSafeIdentifier("CSMain").Should().Be("CSMain");
            Action bad = () => PathSecurity.RequireSafeIdentifier("main; rm -rf /");
            bad.Should().Throw<ArgumentException>();
        }
    }
}
