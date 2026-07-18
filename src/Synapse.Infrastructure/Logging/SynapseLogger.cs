using System;
using System.IO;
using System.Text;
using System.Threading;
using Synapse.Core.Security;

namespace Synapse.Infrastructure.Logging
{
    public enum LogLevel
    {
        Trace = 0,
        Debug = 1,
        Information = 2,
        Warning = 3,
        Error = 4,
        Fatal = 5
    }

    public interface ISynapseLogger
    {
        void Log(LogLevel level, string category, string message, Exception? exception = null);
        void Trace(string category, string message) => Log(LogLevel.Trace, category, message);
        void Debug(string category, string message) => Log(LogLevel.Debug, category, message);
        void Info(string category, string message) => Log(LogLevel.Information, category, message);
        void Warn(string category, string message) => Log(LogLevel.Warning, category, message);
        void Error(string category, string message, Exception? exception = null) => Log(LogLevel.Error, category, message, exception);
        void Fatal(string category, string message, Exception? exception = null) => Log(LogLevel.Fatal, category, message, exception);
    }

    public sealed class SynapseLogger : ISynapseLogger, IDisposable
    {
        private readonly object _lock = new();
        private readonly TextWriter? _fileWriter;
        private readonly LogLevel _minLevel;
        private readonly bool _consoleEnabled;
        private bool _disposed;

        public SynapseLogger(string? logDirectory = null, LogLevel minLevel = LogLevel.Information, bool consoleEnabled = true)
        {
            _minLevel = minLevel;
            _consoleEnabled = consoleEnabled;

            if (!string.IsNullOrWhiteSpace(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
                var path = Path.Combine(logDirectory, $"synapse-{DateTime.UtcNow:yyyyMMdd-HHmmss}.log");
                _fileWriter = new StreamWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read), Encoding.UTF8)
                {
                    AutoFlush = true
                };
            }
        }

        public static SynapseLogger Default { get; } = new(
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Synapse", "logs"),
            LogLevel.Information,
            consoleEnabled: true);

        public void Log(LogLevel level, string category, string message, Exception? exception = null)
        {
            if (level < _minLevel || _disposed) return;

            var safeMessage = SecretRedactor.Redact(message);
            var line = $"{DateTime.UtcNow:O} [{level.ToString().ToUpperInvariant()}] [{category}] {safeMessage}";
            if (exception != null)
                line += Environment.NewLine + SecretRedactor.Redact(exception.GetType().Name + ": " + exception.Message);

            lock (_lock)
            {
                if (_consoleEnabled)
                {
                    var prev = Console.ForegroundColor;
                    Console.ForegroundColor = level switch
                    {
                        LogLevel.Warning => ConsoleColor.Yellow,
                        LogLevel.Error or LogLevel.Fatal => ConsoleColor.Red,
                        LogLevel.Debug or LogLevel.Trace => ConsoleColor.DarkGray,
                        _ => ConsoleColor.Gray
                    };
                    Console.WriteLine(line);
                    Console.ForegroundColor = prev;
                }

                _fileWriter?.WriteLine(line);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            lock (_lock)
            {
                _fileWriter?.Dispose();
            }
        }
    }
}
