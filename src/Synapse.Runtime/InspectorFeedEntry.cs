using System;

namespace Synapse.Runtime
{
    /// <summary>Single line in the Studio live inspector feed (evolution, living laws, etc.).</summary>
    public sealed record InspectorFeedEntry(
        DateTime Timestamp,
        string Category,
        string Title,
        string Detail);
}
