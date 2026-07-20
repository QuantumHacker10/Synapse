using System;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{
    /// <summary>Injection point for LLM router queries from behavior tree nodes.</summary>
    public static class BehaviorLlmContext
    {
        public static Func<string, SentientEntity, EntityContext, CancellationToken, Task<string>>? QueryAsync { get; set; }

        public static Func<SentientEntity, EntityContext, string, TaskStatus>? ResponseHandler { get; set; }

        public static TaskStatus DefaultHandle(SentientEntity entity, EntityContext context, string response)
        {
            if (ResponseHandler != null)
            {
                try
                { return ResponseHandler(entity, context, response); }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Warn("BehaviorLlmContext", "LLM response handler threw an exception.", ex);
                    return TaskStatus.Failure;
                }
            }

            if (response.Contains("fail", StringComparison.OrdinalIgnoreCase))
                return TaskStatus.Failure;
            return TaskStatus.Success;
        }
    }
}
