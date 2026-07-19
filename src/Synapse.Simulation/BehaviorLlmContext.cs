using System;
using System.Threading;
using System.Threading.Tasks;

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
                catch { return TaskStatus.Failure; }
            }

            if (response.Contains("fail", StringComparison.OrdinalIgnoreCase))
                return TaskStatus.Failure;
            return TaskStatus.Success;
        }
    }
}
