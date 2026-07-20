// =============================================================================
// EntityBehaviorSystem.cs
// GDNN.Sentience - Complete Entity Behavior System for G-DNN Engine
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{

    public class GroupBehaviorSystem
    {
        private readonly Dictionary<string, GroupData> _groups = new();
        private readonly Dictionary<Guid, string> _entityGroups = new();

        public float SeparationRadius { get; set; } = 2f;
        public float AlignmentRadius { get; set; } = 5f;
        public float CohesionRadius { get; set; } = 8f;
        public float SeparationWeight { get; set; } = 1.5f;
        public float AlignmentWeight { get; set; } = 1f;
        public float CohesionWeight { get; set; } = 1f;

        public void RegisterEntity(Guid entityId, string groupName)
        {
            _entityGroups[entityId] = groupName;
            if (!_groups.TryGetValue(groupName, out var g))
            { g = new GroupData { Name = groupName }; _groups[groupName] = g; }
            g.Members.Add(entityId);
        }

        public void UnregisterEntity(Guid entityId)
        {
            if (_entityGroups.TryGetValue(entityId, out var gName) && _groups.TryGetValue(gName, out var g))
                g.Members.Remove(entityId);
            _entityGroups.Remove(entityId);
        }

        public Vector3 CalculateFlocking(SentientEntity entity, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_entityGroups.TryGetValue(entity.EntityId, out var groupName) || !_groups.TryGetValue(groupName, out var group))
                return Vector3.Zero;

            var separation = CalculateSeparation(entity, group, allEntities);
            var alignment = CalculateAlignment(entity, group, allEntities);
            var cohesion = CalculateCohesion(entity, group, allEntities);

            return separation * SeparationWeight + alignment * AlignmentWeight + cohesion * CohesionWeight;
        }

        private Vector3 CalculateSeparation(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var force = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                float dist = entity.DistanceTo(other);
                if (dist < SeparationRadius && dist > 0.01f)
                {
                    force += Vector3.Normalize(entity.Position - other.Position) / dist;
                    count++;
                }
            }
            return count > 0 ? force / count : Vector3.Zero;
        }

        private Vector3 CalculateAlignment(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var avgVel = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                if (entity.DistanceTo(other) < AlignmentRadius)
                { avgVel += other.Velocity; count++; }
            }
            return count > 0 ? Vector3.Normalize(avgVel / count - entity.Velocity) : Vector3.Zero;
        }

        private Vector3 CalculateCohesion(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var center = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                if (entity.DistanceTo(other) < CohesionRadius)
                { center += other.Position; count++; }
            }
            if (count == 0)
                return Vector3.Zero;
            center /= count;
            return Vector3.Normalize(center - entity.Position);
        }

        public Guid? FindLeader(string groupName)
        {
            if (!_groups.TryGetValue(groupName, out var g))
                return null;
            return g.LeaderId;
        }

        public void SetLeader(string groupName, Guid entityId)
        {
            if (_groups.TryGetValue(groupName, out var g))
                g.LeaderId = entityId;
        }

        public Vector3 CalculateFollowFormation(SentientEntity follower, SentientEntity leader, int index, int total)
        {
            var spacing = 2.5f;
            var side = index % 2 == 0 ? 1 : -1;
            var row = (index / 2) + 1;
            return leader.Position + new Vector3(side * row * spacing * 0.5f, 0, -row * spacing);
        }

        public Dictionary<Guid, float> CalculateSwarmInfluence(string groupName, Vector3 target, Dictionary<Guid, SentientEntity> allEntities)
        {
            var influences = new Dictionary<Guid, float>();
            if (!_groups.TryGetValue(groupName, out var group))
                return influences;

            float totalDist = 0;
            var distances = new Dictionary<Guid, float>();
            foreach (var mid in group.Members)
            {
                if (!allEntities.TryGetValue(mid, out var e))
                    continue;
                float d = Vector3.Distance(e.Position, target);
                distances[mid] = d;
                totalDist += d;
            }

            if (totalDist > 0)
            {
                foreach (var (id, d) in distances)
                    influences[id] = 1f - (d / totalDist);
            }
            return influences;
        }

        public Dictionary<string, object> MakeGroupDecision(string groupName, List<(string Option, float Score)> options)
        {
            var results = new Dictionary<string, object>();
            if (!_groups.TryGetValue(groupName, out var group))
                return results;

            int totalVotes = group.Members.Count;
            var votes = new Dictionary<string, int>();

            foreach (var member in group.Members)
            {
                var personality = 0.5f;
                var scored = options.OrderByDescending(o => o.Score * personality).First();
                if (!votes.ContainsKey(scored.Option))
                    votes[scored.Option] = 0;
                votes[scored.Option]++;
            }

            var winner = votes.OrderByDescending(kv => kv.Value).First();
            results["Decision"] = winner.Key;
            results["Votes"] = winner.Value;
            results["TotalVoters"] = totalVotes;
            results["Confidence"] = (float)winner.Value / totalVotes;
            return results;
        }

        public void CommunicateWithinGroup(string groupName, Guid senderId, string message, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group))
                return;
            foreach (var mid in group.Members)
            {
                if (mid == senderId || !allEntities.TryGetValue(mid, out var receiver))
                    continue;
                receiver.SetProperty($"GroupMsg_{senderId}", message);
            }
        }

        public Dictionary<string, float> AnalyzeGroupCohesion(string groupName, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group) || group.Members.Count < 2)
                return new Dictionary<string, float> { { "Cohesion", 0 }, { "AvgDist", 0 }, { "SpeedMatch", 0 } };

            var members = group.Members.Where(id => allEntities.ContainsKey(id)).Select(id => allEntities[id]).ToList();
            var center = members.Aggregate(Vector3.Zero, (sum, e) => sum + e.Position) / members.Count;
            float avgDist = members.Average(e => Vector3.Distance(e.Position, center));
            var avgVel = members.Aggregate(Vector3.Zero, (sum, e) => sum + e.Velocity) / members.Count;
            float speedMatch = 1f - members.Average(e => Vector3.Distance(e.Velocity, avgVel)) / 10f;

            return new Dictionary<string, float>
            {
                { "Cohesion", Math.Clamp(1f - avgDist / CohesionRadius, 0, 1) },
                { "AvgDist", avgDist },
                { "SpeedMatch", Math.Clamp(speedMatch, 0, 1) },
                { "MemberCount", members.Count }
            };
        }

        public void TaskAllocation(string groupName, List<string> tasks, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group))
                return;
            var members = group.Members.Where(id => allEntities.ContainsKey(id)).Select(id => allEntities[id]).ToList();
            members = members.OrderBy(e => e.Needs.GetValueOrDefault("TaskLoad", 0)).ToList();

            for (int i = 0; i < Math.Min(tasks.Count, members.Count); i++)
            {
                members[i].SetProperty("AssignedTask", tasks[i]);
                members[i].SetProperty("TaskAssignedTime", Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond);
            }
        }
    }

    public class GroupData
    {
        public string Name { get; set; } = string.Empty;
        public HashSet<Guid> Members { get; set; } = new();
        public Guid? LeaderId { get; set; }
        public Dictionary<string, object> Properties { get; set; } = new();
    }

}
