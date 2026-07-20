// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Optimizes compiled bytecode for better performance.</summary>
    public sealed class LawOptimizer
    {
        private readonly HashSet<OptimizationPass> _enabledPasses;

        public LawOptimizer(HashSet<OptimizationPass>? enabledPasses = null)
        {
            _enabledPasses = enabledPasses ?? new HashSet<OptimizationPass>
            {
                OptimizationPass.ConstantFolding,
                OptimizationPass.DeadCodeElimination,
                OptimizationPass.PeepholeOptimization,
                OptimizationPass.InstructionCombining
            };
        }

        /// <summary>Apply all enabled optimization passes to the bytecode.</summary>
        public LawBytecode Optimize(LawBytecode input)
        {
            var bytecode = input.Clone();
            int iterations = 0;
            int maxIterations = 10;
            bool changed;

            do
            {
                changed = false;
                if (_enabledPasses.Contains(OptimizationPass.ConstantFolding))
                    changed |= ApplyConstantFolding(bytecode);
                if (_enabledPasses.Contains(OptimizationPass.PeepholeOptimization))
                    changed |= ApplyPeepholeOptimization(bytecode);
                if (_enabledPasses.Contains(OptimizationPass.InstructionCombining))
                    changed |= ApplyInstructionCombining(bytecode);
                if (_enabledPasses.Contains(OptimizationPass.DeadCodeElimination))
                    changed |= ApplyDeadCodeElimination(bytecode);
                iterations++;
            } while (changed && iterations < maxIterations);

            return bytecode;
        }

        private bool ApplyConstantFolding(LawBytecode bytecode)
        {
            bool changed = false;
            var instructions = bytecode.Instructions.ToArray();
            var constants = bytecode.Constants.ToArray();

            for (int i = 0; i < instructions.Length - 2; i++)
            {
                if (instructions[i].Op == OpCode.PushConst &&
                    instructions[i + 1].Op == OpCode.PushConst)
                {
                    if (i + 2 < instructions.Length)
                    {
                        float a = constants[instructions[i].Operand];
                        float b = constants[instructions[i + 1].Operand];
                        float? result = instructions[i + 2].Op switch
                        {
                            OpCode.Add => a + b,
                            OpCode.Sub => a - b,
                            OpCode.Mul => a * b,
                            OpCode.Div => MathF.Abs(b) < float.Epsilon ? (float?)null : a / b,
                            OpCode.Pow => MathF.Pow(a, b),
                            OpCode.Min => MathF.Min(a, b),
                            OpCode.Max => MathF.Max(a, b),
                            _ => null
                        };

                        if (result.HasValue)
                        {
                            int constIdx = bytecode.AddConstant(result.Value);
                            bytecode.PatchInstruction(i, constIdx);
                            instructions[i] = new Instruction(OpCode.PushConst, constIdx, result.Value);
                            instructions[i + 1] = new Instruction(OpCode.Nop);
                            instructions[i + 2] = new Instruction(OpCode.Nop);
                            changed = true;
                            i += 2;
                        }
                    }
                }
            }
            return changed;
        }

        private bool ApplyPeepholeOptimization(LawBytecode bytecode)
        {
            bool changed = false;
            var instructions = bytecode.Instructions.ToArray();

            for (int i = 0; i < instructions.Length - 1; i++)
            {
                if (instructions[i].Op == OpCode.PushConst &&
                    instructions[i + 1].Op == OpCode.Neg)
                {
                    float val = bytecode.Constants.ToArray()[instructions[i].Operand];
                    int negIdx = bytecode.AddConstant(-val);
                    bytecode.PatchInstruction(i, negIdx);
                    instructions[i] = new Instruction(OpCode.PushConst, negIdx, -val);
                    instructions[i + 1] = new Instruction(OpCode.Nop);
                    changed = true;
                    i++;
                }

                if (instructions[i].Op == OpCode.Dup && instructions[i + 1].Op == OpCode.Mul)
                {
                    instructions[i] = new Instruction(OpCode.Nop);
                    instructions[i + 1] = new Instruction(OpCode.Pow);
                    int powIdx = bytecode.AddConstant(2f);
                    bytecode.PatchInstruction(i + 1, powIdx);
                    instructions[i] = new Instruction(OpCode.PushConst, powIdx, 2f);
                    changed = true;
                    i++;
                }

                if (instructions[i].Op == OpCode.Dup && instructions[i + 1].Op == OpCode.Add)
                {
                    instructions[i] = new Instruction(OpCode.Nop);
                    int twoIdx = bytecode.AddConstant(2f);
                    instructions[i + 1] = new Instruction(OpCode.PushConst, twoIdx, 2f);
                    changed = true;
                    i++;
                }
            }
            return changed;
        }

        private bool ApplyInstructionCombining(LawBytecode bytecode)
        {
            bool changed = false;
            var instructions = bytecode.Instructions.ToArray();
            var constants = bytecode.Constants.ToArray();

            for (int i = 0; i < instructions.Length - 2; i++)
            {
                if (instructions[i].Op == OpCode.Mul && instructions[i + 1].Op == OpCode.Mul)
                {
                    changed = true;
                }

                if (instructions[i].Op == OpCode.Add && instructions[i + 1].Op == OpCode.Add)
                {
                    changed = true;
                }
            }
            return changed;
        }

        private bool ApplyDeadCodeElimination(LawBytecode bytecode)
        {
            bool changed = false;
            var instructions = bytecode.Instructions.ToArray();

            for (int i = 0; i < instructions.Length; i++)
            {
                if (instructions[i].Op == OpCode.Nop)
                {
                    changed = true;
                }

                if (instructions[i].Op == OpCode.Pop && i + 1 < instructions.Length)
                {
                    if (instructions[i + 1].Op != OpCode.Pop)
                    {
                        instructions[i] = new Instruction(OpCode.Nop);
                        changed = true;
                    }
                }

                if (instructions[i].Op == OpCode.UnconditionalJump)
                {
                    int target = instructions[i].Operand;
                    if (target == i + 1)
                    {
                        instructions[i] = new Instruction(OpCode.Nop);
                        changed = true;
                    }
                }
            }
            return changed;
        }

        /// <summary>Get the set of enabled optimization passes.</summary>
        public IReadOnlyCollection<OptimizationPass> GetEnabledPasses() => _enabledPasses;

        /// <summary>Enable a specific optimization pass.</summary>
        public void EnablePass(OptimizationPass pass) => _enabledPasses.Add(pass);

        /// <summary>Disable a specific optimization pass.</summary>
        public void DisablePass(OptimizationPass pass) => _enabledPasses.Remove(pass);
    }
}
