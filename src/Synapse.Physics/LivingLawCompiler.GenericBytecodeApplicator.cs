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

    /// <summary>Generic bytecode applicator that executes arbitrary compiled bytecode per cell.</summary>
    public sealed class GenericBytecodeApplicator : LawApplicator
    {
        public GenericBytecodeApplicator(string targetField) : base("GenericBytecode", targetField) { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var component = field.GetComponent(TargetField);
            if (component == null)
                return;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            float dx = config.CellSize;
            float[] constants = bytecode.Constants.ToArray();
            ReadOnlySpan<Instruction> instructions = bytecode.Instructions;
            const int STACK_SIZE = 1024;
            float[] localStack = new float[STACK_SIZE];

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        int sp = 0;
                        _gas.Reset();
                        int ip = 0;
                        while (ip < instructions.Length)
                        {
                            if (!_gas.Consume(1))
                                break;
                            ref readonly Instruction instr = ref instructions[ip];
                            ip++;
                            switch (instr.Op)
                            {
                                case OpCode.PushConst:
                                    localStack[sp++] = constants[instr.Operand];
                                    break;
                                case OpCode.LoadField:
                                    { string? fn = bytecode.FieldNames[instr.Operand]; var comp = fn != null ? field.GetComponent(fn) : null; localStack[sp++] = comp != null ? comp[x, y, z] : 0f; }
                                    break;
                                case OpCode.Add:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a + b; }
                                    break;
                                case OpCode.Sub:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a - b; }
                                    break;
                                case OpCode.Mul:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a * b; }
                                    break;
                                case OpCode.Div:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Abs(b) < float.Epsilon ? 0f : a / b; }
                                    break;
                                case OpCode.Mod:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Abs(b) < float.Epsilon ? 0f : a % b; }
                                    break;
                                case OpCode.Pow:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Pow(a, b); }
                                    break;
                                case OpCode.Neg:
                                    { float a = localStack[--sp]; localStack[sp++] = -a; }
                                    break;
                                case OpCode.Abs:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Abs(a); }
                                    break;
                                case OpCode.Sin:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Sin(a); }
                                    break;
                                case OpCode.Cos:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Cos(a); }
                                    break;
                                case OpCode.Tan:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Tan(a); }
                                    break;
                                case OpCode.Asin:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Asin(MathF.Max(-1f, MathF.Min(1f, a))); }
                                    break;
                                case OpCode.Acos:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Acos(MathF.Max(-1f, MathF.Min(1f, a))); }
                                    break;
                                case OpCode.Atan:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Atan(a); }
                                    break;
                                case OpCode.Atan2:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Atan2(a, b); }
                                    break;
                                case OpCode.Sinh:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Sinh(a); }
                                    break;
                                case OpCode.Cosh:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Cosh(a); }
                                    break;
                                case OpCode.Tanh:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Tanh(a); }
                                    break;
                                case OpCode.Exp:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Exp(a); }
                                    break;
                                case OpCode.Log:
                                    { float a = localStack[--sp]; localStack[sp++] = a > 0f ? MathF.Log(a) : 0f; }
                                    break;
                                case OpCode.Log2:
                                    { float a = localStack[--sp]; localStack[sp++] = a > 0f ? MathF.Log2(a) : 0f; }
                                    break;
                                case OpCode.Log10:
                                    { float a = localStack[--sp]; localStack[sp++] = a > 0f ? MathF.Log10(a) : 0f; }
                                    break;
                                case OpCode.Sqrt:
                                    { float a = localStack[--sp]; localStack[sp++] = a >= 0f ? MathF.Sqrt(a) : 0f; }
                                    break;
                                case OpCode.Cbrt:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Cbrt(a); }
                                    break;
                                case OpCode.Ceil:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Ceiling(a); }
                                    break;
                                case OpCode.Floor:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Floor(a); }
                                    break;
                                case OpCode.Round:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Round(a); }
                                    break;
                                case OpCode.Min:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Min(a, b); }
                                    break;
                                case OpCode.Max:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Max(a, b); }
                                    break;
                                case OpCode.Sign:
                                    { float a = localStack[--sp]; localStack[sp++] = MathF.Sign(a); }
                                    break;
                                case OpCode.GradientX:
                                    localStack[sp++] = ComputeGradientX(component, x, y, z, dx);
                                    break;
                                case OpCode.GradientY:
                                    localStack[sp++] = ComputeGradientY(component, x, y, z, dx);
                                    break;
                                case OpCode.GradientZ:
                                    localStack[sp++] = ComputeGradientZ(component, x, y, z, dx);
                                    break;
                                case OpCode.Laplacian:
                                    localStack[sp++] = ComputeLaplacian(component, x, y, z, dx);
                                    break;
                                case OpCode.Divergence:
                                    { float gz = localStack[--sp]; float gy = localStack[--sp]; float gx = localStack[--sp]; localStack[sp++] = gx + gy + gz; }
                                    break;
                                case OpCode.CurlX:
                                    { float gz = localStack[--sp]; float gy = localStack[--sp]; float gx = localStack[--sp]; localStack[sp++] = gz - gy; }
                                    break;
                                case OpCode.CurlY:
                                    { float gz = localStack[--sp]; float gy = localStack[--sp]; float gx = localStack[--sp]; localStack[sp++] = gx - gz; }
                                    break;
                                case OpCode.CurlZ:
                                    { float gz = localStack[--sp]; float gy = localStack[--sp]; float gx = localStack[--sp]; localStack[sp++] = gy - gx; }
                                    break;
                                case OpCode.Equals:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Abs(a - b) < 1e-6f ? 1f : 0f; }
                                    break;
                                case OpCode.NotEquals:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = MathF.Abs(a - b) >= 1e-6f ? 1f : 0f; }
                                    break;
                                case OpCode.LessThan:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a < b ? 1f : 0f; }
                                    break;
                                case OpCode.GreaterThan:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a > b ? 1f : 0f; }
                                    break;
                                case OpCode.LessOrEqual:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a <= b ? 1f : 0f; }
                                    break;
                                case OpCode.GreaterOrEqual:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a >= b ? 1f : 0f; }
                                    break;
                                case OpCode.LogicalAnd:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = (a != 0f && b != 0f) ? 1f : 0f; }
                                    break;
                                case OpCode.LogicalOr:
                                    { float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = (a != 0f || b != 0f) ? 1f : 0f; }
                                    break;
                                case OpCode.LogicalNot:
                                    { float a = localStack[--sp]; localStack[sp++] = a != 0f ? 0f : 1f; }
                                    break;
                                case OpCode.TernaryJump:
                                    { float cond = localStack[--sp]; if (cond == 0f) ip = instr.Operand; }
                                    break;
                                case OpCode.ConditionalJump:
                                    { float cond = localStack[--sp]; if (cond != 0f) ip = instr.Operand; }
                                    break;
                                case OpCode.UnconditionalJump:
                                    ip = instr.Operand;
                                    break;
                                case OpCode.LoadVar:
                                    localStack[sp++] = 0f;
                                    break;
                                case OpCode.LoadParam:
                                    localStack[sp++] = 0f;
                                    break;
                                case OpCode.Pop:
                                    sp--;
                                    break;
                                case OpCode.Dup:
                                    localStack[sp] = localStack[sp - 1];
                                    sp++;
                                    break;
                                case OpCode.Swap:
                                    { float a = localStack[sp - 1]; localStack[sp - 1] = localStack[sp - 2]; localStack[sp - 2] = a; }
                                    break;
                                case OpCode.Clamp:
                                    { float maxV = localStack[--sp]; float minV = localStack[--sp]; float v = localStack[--sp]; localStack[sp++] = MathF.Max(minV, MathF.Min(maxV, v)); }
                                    break;
                                case OpCode.Lerp:
                                    { float t = localStack[--sp]; float b = localStack[--sp]; float a = localStack[--sp]; localStack[sp++] = a + t * (b - a); }
                                    break;
                                case OpCode.Return:
                                case OpCode.Halt:
                                    goto done;
                                default:
                                    break;
                            }
                        }
                    done:
                        ;
                        if (sp > 0)
                            component[x, y, z] = localStack[--sp];
                    }
            ApplyBoundaryConditions(field, config);
        }
    }    // =========================================================================
}
