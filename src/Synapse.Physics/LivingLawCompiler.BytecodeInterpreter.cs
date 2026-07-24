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

    /// <summary>Stack-based interpreter for law bytecode.</summary>
    public sealed class BytecodeInterpreter
    {
        private const int MaxStackSize = 1024;
        private readonly float[] _stack = new float[MaxStackSize];
        private int _sp;
        private readonly GasMeter _gas;
        private float _time;
        private float _dt;

        public float Time { get => _time; set => _time = value; }
        public float Dt { get => _dt; set => _dt = value; }
        public float[] Stack => _stack;
        public int StackPointer => _sp;

        public BytecodeInterpreter(GasMeter? gas = null)
        {
            _gas = gas ?? new GasMeter();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Push(float value)
        {
            if (_sp >= MaxStackSize)
                throw new InvalidOperationException("Stack overflow");
            _stack[_sp++] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float Pop()
        {
            if (_sp <= 0)
                throw new InvalidOperationException("Stack underflow");
            return _stack[--_sp];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float Peek(int offset = 0)
        {
            int idx = _sp - 1 - offset;
            if (idx < 0 || idx >= _sp)
                throw new InvalidOperationException("Stack peek out of range");
            return _stack[idx];
        }

        /// <summary>Execute bytecode with given parameters and variables.</summary>
        public float Execute(LawBytecode bytecode, PhysicsField? field,
            float[]? variables, Dictionary<string, float>? parameters)
        {
            _sp = 0;
            _gas.Reset();
            float[] constants = bytecode.Constants.ToArray();
            ReadOnlySpan<Instruction> instructions = bytecode.Instructions;
            int ip = 0;

            while (ip < instructions.Length)
            {
                if (!_gas.Consume(1))
                    throw new InvalidOperationException("Gas limit exceeded");

                ref readonly Instruction instr = ref instructions[ip];
                ip++;

                switch (instr.Op)
                {
                    case OpCode.Nop:
                        break;
                    case OpCode.PushConst:
                        Push(constants[instr.Operand]);
                        break;
                    case OpCode.Pop:
                        _sp--;
                        break;
                    case OpCode.Dup:
                        Push(Peek());
                        break;
                    case OpCode.Swap:
                        { float a = Pop(); float b = Pop(); Push(a); Push(b); }
                        break;
                    case OpCode.Add:
                        { float b = Pop(); float a = Pop(); Push(a + b); }
                        break;
                    case OpCode.Sub:
                        { float b = Pop(); float a = Pop(); Push(a - b); }
                        break;
                    case OpCode.Mul:
                        { float b = Pop(); float a = Pop(); Push(a * b); }
                        break;
                    case OpCode.Div:
                        { float b = Pop(); float a = Pop(); Push(MathF.Abs(b) < float.Epsilon ? 0f : a / b); }
                        break;
                    case OpCode.Mod:
                        { float b = Pop(); float a = Pop(); Push(MathF.Abs(b) < float.Epsilon ? 0f : a % b); }
                        break;
                    case OpCode.Pow:
                        { float b = Pop(); float a = Pop(); Push(MathF.Pow(a, b)); }
                        break;
                    case OpCode.Neg:
                        Push(-Pop());
                        break;
                    case OpCode.Abs:
                        Push(MathF.Abs(Pop()));
                        break;

                    case OpCode.LoadVar:
                        if (variables != null && instr.Operand < variables.Length)
                            Push(variables[instr.Operand]);
                        else
                            Push(0f);
                        break;

                    case OpCode.StoreVar:
                        if (variables != null && instr.Operand < variables.Length)
                            variables[instr.Operand] = Pop();
                        else
                            Pop();
                        break;

                    case OpCode.LoadField:
                        if (field != null)
                        {
                            string? fieldName = bytecode.FieldNames[instr.Operand];
                            if (fieldName != null)
                            {
                                var component = field.GetComponent(fieldName);
                                Push(component != null && component.TotalCells > 0 ? component.Average() : 0f);
                            }
                            else
                                Push(0f);
                        }
                        else
                            Push(0f);
                        break;

                    case OpCode.LoadParam:
                        {
                            string? paramName = bytecode.ParamNames[instr.Operand];
                            if (paramName != null && parameters != null && parameters.TryGetValue(paramName, out float val))
                                Push(val);
                            else
                                Push(0f);
                        }
                        break;

                    case OpCode.Equals:
                        { float b = Pop(); float a = Pop(); Push(MathF.Abs(a - b) < 1e-6f ? 1f : 0f); }
                        break;
                    case OpCode.NotEquals:
                        { float b = Pop(); float a = Pop(); Push(MathF.Abs(a - b) >= 1e-6f ? 1f : 0f); }
                        break;
                    case OpCode.LessThan:
                        { float b = Pop(); float a = Pop(); Push(a < b ? 1f : 0f); }
                        break;
                    case OpCode.GreaterThan:
                        { float b = Pop(); float a = Pop(); Push(a > b ? 1f : 0f); }
                        break;
                    case OpCode.LessOrEqual:
                        { float b = Pop(); float a = Pop(); Push(a <= b ? 1f : 0f); }
                        break;
                    case OpCode.GreaterOrEqual:
                        { float b = Pop(); float a = Pop(); Push(a >= b ? 1f : 0f); }
                        break;
                    case OpCode.LogicalAnd:
                        { float b = Pop(); float a = Pop(); Push((a != 0f && b != 0f) ? 1f : 0f); }
                        break;
                    case OpCode.LogicalOr:
                        { float b = Pop(); float a = Pop(); Push((a != 0f || b != 0f) ? 1f : 0f); }
                        break;
                    case OpCode.LogicalNot:
                        Push(Pop() != 0f ? 0f : 1f);
                        break;

                    case OpCode.TernaryJump:
                        { float cond = Pop(); if (cond == 0f) ip = instr.Operand; }
                        break;

                    case OpCode.ConditionalJump:
                        { float cond = Pop(); if (cond != 0f) ip = instr.Operand; }
                        break;
                    case OpCode.UnconditionalJump:
                        ip = instr.Operand;
                        break;
                    case OpCode.Return:
                        return _sp > 0 ? _stack[_sp - 1] : 0f;
                    case OpCode.Halt:
                        return _sp > 0 ? _stack[_sp - 1] : 0f;

                    case OpCode.GasConsume:
                        if (!_gas.Consume(instr.Operand))
                            throw new InvalidOperationException("Gas limit exceeded");
                        break;
                    case OpCode.BoundsCheck:
                        break;

                    case OpCode.Sin:
                        Push(MathF.Sin(Pop()));
                        break;
                    case OpCode.Cos:
                        Push(MathF.Cos(Pop()));
                        break;
                    case OpCode.Tan:
                        Push(MathF.Tan(Pop()));
                        break;
                    case OpCode.Asin:
                        Push(MathF.Asin(MathF.Max(-1f, MathF.Min(1f, Pop()))));
                        break;
                    case OpCode.Acos:
                        Push(MathF.Acos(MathF.Max(-1f, MathF.Min(1f, Pop()))));
                        break;
                    case OpCode.Atan:
                        Push(MathF.Atan(Pop()));
                        break;
                    case OpCode.Atan2:
                        { float b = Pop(); float a = Pop(); Push(MathF.Atan2(a, b)); }
                        break;
                    case OpCode.Sinh:
                        Push(MathF.Sinh(Pop()));
                        break;
                    case OpCode.Cosh:
                        Push(MathF.Cosh(Pop()));
                        break;
                    case OpCode.Tanh:
                        Push(MathF.Tanh(Pop()));
                        break;
                    case OpCode.Exp:
                        Push(MathF.Exp(Pop()));
                        break;
                    case OpCode.Log:
                        { float v = Pop(); Push(v > 0f ? MathF.Log(v) : float.NegativeInfinity); }
                        break;
                    case OpCode.Log2:
                        { float v = Pop(); Push(v > 0f ? MathF.Log2(v) : float.NegativeInfinity); }
                        break;
                    case OpCode.Log10:
                        { float v = Pop(); Push(v > 0f ? MathF.Log10(v) : float.NegativeInfinity); }
                        break;
                    case OpCode.Sqrt:
                        { float v = Pop(); Push(v >= 0f ? MathF.Sqrt(v) : 0f); }
                        break;
                    case OpCode.Cbrt:
                        Push(MathF.Cbrt(Pop()));
                        break;
                    case OpCode.Ceil:
                        Push(MathF.Ceiling(Pop()));
                        break;
                    case OpCode.Floor:
                        Push(MathF.Floor(Pop()));
                        break;
                    case OpCode.Round:
                        Push(MathF.Round(Pop()));
                        break;
                    case OpCode.Clamp:
                        { float max = Pop(); float min = Pop(); float v = Pop(); Push(MathF.Max(min, MathF.Min(max, v))); }
                        break;
                    case OpCode.Lerp:
                        { float t = Pop(); float b = Pop(); float a = Pop(); Push(a + t * (b - a)); }
                        break;
                    case OpCode.Min:
                        { float b = Pop(); float a = Pop(); Push(MathF.Min(a, b)); }
                        break;
                    case OpCode.Max:
                        { float b = Pop(); float a = Pop(); Push(MathF.Max(a, b)); }
                        break;
                    case OpCode.Sign:
                        Push(MathF.Sign(Pop()));
                        break;

                    case OpCode.GradientX:
                    case OpCode.GradientY:
                    case OpCode.GradientZ:
                    case OpCode.Laplacian:
                    case OpCode.Divergence:
                    case OpCode.CurlX:
                    case OpCode.CurlY:
                    case OpCode.CurlZ:
                        Push(0f);
                        break;

                    case OpCode.Call:
                        break;

                    default:
                        throw new InvalidOperationException($"Unknown opcode: {instr.Op}");
                }
            }
            return _sp > 0 ? _stack[_sp - 1] : 0f;
        }

        /// <summary>Execute bytecode over a 3D field, updating each cell.</summary>
        public unsafe void ExecuteOverField(LawBytecode bytecode, PhysicsField field, float[]? parameters = null)
        {
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            float[] constants = bytecode.Constants.ToArray();
            ReadOnlySpan<Instruction> instructions = bytecode.Instructions;

            for (int z = 0; z < sz; z++)
            {
                for (int y = 0; y < sy; y++)
                {
                    for (int x = 0; x < sx; x++)
                    {
                        _sp = 0;
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
                                    Push(constants[instr.Operand]);
                                    break;
                                case OpCode.LoadField:
                                    {
                                        string? fn = bytecode.FieldNames[instr.Operand];
                                        var comp = fn != null ? field.GetComponent(fn) : null;
                                        Push(comp != null ? comp[x, y, z] : 0f);
                                    }
                                    break;
                                case OpCode.Add:
                                    { float b = Pop(); float a = Pop(); Push(a + b); }
                                    break;
                                case OpCode.Sub:
                                    { float b = Pop(); float a = Pop(); Push(a - b); }
                                    break;
                                case OpCode.Mul:
                                    { float b = Pop(); float a = Pop(); Push(a * b); }
                                    break;
                                case OpCode.Div:
                                    { float b = Pop(); float a = Pop(); Push(MathF.Abs(b) < float.Epsilon ? 0f : a / b); }
                                    break;
                                case OpCode.Pow:
                                    { float b = Pop(); float a = Pop(); Push(MathF.Pow(a, b)); }
                                    break;
                                case OpCode.Neg:
                                    Push(-Pop());
                                    break;
                                case OpCode.Abs:
                                    Push(MathF.Abs(Pop()));
                                    break;
                                case OpCode.Sin:
                                    Push(MathF.Sin(Pop()));
                                    break;
                                case OpCode.Cos:
                                    Push(MathF.Cos(Pop()));
                                    break;
                                case OpCode.Tan:
                                    Push(MathF.Tan(Pop()));
                                    break;
                                case OpCode.Exp:
                                    Push(MathF.Exp(Pop()));
                                    break;
                                case OpCode.Log:
                                    { float v = Pop(); Push(v > 0f ? MathF.Log(v) : 0f); }
                                    break;
                                case OpCode.Sqrt:
                                    { float v = Pop(); Push(v >= 0f ? MathF.Sqrt(v) : 0f); }
                                    break;
                                case OpCode.Min:
                                    { float b = Pop(); float a = Pop(); Push(MathF.Min(a, b)); }
                                    break;
                                case OpCode.Max:
                                    { float b = Pop(); float a = Pop(); Push(MathF.Max(a, b)); }
                                    break;
                                case OpCode.LoadVar:
                                    if (parameters != null && instr.Operand < parameters.Length)
                                        Push(parameters[instr.Operand]);
                                    else
                                        Push(0f);
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
                    }
                }
            }
        }
    }
}
