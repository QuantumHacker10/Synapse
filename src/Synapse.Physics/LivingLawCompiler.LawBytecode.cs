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

    /// <summary>Compiled bytecode for a law expression.</summary>
    public sealed class LawBytecode
    {
        private Instruction[] _instructions;
        private float[] _constants;
        private string[] _variableNames;
        private string[] _fieldNames;
        private string[] _paramNames;
        private int _instructionCount;

        public ReadOnlySpan<Instruction> Instructions => _instructions.AsSpan(0, _instructionCount);
        public ReadOnlySpan<float> Constants => _constants;
        public ReadOnlySpan<string> VariableNames => _variableNames;
        public ReadOnlySpan<string> FieldNames => _fieldNames;
        public ReadOnlySpan<string> ParamNames => _paramNames;
        public int InstructionCount => _instructionCount;
        public int StackDepth { get; set; }
        public Dimension ResultDimension { get; set; } = Dimension.Scalar;
        public string OriginalExpression { get; set; } = "";

        public LawBytecode(int maxInstructions = 4096, int maxConstants = 256,
            int maxVars = 64, int maxFields = 32, int maxParams = 32)
        {
            _instructions = new Instruction[maxInstructions];
            _constants = new float[maxConstants];
            _variableNames = new string[maxVars];
            _fieldNames = new string[maxFields];
            _paramNames = new string[maxParams];
            _instructionCount = 0;
        }

        public int AddInstruction(OpCode op, int operand = 0, float floatOperand = 0f)
        {
            if (_instructionCount >= _instructions.Length)
            {
                var newArr = new Instruction[_instructions.Length * 2];
                Array.Copy(_instructions, newArr, _instructions.Length);
                _instructions = newArr;
            }
            _instructions[_instructionCount] = new Instruction(op, operand, floatOperand);
            return _instructionCount++;
        }

        public int AddConstant(float value)
        {
            for (int i = 0; i < _constants.Length; i++)
            {
                if (_constants[i] == value)
                    return i;
            }
            for (int i = 0; i < _constants.Length; i++)
            {
                if (_constants[i] == 0f)
                {
                    _constants[i] = value;
                    return i;
                }
            }
            return -1;
        }

        public int AddVariable(string name)
        {
            for (int i = 0; i < _variableNames.Length; i++)
            {
                if (_variableNames[i] == name)
                    return i;
                if (_variableNames[i] == null)
                { _variableNames[i] = name; return i; }
            }
            return -1;
        }

        public int AddField(string name)
        {
            for (int i = 0; i < _fieldNames.Length; i++)
            {
                if (_fieldNames[i] == name)
                    return i;
                if (_fieldNames[i] == null)
                { _fieldNames[i] = name; return i; }
            }
            return -1;
        }

        public int AddParam(string name)
        {
            for (int i = 0; i < _paramNames.Length; i++)
            {
                if (_paramNames[i] == name)
                    return i;
                if (_paramNames[i] == null)
                { _paramNames[i] = name; return i; }
            }
            return -1;
        }

        public void PatchInstruction(int index, int operand)
        {
            if (index >= 0 && index < _instructionCount)
            {
                var instr = _instructions[index];
                _instructions[index] = new Instruction(instr.Op, operand, instr.FloatOperand);
            }
        }

        public LawBytecode Clone()
        {
            var clone = new LawBytecode(_instructions.Length, _constants.Length,
                _variableNames.Length, _fieldNames.Length, _paramNames.Length);
            Array.Copy(_instructions, clone._instructions, _instructionCount);
            clone._instructionCount = _instructionCount;
            Array.Copy(_constants, clone._constants, _constants.Length);
            Array.Copy(_variableNames, clone._variableNames, _variableNames.Length);
            Array.Copy(_fieldNames, clone._fieldNames, _fieldNames.Length);
            Array.Copy(_paramNames, clone._paramNames, _paramNames.Length);
            clone.StackDepth = StackDepth;
            clone.ResultDimension = ResultDimension;
            clone.OriginalExpression = OriginalExpression;
            return clone;
        }
    }
}
