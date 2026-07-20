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
    // =========================================================================
    // Supporting types
    // =========================================================================

    /// <summary>Physical dimension exponents for dimensional analysis.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Dimension : IEquatable<Dimension>
    {
        public readonly float Mass;
        public readonly float Length;
        public readonly float Time;
        public readonly float Temperature;
        public readonly float Amount;
        public readonly float Current;
        public readonly float Luminous;
        public readonly float Information;

        public Dimension(float mass, float length, float time, float temperature,
            float amount, float current, float luminous, float information)
        {
            Mass = mass;
            Length = length;
            Time = time;
            Temperature = temperature;
            Amount = amount;
            Current = current;
            Luminous = luminous;
            Information = information;
        }

        public static readonly Dimension Scalar = new(0, 0, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension LengthD = new(0, 1, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension TimeD = new(0, 0, 1, 0, 0, 0, 0, 0);
        public static readonly Dimension MassD = new(1, 0, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension Velocity = new(0, 1, -1, 0, 0, 0, 0, 0);
        public static readonly Dimension Acceleration = new(0, 1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Force = new(1, 1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Energy = new(1, 2, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Power = new(1, 2, -3, 0, 0, 0, 0, 0);
        public static readonly Dimension Pressure = new(1, -1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension TemperatureD = new(0, 0, 0, 1, 0, 0, 0, 0);
        public static readonly Dimension Density = new(1, -3, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension Viscosity = new(1, -1, -1, 0, 0, 0, 0, 0);

        public Dimension Multiply(Dimension other) => new(
            Mass + other.Mass, Length + other.Length, Time + other.Time,
            Temperature + other.Temperature, Amount + other.Amount,
            Current + other.Current, Luminous + other.Luminous, Information + other.Information);

        public Dimension Divide(Dimension other) => new(
            Mass - other.Mass, Length - other.Length, Time - other.Time,
            Temperature - other.Temperature, Amount - other.Amount,
            Current - other.Current, Luminous - other.Luminous, Information - other.Information);

        public Dimension Pow(float exp) => new(
            Mass * exp, Length * exp, Time * exp, Temperature * exp,
            Amount * exp, Current * exp, Luminous * exp, Information * exp);

        public bool IsCompatible(Dimension other) =>
            MathF.Abs(Mass - other.Mass) < 1e-6f && MathF.Abs(Length - other.Length) < 1e-6f &&
            MathF.Abs(Time - other.Time) < 1e-6f && MathF.Abs(Temperature - other.Temperature) < 1e-6f &&
            MathF.Abs(Amount - other.Amount) < 1e-6f && MathF.Abs(Current - other.Current) < 1e-6f &&
            MathF.Abs(Luminous - other.Luminous) < 1e-6f && MathF.Abs(Information - other.Information) < 1e-6f;

        public bool IsDimensionless => MathF.Abs(Mass) < 1e-6f && MathF.Abs(Length) < 1e-6f &&
            MathF.Abs(Time) < 1e-6f && MathF.Abs(Temperature) < 1e-6f && MathF.Abs(Amount) < 1e-6f &&
            MathF.Abs(Current) < 1e-6f && MathF.Abs(Luminous) < 1e-6f && MathF.Abs(Information) < 1e-6f;

        public bool Equals(Dimension other) => IsCompatible(other);
        public override bool Equals(object? obj) => obj is Dimension d && Equals(d);
        public override int GetHashCode() => HashCode.Combine(Mass, Length, Time, Temperature, Amount, Current, Luminous, Information);
        public override string ToString() => $"[M^{Mass} L^{Length} T^{Time} Θ^{Temperature} N^{Amount} I^{Current} J^{Luminous}]";

        public static Dimension operator +(Dimension a, Dimension b) => a.Multiply(b);
        public static Dimension operator -(Dimension a, Dimension b) => a.Divide(b);
        public static bool operator ==(Dimension a, Dimension b) => a.Equals(b);
        public static bool operator !=(Dimension a, Dimension b) => !a.Equals(b);
    }

    /// <summary>Variable types for the expression system.</summary>
    public enum VariableType
    {
        Scalar = 0, Vector2 = 1, Vector3 = 2, Field = 3, Tensor = 4
    }

    /// <summary>Represents a typed variable in the expression system.</summary>
    public readonly struct TypedVariable : IEquatable<TypedVariable>
    {
        public readonly string Name;
        public readonly VariableType Type;
        public readonly Dimension Dim;

        public TypedVariable(string name, VariableType type, Dimension dim)
        {
            Name = name;
            Type = type;
            Dim = dim;
        }

        public bool Equals(TypedVariable other) => Name == other.Name && Type == other.Type && Dim.Equals(other.Dim);
        public override bool Equals(object? obj) => obj is TypedVariable tv && Equals(tv);
        public override int GetHashCode() => HashCode.Combine(Name, Type, Dim);
        public override string ToString() => $"{Type} {Name} {Dim}";
    }

    /// <summary>Result of a compilation or validation operation.</summary>
    public readonly struct CompilationResult
    {
        public readonly bool Success;
        public readonly string Message;
        public readonly string[] Errors;
        public readonly string[] Warnings;
        public readonly LawBytecode? Bytecode;
        public readonly int InstructionCount;
        public readonly long CompilationTimeMs;

        public CompilationResult(bool success, string message, string[] errors, string[] warnings,
            LawBytecode? bytecode, int instructionCount, long compilationTimeMs)
        {
            Success = success;
            Message = message;
            Errors = errors;
            Warnings = warnings;
            Bytecode = bytecode;
            InstructionCount = instructionCount;
            CompilationTimeMs = compilationTimeMs;
        }

        public static CompilationResult Ok(string msg, LawBytecode bc, int ins, long ms) =>
            new(true, msg, Array.Empty<string>(), Array.Empty<string>(), bc, ins, ms);
        public static CompilationResult Fail(string msg, string[] errors) =>
            new(false, msg, errors, Array.Empty<string>(), null, 0, 0);
    }

    /// <summary>Result of a law validation.</summary>
    public readonly struct ValidationResult
    {
        public readonly bool IsValid;
        public readonly string[] Errors;
        public readonly string[] Warnings;
        public readonly Dimension[] TermDimensions;
        public readonly bool DimensionallyConsistent;
        public readonly float StabilityCflRatio;

        public ValidationResult(bool isValid, string[] errors, string[] warnings,
            Dimension[] termDimensions, bool dimensionallyConsistent, float stabilityCflRatio)
        {
            IsValid = isValid;
            Errors = errors;
            Warnings = warnings;
            TermDimensions = termDimensions;
            DimensionallyConsistent = dimensionallyConsistent;
            StabilityCflRatio = stabilityCflRatio;
        }

        public static ValidationResult Valid() =>
            new(true, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<Dimension>(), true, 0f);
    }

    /// <summary>Comparison result between two law versions.</summary>
    public readonly struct ComparisonResult
    {
        public readonly float MaxDivergence;
        public readonly float MeanDivergence;
        public readonly float RootMeanSquareError;
        public readonly float KolmogorovSmirnovStatistic;
        public readonly float StructuralSimilarity;
        public readonly int ExpressionEditDistance;
        public readonly string[] Differences;
        public readonly bool PhysicallyEquivalent;

        public ComparisonResult(float maxDiv, float meanDiv, float rmse, float ks, float ssim,
            int editDist, string[] diffs, bool physEq)
        {
            MaxDivergence = maxDiv;
            MeanDivergence = meanDiv;
            RootMeanSquareError = rmse;
            KolmogorovSmirnovStatistic = ks;
            StructuralSimilarity = ssim;
            ExpressionEditDistance = editDist;
            Differences = diffs;
            PhysicallyEquivalent = physEq;
        }
    }

    // =========================================================================
    // LawLibrary — repository of physical laws
    // =========================================================================

    /// <summary>Entry in the law library.</summary>
    public sealed class LawEntry
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Category { get; set; } = "";
        public string Expression { get; set; } = "";
        public Dictionary<string, string> Parameters { get; set; } = new();
        public Dictionary<string, float> Constants { get; set; } = new();
        public Dimension ResultDimension { get; set; } = Dimension.Scalar;
        public string Description { get; set; } = "";
    }

    /// <summary>Repository of physical laws that can be loaded and compiled.</summary>
    public sealed class LawLibrary
    {
        private readonly ConcurrentDictionary<string, LawEntry> _entries = new();
        private readonly List<LawEntry> _allEntries = new();

        public IReadOnlyList<LawEntry> AllEntries => _allEntries;

        public void Register(LawEntry entry)
        {
            _entries[entry.Id] = entry;
            lock (_allEntries)
            { _allEntries.Add(entry); }
        }

        public LawEntry? GetLaw(string id) =>
            _entries.TryGetValue(id, out var entry) ? entry : null;

        public IReadOnlyList<LawEntry> SearchByCategory(string category)
        {
            var results = new List<LawEntry>();
            foreach (var e in _allEntries)
                if (string.Equals(e.Category, category, StringComparison.OrdinalIgnoreCase))
                    results.Add(e);
            return results;
        }

        public IReadOnlyList<LawEntry> SearchByName(string query)
        {
            var results = new List<LawEntry>();
            foreach (var e in _allEntries)
                if (e.Name.Contains(query, StringComparison.OrdinalIgnoreCase))
                    results.Add(e);
            return results;
        }

        public static LawLibrary LoadFromJson(string filePath)
        {
            var library = new LawLibrary();
            var json = File.ReadAllText(filePath);
            var entries = JsonSerializer.Deserialize<LawEntry[]>(json);
            if (entries != null)
                foreach (var entry in entries)
                    library.Register(entry);
            return library;
        }

        public static LawLibrary LoadBuiltIn()
        {
            var library = new LawLibrary();
            library.Register(new LawEntry
            {
                Id = "navier_stokes_x",
                Name = "Navier-Stokes (X)",
                Category = "FluidDynamics",
                Expression = "rho * (dv/dt + v·∇v) = -∇P + μ*∇²v + f",
                Description = "Navier-Stokes momentum equation, X component",
                Parameters = new() { { "v", "velocity" }, { "P", "pressure" }, { "rho", "density" }, { "mu", "viscosity" } },
                Constants = new() { { "mu", 1.002e-3f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "heat_equation",
                Name = "Heat Equation",
                Category = "ThermalDynamics",
                Expression = "∂T/∂t = α*∇²T",
                Description = "Heat diffusion equation",
                Parameters = new() { { "T", "temperature" }, { "alpha", "thermal_diffusivity" } },
                Constants = new() { { "alpha", 1.43e-4f } },
                ResultDimension = Dimension.TemperatureD.Divide(Dimension.TimeD)
            });
            library.Register(new LawEntry
            {
                Id = "wave_equation",
                Name = "Wave Equation",
                Category = "WaveDynamics",
                Expression = "∂²u/∂t² = c²*∇²u",
                Description = "Wave propagation equation",
                Parameters = new() { { "u", "displacement" }, { "c", "wave_speed" } },
                Constants = new() { { "c", 340.0f } },
                ResultDimension = Dimension.Acceleration
            });
            library.Register(new LawEntry
            {
                Id = "ideal_gas",
                Name = "Ideal Gas Law",
                Category = "Thermodynamics",
                Expression = "P = ρ*R*T",
                Description = "Ideal gas equation of state",
                Parameters = new() { { "rho", "density" }, { "R", "specific_gas_constant" }, { "T", "temperature" } },
                Constants = new() { { "R", 287.058f } },
                ResultDimension = Dimension.Pressure
            });
            library.Register(new LawEntry
            {
                Id = "fourier_law",
                Name = "Fourier's Law",
                Category = "ThermalDynamics",
                Expression = "q = -k*∇T",
                Description = "Heat conduction equation",
                Parameters = new() { { "k", "thermal_conductivity" }, { "T", "temperature" } },
                Constants = new() { { "k", 205.0f } },
                ResultDimension = Dimension.Power.Divide(new Dimension(0, 2, 0, 0, 0, 0, 0, 0))
            });
            library.Register(new LawEntry
            {
                Id = "coulomb_force",
                Name = "Coulomb Force",
                Category = "Electrodynamics",
                Expression = "F = k_e * q1 * q2 / r²",
                Description = "Electrostatic force between charges",
                Parameters = new() { { "q1", "charge" }, { "q2", "charge" }, { "r", "distance" } },
                Constants = new() { { "ke", 8.9875517923e9f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "gravity_newton",
                Name = "Newtonian Gravity",
                Category = "Gravitation",
                Expression = "F = G * m1 * m2 / r²",
                Description = "Newton's law of universal gravitation",
                Parameters = new() { { "m1", "mass" }, { "m2", "mass" }, { "r", "distance" } },
                Constants = new() { { "G", 6.674e-11f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "hooke_law",
                Name = "Hooke's Law",
                Category = "Elasticity",
                Expression = "F = -k * Δx",
                Description = "Linear elastic restoring force",
                Parameters = new() { { "k", "spring_constant" }, { "x", "displacement" } },
                Constants = new() { { "k", 100.0f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "ohms_law",
                Name = "Ohm's Law",
                Category = "Electrodynamics",
                Expression = "V = I * R",
                Description = "Voltage-current relationship",
                Parameters = new() { { "I", "current" }, { "R", "resistance" } },
                Constants = new() { { "R", 1.0f } },
                ResultDimension = new Dimension(1, 2, -3, 0, 0, -1, 0, 0)
            });
            library.Register(new LawEntry
            {
                Id = "stefan_boltzmann",
                Name = "Stefan-Boltzmann Law",
                Category = "ThermalDynamics",
                Expression = "j = σ * T⁴",
                Description = "Thermal radiation power",
                Parameters = new() { { "T", "temperature" } },
                Constants = new() { { "sigma", 5.670374419e-8f } },
                ResultDimension = new Dimension(1, 0, -3, 0, 0, 0, 0, 0)
            });
            return library;
        }
    }

    // =========================================================================
    // FieldGrid — simple physics field grid
    // =========================================================================

    /// <summary>Represents a 3D field grid for physics simulation.</summary>
    public sealed class FieldGrid
    {
        public int SizeX { get; }
        public int SizeY { get; }
        public int SizeZ { get; }
        public float CellSize { get; }
        private float[] _data;

        public FieldGrid(int sx, int sy, int sz, float cellSize = 1.0f)
        {
            SizeX = sx;
            SizeY = sy;
            SizeZ = sz;
            CellSize = cellSize;
            _data = new float[sx * sy * sz];
        }

        public float this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[x + SizeX * (y + SizeY * z)];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[x + SizeX * (y + SizeY * z)] = value;
        }

        public float[] Data => _data;
        public int TotalCells => SizeX * SizeY * SizeZ;
        public void Clear() => Array.Clear(_data, 0, _data.Length);

        public void CopyFrom(FieldGrid other)
        {
            if (other.TotalCells != TotalCells)
                throw new ArgumentException("Grid sizes mismatch");
            Array.Copy(other._data, _data, _data.Length);
        }

        public float Max() => _data.Length > 0 ? _data.AsSpan().ToArray().Max() : 0f;
        public float Min() => _data.Length > 0 ? _data.AsSpan().ToArray().Min() : 0f;

        public float Average()
        {
            if (_data.Length == 0)
                return 0f;
            double sum = 0;
            for (int i = 0; i < _data.Length; i++)
                sum += _data[i];
            return (float)(sum / _data.Length);
        }

        public float StandardDeviation()
        {
            if (_data.Length == 0)
                return 0f;
            float avg = Average();
            double variance = 0;
            for (int i = 0; i < _data.Length; i++)
            {
                double diff = _data[i] - avg;
                variance += diff * diff;
            }
            return MathF.Sqrt((float)(variance / _data.Length));
        }

        public void AddScaled(FieldGrid other, float scale)
        {
            if (other.TotalCells != TotalCells)
                throw new ArgumentException("Grid sizes mismatch");
            for (int i = 0; i < _data.Length; i++)
                _data[i] += other._data[i] * scale;
        }

        public FieldGrid Clone()
        {
            var clone = new FieldGrid(SizeX, SizeY, SizeZ, CellSize);
            Array.Copy(_data, clone._data, _data.Length);
            return clone;
        }
    }

    /// <summary>A physics field with multiple grid components.</summary>
    public sealed class PhysicsField
    {
        public string Name { get; set; } = "";
        public Dictionary<string, FieldGrid> Components { get; } = new();
        public FieldGrid Temperature { get; set; } = null!;
        public FieldGrid Pressure { get; set; } = null!;
        public FieldGrid Density { get; set; } = null!;
        public FieldGrid VelocityX { get; set; } = null!;
        public FieldGrid VelocityY { get; set; } = null!;
        public FieldGrid VelocityZ { get; set; } = null!;
        public int GridSize { get; }
        public float Time { get; set; }

        public PhysicsField(int gridSize, string name = "default")
        {
            Name = name;
            GridSize = gridSize;
            Temperature = new FieldGrid(gridSize, gridSize, gridSize);
            Pressure = new FieldGrid(gridSize, gridSize, gridSize);
            Density = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityX = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityY = new FieldGrid(gridSize, gridSize, gridSize);
            VelocityZ = new FieldGrid(gridSize, gridSize, gridSize);
        }

        public FieldGrid GetComponent(string name) => name switch
        {
            "T" or "temperature" or "Temperature" => Temperature,
            "P" or "pressure" or "Pressure" => Pressure,
            "rho" or "density" or "Density" => Density,
            "vx" or "VelocityX" or "u" => VelocityX,
            "vy" or "VelocityY" or "v" => VelocityY,
            "vz" or "VelocityZ" or "w" => VelocityZ,
            _ => Components.TryGetValue(name, out var comp) ? comp : Temperature
        };

        public PhysicsField Clone()
        {
            var clone = new PhysicsField(GridSize, Name + "_clone");
            clone.Temperature = Temperature.Clone();
            clone.Pressure = Pressure.Clone();
            clone.Density = Density.Clone();
            clone.VelocityX = VelocityX.Clone();
            clone.VelocityY = VelocityY.Clone();
            clone.VelocityZ = VelocityZ.Clone();
            clone.Time = Time;
            foreach (var kv in Components)
                clone.Components[kv.Key] = kv.Value.Clone();
            return clone;
        }
    }
    // =========================================================================
    // LawBytecode — stack-based bytecode for compiled laws
    // =========================================================================

    /// <summary>Opcodes for the bytecode VM.</summary>
    public enum OpCode : byte
    {
        Nop = 0,
        PushConst = 1, Pop = 2, Dup = 3, Swap = 4,
        Add = 10, Sub = 11, Mul = 12, Div = 13, Mod = 14, Pow = 15, Neg = 16, Abs = 17,
        LoadVar = 30, StoreVar = 31, LoadField = 32, LoadParam = 33,
        Equals = 40, NotEquals = 41, LessThan = 42, GreaterThan = 43,
        LessOrEqual = 44, GreaterOrEqual = 45, LogicalAnd = 46, LogicalOr = 47,
        LogicalNot = 48, TernaryJump = 49,
        Sin = 60, Cos = 61, Tan = 62, Asin = 63, Acos = 64, Atan = 65, Atan2 = 66,
        Sinh = 67, Cosh = 68, Tanh = 69, Exp = 70, Log = 71, Log2 = 72, Log10 = 73,
        Sqrt = 74, Cbrt = 75, Ceil = 76, Floor = 77, Round = 78, Clamp = 79, Lerp = 80,
        Min = 81, Max = 82, Sign = 83,
        GradientX = 90, GradientY = 91, GradientZ = 92, Laplacian = 93,
        Divergence = 94, CurlX = 95, CurlY = 96, CurlZ = 97,
        ConditionalJump = 100, UnconditionalJump = 101, Call = 102, Return = 103,
        GasConsume = 110, BoundsCheck = 111,
        Halt = 255
    }

    /// <summary>A single bytecode instruction.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Instruction
    {
        public readonly OpCode Op;
        public readonly int Operand;
        public readonly float FloatOperand;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Instruction(OpCode op, int operand = 0, float floatOperand = 0f)
        {
            Op = op;
            Operand = operand;
            FloatOperand = floatOperand;
        }

        public override string ToString() => Op switch
        {
            OpCode.PushConst => $"PushConst({FloatOperand})",
            OpCode.LoadVar => $"LoadVar({Operand})",
            OpCode.LoadField => $"LoadField({Operand})",
            OpCode.LoadParam => $"LoadParam({Operand})",
            OpCode.ConditionalJump => $"CondJump({Operand})",
            OpCode.UnconditionalJump => $"Jump({Operand})",
            OpCode.GasConsume => $"Gas({Operand})",
            OpCode.BoundsCheck => $"BoundsCheck({Operand})",
            OpCode.Call => $"Call({Operand})",
            _ => Op.ToString()
        };
    }

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

    // =========================================================================
    // BytecodeInterpreter — stack-based VM for executing bytecode
    // =========================================================================

    /// <summary>Gas metering configuration for preventing infinite loops.</summary>
    public sealed class GasMeter
    {
        private long _gasRemaining;
        public long GasRemaining => _gasRemaining;
        public long MaxGas { get; }
        public long GasUsed => MaxGas - _gasRemaining;

        public GasMeter(long maxGas = 1_000_000)
        {
            MaxGas = maxGas;
            _gasRemaining = maxGas;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Consume(int amount)
        {
            _gasRemaining -= amount;
            return _gasRemaining >= 0;
        }

        public void Reset() => _gasRemaining = MaxGas;
    }

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
    // =========================================================================
    // LawExpressionParser — complete expression parser
    // =========================================================================

    /// <summary>Token types for the expression parser.</summary>
    public enum TokenType
    {
        Number, Identifier,
        Plus, Minus, Star, Slash, Percent, Caret,
        LeftParen, RightParen, LeftBracket, RightBracket,
        Comma, Semicolon, Dot,
        Equals, NotEquals, Less, Greater, LessOrEqual, GreaterOrEqual,
        And, Or, Not,
        Question, Colon, Assign,
        Eof, Error
    }

    /// <summary>A parsed token.</summary>
    public readonly struct Token
    {
        public readonly TokenType Type;
        public readonly string Value;
        public readonly int Line;
        public readonly int Column;
        public Token(TokenType type, string value, int line = 0, int col = 0)
        { Type = type; Value = value; Line = line; Column = col; }
    }

    /// <summary>AST node types.</summary>
    public enum NodeType
    {
        NumberLiteral, Identifier, BinaryExpression, UnaryExpression,
        TernaryExpression, FunctionCall, FieldAccess, IndexAccess,
        Assignment, Block, VariableDeclaration
    }

    /// <summary>AST node for the parsed expression.</summary>
    public sealed class AstNode
    {
        public NodeType Type { get; init; }
        public string? Value { get; init; }
        public float NumericValue { get; init; }
        public AstNode? Left { get; init; }
        public AstNode? Right { get; init; }
        public AstNode? Middle { get; init; }
        public List<AstNode>? Children { get; init; }
        public Dimension InferredDimension { get; set; } = Dimension.Scalar;
    }

    /// <summary>Complete expression parser supporting arithmetic, functions, field access, ternary, etc.</summary>
    public sealed class LawExpressionParser
    {
        private readonly string _input;
        private int _pos;
        private int _line;
        private int _col;
        private Token _current;
        private readonly List<string> _errors = new();
        private readonly List<string> _warnings = new();

        private readonly Dictionary<string, TypedVariable> _knownVariables = new()
        {
            ["x"] = new("x", VariableType.Scalar, Dimension.LengthD),
            ["y"] = new("y", VariableType.Scalar, Dimension.LengthD),
            ["z"] = new("z", VariableType.Scalar, Dimension.LengthD),
            ["t"] = new("t", VariableType.Scalar, Dimension.TimeD),
            ["T"] = new("T", VariableType.Scalar, Dimension.TemperatureD),
            ["P"] = new("P", VariableType.Scalar, Dimension.Pressure),
            ["rho"] = new("rho", VariableType.Scalar, Dimension.Density),
            ["v"] = new("v", VariableType.Scalar, Dimension.Velocity),
            ["u"] = new("u", VariableType.Scalar, Dimension.Velocity),
            ["w"] = new("w", VariableType.Scalar, Dimension.Velocity),
            ["F"] = new("F", VariableType.Scalar, Dimension.Force),
            ["E"] = new("E", VariableType.Scalar, Dimension.Energy),
            ["k"] = new("k", VariableType.Scalar, new Dimension(0, 2, -1, 0, 0, 0, 0, 0)),
            ["mu"] = new("mu", VariableType.Scalar, Dimension.Viscosity),
            ["alpha"] = new("alpha", VariableType.Scalar, new Dimension(0, 2, -1, 0, 0, 0, 0, 0)),
            ["sigma"] = new("sigma", VariableType.Scalar, new Dimension(1, 0, -3, -4, 0, 0, 0, 0)),
            ["G"] = new("G", VariableType.Scalar, new Dimension(-1, 3, -2, 0, 0, 0, 0, 0)),
            ["R"] = new("R", VariableType.Scalar, new Dimension(0, 2, -2, -1, 0, 0, 0, 0)),
            ["c"] = new("c", VariableType.Scalar, Dimension.Velocity),
            ["I"] = new("I", VariableType.Scalar, new Dimension(0, 0, 0, 0, 0, 1, 0, 0)),
            ["V"] = new("V", VariableType.Scalar, new Dimension(1, 2, -3, 0, 0, -1, 0, 0)),
            ["q"] = new("q", VariableType.Scalar, new Dimension(0, 2, -3, 0, 0, 0, 0, 0)),
            ["dt"] = new("dt", VariableType.Scalar, Dimension.TimeD),
            ["dx"] = new("dx", VariableType.Scalar, Dimension.LengthD),
            ["dy"] = new("dy", VariableType.Scalar, Dimension.LengthD),
            ["dz"] = new("dz", VariableType.Scalar, Dimension.LengthD),
            ["m1"] = new("m1", VariableType.Scalar, Dimension.MassD),
            ["m2"] = new("m2", VariableType.Scalar, Dimension.MassD),
            ["r"] = new("r", VariableType.Scalar, Dimension.LengthD),
            ["q1"] = new("q1", VariableType.Scalar, new Dimension(0, 0, 0, 0, 0, 1, 0, 0)),
            ["q2"] = new("q2", VariableType.Scalar, new Dimension(0, 0, 0, 0, 0, 1, 0, 0)),
        };

        public IReadOnlyList<string> Errors => _errors;
        public IReadOnlyList<string> Warnings => _warnings;

        public LawExpressionParser(string input)
        {
            _input = input ?? throw new ArgumentNullException(nameof(input));
            _pos = 0;
            _line = 1;
            _col = 1;
            _current = NextToken();
        }

        private char PeekChar(int offset = 0)
        {
            int idx = _pos + offset;
            return idx < _input.Length ? _input[idx] : '\0';
        }

        private char Advance()
        {
            char c = _input[_pos++];
            if (c == '\n')
            { _line++; _col = 1; }
            else
                _col++;
            return c;
        }

        private Token MakeTwoCharToken(TokenType type, string value, int line, int col)
        {
            Advance();
            return new Token(type, value, line, col);
        }

        private void SkipWhitespace()
        {
            while (_pos < _input.Length && char.IsWhiteSpace(_input[_pos]))
                Advance();
        }

        private Token NextToken()
        {
            SkipWhitespace();
            if (_pos >= _input.Length)
                return new Token(TokenType.Eof, "", _line, _col);

            int startLine = _line, startCol = _col;
            char c = PeekChar();

            if (char.IsDigit(c) || (c == '.' && _pos + 1 < _input.Length && char.IsDigit(PeekChar(1))))
            {
                var sb = new StringBuilder();
                while (_pos < _input.Length && (char.IsDigit(PeekChar()) || PeekChar() == '.' ||
                    PeekChar() == 'e' || PeekChar() == 'E' || PeekChar() == '+' || PeekChar() == '-'))
                {
                    if ((PeekChar() == '+' || PeekChar() == '-') && sb.Length > 0 && sb[^1] != 'e' && sb[^1] != 'E')
                        break;
                    sb.Append(Advance());
                }
                return new Token(TokenType.Number, sb.ToString(), startLine, startCol);
            }

            if (char.IsLetter(c) || c == '_')
            {
                var sb = new StringBuilder();
                while (_pos < _input.Length && (char.IsLetterOrDigit(PeekChar()) || PeekChar() == '_'))
                    sb.Append(Advance());
                string word = sb.ToString();
                return word switch
                {
                    "and" or "&&" => new Token(TokenType.And, word, startLine, startCol),
                    "or" or "||" => new Token(TokenType.Or, word, startLine, startCol),
                    "not" or "!" => new Token(TokenType.Not, word, startLine, startCol),
                    _ => new Token(TokenType.Identifier, word, startLine, startCol)
                };
            }

            Advance();
            return c switch
            {
                '+' => new Token(TokenType.Plus, "+", startLine, startCol),
                '-' => new Token(TokenType.Minus, "-", startLine, startCol),
                '*' => new Token(TokenType.Star, "*", startLine, startCol),
                '/' => new Token(TokenType.Slash, "/", startLine, startCol),
                '%' => new Token(TokenType.Percent, "%", startLine, startCol),
                '^' => new Token(TokenType.Caret, "^", startLine, startCol),
                '(' => new Token(TokenType.LeftParen, "(", startLine, startCol),
                ')' => new Token(TokenType.RightParen, ")", startLine, startCol),
                '[' => new Token(TokenType.LeftBracket, "[", startLine, startCol),
                ']' => new Token(TokenType.RightBracket, "]", startLine, startCol),
                ',' => new Token(TokenType.Comma, ",", startLine, startCol),
                ';' => new Token(TokenType.Semicolon, ";", startLine, startCol),
                '.' => new Token(TokenType.Dot, ".", startLine, startCol),
                '=' when PeekChar() == '=' => MakeTwoCharToken(TokenType.Equals, "==", startLine, startCol),
                '=' => new Token(TokenType.Assign, "=", startLine, startCol),
                '!' when PeekChar() == '=' => MakeTwoCharToken(TokenType.NotEquals, "!=", startLine, startCol),
                '<' when PeekChar() == '=' => MakeTwoCharToken(TokenType.LessOrEqual, "<=", startLine, startCol),
                '<' => new Token(TokenType.Less, "<", startLine, startCol),
                '>' when PeekChar() == '=' => MakeTwoCharToken(TokenType.GreaterOrEqual, ">=", startLine, startCol),
                '>' => new Token(TokenType.Greater, ">", startLine, startCol),
                '&' when PeekChar() == '&' => MakeTwoCharToken(TokenType.And, "&&", startLine, startCol),
                '|' when PeekChar() == '|' => MakeTwoCharToken(TokenType.Or, "||", startLine, startCol),
                '?' => new Token(TokenType.Question, "?", startLine, startCol),
                ':' => new Token(TokenType.Colon, ":", startLine, startCol),
                _ => new Token(TokenType.Error, c.ToString(), startLine, startCol)
            };
        }

        private void Match(TokenType type)
        {
            if (_current.Type == type)
                _current = NextToken();
            else
                _errors.Add($"Expected {type} at line {_line}:{_col}, got {_current.Type}('{_current.Value}')");
        }

        private bool Check(TokenType type) => _current.Type == type;
        private bool Check(TokenType t1, TokenType t2) => _current.Type == t1 || _current.Type == t2;
        private bool Check(TokenType t1, TokenType t2, TokenType t3) => _current.Type == t1 || _current.Type == t2 || _current.Type == t3;
        private bool Check(TokenType t1, TokenType t2, TokenType t3, TokenType t4) =>
            _current.Type == t1 || _current.Type == t2 || _current.Type == t3 || _current.Type == t4;

        private bool MatchOptional(TokenType type)
        {
            if (_current.Type == type)
            { _current = NextToken(); return true; }
            return false;
        }

        private AstNode ParseExpression() => ParseTernary();

        private AstNode ParseTernary()
        {
            var condition = ParseOr();
            if (Check(TokenType.Question))
            {
                Match(TokenType.Question);
                var trueExpr = ParseExpression();
                Match(TokenType.Colon);
                var falseExpr = ParseExpression();
                return new AstNode { Type = NodeType.TernaryExpression, Left = condition, Right = trueExpr, Middle = falseExpr };
            }
            return condition;
        }

        private AstNode ParseOr()
        {
            var left = ParseAnd();
            while (Check(TokenType.Or))
            {
                Match(TokenType.Or);
                var right = ParseAnd();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = "||", Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseAnd()
        {
            var left = ParseEquality();
            while (Check(TokenType.And))
            {
                Match(TokenType.And);
                var right = ParseEquality();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = "&&", Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseEquality()
        {
            var left = ParseComparison();
            while (Check(TokenType.Equals, TokenType.NotEquals))
            {
                string op = _current.Value;
                Match(_current.Type);
                var right = ParseComparison();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = op, Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseComparison()
        {
            var left = ParseAddSub();
            while (Check(TokenType.Less, TokenType.Greater, TokenType.LessOrEqual, TokenType.GreaterOrEqual))
            {
                string op = _current.Value;
                Match(_current.Type);
                var right = ParseAddSub();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = op, Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseAddSub()
        {
            var left = ParseMulDiv();
            while (Check(TokenType.Plus, TokenType.Minus))
            {
                string op = _current.Value;
                Match(_current.Type);
                var right = ParseMulDiv();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = op, Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseMulDiv()
        {
            var left = ParseUnary();
            while (Check(TokenType.Star, TokenType.Slash, TokenType.Percent))
            {
                string op = _current.Value;
                Match(_current.Type);
                var right = ParseUnary();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = op, Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParseUnary()
        {
            if (Check(TokenType.Minus))
            {
                Match(TokenType.Minus);
                var expr = ParsePower();
                return new AstNode { Type = NodeType.UnaryExpression, Value = "-", Left = expr };
            }
            if (Check(TokenType.Not))
            {
                Match(TokenType.Not);
                var expr = ParsePower();
                return new AstNode { Type = NodeType.UnaryExpression, Value = "!", Left = expr };
            }
            return ParsePower();
        }

        private AstNode ParsePower()
        {
            var left = ParsePrimary();
            if (Check(TokenType.Caret))
            {
                Match(TokenType.Caret);
                var right = ParseUnary();
                left = new AstNode { Type = NodeType.BinaryExpression, Value = "^", Left = left, Right = right };
            }
            return left;
        }

        private AstNode ParsePrimary()
        {
            if (Check(TokenType.Number))
            {
                string val = _current.Value;
                Match(TokenType.Number);
                float num = float.Parse(val, CultureInfo.InvariantCulture);
                return new AstNode { Type = NodeType.NumberLiteral, Value = val, NumericValue = num };
            }

            if (Check(TokenType.LeftParen))
            {
                Match(TokenType.LeftParen);
                var expr = ParseExpression();
                Match(TokenType.RightParen);
                return expr;
            }

            if (Check(TokenType.Identifier))
            {
                string name = _current.Value;
                Match(TokenType.Identifier);

                if (Check(TokenType.Dot) && name == "field")
                {
                    Match(TokenType.Dot);
                    string fieldName = _current.Value;
                    Match(TokenType.Identifier);
                    return new AstNode
                    {
                        Type = NodeType.FieldAccess,
                        Value = fieldName,
                        Left = new AstNode { Type = NodeType.Identifier, Value = "field" }
                    };
                }

                if (Check(TokenType.LeftParen))
                {
                    Match(TokenType.LeftParen);
                    var args = new List<AstNode>();
                    while (!Check(TokenType.RightParen) && !Check(TokenType.Eof))
                    {
                        args.Add(ParseExpression());
                        if (!MatchOptional(TokenType.Comma))
                            break;
                    }
                    Match(TokenType.RightParen);
                    return new AstNode { Type = NodeType.FunctionCall, Value = name, Children = args };
                }

                return new AstNode { Type = NodeType.Identifier, Value = name };
            }

            _errors.Add($"Unexpected token '{_current.Value}' at line {_line}:{_col}");
            var errorNode = new AstNode { Type = NodeType.NumberLiteral, Value = "0", NumericValue = 0 };
            _current = NextToken();
            return errorNode;
        }

        private void InferDimensions(AstNode node)
        {
            if (node == null)
                return;
            switch (node.Type)
            {
                case NodeType.NumberLiteral:
                    node.InferredDimension = Dimension.Scalar;
                    break;
                case NodeType.Identifier:
                    if (_knownVariables.TryGetValue(node.Value ?? "", out var tv))
                        node.InferredDimension = tv.Dim;
                    else
                        node.InferredDimension = Dimension.Scalar;
                    break;
                case NodeType.FieldAccess:
                    node.InferredDimension = Dimension.Scalar;
                    break;
                case NodeType.BinaryExpression:
                    InferDimensions(node.Left);
                    InferDimensions(node.Right);
                    node.InferredDimension = node.Value switch
                    {
                        "+" or "-" => node.Left!.InferredDimension,
                        "*" => node.Left!.InferredDimension.Multiply(node.Right!.InferredDimension),
                        "/" => node.Left!.InferredDimension.Divide(node.Right!.InferredDimension),
                        "^" => node.Left!.InferredDimension.Pow(node.Right!.NumericValue),
                        _ => Dimension.Scalar
                    };
                    break;
                case NodeType.UnaryExpression:
                    InferDimensions(node.Left);
                    node.InferredDimension = node.Left!.InferredDimension;
                    break;
                case NodeType.TernaryExpression:
                    InferDimensions(node.Left);
                    InferDimensions(node.Right);
                    InferDimensions(node.Middle);
                    node.InferredDimension = node.Right!.InferredDimension;
                    break;
                case NodeType.FunctionCall:
                    if (node.Children != null)
                        foreach (var child in node.Children)
                            InferDimensions(child);
                    node.InferredDimension = node.Children?.Count > 0 ? node.Children[0].InferredDimension : Dimension.Scalar;
                    break;
                default:
                    node.InferredDimension = Dimension.Scalar;
                    break;
            }
        }

        /// <summary>Parse the input expression into an AST.</summary>
        public AstNode Parse()
        {
            _errors.Clear();
            _warnings.Clear();
            var ast = ParseExpression();
            InferDimensions(ast);
            if (_current.Type != TokenType.Eof)
                _warnings.Add($"Unexpected tokens after expression at line {_line}:{_col}");
            return ast;
        }

        /// <summary>Compile an AST to bytecode.</summary>
        public LawBytecode CompileToBytecode(AstNode? ast = null)
        {
            ast ??= Parse();
            var bytecode = new LawBytecode();
            var varDict = new Dictionary<string, int>();
            var fieldDict = new Dictionary<string, int>();
            var paramMap = new Dictionary<string, int>();
            CompileNode(ast, bytecode, varDict, fieldDict, paramMap);
            bytecode.AddInstruction(OpCode.Return);
            bytecode.ResultDimension = ast.InferredDimension;
            bytecode.OriginalExpression = _input;
            return bytecode;
        }

        private void CompileNode(AstNode node, LawBytecode bytecode,
            Dictionary<string, int> varDict, Dictionary<string, int> fieldDict,
            Dictionary<string, int> paramMap)
        {
            switch (node.Type)
            {
                case NodeType.NumberLiteral:
                    {
                        int cidx = bytecode.AddConstant(node.NumericValue);
                        bytecode.AddInstruction(OpCode.PushConst, cidx, node.NumericValue);
                    }
                    break;
                case NodeType.Identifier:
                    {
                        string name = node.Value ?? "";
                        if (_knownVariables.ContainsKey(name))
                        {
                            if (!varDict.TryGetValue(name, out int vidx))
                            { vidx = bytecode.AddVariable(name); varDict[name] = vidx; }
                            bytecode.AddInstruction(OpCode.LoadVar, vidx);
                        }
                        else
                        {
                            if (!paramMap.TryGetValue(name, out int pidx))
                            { pidx = bytecode.AddParam(name); paramMap[name] = pidx; }
                            bytecode.AddInstruction(OpCode.LoadParam, pidx);
                        }
                    }
                    break;
                case NodeType.FieldAccess:
                    {
                        string fieldName = node.Value ?? "";
                        if (!fieldDict.TryGetValue(fieldName, out int fidx))
                        { fidx = bytecode.AddField(fieldName); fieldDict[fieldName] = fidx; }
                        bytecode.AddInstruction(OpCode.LoadField, fidx);
                    }
                    break;
                case NodeType.BinaryExpression:
                    CompileNode(node.Left!, bytecode, varDict, fieldDict, paramMap);
                    CompileNode(node.Right!, bytecode, varDict, fieldDict, paramMap);
                    bytecode.AddInstruction(node.Value switch
                    {
                        "+" => OpCode.Add,
                        "-" => OpCode.Sub,
                        "*" => OpCode.Mul,
                        "/" => OpCode.Div,
                        "%" => OpCode.Mod,
                        "^" => OpCode.Pow,
                        "==" => OpCode.Equals,
                        "!=" => OpCode.NotEquals,
                        "<" => OpCode.LessThan,
                        ">" => OpCode.GreaterThan,
                        "<=" => OpCode.LessOrEqual,
                        ">=" => OpCode.GreaterOrEqual,
                        "&&" => OpCode.LogicalAnd,
                        "||" => OpCode.LogicalOr,
                        _ => OpCode.Nop
                    });
                    break;
                case NodeType.UnaryExpression:
                    CompileNode(node.Left!, bytecode, varDict, fieldDict, paramMap);
                    if (node.Value == "-")
                        bytecode.AddInstruction(OpCode.Neg);
                    else if (node.Value == "!")
                        bytecode.AddInstruction(OpCode.LogicalNot);
                    break;
                case NodeType.TernaryExpression:
                    {
                        CompileNode(node.Left!, bytecode, varDict, fieldDict, paramMap);
                        int condJumpIdx = bytecode.AddInstruction(OpCode.TernaryJump, 0);
                        CompileNode(node.Right!, bytecode, varDict, fieldDict, paramMap);
                        int uncondJumpIdx = bytecode.AddInstruction(OpCode.UnconditionalJump, 0);
                        bytecode.PatchInstruction(condJumpIdx, bytecode.InstructionCount);
                        CompileNode(node.Middle!, bytecode, varDict, fieldDict, paramMap);
                        bytecode.PatchInstruction(uncondJumpIdx, bytecode.InstructionCount);
                    }
                    break;
                case NodeType.FunctionCall:
                    {
                        string funcName = (node.Value ?? "").ToLowerInvariant();
                        var args = node.Children ?? new();
                        foreach (var arg in args)
                            CompileNode(arg, bytecode, varDict, fieldDict, paramMap);
                        bytecode.AddInstruction(funcName switch
                        {
                            "sin" => OpCode.Sin,
                            "cos" => OpCode.Cos,
                            "tan" => OpCode.Tan,
                            "asin" => OpCode.Asin,
                            "acos" => OpCode.Acos,
                            "atan" => OpCode.Atan,
                            "atan2" => OpCode.Atan2,
                            "sinh" => OpCode.Sinh,
                            "cosh" => OpCode.Cosh,
                            "tanh" => OpCode.Tanh,
                            "exp" => OpCode.Exp,
                            "log" => OpCode.Log,
                            "log2" => OpCode.Log2,
                            "log10" => OpCode.Log10,
                            "sqrt" => OpCode.Sqrt,
                            "cbrt" => OpCode.Cbrt,
                            "ceil" => OpCode.Ceil,
                            "floor" => OpCode.Floor,
                            "round" => OpCode.Round,
                            "abs" => OpCode.Abs,
                            "sign" => OpCode.Sign,
                            "min" => OpCode.Min,
                            "max" => OpCode.Max,
                            "clamp" => OpCode.Clamp,
                            "lerp" => OpCode.Lerp,
                            "grad_x" => OpCode.GradientX,
                            "grad_y" => OpCode.GradientY,
                            "grad_z" => OpCode.GradientZ,
                            "laplacian" => OpCode.Laplacian,
                            "divergence" => OpCode.Divergence,
                            "curl_x" => OpCode.CurlX,
                            "curl_y" => OpCode.CurlY,
                            "curl_z" => OpCode.CurlZ,
                            _ => OpCode.Nop
                        });
                    }
                    break;
                case NodeType.Block:
                    if (node.Children != null)
                        foreach (var child in node.Children)
                            CompileNode(child, bytecode, varDict, fieldDict, paramMap);
                    break;
                default:
                    break;
            }
        }
    }
    // =========================================================================
    // LawVersionTree — version tree for law modifications
    // =========================================================================

    /// <summary>A node in the law version tree.</summary>
    public sealed class LawVersionNode
    {
        public string VersionId { get; set; } = Guid.NewGuid().ToString("N")[..8];
        public string Expression { get; set; } = "";
        public string Description { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public LawVersionNode? Parent { get; set; }
        public List<LawVersionNode> Children { get; } = new();
        public LawBytecode? CompiledBytecode { get; set; }
        public bool IsActive { get; set; }
        public Dictionary<string, string> Metadata { get; } = new();
    }

    /// <summary>Tree structure for managing law versions, forks, merges, and diffs.</summary>
    public sealed class LawVersionTree
    {
        private readonly LawVersionNode _root;
        private readonly Dictionary<string, LawVersionNode> _nodes = new();
        private LawVersionNode _current;
        private int _versionCounter;

        public LawVersionNode Root => _root;
        public LawVersionNode Current => _current;
        public int VersionCount => _nodes.Count;

        public LawVersionTree(string initialExpression)
        {
            _root = new LawVersionNode
            {
                VersionId = "v0",
                Expression = initialExpression,
                Description = "Initial version",
                Timestamp = DateTime.UtcNow,
                IsActive = true
            };
            _nodes[_root.VersionId] = _root;
            _current = _root;
            _versionCounter = 1;
        }

        /// <summary>Create a new version from a modified expression.</summary>
        public LawVersionNode Commit(string expression, string description = "")
        {
            var node = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}",
                Expression = expression,
                Description = description,
                Timestamp = DateTime.UtcNow,
                Parent = _current
            };
            _current.Children.Add(node);
            _nodes[node.VersionId] = node;
            _current = node;
            return node;
        }

        /// <summary>Fork from the current version to create a branch.</summary>
        public LawVersionNode Fork(string expression, string branchName = "")
        {
            var node = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}_fork",
                Expression = expression,
                Description = $"Fork: {branchName}",
                Timestamp = DateTime.UtcNow,
                Parent = _current
            };
            _current.Children.Add(node);
            _nodes[node.VersionId] = node;
            return node;
        }

        /// <summary>Merge two version branches.</summary>
        public LawVersionNode Merge(string sourceId, string targetId, string mergedExpression)
        {
            if (!_nodes.TryGetValue(sourceId, out var source))
                throw new ArgumentException($"Source version {sourceId} not found");
            if (!_nodes.TryGetValue(targetId, out var target))
                throw new ArgumentException($"Target version {targetId} not found");

            var mergeNode = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}_merge",
                Expression = mergedExpression,
                Description = $"Merge {sourceId} into {targetId}",
                Timestamp = DateTime.UtcNow,
                Parent = target
            };
            target.Children.Add(mergeNode);
            _nodes[mergeNode.VersionId] = mergeNode;
            _current = mergeNode;
            return mergeNode;
        }

        /// <summary>Rollback to a specific version.</summary>
        public LawVersionNode Rollback(string versionId)
        {
            if (!_nodes.TryGetValue(versionId, out var node))
                throw new ArgumentException($"Version {versionId} not found");
            _current = node;
            return node;
        }

        /// <summary>Rollback to a specific version by index in the path from root.</summary>
        public LawVersionNode RollbackToIndex(int index)
        {
            var path = GetPath(_current);
            if (index < 0 || index >= path.Count)
                throw new ArgumentOutOfRangeException(nameof(index));
            _current = path[index];
            return _current;
        }

        /// <summary>Get the path from root to a specific node.</summary>
        public List<LawVersionNode> GetPath(LawVersionNode? target = null)
        {
            target ??= _current;
            var path = new List<LawVersionNode>();
            var node = target;
            while (node != null)
            { path.Add(node); node = node.Parent; }
            path.Reverse();
            return path;
        }

        /// <summary>Get all versions in the tree (breadth-first).</summary>
        public List<LawVersionNode> GetAllVersions()
        {
            var result = new List<LawVersionNode>();
            var queue = new Queue<LawVersionNode>();
            queue.Enqueue(_root);
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                result.Add(node);
                foreach (var child in node.Children)
                    queue.Enqueue(child);
            }
            return result;
        }

        /// <summary>Compute Levenshtein edit distance between two strings.</summary>
        public static int ComputeEditDistance(string a, string b)
        {
            int m = a.Length, n = b.Length;
            var dp = new int[m + 1, n + 1];
            for (int i = 0; i <= m; i++)
                dp[i, 0] = i;
            for (int j = 0; j <= n; j++)
                dp[0, j] = j;
            for (int i = 1; i <= m; i++)
                for (int j = 1; j <= n; j++)
                {
                    int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                    dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
                }
            return dp[m, n];
        }

        /// <summary>Compute structural similarity (Jaccard) between two expressions.</summary>
        public static float ComputeStructuralSimilarity(string a, string b)
        {
            var tokensA = TokenizeExpression(a);
            var tokensB = TokenizeExpression(b);
            var setA = new HashSet<string>(tokensA);
            var setB = new HashSet<string>(tokensB);
            if (setA.Count == 0 && setB.Count == 0)
                return 1f;
            int intersection = setA.Intersect(setB).Count();
            int union = setA.Union(setB).Count();
            return union > 0 ? (float)intersection / union : 0f;
        }

        private static List<string> TokenizeExpression(string expr)
        {
            var tokens = new List<string>();
            var sb = new StringBuilder();
            foreach (char c in expr)
            {
                if (char.IsLetterOrDigit(c) || c == '_' || c == '.')
                    sb.Append(c);
                else if (sb.Length > 0)
                { tokens.Add(sb.ToString()); sb.Clear(); }
            }
            if (sb.Length > 0)
                tokens.Add(sb.ToString());
            return tokens;
        }

        /// <summary>Compute edit distance between two version nodes.</summary>
        public int CompareVersions(string versionIdA, string versionIdB)
        {
            if (!_nodes.TryGetValue(versionIdA, out var a))
                throw new ArgumentException($"Version {versionIdA} not found");
            if (!_nodes.TryGetValue(versionIdB, out var b))
                throw new ArgumentException($"Version {versionIdB} not found");
            return ComputeEditDistance(a.Expression, b.Expression);
        }

        /// <summary>Export version history.</summary>
        public List<(string Id, string Expression, DateTime Timestamp, string Description)> ExportHistory()
        {
            return GetAllVersions().OrderBy(n => n.Timestamp)
                .Select(n => (n.VersionId, n.Expression, n.Timestamp, n.Description)).ToList();
        }

        /// <summary>Find the most recent common ancestor of two versions.</summary>
        public LawVersionNode? FindCommonAncestor(string versionIdA, string versionIdB)
        {
            if (!_nodes.TryGetValue(versionIdA, out var a))
                return null;
            if (!_nodes.TryGetValue(versionIdB, out var b))
                return null;
            var ancestorsA = new HashSet<string>();
            var node = a;
            while (node != null)
            { ancestorsA.Add(node.VersionId); node = node.Parent; }
            node = b;
            while (node != null)
            {
                if (ancestorsA.Contains(node.VersionId))
                    return node;
                node = node.Parent;
            }
            return null;
        }

        /// <summary>Get a version node by ID.</summary>
        public LawVersionNode? GetVersion(string versionId)
        {
            return _nodes.TryGetValue(versionId, out var node) ? node : null;
        }

        /// <summary>List all branch tips (leaves).</summary>
        public List<LawVersionNode> GetBranchTips()
        {
            return GetAllVersions().Where(n => n.Children.Count == 0).ToList();
        }

        /// <summary>Get depth of a version from root.</summary>
        public int GetDepth(string versionId)
        {
            return GetPath(GetVersion(versionId)).Count;
        }
    }
    // =========================================================================
    // LawApplicator — applies compiled laws to physics fields
    // =========================================================================

    /// <summary>Boundary conditions for law application.</summary>
    public enum BoundaryCondition
    {
        Periodic, Dirichlet, Neumann, Radiation
    }

    /// <summary>Applies a compiled law to a PhysicsField. Base class for specific applicators.</summary>
    public abstract class LawApplicator
    {
        public string Name { get; }
        public string TargetField { get; }
        protected readonly BytecodeInterpreter _interpreter;
        protected readonly GasMeter _gas;

        protected LawApplicator(string name, string targetField, long gasLimit = 10_000_000)
        {
            Name = name;
            TargetField = targetField;
            _gas = new GasMeter(gasLimit);
            _interpreter = new BytecodeInterpreter(_gas);
        }

        /// <summary>Apply the compiled law to the field at the given time step.</summary>
        public abstract void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config);

        /// <summary>Apply boundary conditions to the field after the law is applied.</summary>
        protected void ApplyBoundaryConditions(PhysicsField field, LawCompilerConfig config)
        {
            int size = field.GridSize;
            var component = field.GetComponent(TargetField);
            if (component == null)
                return;

            switch (config.BoundaryCondition)
            {
                case BoundaryCondition.Periodic:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[size - 1, y, z];
                            component[size - 1, y, z] = component[0, y, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = component[x, size - 1, z];
                            component[x, size - 1, z] = component[x, 0, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = component[x, y, size - 1];
                            component[x, y, size - 1] = component[x, y, 0];
                        }
                    break;

                case BoundaryCondition.Dirichlet:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = config.BoundaryValue;
                            component[size - 1, y, z] = config.BoundaryValue;
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = config.BoundaryValue;
                            component[x, size - 1, z] = config.BoundaryValue;
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = config.BoundaryValue;
                            component[x, y, size - 1] = config.BoundaryValue;
                        }
                    break;

                case BoundaryCondition.Neumann:
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[1, y, z];
                            component[size - 1, y, z] = component[size - 2, y, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int z = 0; z < size; z++)
                        {
                            component[x, 0, z] = component[x, 1, z];
                            component[x, size - 1, z] = component[x, size - 2, z];
                        }
                    for (int x = 0; x < size; x++)
                        for (int y = 0; y < size; y++)
                        {
                            component[x, y, 0] = component[x, y, 1];
                            component[x, y, size - 1] = component[x, y, size - 2];
                        }
                    break;

                case BoundaryCondition.Radiation:
                    float h = config.BoundaryValue;
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                        {
                            component[0, y, z] = component[1, y, z] / (1f + h);
                            component[size - 1, y, z] = component[size - 2, y, z] / (1f + h);
                        }
                    break;
            }
        }

        protected float ComputeGradientX(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX;
            float left = x > 0 ? field[x - 1, y, z] : field[0, y, z];
            float right = x < sx - 1 ? field[x + 1, y, z] : field[sx - 1, y, z];
            return (right - left) / (2f * dx);
        }

        protected float ComputeGradientY(FieldGrid field, int x, int y, int z, float dy)
        {
            int sy = field.SizeY;
            float bottom = y > 0 ? field[x, y - 1, z] : field[x, 0, z];
            float top = y < sy - 1 ? field[x, y + 1, z] : field[x, sy - 1, z];
            return (top - bottom) / (2f * dy);
        }

        protected float ComputeGradientZ(FieldGrid field, int x, int y, int z, float dz)
        {
            int sz = field.SizeZ;
            float back = z > 0 ? field[x, y, z - 1] : field[x, y, 0];
            float front = z < sz - 1 ? field[x, y, z + 1] : field[x, y, sz - 1];
            return (front - back) / (2f * dz);
        }

        protected float ComputeLaplacian(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX, sy = field.SizeY, sz = field.SizeZ;
            float center = field[x, y, z];
            float left = x > 0 ? field[x - 1, y, z] : center;
            float right = x < sx - 1 ? field[x + 1, y, z] : center;
            float bottom = y > 0 ? field[x, y - 1, z] : center;
            float top = y < sy - 1 ? field[x, y + 1, z] : center;
            float back = z > 0 ? field[x, y, z - 1] : center;
            float front = z < sz - 1 ? field[x, y, z + 1] : center;
            float invDx2 = 1f / (dx * dx);
            return (left + right + bottom + top + back + front - 6f * center) * invDx2;
        }

        protected float ComputeDivergence3D(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientX(vx, x, y, z, dx) + ComputeGradientY(vy, x, y, z, dx) + ComputeGradientZ(vz, x, y, z, dx);
        }

        protected float ComputeCurlX(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientY(vz, x, y, z, dx) - ComputeGradientZ(vy, x, y, z, dx);
        }

        protected float ComputeCurlY(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientZ(vx, x, y, z, dx) - ComputeGradientX(vz, x, y, z, dx);
        }

        protected float ComputeCurlZ(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return ComputeGradientX(vy, x, y, z, dx) - ComputeGradientY(vx, x, y, z, dx);
        }
    }

    /// <summary>Heat equation applicator: ∂T/∂t = α*∇²T.</summary>
    public sealed class HeatApplicator : LawApplicator
    {
        public HeatApplicator() : base("HeatEquation", "temperature") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var temp = field.Temperature;
            var tempNew = temp.Clone();
            float alpha = config.Constants.TryGetValue("alpha", out float a) ? a : 1.43e-4f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(temp, x, y, z, dx);
                        tempNew[x, y, z] = MathF.Max(0f, temp[x, y, z] + alpha * dt * laplacian);
                    }

            field.Temperature.CopyFrom(tempNew);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Wave equation applicator: ∂²u/∂t² = c²*∇²u.</summary>
    public sealed class WaveApplicator : LawApplicator
    {
        private FieldGrid? _previousState;

        public WaveApplicator() : base("WaveEquation", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var velocity = field.VelocityX;
            if (_previousState == null)
            { _previousState = velocity.Clone(); return; }

            float c = config.Constants.TryGetValue("c", out float cv) ? cv : 340f;
            float dx = config.CellSize;
            float dt2 = dt * dt, c2 = c * c;
            var newState = velocity.Clone();
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(velocity, x, y, z, dx);
                        float uPrev = _previousState[x, y, z];
                        float uCurr = velocity[x, y, z];
                        newState[x, y, z] = 2f * uCurr - uPrev + c2 * dt2 * laplacian;
                    }

            _previousState = velocity.Clone();
            velocity.CopyFrom(newState);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Elasticity applicator: F = -k*Δx.</summary>
    public sealed class ElasticityApplicator : LawApplicator
    {
        public ElasticityApplicator() : base("Elasticity", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float k = config.Constants.TryGetValue("k", out float kv) ? kv : 100f;
            float mass = config.Constants.TryGetValue("m", out float mv) ? mv : 1f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            var vx = field.VelocityX;
            var vy = field.VelocityY;
            var vz = field.VelocityZ;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float div = ComputeDivergence3D(vx, vy, vz, x, y, z, dx);
                        float accel = -k * div / mass;
                        vx[x, y, z] += accel * dt * 0.5f;
                        vy[x, y, z] += accel * dt * 0.5f;
                        vz[x, y, z] += accel * dt * 0.5f;
                    }

            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Advection applicator using upwind scheme.</summary>
    public sealed class AdvectionApplicator : LawApplicator
    {
        public AdvectionApplicator() : base("Advection", "density") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var density = field.Density;
            var densityNew = density.Clone();
            float vx = config.Constants.TryGetValue("vx", out float vxv) ? vxv : 1f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float dudx = vx > 0
                            ? (density[x, y, z] - density[x - 1, y, z]) / dx
                            : (density[x + 1, y, z] - density[x, y, z]) / dx;
                        densityNew[x, y, z] = density[x, y, z] - vx * dt * dudx;
                    }

            field.Density.CopyFrom(densityNew);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Diffusion applicator using explicit finite differences.</summary>
    public sealed class DiffusionApplicator : LawApplicator
    {
        public DiffusionApplicator() : base("Diffusion", "density") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            var density = field.Density;
            var densityNew = density.Clone();
            float diffusivity = config.Constants.TryGetValue("D", out float d) ? d : 1e-5f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(density, x, y, z, dx);
                        densityNew[x, y, z] = density[x, y, z] + diffusivity * dt * laplacian;
                    }

            field.Density.CopyFrom(densityNew);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Incompressible Navier-Stokes applicator (simplified).</summary>
    public sealed class IncompressibleNSApplicator : LawApplicator
    {
        public IncompressibleNSApplicator() : base("IncompressibleNS", "velocity") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float viscosity = config.Constants.TryGetValue("mu", out float mu) ? mu : 1.002e-3f;
            float density = config.Constants.TryGetValue("rho", out float rho) ? rho : 1000f;
            float dx = config.CellSize;
            float nu = viscosity / density;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            var vx = field.VelocityX;
            var vy = field.VelocityY;
            var vz = field.VelocityZ;
            var vxNew = vx.Clone();
            var vyNew = vy.Clone();
            var vzNew = vz.Clone();

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float lapVx = ComputeLaplacian(vx, x, y, z, dx);
                        float lapVy = ComputeLaplacian(vy, x, y, z, dx);
                        float lapVz = ComputeLaplacian(vz, x, y, z, dx);
                        float dVxDx = ComputeGradientX(vx, x, y, z, dx);
                        float dVyDy = ComputeGradientY(vy, x, y, z, dx);
                        float dVzDz = ComputeGradientZ(vz, x, y, z, dx);

                        vxNew[x, y, z] = vx[x, y, z] + dt * (nu * lapVx - vx[x, y, z] * dVxDx);
                        vyNew[x, y, z] = vy[x, y, z] + dt * (nu * lapVy - vy[x, y, z] * dVyDy);
                        vzNew[x, y, z] = vz[x, y, z] + dt * (nu * lapVz - vz[x, y, z] * dVzDz);
                    }

            vx.CopyFrom(vxNew);
            vy.CopyFrom(vyNew);
            vz.CopyFrom(vzNew);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Electromagnetic wave applicator (Maxwell-like).</summary>
    public sealed class ElectromagneticApplicator : LawApplicator
    {
        public ElectromagneticApplicator() : base("Electromagnetic", "temperature") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float epsilon = config.Constants.TryGetValue("eps", out float eps) ? eps : 8.854e-12f;
            float mu0 = config.Constants.TryGetValue("mu0", out float mu0v) ? mu0v : 1.25663706212e-6f;
            float dx = config.CellSize;
            float c = 1f / MathF.Sqrt(epsilon * mu0);
            float dx2 = dx * dx;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;

            var temp = field.Temperature;
            var tempNew = temp.Clone();

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(temp, x, y, z, dx);
                        tempNew[x, y, z] = 2f * temp[x, y, z] - temp[x, y, z] + c * c * dt * dt * laplacian;
                    }

            field.Temperature.CopyFrom(tempNew);
            ApplyBoundaryConditions(field, config);
        }
    }

    /// <summary>Gravity field applicator.</summary>
    public sealed class GravityApplicator : LawApplicator
    {
        public GravityApplicator() : base("Gravity", "density") { }

        public override void Apply(LawBytecode bytecode, PhysicsField field, float dt, LawCompilerConfig config)
        {
            float G = config.Constants.TryGetValue("G", out float gv) ? gv : 6.674e-11f;
            float dx = config.CellSize;
            int sx = field.GridSize, sy = field.GridSize, sz = field.GridSize;
            var density = field.Density;
            var densityNew = density.Clone();

            float totalMass = 0f;
            for (int z = 0; z < sz; z++)
                for (int y = 0; y < sy; y++)
                    for (int x = 0; x < sx; x++)
                        totalMass += density[x, y, z] * dx * dx * dx;

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        float laplacian = ComputeLaplacian(density, x, y, z, dx);
                        float gravityPotential = -G * totalMass / MathF.Max(MathF.Abs(density[x, y, z]) * dx * dx * dx, 1e-10f);
                        densityNew[x, y, z] = density[x, y, z] + dt * gravityPotential * laplacian * 0.001f;
                    }

            field.Density.CopyFrom(densityNew);
            ApplyBoundaryConditions(field, config);
        }
    }

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
    // LawValidation — dimensional analysis, stability, consistency
    // =========================================================================

    /// <summary>Validates law expressions for dimensional consistency, stability, and correctness.</summary>
    public sealed class LawValidation
    {
        private readonly LawLibrary _library;

        public LawValidation(LawLibrary library)
        {
            _library = library;
        }

        /// <summary>Perform full dimensional analysis on a law expression.</summary>
        public static ValidationResult ValidateDimensional(AstNode ast, LawEntry? knownLaw = null)
        {
            var errors = new List<string>();
            var warnings = new List<string>();
            var termDimensions = new List<Dimension>();
            bool consistent = true;

            CollectTermDimensions(ast, termDimensions);

            if (termDimensions.Count > 1)
            {
                var firstDim = termDimensions[0];
                for (int i = 1; i < termDimensions.Count; i++)
                {
                    if (!firstDim.IsCompatible(termDimensions[i]))
                    {
                        errors.Add($"Dimensional inconsistency: term {i} has dimension {termDimensions[i]} but expected {firstDim}");
                        consistent = false;
                    }
                }
            }

            if (knownLaw != null)
            {
                if (!knownLaw.ResultDimension.IsDimensionless && termDimensions.Count > 0)
                {
                    if (!termDimensions[0].IsCompatible(knownLaw.ResultDimension))
                        warnings.Add($"Result dimension {termDimensions[0]} differs from expected {knownLaw.ResultDimension}");
                }
            }

            return new ValidationResult(errors.Count == 0, errors.ToArray(), warnings.ToArray(),
                termDimensions.ToArray(), consistent, 0f);
        }

        private static void CollectTermDimensions(AstNode node, List<Dimension> dims)
        {
            if (node == null)
                return;
            switch (node.Type)
            {
                case NodeType.BinaryExpression:
                    if (node.Value is "+" or "-")
                    {
                        CollectTermDimensions(node.Left!, dims);
                        CollectTermDimensions(node.Right!, dims);
                    }
                    else
                        dims.Add(node.InferredDimension);
                    break;
                case NodeType.NumberLiteral:
                    dims.Add(Dimension.Scalar);
                    break;
                case NodeType.Identifier:
                    dims.Add(node.InferredDimension);
                    break;
                case NodeType.FunctionCall:
                    dims.Add(node.InferredDimension);
                    break;
                default:
                    dims.Add(node.InferredDimension);
                    break;
            }
        }

        /// <summary>Check CFL stability condition for explicit time-stepping schemes.</summary>
        public static float ComputeCflRatio(PhysicsField field, float dt, float dx, float maxWaveSpeed)
        {
            if (dx <= 0f || maxWaveSpeed <= 0f)
                return 0f;
            return maxWaveSpeed * dt / dx;
        }

        /// <summary>Check if a simulation is stable given the CFL condition.</summary>
        public static bool IsStable(float cflRatio, float cflLimit = 1.0f) => cflRatio <= cflLimit;

        /// <summary>Validate parameter ranges.</summary>
        public static ValidationResult ValidateParameters(Dictionary<string, float> parameters, Dictionary<string, (float Min, float Max)> ranges)
        {
            var errors = new List<string>();
            var warnings = new List<string>();
            foreach (var kv in parameters)
            {
                if (ranges.TryGetValue(kv.Key, out var range))
                {
                    if (kv.Value < range.Min || kv.Value > range.Max)
                        errors.Add($"Parameter '{kv.Key}' = {kv.Value} is outside valid range [{range.Min}, {range.Max}]");
                }
            }
            return new ValidationResult(errors.Count == 0, errors.ToArray(), warnings.ToArray(),
                Array.Empty<Dimension>(), true, 0f);
        }

        /// <summary>Verify that the equation reduces to known limits.</summary>
        public static bool CheckLimitConsistency(string expression, string limitCase, float expectedValue, float tolerance = 1e-3f)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            var bytecode = parser.CompileToBytecode(ast);
            var interpreter = new BytecodeInterpreter();
            float result = interpreter.Execute(bytecode, null, null, new Dictionary<string, float>
            {
                ["rho"] = limitCase == "incompressible" ? 1000f : 1.225f,
                ["v"] = limitCase == "low_speed" ? 0.1f : 100f,
                ["T"] = limitCase == "low_temp" ? 273.15f : 300f,
                ["P"] = 101325f,
                ["mu"] = 0.001f,
                ["c"] = 340f,
                ["alpha"] = 1.43e-4f,
                ["k"] = 205f,
                ["G"] = 6.674e-11f,
                ["sigma"] = 5.670374419e-8f,
                ["R"] = 287.058f
            });
            return MathF.Abs(result - expectedValue) < tolerance;
        }

        /// <summary>Validate stability for a specific equation type.</summary>
        public ValidationResult ValidateStability(string expression, string equationType, PhysicsField? testField = null)
        {
            var errors = new List<string>();
            var warnings = new List<string>();
            var termDimensions = Array.Empty<Dimension>();

            float cflRatio = 0f;
            var field = testField ?? CreateDefaultTestField();

            switch (equationType.ToLowerInvariant())
            {
                case "heat":
                    float alpha = 1.43e-4f;
                    cflRatio = alpha * 0.001f / (1.0f * 1.0f);
                    if (cflRatio > 0.5f)
                        warnings.Add($"CFL ratio {cflRatio:F4} may be unstable for heat equation (limit ~0.5)");
                    break;

                case "wave":
                    float c = 340f;
                    cflRatio = c * 0.001f / 1.0f;
                    if (cflRatio > 1.0f)
                        errors.Add($"CFL ratio {cflRatio:F4} exceeds stability limit for wave equation (limit = 1.0)");
                    break;

                case "advection":
                    float vx = 1.0f;
                    cflRatio = vx * 0.001f / 1.0f;
                    if (cflRatio > 1.0f)
                        errors.Add($"CFL ratio {cflRatio:F4} exceeds stability limit for advection (limit = 1.0)");
                    break;

                case "diffusion":
                    float D = 1e-5f;
                    cflRatio = D * 0.001f / (1.0f * 1.0f);
                    if (cflRatio > 0.5f)
                        warnings.Add($"CFL ratio {cflRatio:F4} may be unstable for diffusion (limit ~0.5)");
                    break;

                case "navier_stokes":
                case "navier-stokes":
                    float nu = 1.002e-6f;
                    float cflNs = 1.0f * 0.001f / 1.0f;
                    float viscousLimit = nu * 0.001f / (1.0f * 1.0f);
                    cflRatio = MathF.Max(cflNs, viscousLimit);
                    if (cflRatio > 0.5f)
                        warnings.Add($"CFL ratio {cflRatio:F4} may be unstable for Navier-Stokes");
                    break;

                default:
                    warnings.Add($"No stability analysis available for equation type '{equationType}'");
                    break;
            }

            return new ValidationResult(errors.Count == 0, errors.ToArray(), warnings.ToArray(),
                termDimensions, true, cflRatio);
        }

        private PhysicsField CreateDefaultTestField()
        {
            int size = 32;
            var field = new PhysicsField(size, "validation_test");
            float cx = size / 2f;
            for (int z = 0; z < size; z++)
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (size * size));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }
            return field;
        }

        /// <summary>Comprehensive validation of a law expression.</summary>
        public ValidationResult ComprehensiveValidate(string expression, string? lawId = null, PhysicsField? testField = null)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();

            if (parser.Errors.Count > 0)
                return new ValidationResult(false, parser.Errors.ToArray(), parser.Warnings.ToArray(),
                    Array.Empty<Dimension>(), false, 0f);

            LawEntry? knownLaw = lawId != null ? _library.GetLaw(lawId) : null;
            var dimResult = ValidateDimensional(ast, knownLaw);

            var allWarnings = new List<string>(dimResult.Warnings);
            allWarnings.AddRange(parser.Warnings);

            ValidationResult stabilityResult = ValidationResult.Valid();
            if (knownLaw != null)
            {
                stabilityResult = ValidateStability(expression, knownLaw.Category, testField);
                allWarnings.AddRange(stabilityResult.Warnings);
            }

            return new ValidationResult(
                dimResult.IsValid && stabilityResult.IsValid,
                dimResult.Errors.Concat(stabilityResult.Errors).ToArray(),
                allWarnings.ToArray(),
                dimResult.TermDimensions,
                dimResult.DimensionallyConsistent,
                stabilityResult.StabilityCflRatio);
        }

        /// <summary>Check that all variables in the expression are physically meaningful.</summary>
        public static ValidationResult ValidatePhysicalMeaningfulness(string expression)
        {
            var errors = new List<string>();
            var warnings = new List<string>();
            var knownVars = new HashSet<string> { "x", "y", "z", "t", "T", "P", "rho", "v", "u", "w", "F", "E", "k", "mu", "alpha", "sigma", "G", "R", "c", "I", "V", "q", "dt", "dx", "dy", "dz", "m1", "m2", "r", "q1", "q2" };

            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            CollectIdentifiers(ast, knownVars, errors, warnings);

            return new ValidationResult(errors.Count == 0, errors.ToArray(), warnings.ToArray(),
                Array.Empty<Dimension>(), true, 0f);
        }

        private static void CollectIdentifiers(AstNode node, HashSet<string> known, List<string> errors, List<string> warnings)
        {
            if (node == null)
                return;
            switch (node.Type)
            {
                case NodeType.Identifier:
                    if (!known.Contains(node.Value ?? "") && node.Value != "field")
                        warnings.Add($"Unknown variable '{node.Value}' may not have physical meaning");
                    break;
                case NodeType.FieldAccess:
                    break;
                default:
                    if (node.Left != null)
                        CollectIdentifiers(node.Left, known, errors, warnings);
                    if (node.Right != null)
                        CollectIdentifiers(node.Right, known, errors, warnings);
                    if (node.Middle != null)
                        CollectIdentifiers(node.Middle, known, errors, warnings);
                    if (node.Children != null)
                        foreach (var child in node.Children)
                            CollectIdentifiers(child, known, errors, warnings);
                    break;
            }
        }
    }    // =========================================================================
    // LawModificationEngine — natural language and math editing
    // =========================================================================

    /// <summary>Modification types for the law modification engine.</summary>
    public enum ModificationType
    {
        ModifyConstant, ModifyOperator, AddCouplingTerm, ScaleVariable,
        ReplaceSubexpression, SimplifyExpression, InvertSign, Exponentiate,
        AddDamping, AddForcing, Linearize, Nonlinearize, Custom
    }

    /// <summary>Describes a modification to be applied to a law expression.</summary>
    public sealed class LawModification
    {
        public ModificationType Type { get; set; }
        public string? TargetExpression { get; set; }
        public string? ReplacementExpression { get; set; }
        public float? ConstantValue { get; set; }
        public string? VariableName { get; set; }
        public float ScaleFactor { get; set; } = 1f;
        public string? CouplingTerm { get; set; }
        public string? Description { get; set; }
        public Dictionary<string, float>? Metadata { get; set; }
    }

    /// <summary>Modification record for tracking changes.</summary>
    public sealed class ModificationRecord
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];
        public LawModification Modification { get; set; } = new();
        public string OriginalExpression { get; set; } = "";
        public string ResultExpression { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string? AppliedBy { get; set; }
    }

    /// <summary>Engine for modifying law expressions through various methods.</summary>
    public sealed class LawModificationEngine
    {
        private readonly LawLibrary _library;
        private readonly List<ModificationRecord> _history = new();
        public IReadOnlyList<ModificationRecord> History => _history;

        public LawModificationEngine(LawLibrary library) { _library = library; }

        /// <summary>Modify a constant value in the expression.</summary>
        public string ModifyConstant(string expression, string constantName, float newValue)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            return ReplaceConstantInAst(ast, constantName, newValue);
        }

        private string ReplaceConstantInAst(AstNode node, string name, float newValue)
        {
            if (node == null)
                return "";
            return node.Type switch
            {
                NodeType.NumberLiteral => node.Value ?? "0",
                NodeType.Identifier => node.Value == name
                    ? newValue.ToString(CultureInfo.InvariantCulture) : node.Value ?? "",
                NodeType.BinaryExpression =>
                    $"({ReplaceConstantInAst(node.Left!, name, newValue)} {node.Value} {ReplaceConstantInAst(node.Right!, name, newValue)})",
                NodeType.UnaryExpression =>
                    $"({node.Value}{ReplaceConstantInAst(node.Left!, name, newValue)})",
                NodeType.FunctionCall =>
                    $"{node.Value}({string.Join(", ", node.Children?.Select(c => ReplaceConstantInAst(c, name, newValue)) ?? Array.Empty<string>())})",
                NodeType.FieldAccess => $"field.{node.Value}",
                NodeType.TernaryExpression =>
                    $"({ReplaceConstantInAst(node.Left!, name, newValue)} ? {ReplaceConstantInAst(node.Right!, name, newValue)} : {ReplaceConstantInAst(node.Middle!, name, newValue)})",
                _ => ""
            };
        }

        /// <summary>Replace one operator with another.</summary>
        public string ModifyOperator(string expression, string targetOp, string replacementOp)
        {
            return expression.Replace(targetOp, replacementOp);
        }

        /// <summary>Add a coupling term to the expression.</summary>
        public string AddCouplingTerm(string expression, string couplingTerm, float couplingStrength = 1.0f)
        {
            string strengthStr = couplingStrength == 1.0f ? "" : couplingStrength.ToString(CultureInfo.InvariantCulture) + "*";
            return $"({expression}) + {strengthStr}({couplingTerm})";
        }

        /// <summary>Scale a variable in the expression.</summary>
        public string ScaleVariable(string expression, string variableName, float scaleFactor)
        {
            return expression.Replace(variableName, $"({scaleFactor}*{variableName})");
        }

        /// <summary>Invert the sign of a term.</summary>
        public string InvertSign(string expression, string termName)
        {
            return expression.Replace(termName, $"(-{termName})");
        }

        /// <summary>Exponentiate a variable.</summary>
        public string Exponentiate(string expression, string variableName, float exponent)
        {
            return expression.Replace(variableName, $"({variableName}^{exponent.ToString(CultureInfo.InvariantCulture)})");
        }

        /// <summary>Add damping term to an equation.</summary>
        public string AddDamping(string expression, string variableName, float dampingCoeff)
        {
            string dampingTerm = $"{dampingCoeff.ToString(CultureInfo.InvariantCulture)}*{variableName}";
            return $"({expression}) - {dampingTerm}";
        }

        /// <summary>Add external forcing term.</summary>
        public string AddForcing(string expression, string forcingExpression, float strength = 1.0f)
        {
            string strengthStr = strength == 1.0f ? "" : strength.ToString(CultureInfo.InvariantCulture) + "*";
            return $"({expression}) + {strengthStr}({forcingExpression})";
        }

        /// <summary>Linearize around an operating point.</summary>
        public string Linearize(string expression, string variableName, float operatingPoint)
        {
            string result = expression;
            result = result.Replace($"{variableName}^2", $"(2*{operatingPoint.ToString(CultureInfo.InvariantCulture)}*{variableName})");
            result = result.Replace($"{variableName}^3", $"(3*{MathF.Pow(operatingPoint, 2).ToString(CultureInfo.InvariantCulture)}*{variableName})");
            return result;
        }

        /// <summary>Add nonlinearity (power law).</summary>
        public string Nonlinearize(string expression, string variableName, float exponent)
        {
            return expression.Replace(variableName, $"abs({variableName})^{exponent.ToString(CultureInfo.InvariantCulture)}*sign({variableName})");
        }

        /// <summary>Apply a modification from a LawModification object.</summary>
        public string ApplyModification(string expression, LawModification modification)
        {
            string result = modification.Type switch
            {
                ModificationType.ModifyConstant when modification.ConstantValue.HasValue && modification.VariableName != null =>
                    ModifyConstant(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.ModifyOperator when modification.TargetExpression != null && modification.ReplacementExpression != null =>
                    ModifyOperator(expression, modification.TargetExpression, modification.ReplacementExpression),
                ModificationType.AddCouplingTerm when modification.CouplingTerm != null =>
                    AddCouplingTerm(expression, modification.CouplingTerm, modification.ScaleFactor),
                ModificationType.ScaleVariable when modification.VariableName != null =>
                    ScaleVariable(expression, modification.VariableName, modification.ScaleFactor),
                ModificationType.ReplaceSubexpression when modification.TargetExpression != null && modification.ReplacementExpression != null =>
                    expression.Replace(modification.TargetExpression, modification.ReplacementExpression),
                ModificationType.SimplifyExpression => SimplifyExpression(expression),
                ModificationType.InvertSign when modification.VariableName != null =>
                    InvertSign(expression, modification.VariableName),
                ModificationType.Exponentiate when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Exponentiate(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.AddDamping when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    AddDamping(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.AddForcing when modification.CouplingTerm != null =>
                    AddForcing(expression, modification.CouplingTerm, modification.ScaleFactor),
                ModificationType.Linearize when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Linearize(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.Nonlinearize when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Nonlinearize(expression, modification.VariableName, modification.ConstantValue.Value),
                _ => expression
            };

            _history.Add(new ModificationRecord
            {
                Modification = modification,
                OriginalExpression = expression,
                ResultExpression = result,
                Timestamp = DateTime.UtcNow
            });
            return result;
        }

        /// <summary>Apply a sequence of modifications.</summary>
        public string ApplyModifications(string expression, IReadOnlyList<LawModification> modifications)
        {
            string result = expression;
            foreach (var mod in modifications)
                result = ApplyModification(result, mod);
            return result;
        }

        /// <summary>Simplify an expression (basic algebraic simplifications).</summary>
        public string SimplifyExpression(string expression)
        {
            string result = expression;
            result = System.Text.RegularExpressions.Regex.Replace(result, @"--", "+");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\d+\.?\d*)\s*\*\s*1\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b1\s*\*\s*(\d+\.?\d*)", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\d+\.?\d*)\s*\*\s*0\b", "0");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b0\s*\*\s*(\d+\.?\d*)", "0");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\+\s*0\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b0\s*\+", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\^\s*1\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\w+)\s*\^\s*0", "1");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\(\s*(\w+)\s*\)", "");
            return result;
        }

        /// <summary>Parse natural language modification instruction and apply it.</summary>
        public string ApplyNaturalLanguageModification(string expression, string instruction)
        {
            string lower = instruction.ToLowerInvariant();

            var increaseMatch = System.Text.RegularExpressions.Regex.Match(lower, @"increase\s+(\w+)\s+by\s+(\d+)%");
            if (increaseMatch.Success)
            {
                string varName = increaseMatch.Groups[1].Value;
                float percent = float.Parse(increaseMatch.Groups[2].Value) / 100f;
                return ScaleVariable(expression, varName, 1f + percent);
            }

            var decreaseMatch = System.Text.RegularExpressions.Regex.Match(lower, @"decrease\s+(\w+)\s+by\s+(\d+)%");
            if (decreaseMatch.Success)
            {
                string varName = decreaseMatch.Groups[1].Value;
                float percent = float.Parse(decreaseMatch.Groups[2].Value) / 100f;
                return ScaleVariable(expression, varName, 1f - percent);
            }

            var setMatch = System.Text.RegularExpressions.Regex.Match(lower, @"set\s+(\w+)\s+to\s+([\d.]+)");
            if (setMatch.Success)
            {
                string varName = setMatch.Groups[1].Value;
                float value = float.Parse(setMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return ModifyConstant(expression, varName, value);
            }

            var replaceMatch = System.Text.RegularExpressions.Regex.Match(lower, @"replace\s+(\w+)\s+with\s+(\w+)");
            if (replaceMatch.Success)
            {
                string target = replaceMatch.Groups[1].Value;
                string replacement = replaceMatch.Groups[2].Value;
                return ModifyOperator(expression, target, replacement);
            }

            var couplingMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+coupling\s+with\s+(\w+)");
            if (couplingMatch.Success)
            {
                string coupling = couplingMatch.Groups[1].Value;
                return AddCouplingTerm(expression, coupling);
            }

            var dampingMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+damping\s+to\s+(\w+)\s+with\s+coefficient\s+([\d.]+)");
            if (dampingMatch.Success)
            {
                string varName = dampingMatch.Groups[1].Value;
                float coeff = float.Parse(dampingMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return AddDamping(expression, varName, coeff);
            }

            var forceMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+forcing\s+(.+)");
            if (forceMatch.Success)
            {
                string forcing = forceMatch.Groups[1].Value;
                return AddForcing(expression, forcing);
            }

            var linMatch = System.Text.RegularExpressions.Regex.Match(lower, @"linearize\s+(\w+)\s+around\s+([\d.]+)");
            if (linMatch.Success)
            {
                string varName = linMatch.Groups[1].Value;
                float op = float.Parse(linMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return Linearize(expression, varName, op);
            }

            var invMatch = System.Text.RegularExpressions.Regex.Match(lower, @"invert\s+sign\s+of\s+(\w+)");
            if (invMatch.Success)
                return InvertSign(expression, invMatch.Groups[1].Value);

            if (lower.Contains("simplify"))
                return SimplifyExpression(expression);
            if (lower.Contains("double"))
            {
                var doubleVarMatch = System.Text.RegularExpressions.Regex.Match(lower, @"double\s+(\w+)");
                if (doubleVarMatch.Success)
                    return ScaleVariable(expression, doubleVarMatch.Groups[1].Value, 2f);
            }
            if (lower.Contains("half"))
            {
                var halfVarMatch = System.Text.RegularExpressions.Regex.Match(lower, @"half\s+(\w+)");
                if (halfVarMatch.Success)
                    return ScaleVariable(expression, halfVarMatch.Groups[1].Value, 0.5f);
            }

            return expression;
        }

        /// <summary>Get modification history for an expression.</summary>
        public IReadOnlyList<ModificationRecord> GetHistoryForExpression(string expression)
        {
            return _history.Where(r => r.OriginalExpression == expression || r.ResultExpression == expression).ToList();
        }

        /// <summary>Undo the last modification.</summary>
        public string? UndoLastModification()
        {
            if (_history.Count == 0)
                return null;
            var last = _history[^1];
            _history.RemoveAt(_history.Count - 1);
            return last.OriginalExpression;
        }
    }    // =========================================================================
    // LawCompilerConfig — configuration for the compiler
    // =========================================================================

    /// <summary>Solver methods available.</summary>
    public enum SolverMethod
    {
        ExplicitEuler, ImplicitEuler, RungeKutta4, Leapfrog,
        SymplecticEuler, AdamsBashforth, AdamsMoulton, CrankNicolson
    }

    /// <summary>Configuration for the Living Law Compiler.</summary>
    public sealed class LawCompilerConfig
    {
        /// <summary>Simulation tolerance (convergence threshold).</summary>
        public float Tolerance { get; set; } = 1e-6f;
        /// <summary>Maximum iterations for iterative solvers.</summary>
        public int MaxIterations { get; set; } = 1000;
        /// <summary>Time step for explicit schemes.</summary>
        public float TimeStep { get; set; } = 0.001f;
        /// <summary>Cell size for spatial discretization.</summary>
        public float CellSize { get; set; } = 1.0f;
        /// <summary>Boundary condition type.</summary>
        public BoundaryCondition BoundaryCondition { get; set; } = BoundaryCondition.Periodic;
        /// <summary>Boundary value for Dirichlet conditions.</summary>
        public float BoundaryValue { get; set; } = 0f;
        /// <summary>Gas limit for bytecode execution.</summary>
        public long GasLimit { get; set; } = 10_000_000;
        /// <summary>CFL stability limit.</summary>
        public float CflLimit { get; set; } = 1.0f;
        /// <summary>Physical constants used in the simulation.</summary>
        public Dictionary<string, float> Constants { get; set; } = new()
        {
            ["alpha"] = 1.43e-4f,
            ["c"] = 340f,
            ["k"] = 100f,
            ["m"] = 1f,
            ["G"] = 6.674e-11f,
            ["R"] = 287.058f,
            ["sigma"] = 5.670374419e-8f,
            ["mu"] = 1.002e-3f,
            ["rho"] = 1.225f,
            ["D"] = 1e-5f,
            ["eps"] = 8.854e-12f,
            ["mu0"] = 1.25663706212e-6f,
            ["vx"] = 1.0f,
            ["vy"] = 0f,
            ["vz"] = 0f,
            ["ke"] = 8.9875517923e9f,
            ["h"] = 0.01f,
        };
        /// <summary>Coupling parameters between fields.</summary>
        public Dictionary<string, float> CouplingParameters { get; set; } = new();
        /// <summary>Solver method selection.</summary>
        public SolverMethod Solver { get; set; } = SolverMethod.ExplicitEuler;
        /// <summary>Number of grid cells per dimension.</summary>
        public int GridSize { get; set; } = 64;
        /// <summary>Enable hot-reload of law expressions.</summary>
        public bool EnableHotReload { get; set; } = true;
        /// <summary>Compile with validation enabled.</summary>
        public bool EnableValidation { get; set; } = true;
        /// <summary>Maximum bytecode instructions.</summary>
        public int MaxBytecodeInstructions { get; set; } = 4096;
        /// <summary>Enable JIT-like specialization for hot loops.</summary>
        public bool EnableJitSpecialization { get; set; } = false;
        /// <summary>Number of parallel threads for field operations.</summary>
        public int Parallelism { get; set; } = 1;
        /// <summary>Output directory for compilation artifacts.</summary>
        public string? OutputDirectory { get; set; }

        /// <summary>Create a copy of this configuration.</summary>
        public LawCompilerConfig Clone()
        {
            return new LawCompilerConfig
            {
                Tolerance = Tolerance,
                MaxIterations = MaxIterations,
                TimeStep = TimeStep,
                CellSize = CellSize,
                BoundaryCondition = BoundaryCondition,
                BoundaryValue = BoundaryValue,
                GasLimit = GasLimit,
                CflLimit = CflLimit,
                Solver = Solver,
                GridSize = GridSize,
                EnableHotReload = EnableHotReload,
                EnableValidation = EnableValidation,
                MaxBytecodeInstructions = MaxBytecodeInstructions,
                EnableJitSpecialization = EnableJitSpecialization,
                Parallelism = Parallelism,
                OutputDirectory = OutputDirectory,
                Constants = new Dictionary<string, float>(Constants),
                CouplingParameters = new Dictionary<string, float>(CouplingParameters)
            };
        }

        /// <summary>Get a preset configuration for a specific physics domain.</summary>
        public static LawCompilerConfig ForDomain(string domain) => domain.ToLowerInvariant() switch
        {
            "heat" or "thermal" => new LawCompilerConfig
            {
                TimeStep = 0.001f,
                CellSize = 1f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Dirichlet,
                BoundaryValue = 300f,
                CflLimit = 0.5f,
                Constants = new() { ["alpha"] = 1.43e-4f, ["k"] = 205f }
            },
            "wave" or "acoustic" => new LawCompilerConfig
            {
                TimeStep = 0.0001f,
                CellSize = 0.1f,
                GridSize = 128,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.9f,
                Constants = new() { ["c"] = 340f }
            },
            "fluid" or "navier_stokes" => new LawCompilerConfig
            {
                TimeStep = 0.0001f,
                CellSize = 0.01f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.5f,
                Solver = SolverMethod.RungeKutta4,
                Constants = new() { ["mu"] = 1.002e-3f, ["rho"] = 1000f }
            },
            "elasticity" or "solid" => new LawCompilerConfig
            {
                TimeStep = 0.001f,
                CellSize = 1f,
                GridSize = 32,
                BoundaryCondition = BoundaryCondition.Neumann,
                Constants = new() { ["k"] = 100f, ["m"] = 1f }
            },
            "electromagnetic" or "em" => new LawCompilerConfig
            {
                TimeStep = 1e-10f,
                CellSize = 1e-3f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.95f,
                Constants = new() { ["eps"] = 8.854e-12f, ["mu0"] = 1.25663706212e-6f }
            },
            "gravity" or "gravitation" => new LawCompilerConfig
            {
                TimeStep = 0.01f,
                CellSize = 1e6f,
                GridSize = 32,
                BoundaryCondition = BoundaryCondition.Periodic,
                Constants = new() { ["G"] = 6.674e-11f }
            },
            _ => new LawCompilerConfig()
        };
    }    // =========================================================================
    // LawComparison — compare simulation results
    // =========================================================================

    /// <summary>Compares simulation results between different law versions.</summary>
    public sealed class LawComparison
    {
        /// <summary>Compare two field grids point-wise.</summary>
        public static ComparisonResult CompareFields(FieldGrid fieldA, FieldGrid fieldB, int expressionEditDistance = 0)
        {
            if (fieldA.TotalCells != fieldB.TotalCells)
                throw new ArgumentException("Field sizes must match");

            int n = fieldA.TotalCells;
            float maxDiv = 0f;
            double sumDiv = 0, sumSqDiff = 0;
            var diffs = new List<string>();

            for (int i = 0; i < n; i++)
            {
                float diff = MathF.Abs(fieldA.Data[i] - fieldB.Data[i]);
                if (diff > maxDiv)
                    maxDiv = diff;
                sumDiv += diff;
                sumSqDiff += (double)diff * diff;
            }

            float meanDiv = (float)(sumDiv / n);
            float rmse = MathF.Sqrt((float)(sumSqDiff / n));

            // Kolmogorov-Smirnov statistic
            var valuesA = new float[n];
            var valuesB = new float[n];
            Array.Copy(fieldA.Data, valuesA, n);
            Array.Copy(fieldB.Data, valuesB, n);
            Array.Sort(valuesA);
            Array.Sort(valuesB);

            float ksMax = 0f;
            for (int i = 0; i < n; i++)
            {
                float cdfA = (float)(i + 1) / n;
                int bIdx = Array.BinarySearch(valuesB, valuesA[i]);
                if (bIdx < 0)
                    bIdx = ~bIdx;
                float cdfB = (float)(bIdx + 1) / n;
                float ksDiff = MathF.Abs(cdfA - cdfB);
                if (ksDiff > ksMax)
                    ksMax = ksDiff;
            }

            // SSIM approximation based on statistics
            float muA = fieldA.Average(), muB = fieldB.Average();
            float sigA = fieldA.StandardDeviation(), sigB = fieldB.StandardDeviation();
            float sigAB = 0f;
            for (int i = 0; i < n; i++)
                sigAB += (fieldA.Data[i] - muA) * (fieldB.Data[i] - muB);
            sigAB /= n;

            float c1 = 0.01f * 0.01f, c2 = 0.03f * 0.03f;
            float ssim = ((2f * muA * muB + c1) * (2f * sigAB + c2)) /
                         ((muA * muA + muB * muB + c1) * (sigA * sigA + sigB * sigB + c2));

            if (maxDiv > 0.01f)
                diffs.Add($"Max divergence: {maxDiv:F6}");
            if (meanDiv > 0.001f)
                diffs.Add($"Mean divergence: {meanDiv:F6}");
            if (rmse > 0.01f)
                diffs.Add($"RMSE: {rmse:F6}");
            if (ksMax > 0.1f)
                diffs.Add($"KS statistic: {ksMax:F4} (distributions differ significantly)");
            if (MathF.Abs(muA - muB) > 0.01f)
                diffs.Add($"Mean difference: {muA:F4} vs {muB:F4}");
            if (MathF.Abs(sigA - sigB) > 0.01f)
                diffs.Add($"Std dev difference: {sigA:F4} vs {sigB:F4}");

            bool physEq = maxDiv < 0.01f && rmse < 0.01f && ksMax < 0.05f;
            return new ComparisonResult(maxDiv, meanDiv, rmse, ksMax, ssim, expressionEditDistance, diffs.ToArray(), physEq);
        }

        /// <summary>Compare two physics fields.</summary>
        public static ComparisonResult ComparePhysicsFields(PhysicsField fieldA, PhysicsField fieldB)
        {
            var tempComp = CompareFields(fieldA.Temperature, fieldB.Temperature);
            var presComp = CompareFields(fieldA.Pressure, fieldB.Pressure);
            var densComp = CompareFields(fieldA.Density, fieldB.Density);

            float maxDiv = MathF.Max(tempComp.MaxDivergence, MathF.Max(presComp.MaxDivergence, densComp.MaxDivergence));
            float meanDiv = (tempComp.MeanDivergence + presComp.MeanDivergence + densComp.MeanDivergence) / 3f;
            float rmse = MathF.Sqrt((tempComp.RootMeanSquareError * tempComp.RootMeanSquareError +
                presComp.RootMeanSquareError * presComp.RootMeanSquareError +
                densComp.RootMeanSquareError * densComp.RootMeanSquareError) / 3f);
            float ks = MathF.Max(tempComp.KolmogorovSmirnovStatistic,
                MathF.Max(presComp.KolmogorovSmirnovStatistic, densComp.KolmogorovSmirnovStatistic));
            float ssim = (tempComp.StructuralSimilarity + presComp.StructuralSimilarity + densComp.StructuralSimilarity) / 3f;

            var diffs = new List<string>();
            diffs.AddRange(tempComp.Differences.Select(d => $"Temperature: {d}"));
            diffs.AddRange(presComp.Differences.Select(d => $"Pressure: {d}"));
            diffs.AddRange(densComp.Differences.Select(d => $"Density: {d}"));
            bool physEq = tempComp.PhysicallyEquivalent && presComp.PhysicallyEquivalent && densComp.PhysicallyEquivalent;
            return new ComparisonResult(maxDiv, meanDiv, rmse, ks, ssim, 0, diffs.ToArray(), physEq);
        }

        /// <summary>Compare two simulation snapshots over time.</summary>
        public static List<ComparisonResult> CompareTimeSeries(
            List<PhysicsField> snapshotsA, List<PhysicsField> snapshotsB)
        {
            int count = Math.Min(snapshotsA.Count, snapshotsB.Count);
            var results = new List<ComparisonResult>(count);
            for (int i = 0; i < count; i++)
                results.Add(ComparePhysicsFields(snapshotsA[i], snapshotsB[i]));
            return results;
        }

        /// <summary>Compute divergence field between two grids.</summary>
        public static FieldGrid ComputeDivergenceField(FieldGrid a, FieldGrid b)
        {
            if (a.SizeX != b.SizeX || a.SizeY != b.SizeY || a.SizeZ != b.SizeZ)
                throw new ArgumentException("Grid sizes must match");
            var div = new FieldGrid(a.SizeX, a.SizeY, a.SizeZ);
            for (int z = 0; z < a.SizeZ; z++)
                for (int y = 0; y < a.SizeY; y++)
                    for (int x = 0; x < a.SizeX; x++)
                        div[x, y, z] = a[x, y, z] - b[x, y, z];
            return div;
        }

        /// <summary>Compute L2 norm of the difference between two fields.</summary>
        public static float ComputeL2Norm(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            double sum = 0;
            for (int i = 0; i < a.TotalCells; i++)
            {
                double diff = a.Data[i] - b.Data[i];
                sum += diff * diff;
            }
            return MathF.Sqrt((float)(sum / a.TotalCells));
        }

        /// <summary>Compute L-infinity norm (max absolute difference).</summary>
        public static float ComputeLInfNorm(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            float maxDiff = 0f;
            for (int i = 0; i < a.TotalCells; i++)
            {
                float diff = MathF.Abs(a.Data[i] - b.Data[i]);
                if (diff > maxDiff)
                    maxDiff = diff;
            }
            return maxDiff;
        }

        /// <summary>Compute correlation coefficient between two fields.</summary>
        public static float ComputeCorrelation(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            int n = a.TotalCells;
            float muA = a.Average(), muB = b.Average();
            float num = 0f, denA = 0f, denB = 0f;
            for (int i = 0; i < n; i++)
            {
                float dA = a.Data[i] - muA;
                float dB = b.Data[i] - muB;
                num += dA * dB;
                denA += dA * dA;
                denB += dB * dB;
            }
            float den = MathF.Sqrt(denA * denB);
            return den > 0f ? num / den : 0f;
        }

        /// <summary>Compute energy norm of the difference.</summary>
        public static float ComputeEnergyNorm(FieldGrid a, FieldGrid b, float dx)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            double sum = 0;
            float invDx2 = 1f / (dx * dx);
            for (int z = 1; z < a.SizeZ - 1; z++)
                for (int y = 1; y < a.SizeY - 1; y++)
                    for (int x = 1; x < a.SizeX - 1; x++)
                    {
                        float valA = a[x, y, z];
                        float valB = b[x, y, z];
                        sum += (valA - valB) * (valA - valB) * dx * dx * dx;
                    }
            return MathF.Sqrt((float)sum);
        }

        /// <summary>Compute spectral analysis difference.</summary>
        public static float ComputeSpectralDifference(FieldGrid a, FieldGrid b)
        {
            if (a.TotalCells != b.TotalCells)
                throw new ArgumentException("Field sizes must match");
            float muA = a.Average(), muB = b.Average();
            float varA = 0f, varB = 0f, covAB = 0f;
            for (int i = 0; i < a.TotalCells; i++)
            {
                float dA = a.Data[i] - muA;
                float dB = b.Data[i] - muB;
                varA += dA * dA;
                varB += dB * dB;
                covAB += dA * dB;
            }
            float n = a.TotalCells;
            varA /= n;
            varB /= n;
            covAB /= n;
            return MathF.Sqrt(MathF.Max(0f, varA + varB - 2f * covAB));
        }
    }    // =========================================================================
    // LawInventor — creates new laws from templates and parameters
    // =========================================================================

    /// <summary>Template for inventing new physical laws.</summary>
    public sealed class LawTemplate
    {
        public string Name { get; set; } = "";
        public string Category { get; set; } = "";
        public string ExpressionTemplate { get; set; } = "";
        public Dictionary<string, string> VariableDescriptions { get; set; } = new();
        public Dictionary<string, float> DefaultValues { get; set; } = new();
        public List<string> Constraints { get; set; } = new();
        public Dimension ExpectedDimension { get; set; } = Dimension.Scalar;
    }

    /// <summary>Creates new physical laws from templates and parameter exploration.</summary>
    public sealed class LawInventor
    {
        private readonly LawLibrary _library;
        private readonly LawExpressionParser _parser;
        private readonly List<LawTemplate> _templates = new();

        public IReadOnlyList<LawTemplate> Templates => _templates;

        public LawInventor(LawLibrary library)
        {
            _library = library;
            _parser = new LawExpressionParser("");
            RegisterBuiltInTemplates();
        }

        private void RegisterBuiltInTemplates()
        {
            _templates.Add(new LawTemplate
            {
                Name = "Conservation Law",
                Category = "Conservation",
                ExpressionTemplate = "∂({var})/∂t + ∇·({var}*v) = {source}",
                VariableDescriptions = new() { ["var"] = "conserved quantity", ["v"] = "velocity", ["source"] = "source term" },
                ExpectedDimension = new Dimension(0, -3, -1, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Diffusion-Reaction",
                Category = "ReactionDiffusion",
                ExpressionTemplate = "∂u/∂t = D*∇²u + f(u)",
                VariableDescriptions = new() { ["u"] = "concentration", ["D"] = "diffusivity", ["f"] = "reaction rate" },
                ExpectedDimension = new Dimension(0, -3, -1, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Dispersion Relation",
                Category = "WaveDynamics",
                ExpressionTemplate = "ω² = {coeff}*k^n",
                VariableDescriptions = new() { ["ω"] = "angular frequency", ["k"] = "wavenumber", ["n"] = "power" },
                ExpectedDimension = new Dimension(0, 0, -2, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Power Law",
                Category = "Empirical",
                ExpressionTemplate = "{output} = {coeff}*{input}^{exponent}",
                VariableDescriptions = new() { ["output"] = "output", ["input"] = "input", ["coeff"] = "coefficient", ["exponent"] = "exponent" },
                ExpectedDimension = Dimension.Scalar
            });
            _templates.Add(new LawTemplate
            {
                Name = "Exponential Decay",
                Category = "Kinetics",
                ExpressionTemplate = "{var} = {initial}*exp(-{rate}*t)",
                VariableDescriptions = new() { ["var"] = "variable", ["initial"] = "initial value", ["rate"] = "decay rate" },
                ExpectedDimension = Dimension.Scalar
            });
            _templates.Add(new LawTemplate
            {
                Name = "Coupled Oscillators",
                Category = "Dynamics",
                ExpressionTemplate = "m*ẍ + c*ẋ + k*x = F_ext + {coupling}*y",
                VariableDescriptions = new() { ["m"] = "mass", ["c"] = "damping", ["k"] = "stiffness", ["F_ext"] = "external force", ["coupling"] = "coupling constant", ["y"] = "coupled variable" },
                ExpectedDimension = Dimension.Force
            });
        }

        /// <summary>Invent a law from a template with specific parameter values.</summary>
        public LawEntry InventFromTemplate(string templateName, Dictionary<string, float> parameters, string? lawId = null)
        {
            var template = _templates.FirstOrDefault(t => t.Name == templateName);
            if (template == null)
                throw new ArgumentException($"Template '{templateName}' not found");

            string expression = template.ExpressionTemplate;
            foreach (var kv in parameters)
            {
                expression = expression.Replace($"{{{kv.Key}}}", kv.Value.ToString(CultureInfo.InvariantCulture));
            }

            var entry = new LawEntry
            {
                Id = lawId ?? $"invented_{Guid.NewGuid().ToString("N")[..8]}",
                Name = $"Invented: {templateName}",
                Category = template.Category,
                Expression = expression,
                Description = $"Invented from template '{templateName}'",
                ResultDimension = template.ExpectedDimension
            };

            _library.Register(entry);
            return entry;
        }

        /// <summary>Explore parameter space to find valid laws.</summary>
        public List<LawEntry> ExploreParameterSpace(string templateName,
            Dictionary<string, (float Min, float Max, float Step)> parameterRanges,
            Func<string, bool> validator, int maxResults = 100)
        {
            var results = new List<LawEntry>();
            var template = _templates.FirstOrDefault(t => t.Name == templateName);
            if (template == null)
                return results;

            var paramNames = parameterRanges.Keys.ToList();
            var ranges = paramNames.Select(n => parameterRanges[n]).ToList();
            int[] indices = new int[paramNames.Count];
            int[] counts = ranges.Select(r => (int)((r.Max - r.Min) / r.Step) + 1).ToArray();
            int totalCombinations = counts.Aggregate(1, (a, b) => a * b);

            for (int combo = 0; combo < totalCombinations && results.Count < maxResults; combo++)
            {
                var parameters = new Dictionary<string, float>();
                int tempCombo = combo;
                for (int i = 0; i < paramNames.Count; i++)
                {
                    indices[i] = tempCombo % counts[i];
                    tempCombo /= counts[i];
                    float value = ranges[i].Min + indices[i] * ranges[i].Step;
                    parameters[paramNames[i]] = value;
                }

                string expression = template.ExpressionTemplate;
                foreach (var kv in parameters)
                    expression = expression.Replace($"{{{kv.Key}}}", kv.Value.ToString(CultureInfo.InvariantCulture));

                if (validator(expression))
                {
                    var entry = new LawEntry
                    {
                        Id = $"explored_{Guid.NewGuid().ToString("N")[..8]}",
                        Name = $"Explored: {templateName}",
                        Category = template.Category,
                        Expression = expression,
                        Description = $"Parameter exploration from '{templateName}'"
                    };
                    _library.Register(entry);
                    results.Add(entry);
                }
            }
            return results;
        }

        /// <summary>Generate variations of an existing law.</summary>
        public List<string> GenerateVariations(string expression, int count = 10)
        {
            var variations = new List<string>();
            var rng = new Random();
            var operations = new Func<string, string>[]
            {
                e => $"({e}) * (1 + 0.1*sin(t))",
                e => $"({e})^1.1",
                e => $"({e}) * exp(-0.01*t)",
                e => $"({e}) + 0.01*∇²({e})",
                e => $"({e}) * (1 - 0.05*rho/1000)",
                e => $"clamp({e}, -1000, 1000)",
                e => $"({e}) * (1 + 0.05*sign(sin(2*3.14159*x)))",
                e => $"tanh({e}/100)*100",
                e => $"({e}) * exp(-abs(x)/100)",
                e => $"({e}) * (1 + 0.1*cos(y*0.1))",
            };

            for (int i = 0; i < count; i++)
            {
                int opIdx = rng.Next(operations.Length);
                variations.Add(operations[opIdx](expression));
            }
            return variations;
        }

        /// <summary>Blend two law expressions.</summary>
        public string BlendLaws(string expressionA, string expressionB, float weightA = 0.5f)
        {
            float weightB = 1f - weightA;
            return $"({weightA.ToString(CultureInfo.InvariantCulture)}*({expressionA})) + ({weightB.ToString(CultureInfo.InvariantCulture)}*({expressionB}))";
        }

        /// <summary>Apply dimensional constraints to generate valid expressions.</summary>
        public List<string> ApplyDimensionalConstraint(string variableName, Dimension targetDim)
        {
            var results = new List<string>();
            var dimStr = targetDim.ToString();

            if (targetDim.IsCompatible(Dimension.Velocity))
                results.Add($"{variableName} = dx/dt");
            else if (targetDim.IsCompatible(Dimension.Acceleration))
                results.Add($"{variableName} = dv/dt");
            else if (targetDim.IsCompatible(Dimension.Force))
                results.Add($"{variableName} = m*{variableName}_accel");
            else if (targetDim.IsCompatible(Dimension.Energy))
                results.Add($"{variableName} = 0.5*m*v^2");
            else if (targetDim.IsCompatible(Dimension.Pressure))
                results.Add($"{variableName} = F/A");
            else if (targetDim.IsCompatible(Dimension.Density))
                results.Add($"{variableName} = m/V");
            else if (targetDim.Time > 0)
                results.Add($"{variableName} = t^{targetDim.Time.ToString(CultureInfo.InvariantCulture)}");
            else if (targetDim.Length > 0)
                results.Add($"{variableName} = x^{targetDim.Length.ToString(CultureInfo.InvariantCulture)}");
            else
                results.Add($"{variableName} = constant");

            return results;
        }
    }

    // =========================================================================
    // LawSimulationRunner — runs simulations with compiled laws
    // =========================================================================

    /// <summary>Configuration for a simulation run.</summary>
    public sealed class SimulationConfig
    {
        public float Duration { get; set; } = 1.0f;
        public float TimeStep { get; set; } = 0.001f;
        public int GridSize { get; set; } = 64;
        public bool RecordHistory { get; set; } = true;
        public int HistoryInterval { get; set; } = 10;
        public Func<int, PhysicsField, bool>? StopCondition { get; set; }
    }

    /// <summary>Result of a simulation run.</summary>
    public sealed class SimulationResult
    {
        public List<PhysicsField> Snapshots { get; set; } = new();
        public List<float> TimeSteps { get; set; } = new();
        public List<float> EnergyHistory { get; set; } = new();
        public List<float> ErrorHistory { get; set; } = new();
        public float TotalTime { get; set; }
        public bool Converged { get; set; }
        public int Iterations { get; set; }
        public string? ErrorMessage { get; set; }
    }

    /// <summary>Runs simulations with compiled law bytecodes.</summary>
    public sealed class LawSimulationRunner
    {
        private readonly LivingLawCompiler _compiler;

        public LawSimulationRunner(LivingLawCompiler compiler)
        {
            _compiler = compiler;
        }

        /// <summary>Run a simulation with a single compiled law.</summary>
        public SimulationResult RunSimulation(string lawId, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            config ??= new SimulationConfig();
            var result = new SimulationResult();
            var field = initialField ?? CreateTestField(config.GridSize);

            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(0f);
                result.EnergyHistory.Add(ComputeFieldEnergy(field));
            }

            var sw = Stopwatch.StartNew();
            int totalSteps = (int)(config.Duration / config.TimeStep);

            try
            {
                for (int step = 0; step < totalSteps; step++)
                {
                    _compiler.ApplyLaw(lawId, field, config.TimeStep);
                    field.Time += config.TimeStep;
                    result.Iterations = step + 1;

                    if (config.RecordHistory && step % config.HistoryInterval == 0)
                    {
                        result.Snapshots.Add(field.Clone());
                        result.TimeSteps.Add(field.Time);
                        result.EnergyHistory.Add(ComputeFieldEnergy(field));
                    }

                    float error = ComputeFieldEnergy(field);
                    result.ErrorHistory.Add(error);

                    if (config.StopCondition != null && config.StopCondition(step, field))
                    {
                        result.Converged = true;
                        break;
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                    {
                        result.ErrorMessage = $"Simulation diverged at step {step}";
                        break;
                    }
                }

                if (result.Converged == false && result.ErrorMessage == null)
                    result.Converged = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
            }

            sw.Stop();
            result.TotalTime = (float)sw.Elapsed.TotalSeconds;
            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(field.Time);
            }
            return result;
        }

        /// <summary>Run a simulation with multiple coupled laws.</summary>
        public SimulationResult RunCoupledSimulation(IReadOnlyList<string> lawIds, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            config ??= new SimulationConfig();
            var result = new SimulationResult();
            var field = initialField ?? CreateTestField(config.GridSize);

            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(0f);
                result.EnergyHistory.Add(ComputeFieldEnergy(field));
            }

            var sw = Stopwatch.StartNew();
            int totalSteps = (int)(config.Duration / config.TimeStep);

            try
            {
                for (int step = 0; step < totalSteps; step++)
                {
                    foreach (var lawId in lawIds)
                    {
                        _compiler.ApplyLaw(lawId, field, config.TimeStep);
                    }
                    field.Time += config.TimeStep;
                    result.Iterations = step + 1;

                    if (config.RecordHistory && step % config.HistoryInterval == 0)
                    {
                        result.Snapshots.Add(field.Clone());
                        result.TimeSteps.Add(field.Time);
                        result.EnergyHistory.Add(ComputeFieldEnergy(field));
                    }

                    float error = ComputeFieldEnergy(field);
                    result.ErrorHistory.Add(error);

                    if (config.StopCondition != null && config.StopCondition(step, field))
                    {
                        result.Converged = true;
                        break;
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                    {
                        result.ErrorMessage = $"Simulation diverged at step {step}";
                        break;
                    }
                }

                if (result.Converged == false && result.ErrorMessage == null)
                    result.Converged = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
            }

            sw.Stop();
            result.TotalTime = (float)sw.Elapsed.TotalSeconds;
            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(field.Time);
            }
            return result;
        }

        /// <summary>Compare two simulation runs.</summary>
        public ComparisonResult CompareSimulations(SimulationResult a, SimulationResult b)
        {
            int count = Math.Min(a.Snapshots.Count, b.Snapshots.Count);
            if (count == 0)
                return new ComparisonResult(0, 0, 0, 0, 0, 0, Array.Empty<string>(), true);

            float maxDiv = 0f, sumDiv = 0f;
            var diffs = new List<string>();

            for (int i = 0; i < count; i++)
            {
                var comp = LawComparison.ComparePhysicsFields(a.Snapshots[i], b.Snapshots[i]);
                if (comp.MaxDivergence > maxDiv)
                    maxDiv = comp.MaxDivergence;
                sumDiv += comp.MeanDivergence;
            }

            float meanDiv = sumDiv / count;
            float rmse = MathF.Sqrt(sumDiv * sumDiv / count);
            diffs.Add($"Compared {count} snapshots");
            if (MathF.Abs(a.TotalTime - b.TotalTime) > 0.01f)
                diffs.Add($"Total time difference: {a.TotalTime:F3}s vs {b.TotalTime:F3}s");

            return new ComparisonResult(maxDiv, meanDiv, rmse, 0, 0, 0, diffs.ToArray(),
                maxDiv < 0.01f && rmse < 0.01f);
        }

        private PhysicsField CreateTestField(int size)
        {
            var field = new PhysicsField(size, "simulation");
            float cx = size / 2f;
            for (int z = 0; z < size; z++)
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (size * size));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }
            return field;
        }

        private float ComputeFieldEnergy(PhysicsField field)
        {
            float energy = 0f;
            var temp = field.Temperature;
            for (int i = 0; i < temp.TotalCells; i++)
                energy += temp.Data[i] * temp.Data[i];
            return energy / temp.TotalCells;
        }
    }    // =========================================================================
    // LivingLawCompiler — core compiler orchestrating everything
    // =========================================================================

    /// <summary>The Living Law Compiler: loads, compiles, modifies, and applies physical laws.</summary>
    public sealed class LivingLawCompiler
    {
        private readonly LawLibrary _library;
        private readonly LawCompilerConfig _config;
        private readonly LawModificationEngine _modificationEngine;
        private readonly LawValidation _validation;
        private readonly LawInventor _inventor;
        private readonly LawSimulationRunner _simulationRunner;
        private readonly Dictionary<string, LawBytecode> _compiledCache = new();
        private readonly Dictionary<string, LawVersionTree> _versionTrees = new();
        private readonly Dictionary<string, LawApplicator> _applicators = new();
        private readonly ConcurrentDictionary<string, CompilationResult> _compilationResults = new();
        private long _totalCompilationTimeMs;
        private int _totalCompilations;

        public LawLibrary Library => _library;
        public LawCompilerConfig Config => _config;
        public LawModificationEngine ModificationEngine => _modificationEngine;
        public LawValidation Validation => _validation;
        public LawInventor Inventor => _inventor;
        public LawSimulationRunner SimulationRunner => _simulationRunner;
        public LawEventSystem Events { get; } = new();
        public long TotalCompilationTimeMs => Interlocked.Read(ref _totalCompilationTimeMs);
        public int TotalCompilations => Interlocked.CompareExchange(ref _totalCompilations, 0, 0);

        public LivingLawCompiler(LawCompilerConfig? config = null)
        {
            _config = config ?? new LawCompilerConfig();
            _library = LawLibrary.LoadBuiltIn();
            _modificationEngine = new LawModificationEngine(_library);
            _validation = new LawValidation(_library);
            _inventor = new LawInventor(_library);
            _simulationRunner = new LawSimulationRunner(this);
            RegisterDefaultApplicators();
        }

        public LivingLawCompiler(LawLibrary library, LawCompilerConfig? config = null)
        {
            _config = config ?? new LawCompilerConfig();
            _library = library;
            _modificationEngine = new LawModificationEngine(_library);
            _validation = new LawValidation(_library);
            _inventor = new LawInventor(_library);
            _simulationRunner = new LawSimulationRunner(this);
            RegisterDefaultApplicators();
        }

        private void RegisterDefaultApplicators()
        {
            _applicators["heat"] = new HeatApplicator();
            _applicators["wave"] = new WaveApplicator();
            _applicators["elasticity"] = new ElasticityApplicator();
            _applicators["advection"] = new AdvectionApplicator();
            _applicators["diffusion"] = new DiffusionApplicator();
            _applicators["incompressible_ns"] = new IncompressibleNSApplicator();
            _applicators["electromagnetic"] = new ElectromagneticApplicator();
            _applicators["gravity"] = new GravityApplicator();
            _applicators["generic"] = new GenericBytecodeApplicator("temperature");
        }

        /// <summary>Register a custom applicator.</summary>
        public void RegisterApplicator(string key, LawApplicator applicator)
        {
            _applicators[key] = applicator;
        }

        /// <summary>Load a law from the library by ID.</summary>
        public LawEntry? LoadLaw(string lawId) => _library.GetLaw(lawId);

        /// <summary>Compile a law expression string.</summary>
        public CompilationResult Compile(string expression, string? lawId = null)
        {
            var sw = Stopwatch.StartNew();
            Events.Raise(LawEventType.CompilationStarted, lawId, expression);
            try
            {
                var parser = new LawExpressionParser(expression);
                var ast = parser.Parse();

                if (parser.Errors.Count > 0)
                {
                    sw.Stop();
                    Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                    Interlocked.Increment(ref _totalCompilations);
                    var errors = parser.Errors.ToArray();
                    Events.Raise(LawEventType.CompilationFailed, lawId, expression,
                        string.Join("; ", errors));
                    return CompilationResult.Fail("Parse errors", errors);
                }

                var bytecode = parser.CompileToBytecode(ast);
                bytecode.OriginalExpression = expression;

                if (_config.EnableValidation)
                {
                    var valResult = LawValidation.ValidateDimensional(ast);
                    if (!valResult.IsValid)
                    {
                        sw.Stop();
                        Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                        Interlocked.Increment(ref _totalCompilations);
                        Events.Raise(LawEventType.ValidationFailed, lawId, expression,
                            string.Join("; ", valResult.Errors));
                        return CompilationResult.Fail("Validation errors", valResult.Errors);
                    }
                }

                sw.Stop();
                Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                Interlocked.Increment(ref _totalCompilations);

                string cacheKey = lawId ?? expression;
                _compiledCache[cacheKey] = bytecode;

                if (!string.IsNullOrEmpty(lawId))
                    CreateVersionTree(lawId, expression);

                var result = CompilationResult.Ok(
                    $"Compiled successfully in {sw.ElapsedMilliseconds}ms",
                    bytecode, bytecode.InstructionCount, sw.ElapsedMilliseconds);
                _compilationResults[cacheKey] = result;
                Events.Raise(LawEventType.CompilationCompleted, lawId, expression,
                    $"{result.InstructionCount} ops, {sw.ElapsedMilliseconds} ms");
                return result;
            }
            catch (Exception ex)
            {
                sw.Stop();
                Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                Interlocked.Increment(ref _totalCompilations);
                Events.Raise(LawEventType.CompilationFailed, lawId, expression, ex.Message);
                return CompilationResult.Fail($"Compilation failed: {ex.Message}", new[] { ex.Message });
            }
        }

        /// <summary>Load and compile a law from the library.</summary>
        public CompilationResult CompileFromLibrary(string lawId)
        {
            var law = _library.GetLaw(lawId);
            if (law == null)
                return CompilationResult.Fail($"Law '{lawId}' not found", new[] { $"Law '{lawId}' not found in library" });
            return Compile(law.Expression, lawId);
        }

        /// <summary>Hot-reload: modify a law expression and recompile without stopping.</summary>
        public CompilationResult HotReload(string lawId, string newExpression)
        {
            Events.Raise(LawEventType.HotReloadTriggered, lawId, newExpression);
            var law = _library.GetLaw(lawId);
            if (law != null)
                law.Expression = newExpression;

            if (_versionTrees.TryGetValue(lawId, out var tree))
                tree.Commit(newExpression, $"Hot-reload at {DateTime.UtcNow:HH:mm:ss}");

            _compiledCache.Remove(lawId);
            var result = Compile(newExpression, lawId);
            Events.Raise(LawEventType.HotReloadCompleted, lawId, newExpression, result.Message);
            return result;
        }

        /// <summary>Create a version tree for a law expression.</summary>
        public LawVersionTree CreateVersionTree(string lawId, string? initialExpression = null)
        {
            if (_versionTrees.TryGetValue(lawId, out var existing))
                return existing;

            string expr = initialExpression ?? "";
            if (string.IsNullOrEmpty(expr))
            {
                var law = _library.GetLaw(lawId);
                if (law != null)
                    expr = law.Expression;
            }

            var tree = new LawVersionTree(expr);
            _versionTrees[lawId] = tree;
            return tree;
        }

        /// <summary>Get the version tree for a law.</summary>
        public LawVersionTree? GetVersionTree(string lawId) =>
            _versionTrees.TryGetValue(lawId, out var tree) ? tree : null;

        /// <summary>Apply a compiled law to a physics field.</summary>
        public void ApplyLaw(string lawId, PhysicsField field, float? dt = null)
        {
            if (!_compiledCache.TryGetValue(lawId, out var bytecode))
            {
                var result = CompileFromLibrary(lawId);
                if (!result.Success || result.Bytecode == null)
                    throw new InvalidOperationException($"Failed to compile law '{lawId}': {result.Message}");
                bytecode = result.Bytecode;
            }

            float timeStep = dt ?? _config.TimeStep;
            string applicatorKey = DetermineApplicatorKey(lawId);
            if (_applicators.TryGetValue(applicatorKey, out var applicator))
                applicator.Apply(bytecode, field, timeStep, _config);
            else
            {
                var generic = new GenericBytecodeApplicator("temperature");
                generic.Apply(bytecode, field, timeStep, _config);
            }
        }

        /// <summary>Apply a custom compiled bytecode to a field.</summary>
        public void ApplyBytecode(LawBytecode bytecode, PhysicsField field, string targetField = "temperature", float? dt = null)
        {
            float timeStep = dt ?? _config.TimeStep;
            var applicator = new GenericBytecodeApplicator(targetField);
            applicator.Apply(bytecode, field, timeStep, _config);
        }

        private string DetermineApplicatorKey(string lawId)
        {
            var law = _library.GetLaw(lawId);
            if (law == null)
                return "generic";
            return law.Category.ToLowerInvariant() switch
            {
                "thermaldynamics" or "thermal" => "heat",
                "wavedynamics" or "wave" or "acoustic" => "wave",
                "elasticity" or "solid" => "elasticity",
                "fluiddynamics" or "fluid" or "navier_stokes" => "incompressible_ns",
                "electrodynamics" or "em" or "electromagnetic" => "electromagnetic",
                "gravitation" or "gravity" => "gravity",
                _ => "generic"
            };
        }

        /// <summary>Validate a law expression.</summary>
        public ValidationResult Validate(string expression, string? lawId = null)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            LawEntry? knownLaw = lawId != null ? _library.GetLaw(lawId) : null;
            return LawValidation.ValidateDimensional(ast, knownLaw);
        }

        /// <summary>Comprehensive validation of a law.</summary>
        public ValidationResult ComprehensiveValidate(string expression, string? lawId = null, PhysicsField? testField = null)
        {
            return _validation.ComprehensiveValidate(expression, lawId, testField);
        }

        /// <summary>Modify a law using the modification engine.</summary>
        public string ModifyLaw(string lawId, LawModification modification)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            string newExpression = _modificationEngine.ApplyModification(law.Expression, modification);
            law.Expression = newExpression;
            return newExpression;
        }

        /// <summary>Modify a law using natural language instruction.</summary>
        public string ModifyLawNaturalLanguage(string lawId, string instruction)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            string newExpression = _modificationEngine.ApplyNaturalLanguageModification(law.Expression, instruction);
            law.Expression = newExpression;
            return newExpression;
        }

        /// <summary>Compare two law versions applied to the same field.</summary>
        public ComparisonResult CompareLawVersions(string expressionA, string expressionB, PhysicsField? testField = null)
        {
            var field = testField ?? CreateTestField();
            var fieldB = field.Clone();

            var resultA = Compile(expressionA);
            var resultB = Compile(expressionB);

            if (resultA.Success && resultA.Bytecode != null)
                ApplyBytecode(resultA.Bytecode, field);
            if (resultB.Success && resultB.Bytecode != null)
                ApplyBytecode(resultB.Bytecode, fieldB);

            int editDist = LawVersionTree.ComputeEditDistance(expressionA, expressionB);
            return LawComparison.CompareFields(field.Temperature, fieldB.Temperature, editDist);
        }

        /// <summary>Compare two law library entries applied to a field.</summary>
        public ComparisonResult CompareLaws(string lawIdA, string lawIdB, PhysicsField? testField = null)
        {
            var lawA = _library.GetLaw(lawIdA) ?? throw new ArgumentException($"Law '{lawIdA}' not found");
            var lawB = _library.GetLaw(lawIdB) ?? throw new ArgumentException($"Law '{lawIdB}' not found");
            return CompareLawVersions(lawA.Expression, lawB.Expression, testField);
        }

        /// <summary>Run a simulation with a law.</summary>
        public SimulationResult RunSimulation(string lawId, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            return _simulationRunner.RunSimulation(lawId, initialField, config);
        }

        /// <summary>Run a coupled simulation with multiple laws.</summary>
        public SimulationResult RunCoupledSimulation(IReadOnlyList<string> lawIds, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            return _simulationRunner.RunCoupledSimulation(lawIds, initialField, config);
        }

        /// <summary>Create a test field with some initial conditions.</summary>
        private PhysicsField CreateTestField()
        {
            int size = _config.GridSize;
            var field = new PhysicsField(size, "test");
            float cx = size / 2f;
            for (int z = 0; z < size; z++)
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (size * size));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }
            return field;
        }

        /// <summary>Get compilation statistics.</summary>
        public (int TotalCompilations, long TotalTimeMs, float AvgTimeMs) GetStatistics()
        {
            int total = Interlocked.CompareExchange(ref _totalCompilations, 0, 0);
            long time = Interlocked.Read(ref _totalCompilationTimeMs);
            return (total, time, total > 0 ? (float)time / total : 0f);
        }

        /// <summary>Clear the compilation cache.</summary>
        public void ClearCache() { _compiledCache.Clear(); _compilationResults.Clear(); }

        /// <summary>Get all cached compiled bytecodes.</summary>
        public IReadOnlyDictionary<string, LawBytecode> GetCachedBytecodes() => _compiledCache;

        /// <summary>Compile and execute a law expression with given parameters.</summary>
        public float Evaluate(string expression, PhysicsField? field, Dictionary<string, float>? parameters = null)
        {
            var result = Compile(expression);
            if (!result.Success || result.Bytecode == null)
                throw new InvalidOperationException($"Failed to compile: {result.Message}");
            var interpreter = new BytecodeInterpreter(new GasMeter(_config.GasLimit));
            return interpreter.Execute(result.Bytecode, field, null, parameters);
        }

        /// <summary>Compile and evaluate a law from the library.</summary>
        public float EvaluateLaw(string lawId, PhysicsField? field, Dictionary<string, float>? parameters = null)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            var allParams = new Dictionary<string, float>(law.Constants);
            if (parameters != null)
                foreach (var kv in parameters)
                    allParams[kv.Key] = kv.Value;
            return Evaluate(law.Expression, field, allParams);
        }

        /// <summary>Batch compile multiple expressions.</summary>
        public CompilationResult[] CompileBatch(string[] expressions)
        {
            var results = new CompilationResult[expressions.Length];
            for (int i = 0; i < expressions.Length; i++)
                results[i] = Compile(expressions[i]);
            return results;
        }

        /// <summary>Batch compile all laws in a category.</summary>
        public CompilationResult[] CompileCategory(string category)
        {
            var laws = _library.SearchByCategory(category);
            var results = new CompilationResult[laws.Count];
            for (int i = 0; i < laws.Count; i++)
                results[i] = Compile(laws[i].Expression, laws[i].Id);
            return results;
        }

        /// <summary>Get a list of all available applicator keys.</summary>
        public IReadOnlyCollection<string> GetApplicatorKeys() => _applicators.Keys;

        /// <summary>Export the compiler state as JSON.</summary>
        public string ExportState()
        {
            var state = new
            {
                Config = new
                {
                    _config.Tolerance,
                    _config.MaxIterations,
                    _config.TimeStep,
                    _config.CellSize,
                    BoundaryCondition = _config.BoundaryCondition.ToString(),
                    _config.BoundaryValue,
                    _config.GasLimit,
                    _config.CflLimit,
                    _config.GridSize,
                    _config.EnableHotReload,
                    _config.EnableValidation,
                    Solver = _config.Solver.ToString()
                },
                Statistics = GetStatistics(),
                CachedLaws = _compiledCache.Keys.ToArray(),
                VersionTrees = _versionTrees.Keys.ToArray(),
                ModificationHistory = _modificationEngine.History.Count,
                AvailableLaws = _library.AllEntries.Select(e => new { e.Id, e.Name, e.Category }).ToArray(),
                Applicators = _applicators.Keys.ToArray()
            };
            return JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>Invent a new law from a template.</summary>
        public LawEntry InventLaw(string templateName, Dictionary<string, float> parameters, string? lawId = null)
        {
            return _inventor.InventFromTemplate(templateName, parameters, lawId);
        }

        /// <summary>Generate variations of a law expression.</summary>
        public List<string> GenerateVariations(string expression, int count = 10)
        {
            return _inventor.GenerateVariations(expression, count);
        }

        /// <summary>Blend two law expressions.</summary>
        public string BlendLaws(string expressionA, string expressionB, float weightA = 0.5f)
        {
            return _inventor.BlendLaws(expressionA, expressionB, weightA);
        }

        /// <summary>Export all version trees to a dictionary.</summary>
        public Dictionary<string, List<(string Id, string Expression, DateTime Timestamp, string Description)>> ExportAllVersionHistories()
        {
            var result = new Dictionary<string, List<(string, string, DateTime, string)>>();
            foreach (var kv in _versionTrees)
                result[kv.Key] = kv.Value.ExportHistory();
            return result;
        }

        /// <summary>Import a law expression and create a new entry.</summary>
        public LawEntry ImportLaw(string id, string name, string category, string expression, string description = "")
        {
            var entry = new LawEntry
            {
                Id = id,
                Name = name,
                Category = category,
                Expression = expression,
                Description = description
            };
            _library.Register(entry);
            return entry;
        }

        /// <summary>Get all laws in the library.</summary>
        public IReadOnlyList<LawEntry> GetAllLaws() => _library.AllEntries;

        /// <summary>Search laws by name.</summary>
        public IReadOnlyList<LawEntry> SearchLaws(string query) => _library.SearchByName(query);

        /// <summary>Search laws by category.</summary>
        public IReadOnlyList<LawEntry> SearchLawsByCategory(string category) => _library.SearchByCategory(category);

        /// <summary>Remove a compiled law from the cache.</summary>
        public bool RemoveFromCache(string lawId) => _compiledCache.Remove(lawId);

        /// <summary>Check if a law is compiled and cached.</summary>
        public bool IsCompiled(string lawId) => _compiledCache.ContainsKey(lawId);

        /// <summary>Get the compiled bytecode for a law.</summary>
        public LawBytecode? GetCompiledBytecode(string lawId) =>
            _compiledCache.TryGetValue(lawId, out var bc) ? bc : null;

        /// <summary>Reset the compiler state (clear caches, reset statistics).</summary>
        public void Reset()
        {
            ClearCache();
            _versionTrees.Clear();
            Interlocked.Exchange(ref _totalCompilationTimeMs, 0);
            Interlocked.Exchange(ref _totalCompilations, 0);
        }
    }

    // =========================================================================
    // LawSerializer — serialization/deserialization of laws and bytecode
    // =========================================================================

    /// <summary>Serializes and deserializes law entries, bytecodes, and version trees.</summary>
    public sealed class LawSerializer
    {
        private static readonly JsonSerializerOptions _options = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>Serialize a law entry to JSON.</summary>
        public static string SerializeLawEntry(LawEntry entry) => JsonSerializer.Serialize(entry, _options);

        /// <summary>Deserialize a law entry from JSON.</summary>
        public static LawEntry? DeserializeLawEntry(string json) => JsonSerializer.Deserialize<LawEntry>(json, _options);

        /// <summary>Serialize multiple law entries to JSON.</summary>
        public static string SerializeLawEntries(IReadOnlyList<LawEntry> entries) =>
            JsonSerializer.Serialize(entries, _options);

        /// <summary>Deserialize law entries from JSON.</summary>
        public static LawEntry[] DeserializeLawEntries(string json) =>
            JsonSerializer.Deserialize<LawEntry[]>(json, _options) ?? Array.Empty<LawEntry>();

        /// <summary>Save a law library to a JSON file.</summary>
        public static void SaveLibrary(LawLibrary library, string filePath)
        {
            var json = SerializeLawEntries(library.AllEntries);
            File.WriteAllText(filePath, json);
        }

        /// <summary>Load a law library from a JSON file.</summary>
        public static LawLibrary LoadLibrary(string filePath)
        {
            var library = new LawLibrary();
            var entries = DeserializeLawEntries(File.ReadAllText(filePath));
            foreach (var entry in entries)
                library.Register(entry);
            return library;
        }

        /// <summary>Serialize version tree history to JSON.</summary>
        public static string SerializeVersionHistory(LawVersionTree tree)
        {
            var history = tree.ExportHistory();
            var records = history.Select(h => new
            {
                h.Id,
                h.Expression,
                h.Timestamp,
                h.Description
            });
            return JsonSerializer.Serialize(records, _options);
        }

        /// <summary>Serialize compilation results.</summary>
        public static string SerializeCompilationResult(CompilationResult result) =>
            JsonSerializer.Serialize(new
            {
                result.Success,
                result.Message,
                result.Errors,
                result.Warnings,
                result.InstructionCount,
                result.CompilationTimeMs,
                ResultDimension = result.Bytecode?.ResultDimension.ToString() ?? "N/A"
            }, _options);

        /// <summary>Serialize comparison results.</summary>
        public static string SerializeComparisonResult(ComparisonResult result) =>
            JsonSerializer.Serialize(new
            {
                result.MaxDivergence,
                result.MeanDivergence,
                result.RootMeanSquareError,
                result.KolmogorovSmirnovStatistic,
                result.StructuralSimilarity,
                result.ExpressionEditDistance,
                result.Differences,
                result.PhysicallyEquivalent
            }, _options);

        /// <summary>Serialize simulation results.</summary>
        public static string SerializeSimulationResult(SimulationResult result) =>
            JsonSerializer.Serialize(new
            {
                result.TotalTime,
                result.Converged,
                result.Iterations,
                result.ErrorMessage,
                SnapshotCount = result.Snapshots.Count,
                result.TimeSteps,
                result.EnergyHistory,
                result.ErrorHistory
            }, _options);

        /// <summary>Export a law to a compact string format.</summary>
        public static string ExportCompact(LawEntry entry)
        {
            return $"ID:{entry.Id}|CAT:{entry.Category}|NAME:{entry.Name}|EXPR:{entry.Expression}";
        }

        /// <summary>Import a law from a compact string format.</summary>
        public static LawEntry? ImportCompact(string compact)
        {
            var parts = compact.Split('|');
            if (parts.Length < 4)
                return null;
            var entry = new LawEntry();
            foreach (var part in parts)
            {
                var kv = part.Split(':', 2);
                if (kv.Length != 2)
                    continue;
                switch (kv[0])
                {
                    case "ID":
                        entry.Id = kv[1];
                        break;
                    case "CAT":
                        entry.Category = kv[1];
                        break;
                    case "NAME":
                        entry.Name = kv[1];
                        break;
                    case "EXPR":
                        entry.Expression = kv[1];
                        break;
                    case "DESC":
                        entry.Description = kv[1];
                        break;
                }
            }
            return entry;
        }

        /// <summary>Serialize a PhysicsField to JSON (for checkpointing).</summary>
        public static string SerializeField(PhysicsField field)
        {
            return JsonSerializer.Serialize(new
            {
                field.Name,
                field.GridSize,
                field.Time,
                Temperature = field.Temperature.Data,
                Pressure = field.Pressure.Data,
                Density = field.Density.Data,
                VelocityX = field.VelocityX.Data,
                VelocityY = field.VelocityY.Data,
                VelocityZ = field.VelocityZ.Data
            }, _options);
        }

        /// <summary>Deserialize a PhysicsField from JSON.</summary>
        public static PhysicsField? DeserializeField(string json)
        {
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            int gridSize = root.GetProperty("gridSize").GetInt32();
            string name = root.GetProperty("name").GetString() ?? "deserialized";
            float time = root.GetProperty("time").GetSingle();
            var field = new PhysicsField(gridSize, name);
            field.Time = time;

            if (root.TryGetProperty("temperature", out var tempEl))
                CopyJsonArrayToGrid(tempEl, field.Temperature);
            if (root.TryGetProperty("pressure", out var presEl))
                CopyJsonArrayToGrid(presEl, field.Pressure);
            if (root.TryGetProperty("density", out var densEl))
                CopyJsonArrayToGrid(densEl, field.Density);
            if (root.TryGetProperty("velocityX", out var vxEl))
                CopyJsonArrayToGrid(vxEl, field.VelocityX);
            if (root.TryGetProperty("velocityY", out var vyEl))
                CopyJsonArrayToGrid(vyEl, field.VelocityY);
            if (root.TryGetProperty("velocityZ", out var vzEl))
                CopyJsonArrayToGrid(vzEl, field.VelocityZ);

            return field;
        }

        private static void CopyJsonArrayToGrid(JsonElement array, FieldGrid grid)
        {
            int idx = 0;
            int total = Math.Min(array.GetArrayLength(), grid.TotalCells);
            foreach (var item in array.EnumerateArray())
            {
                if (idx >= total)
                    break;
                int z = idx / (grid.SizeX * grid.SizeY);
                int rem = idx % (grid.SizeX * grid.SizeY);
                int y = rem / grid.SizeX;
                int x = rem % grid.SizeX;
                grid[x, y, z] = item.GetSingle();
                idx++;
            }
        }

        /// <summary>Save a checkpoint of the entire compiler state.</summary>
        public static void SaveCheckpoint(LivingLawCompiler compiler, string directory)
        {
            Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "compiler_state.json"), compiler.ExportState());
            File.WriteAllText(Path.Combine(directory, "library.json"),
                SerializeLawEntries(compiler.GetAllLaws()));
            foreach (var law in compiler.GetAllLaws())
            {
                string lawFile = Path.Combine(directory, $"law_{law.Id}.json");
                File.WriteAllText(lawFile, SerializeLawEntry(law));
            }
        }

        /// <summary>Load a checkpoint of the compiler state.</summary>
        public static (LawLibrary Library, string? StateJson) LoadCheckpoint(string directory)
        {
            string stateJson = File.Exists(Path.Combine(directory, "compiler_state.json"))
                ? File.ReadAllText(Path.Combine(directory, "compiler_state.json")) : "";
            var library = File.Exists(Path.Combine(directory, "library.json"))
                ? LoadLibrary(Path.Combine(directory, "library.json")) : LawLibrary.LoadBuiltIn();
            return (library, stateJson);
        }
    }

    // =========================================================================
    // LawParserFactory — creates parsers with predefined configurations
    // =========================================================================

    /// <summary>Factory for creating configured expression parsers.</summary>
    public static class LawParserFactory
    {
        /// <summary>Create a parser configured for fluid dynamics expressions.</summary>
        public static LawExpressionParser CreateFluidDynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for thermodynamics expressions.</summary>
        public static LawExpressionParser CreateThermodynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for electrodynamics expressions.</summary>
        public static LawExpressionParser CreateElectrodynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for general expressions.</summary>
        public static LawExpressionParser CreateGeneralParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Parse and validate an expression in one step.</summary>
        public static (AstNode Ast, LawBytecode Bytecode, bool IsValid, string[] Errors) ParseAndValidate(string expression)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            bool isValid = parser.Errors.Count == 0;
            LawBytecode bytecode = isValid ? parser.CompileToBytecode(ast) : new LawBytecode();
            return (ast, bytecode, isValid, parser.Errors.ToArray());
        }

        /// <summary>Quick-evaluate a simple expression with given variable values.</summary>
        public static float QuickEvaluate(string expression, Dictionary<string, float>? variables = null)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            var bytecode = parser.CompileToBytecode(ast);
            var interpreter = new BytecodeInterpreter();
            return interpreter.Execute(bytecode, null, null, variables);
        }
    }

    // =========================================================================
    // LawCache — thread-safe caching for compiled bytecodes
    // =========================================================================

    /// <summary>Thread-safe cache for compiled law bytecodes with LRU eviction.</summary>
    public sealed class LawCache
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _cache = new();
        private readonly int _maxEntries;
        private long _hits;
        private long _misses;
        private long _evictions;

        public int Count => _cache.Count;
        public long Hits => Interlocked.Read(ref _hits);
        public long Misses => Interlocked.Read(ref _misses);
        public long Evictions => Interlocked.Read(ref _evictions);

        private sealed class CacheEntry
        {
            public LawBytecode Bytecode { get; set; } = null!;
            public DateTime LastAccessed { get; set; }
            public int AccessCount { get; set; }
        }

        public LawCache(int maxEntries = 1000)
        {
            _maxEntries = maxEntries;
        }

        /// <summary>Try to get a cached bytecode.</summary>
        public bool TryGet(string key, [MaybeNullWhen(false)] out LawBytecode bytecode)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                entry.LastAccessed = DateTime.UtcNow;
                entry.AccessCount++;
                bytecode = entry.Bytecode;
                Interlocked.Increment(ref _hits);
                return true;
            }
            bytecode = null!;
            Interlocked.Increment(ref _misses);
            return false;
        }

        /// <summary>Store a bytecode in the cache.</summary>
        public void Store(string key, LawBytecode bytecode)
        {
            if (_cache.Count >= _maxEntries)
            {
                EvictLRU();
            }
            _cache[key] = new CacheEntry
            {
                Bytecode = bytecode,
                LastAccessed = DateTime.UtcNow,
                AccessCount = 1
            };
        }

        /// <summary>Remove a specific entry from the cache.</summary>
        public bool Remove(string key) => _cache.TryRemove(key, out _);

        /// <summary>Clear the entire cache.</summary>
        public void Clear() { _cache.Clear(); Interlocked.Exchange(ref _hits, 0); Interlocked.Exchange(ref _misses, 0); Interlocked.Exchange(ref _evictions, 0); }

        private void EvictLRU()
        {
            string? oldestKey = null;
            DateTime oldestTime = DateTime.MaxValue;
            foreach (var kv in _cache)
            {
                if (kv.Value.LastAccessed < oldestTime)
                {
                    oldestTime = kv.Value.LastAccessed;
                    oldestKey = kv.Key;
                }
            }
            if (oldestKey != null)
            {
                _cache.TryRemove(oldestKey, out _);
                Interlocked.Increment(ref _evictions);
            }
        }

        /// <summary>Get cache statistics.</summary>
        public (long Hits, long Misses, long Evictions, int Count, float HitRate) GetStats()
        {
            long h = Hits, m = Misses;
            float rate = h + m > 0 ? (float)h / (h + m) : 0f;
            return (h, m, Evictions, Count, rate);
        }

        /// <summary>Get the most frequently accessed entries.</summary>
        public IReadOnlyList<(string Key, int AccessCount)> GetMostAccessed(int count = 10)
        {
            return _cache.OrderByDescending(kv => kv.Value.AccessCount)
                .Take(count)
                .Select(kv => (kv.Key, kv.Value.AccessCount))
                .ToList();
        }
    }

    // =========================================================================
    // LawEventSystem — event system for compiler lifecycle
    // =========================================================================

    /// <summary>Events that can occur during compilation and law processing.</summary>
    public enum LawEventType
    {
        CompilationStarted, CompilationCompleted, CompilationFailed,
        HotReloadTriggered, HotReloadCompleted,
        VersionCreated, VersionRolledBack, VersionForked, VersionMerged,
        LawModified, LawApplied, ValidationCompleted, ValidationFailed,
        CacheHit, CacheMiss, CacheEviction,
        SimulationStarted, SimulationCompleted, SimulationFailed,
        LawInvented, LawImported, LawExported
    }

    /// <summary>Event data for law compiler events.</summary>
    public sealed class LawEventArgs : EventArgs
    {
        public LawEventType EventType { get; init; }
        public string? LawId { get; init; }
        public string? Expression { get; init; }
        public string? Message { get; init; }
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
        public Dictionary<string, object>? Metadata { get; init; }
    }

    /// <summary>Event handler for law compiler events.</summary>
    public delegate void LawEventHandler(object sender, LawEventArgs args);

    /// <summary>Event system for the Living Law Compiler.</summary>
    public sealed class LawEventSystem
    {
        private readonly Dictionary<LawEventType, List<LawEventHandler>> _handlers = new();
        private readonly List<LawEventArgs> _eventLog = new();
        private readonly object _logLock = new();
        private int _maxLogSize;

        public IReadOnlyList<LawEventArgs> EventLog
        {
            get { lock (_logLock) return _eventLog.ToList(); }
        }

        public LawEventSystem(int maxLogSize = 10000)
        {
            _maxLogSize = maxLogSize;
        }

        /// <summary>Subscribe to an event type.</summary>
        public void Subscribe(LawEventType eventType, LawEventHandler handler)
        {
            if (!_handlers.TryGetValue(eventType, out var list))
            {
                list = new List<LawEventHandler>();
                _handlers[eventType] = list;
            }
            list.Add(handler);
        }

        /// <summary>Unsubscribe from an event type.</summary>
        public void Unsubscribe(LawEventType eventType, LawEventHandler handler)
        {
            if (_handlers.TryGetValue(eventType, out var list))
                list.Remove(handler);
        }

        /// <summary>Raise an event.</summary>
        public void Raise(LawEventType eventType, string? lawId = null, string? expression = null, string? message = null, Dictionary<string, object>? metadata = null)
        {
            var args = new LawEventArgs
            {
                EventType = eventType,
                LawId = lawId,
                Expression = expression,
                Message = message,
                Metadata = metadata
            };

            lock (_logLock)
            {
                _eventLog.Add(args);
                if (_eventLog.Count > _maxLogSize)
                    _eventLog.RemoveRange(0, _eventLog.Count - _maxLogSize);
            }

            if (_handlers.TryGetValue(eventType, out var list))
            {
                foreach (var handler in list)
                {
                    try
                    { handler(this, args); }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("LivingLawCompiler", $"Law event handler for '{eventType}' threw an exception.", ex);
                    }
                }
            }
        }

        /// <summary>Get events of a specific type.</summary>
        public IReadOnlyList<LawEventArgs> GetEvents(LawEventType eventType, int maxCount = 100)
        {
            lock (_logLock)
            {
                return _eventLog.Where(e => e.EventType == eventType)
                    .TakeLast(maxCount).ToList();
            }
        }

        /// <summary>Get events for a specific law.</summary>
        public IReadOnlyList<LawEventArgs> GetEventsForLaw(string lawId, int maxCount = 100)
        {
            lock (_logLock)
            {
                return _eventLog.Where(e => e.LawId == lawId)
                    .TakeLast(maxCount).ToList();
            }
        }

        /// <summary>Clear the event log.</summary>
        public void ClearLog()
        {
            lock (_logLock)
            { _eventLog.Clear(); }
        }

        /// <summary>Get event statistics.</summary>
        public Dictionary<LawEventType, int> GetStatistics()
        {
            lock (_logLock)
            {
                return _eventLog.GroupBy(e => e.EventType)
                    .ToDictionary(g => g.Key, g => g.Count());
            }
        }
    }

    // =========================================================================
    // LawGraph — dependency graph between laws
    // =========================================================================

    /// <summary>Represents a dependency relationship between two laws.</summary>
    public sealed class LawDependency
    {
        public string SourceLawId { get; init; } = "";
        public string TargetLawId { get; init; } = "";
        public string DependencyType { get; init; } = "uses";
        public float Strength { get; init; } = 1.0f;
        public string Description { get; init; } = "";
    }

    /// <summary>Directed graph of law dependencies and couplings.</summary>
    public sealed class LawGraph
    {
        private readonly Dictionary<string, List<LawDependency>> _adjacency = new();
        private readonly Dictionary<string, List<LawDependency>> _reverseAdjacency = new();

        /// <summary>Add a dependency between two laws.</summary>
        public void AddDependency(LawDependency dependency)
        {
            if (!_adjacency.TryGetValue(dependency.SourceLawId, out var list))
            {
                list = new List<LawDependency>();
                _adjacency[dependency.SourceLawId] = list;
            }
            list.Add(dependency);

            if (!_reverseAdjacency.TryGetValue(dependency.TargetLawId, out var rList))
            {
                rList = new List<LawDependency>();
                _reverseAdjacency[dependency.TargetLawId] = rList;
            }
            rList.Add(dependency);
        }

        /// <summary>Get all dependencies of a law (laws it depends on).</summary>
        public IReadOnlyList<LawDependency> GetDependencies(string lawId) =>
            _adjacency.TryGetValue(lawId, out var list) ? list : Array.Empty<LawDependency>();

        /// <summary>Get all dependents of a law (laws that depend on it).</summary>
        public IReadOnlyList<LawDependency> GetDependents(string lawId) =>
            _reverseAdjacency.TryGetValue(lawId, out var list) ? list : Array.Empty<LawDependency>();

        /// <summary>Check if a dependency cycle exists.</summary>
        public bool HasCycle()
        {
            var visited = new HashSet<string>();
            var inStack = new HashSet<string>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node) && HasCycleDFS(node, visited, inStack))
                    return true;
            }
            return false;
        }

        private bool HasCycleDFS(string node, HashSet<string> visited, HashSet<string> inStack)
        {
            visited.Add(node);
            inStack.Add(node);

            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    if (!visited.Contains(dep.TargetLawId))
                    {
                        if (HasCycleDFS(dep.TargetLawId, visited, inStack))
                            return true;
                    }
                    else if (inStack.Contains(dep.TargetLawId))
                        return true;
                }
            }

            inStack.Remove(node);
            return false;
        }

        /// <summary>Topological sort of laws.</summary>
        public List<string> TopologicalSort()
        {
            var result = new List<string>();
            var visited = new HashSet<string>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node))
                    TopologicalSortDFS(node, visited, result);
            }

            result.Reverse();
            return result;
        }

        private void TopologicalSortDFS(string node, HashSet<string> visited, List<string> result)
        {
            visited.Add(node);
            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    if (!visited.Contains(dep.TargetLawId))
                        TopologicalSortDFS(dep.TargetLawId, visited, result);
                }
            }
            result.Add(node);
        }

        /// <summary>Find all laws in a connected component.</summary>
        public List<string> GetConnectedComponent(string startLawId)
        {
            var component = new HashSet<string>();
            var queue = new Queue<string>();
            queue.Enqueue(startLawId);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (!component.Add(current))
                    continue;

                if (_adjacency.TryGetValue(current, out var deps))
                    foreach (var dep in deps)
                        if (!component.Contains(dep.TargetLawId))
                            queue.Enqueue(dep.TargetLawId);

                if (_reverseAdjacency.TryGetValue(current, out var rDeps))
                    foreach (var dep in rDeps)
                        if (!component.Contains(dep.SourceLawId))
                            queue.Enqueue(dep.SourceLawId);
            }

            return component.ToList();
        }

        /// <summary>Get the longest dependency chain from a given law.</summary>
        public int GetLongestChain(string lawId)
        {
            var visited = new HashSet<string>();
            return GetLongestChainDFS(lawId, visited);
        }

        private int GetLongestChainDFS(string node, HashSet<string> visited)
        {
            if (!visited.Add(node))
                return 0;
            int maxDepth = 0;

            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    int depth = GetLongestChainDFS(dep.TargetLawId, visited);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
            }

            return maxDepth + 1;
        }

        /// <summary>Export the graph as adjacency list.</summary>
        public Dictionary<string, List<string>> ExportAdjacencyList()
        {
            var result = new Dictionary<string, List<string>>();
            foreach (var kv in _adjacency)
            {
                result[kv.Key] = kv.Value.Select(d => d.TargetLawId).ToList();
            }
            return result;
        }

        /// <summary>Count total edges in the graph.</summary>
        public int EdgeCount => _adjacency.Values.Sum(list => list.Count);

        /// <summary>Count total nodes in the graph.</summary>
        public int NodeCount => _adjacency.Keys.Count;
    }

    // =========================================================================
    // LawBenchmark — performance benchmarking for compilation and execution
    // =========================================================================

    /// <summary>Benchmark result for a single operation.</summary>
    public sealed class BenchmarkResult
    {
        public string OperationName { get; init; } = "";
        public long ElapsedTicks { get; init; }
        public double ElapsedMilliseconds { get; init; }
        public int Iterations { get; init; }
        public double OpsPerSecond { get; init; }
        public long MemoryBytes { get; init; }
        public string? AdditionalInfo { get; init; }
    }

    /// <summary>Benchmarks compilation and execution performance.</summary>
    public sealed class LawBenchmark
    {
        private readonly LivingLawCompiler _compiler;
        private readonly List<BenchmarkResult> _results = new();

        public IReadOnlyList<BenchmarkResult> Results => _results;

        public LawBenchmark(LivingLawCompiler compiler)
        {
            _compiler = compiler;
        }

        /// <summary>Benchmark expression parsing.</summary>
        public BenchmarkResult BenchmarkParsing(string expression, int iterations = 1000)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var parser = new LawExpressionParser(expression);
                parser.Parse();
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Parsing",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Expression: {expression}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark bytecode compilation.</summary>
        public BenchmarkResult BenchmarkCompilation(string expression, int iterations = 1000)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _compiler.Compile(expression);
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Compilation",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Expression: {expression}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark bytecode execution.</summary>
        public BenchmarkResult BenchmarkExecution(string expression, int iterations = 10000)
        {
            var compResult = _compiler.Compile(expression);
            if (!compResult.Success || compResult.Bytecode == null)
                return new BenchmarkResult { OperationName = "Execution (failed)", AdditionalInfo = compResult.Message };

            var interpreter = new BytecodeInterpreter();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                interpreter.Execute(compResult.Bytecode, null, null, new Dictionary<string, float>
                {
                    ["x"] = 1f,
                    ["y"] = 2f,
                    ["z"] = 3f,
                    ["t"] = 0.5f,
                    ["T"] = 300f,
                    ["P"] = 101325f,
                    ["rho"] = 1.225f,
                    ["v"] = 10f,
                    ["c"] = 340f,
                    ["k"] = 100f
                });
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Execution",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Instructions: {compResult.InstructionCount}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark law application to a field.</summary>
        public BenchmarkResult BenchmarkApplication(string lawId, int gridSize = 32, int iterations = 10)
        {
            var field = new PhysicsField(gridSize, "benchmark");
            float cx = gridSize / 2f;
            for (int z = 0; z < gridSize; z++)
                for (int y = 0; y < gridSize; y++)
                    for (int x = 0; x < gridSize; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (gridSize * gridSize));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }

            long memBefore = GC.GetTotalMemory(true);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _compiler.ApplyLaw(lawId, field);
            }
            sw.Stop();
            long memAfter = GC.GetTotalMemory(false);

            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Application",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                MemoryBytes = memAfter - memBefore,
                AdditionalInfo = $"Grid: {gridSize}^3, Law: {lawId}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Run a comprehensive benchmark suite.</summary>
        public List<BenchmarkResult> RunFullBenchmark()
        {
            _results.Clear();
            var expressions = new[]
            {
                "sin(x) + cos(y)", "exp(-x^2)", "sqrt(x^2 + y^2 + z^2)",
                "x*y + y*z + z*x", "log(abs(x) + 1)", "min(max(x, 0), 1)"
            };

            foreach (var expr in expressions)
            {
                BenchmarkParsing(expr, 1000);
                BenchmarkCompilation(expr, 1000);
                BenchmarkExecution(expr, 10000);
            }

            var lawIds = new[] { "heat_equation", "wave_equation", "hooke_law" };
            foreach (var id in lawIds)
            {
                var law = _compiler.LoadLaw(id);
                if (law != null)
                {
                    BenchmarkCompilation(law.Expression, 500);
                    BenchmarkApplication(id, 32, 5);
                }
            }

            return _results.ToList();
        }

        /// <summary>Generate a summary report.</summary>
        public string GenerateReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== Living Law Compiler Benchmark Report ===");
            sb.AppendLine($"Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}");
            sb.AppendLine($"Total benchmarks: {_results.Count}");
            sb.AppendLine();
            foreach (var r in _results)
            {
                sb.AppendLine($"--- {r.OperationName} ---");
                sb.AppendLine($"  Iterations: {r.Iterations}");
                sb.AppendLine($"  Total time: {r.ElapsedMilliseconds:F2} ms");
                sb.AppendLine($"  Per iteration: {r.ElapsedMilliseconds / r.Iterations:F4} ms");
                sb.AppendLine($"  Ops/sec: {r.OpsPerSecond:F0}");
                if (r.MemoryBytes != 0)
                    sb.AppendLine($"  Memory delta: {r.MemoryBytes:N0} bytes");
                if (r.AdditionalInfo != null)
                    sb.AppendLine($"  Info: {r.AdditionalInfo}");
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }

    // =========================================================================
    // LawDimensionalAnalyzer — advanced dimensional analysis
    // =========================================================================

    /// <summary>Unit representation for dimensional analysis.</summary>
    public sealed class PhysicalUnit
    {
        public string Symbol { get; init; } = "";
        public string Name { get; init; } = "";
        public Dimension BaseDimension { get; init; } = Dimension.Scalar;
        public float ConversionFactor { get; init; } = 1.0f;
        public float Offset { get; init; } = 0.0f;

        public float ConvertToBase(float value) => value * ConversionFactor + Offset;
        public float ConvertFromBase(float baseValue) => (baseValue - Offset) / ConversionFactor;
    }

    /// <summary>Registry of physical units for dimensional analysis.</summary>
    public sealed class UnitRegistry
    {
        private readonly Dictionary<string, PhysicalUnit> _units = new();

        public UnitRegistry()
        {
            RegisterDefaults();
        }

        private void RegisterDefaults()
        {
            Register(new PhysicalUnit { Symbol = "m", Name = "meter", BaseDimension = Dimension.LengthD });
            Register(new PhysicalUnit { Symbol = "kg", Name = "kilogram", BaseDimension = Dimension.MassD });
            Register(new PhysicalUnit { Symbol = "s", Name = "second", BaseDimension = Dimension.TimeD });
            Register(new PhysicalUnit { Symbol = "K", Name = "kelvin", BaseDimension = Dimension.TemperatureD });
            Register(new PhysicalUnit { Symbol = "mol", Name = "mole", BaseDimension = new Dimension(0, 0, 0, 0, 1, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "A", Name = "ampere", BaseDimension = new Dimension(0, 0, 0, 0, 0, 1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "cd", Name = "candela", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, 0) });
            Register(new PhysicalUnit { Symbol = "Hz", Name = "hertz", BaseDimension = new Dimension(0, 0, -1, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "N", Name = "newton", BaseDimension = Dimension.Force });
            Register(new PhysicalUnit { Symbol = "Pa", Name = "pascal", BaseDimension = Dimension.Pressure });
            Register(new PhysicalUnit { Symbol = "J", Name = "joule", BaseDimension = Dimension.Energy });
            Register(new PhysicalUnit { Symbol = "W", Name = "watt", BaseDimension = Dimension.Power });
            Register(new PhysicalUnit { Symbol = "V", Name = "volt", BaseDimension = new Dimension(1, 2, -3, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Ω", Name = "ohm", BaseDimension = new Dimension(1, 2, -3, 0, 0, -2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "C", Name = "coulomb", BaseDimension = new Dimension(0, 0, 0, 0, 0, 1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "F", Name = "farad", BaseDimension = new Dimension(-1, -2, 4, 0, 0, 2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "H", Name = "henry", BaseDimension = new Dimension(1, 2, -2, 0, 0, -2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "T", Name = "tesla", BaseDimension = new Dimension(1, 0, -2, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Wb", Name = "weber", BaseDimension = new Dimension(1, 2, -2, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "lm", Name = "lumen", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, 0) });
            Register(new PhysicalUnit { Symbol = "lx", Name = "lux", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, -2) });
            Register(new PhysicalUnit { Symbol = "Bq", Name = "becquerel", BaseDimension = new Dimension(0, 0, -1, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Gy", Name = "gray", BaseDimension = new Dimension(0, 2, -2, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Sv", Name = "sievert", BaseDimension = new Dimension(0, 2, -2, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "kat", Name = "katal", BaseDimension = new Dimension(0, 0, -1, 0, 1, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "m/s", Name = "meters per second", BaseDimension = Dimension.Velocity });
            Register(new PhysicalUnit { Symbol = "m/s²", Name = "meters per second squared", BaseDimension = Dimension.Acceleration });
            Register(new PhysicalUnit { Symbol = "kg/m³", Name = "kilograms per cubic meter", BaseDimension = Dimension.Density });
            Register(new PhysicalUnit { Symbol = "Pa·s", Name = "pascal-second", BaseDimension = Dimension.Viscosity });
            Register(new PhysicalUnit { Symbol = "W/(m·K)", Name = "watts per meter-kelvin", BaseDimension = new Dimension(1, 1, -3, -1, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "m²/s", Name = "square meters per second", BaseDimension = new Dimension(0, 2, -1, 0, 0, 0, 0, 0) });
        }

        public void Register(PhysicalUnit unit) => _units[unit.Symbol] = unit;

        public PhysicalUnit? GetUnit(string symbol) =>
            _units.TryGetValue(symbol, out var unit) ? unit : null;

        public bool UnitExists(string symbol) => _units.ContainsKey(symbol);

        /// <summary>Check if two units are dimensionally compatible.</summary>
        public bool AreCompatible(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return false;
            return a.BaseDimension.IsCompatible(b.BaseDimension);
        }

        /// <summary>Get the product dimension of two units.</summary>
        public Dimension ProductDimension(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return Dimension.Scalar;
            return a.BaseDimension.Multiply(b.BaseDimension);
        }

        /// <summary>Get the quotient dimension of two units.</summary>
        public Dimension QuotientDimension(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return Dimension.Scalar;
            return a.BaseDimension.Divide(b.BaseDimension);
        }
    }

    /// <summary>Advanced dimensional analysis for law expressions.</summary>
    public sealed class LawDimensionalAnalyzer
    {
        private readonly UnitRegistry _unitRegistry;
        private readonly Dictionary<string, Dimension> _variableDimensions = new();

        public LawDimensionalAnalyzer()
        {
            _unitRegistry = new UnitRegistry();
            InitializeVariableDimensions();
        }

        private void InitializeVariableDimensions()
        {
            _variableDimensions["x"] = Dimension.LengthD;
            _variableDimensions["y"] = Dimension.LengthD;
            _variableDimensions["z"] = Dimension.LengthD;
            _variableDimensions["t"] = Dimension.TimeD;
            _variableDimensions["T"] = Dimension.TemperatureD;
            _variableDimensions["P"] = Dimension.Pressure;
            _variableDimensions["rho"] = Dimension.Density;
            _variableDimensions["v"] = Dimension.Velocity;
            _variableDimensions["u"] = Dimension.Velocity;
            _variableDimensions["w"] = Dimension.Velocity;
            _variableDimensions["F"] = Dimension.Force;
            _variableDimensions["E"] = Dimension.Energy;
            _variableDimensions["k"] = new Dimension(0, 2, -1, 0, 0, 0, 0, 0);
            _variableDimensions["mu"] = Dimension.Viscosity;
            _variableDimensions["alpha"] = new Dimension(0, 2, -1, 0, 0, 0, 0, 0);
            _variableDimensions["sigma"] = new Dimension(1, 0, -3, -4, 0, 0, 0, 0);
            _variableDimensions["G"] = new Dimension(-1, 3, -2, 0, 0, 0, 0, 0);
            _variableDimensions["R"] = new Dimension(0, 2, -2, -1, 0, 0, 0, 0);
            _variableDimensions["c"] = Dimension.Velocity;
            _variableDimensions["I"] = new Dimension(0, 0, 0, 0, 0, 1, 0, 0);
            _variableDimensions["q"] = new Dimension(0, 0, 0, 0, 0, 1, 0, 0);
            _variableDimensions["dt"] = Dimension.TimeD;
            _variableDimensions["dx"] = Dimension.LengthD;
            _variableDimensions["dy"] = Dimension.LengthD;
            _variableDimensions["dz"] = Dimension.LengthD;
            _variableDimensions["m1"] = Dimension.MassD;
            _variableDimensions["m2"] = Dimension.MassD;
            _variableDimensions["r"] = Dimension.LengthD;
        }

        /// <summary>Analyze an expression and return dimension of each sub-expression.</summary>
        public Dictionary<AstNode, Dimension> AnalyzeDimensions(AstNode ast)
        {
            var result = new Dictionary<AstNode, Dimension>();
            AnalyzeNode(ast, result);
            return result;
        }

        private Dimension AnalyzeNode(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            if (node == null)
                return Dimension.Scalar;

            Dimension dim = node.Type switch
            {
                NodeType.NumberLiteral => Dimension.Scalar,
                NodeType.Identifier => _variableDimensions.TryGetValue(node.Value ?? "", out var d) ? d : Dimension.Scalar,
                NodeType.FieldAccess => Dimension.Scalar,
                NodeType.BinaryExpression => AnalyzeBinary(node, result),
                NodeType.UnaryExpression => AnalyzeNode(node.Left!, result),
                NodeType.TernaryExpression => AnalyzeNode(node.Right!, result),
                NodeType.FunctionCall => AnalyzeFunction(node, result),
                _ => Dimension.Scalar
            };

            result[node] = dim;
            return dim;
        }

        private Dimension AnalyzeBinary(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            var leftDim = AnalyzeNode(node.Left!, result);
            var rightDim = AnalyzeNode(node.Right!, result);

            return node.Value switch
            {
                "+" or "-" => leftDim,
                "*" => leftDim.Multiply(rightDim),
                "/" => leftDim.Divide(rightDim),
                "^" => leftDim.Pow(rightDim.IsDimensionless ? 1f : 2f),
                "%" => leftDim,
                "==" or "!=" or "<" or ">" or "<=" or ">=" => Dimension.Scalar,
                "&&" or "||" => Dimension.Scalar,
                _ => Dimension.Scalar
            };
        }

        private Dimension AnalyzeFunction(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            string funcName = (node.Value ?? "").ToLowerInvariant();
            if (node.Children != null && node.Children.Count > 0)
                AnalyzeNode(node.Children[0], result);

            return funcName switch
            {
                "sin" or "cos" or "tan" or "asin" or "acos" or "atan" or
                "sinh" or "cosh" or "tanh" => Dimension.Scalar,
                "exp" or "log" or "log2" or "log10" or "sqrt" or "cbrt" => Dimension.Scalar,
                "abs" or "sign" or "ceil" or "floor" or "round" => Dimension.Scalar,
                "min" or "max" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "pow" => Dimension.Scalar,
                "clamp" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "lerp" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "grad_x" or "grad_y" or "grad_z" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                "laplacian" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD.Pow(2)) : Dimension.Scalar,
                "divergence" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                "curl_x" or "curl_y" or "curl_z" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                _ => Dimension.Scalar
            };
        }

        /// <summary>Check if an equation is dimensionally homogeneous.</summary>
        public bool IsDimensionallyHomogeneous(AstNode ast)
        {
            var dims = AnalyzeDimensions(ast);
            if (ast.Type == NodeType.BinaryExpression && ast.Value is "=" or "==")
            {
                if (dims.TryGetValue(ast.Left!, out var leftDim) && dims.TryGetValue(ast.Right!, out var rightDim))
                    return leftDim.IsCompatible(rightDim);
            }
            return true;
        }

        /// <summary>Get the dimension of the entire expression.</summary>
        public Dimension GetExpressionDimension(AstNode ast)
        {
            var dims = AnalyzeDimensions(ast);
            return dims.TryGetValue(ast, out var dim) ? dim : Dimension.Scalar;
        }

        /// <summary>Verify that a unit matches the expected dimension.</summary>
        public bool VerifyUnit(string expression, string expectedUnit)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            var exprDim = GetExpressionDimension(ast);
            var unit = _unitRegistry.GetUnit(expectedUnit);
            if (unit == null)
                return false;
            return exprDim.IsCompatible(unit.BaseDimension);
        }

        /// <summary>Get a list of compatible units for a dimension.</summary>
        public List<string> GetCompatibleUnits(Dimension dim)
        {
            var results = new List<string>();
            foreach (var unitName in _variableDimensions)
            {
                if (unitName.Value.IsCompatible(dim))
                    results.Add(unitName.Key);
            }
            return results;
        }
    }

    // =========================================================================
    // LawOptimizer — optimizes compiled bytecode
    // =========================================================================

    /// <summary>Optimization passes for law bytecode.</summary>
    public enum OptimizationPass
    {
        ConstantFolding,
        DeadCodeElimination,
        PeepholeOptimization,
        CommonSubexpressionElimination,
        InstructionCombining,
        JumpThreading
    }

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

    // =========================================================================
    // LawProfiler — detailed profiling for law operations
    // =========================================================================

    /// <summary>Profile data for a single operation.</summary>
    public sealed class ProfileData
    {
        public string OperationName { get; init; } = "";
        public int CallCount { get; set; }
        public double TotalMilliseconds { get; set; }
        public double MinMilliseconds { get; set; } = double.MaxValue;
        public double MaxMilliseconds { get; set; } = double.MinValue;
        public double LastMilliseconds { get; set; }
        public long TotalBytes { get; set; }
        public double AverageMilliseconds => CallCount > 0 ? TotalMilliseconds / CallCount : 0;
    }

    /// <summary>Profiles law compilation and execution operations.</summary>
    public sealed class LawProfiler
    {
        private readonly ConcurrentDictionary<string, ProfileData> _profiles = new();
        private bool _enabled = true;

        public bool Enabled { get => _enabled; set => _enabled = value; }

        public IReadOnlyDictionary<string, ProfileData> Profiles => _profiles;

        /// <summary>Start timing an operation.</summary>
        public Stopwatch StartProfile(string operationName)
        {
            if (!_enabled)
                return Stopwatch.StartNew();
            return Stopwatch.StartNew();
        }

        /// <summary>Stop timing and record the result.</summary>
        public void StopProfile(string operationName, Stopwatch sw, long allocatedBytes = 0)
        {
            if (!_enabled)
                return;
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;

            var profile = _profiles.GetOrAdd(operationName, _ => new ProfileData { OperationName = operationName });
            lock (profile)
            {
                profile.CallCount++;
                profile.TotalMilliseconds += ms;
                profile.LastMilliseconds = ms;
                profile.TotalBytes += allocatedBytes;
                if (ms < profile.MinMilliseconds)
                    profile.MinMilliseconds = ms;
                if (ms > profile.MaxMilliseconds)
                    profile.MaxMilliseconds = ms;
            }
        }

        /// <summary>Profile a compiled action.</summary>
        public void ProfileAction(string operationName, Action action)
        {
            var sw = StartProfile(action.Method.Name);
            action();
            StopProfile(operationName, sw);
        }

        /// <summary>Profile a compiled function.</summary>
        public T ProfileFunction<T>(string operationName, Func<T> func)
        {
            var sw = StartProfile(operationName);
            T result = func();
            StopProfile(operationName, sw);
            return result;
        }

        /// <summary>Reset all profiling data.</summary>
        public void Reset()
        {
            _profiles.Clear();
        }

        /// <summary>Generate a profiling report.</summary>
        public string GenerateReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== Law Profiler Report ===");
            sb.AppendLine($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}");
            sb.AppendLine($"Total operations: {_profiles.Count}");
            sb.AppendLine();
            sb.AppendLine($"{"Operation",-30} {"Count",8} {"Total(ms)",12} {"Avg(ms)",10} {"Min(ms)",10} {"Max(ms)",10}");
            sb.AppendLine(new string('-', 82));

            foreach (var kv in _profiles.OrderByDescending(p => p.Value.TotalMilliseconds))
            {
                var p = kv.Value;
                sb.AppendLine($"{p.OperationName,-30} {p.CallCount,8} {p.TotalMilliseconds,12:F2} {p.AverageMilliseconds,10:F4} {p.MinMilliseconds,10:F4} {p.MaxMilliseconds,10:F4}");
            }

            sb.AppendLine();
            long totalBytes = _profiles.Values.Sum(p => p.TotalBytes);
            sb.AppendLine($"Total memory allocated: {totalBytes:N0} bytes");
            double totalTime = _profiles.Values.Sum(p => p.TotalMilliseconds);
            sb.AppendLine($"Total time: {totalTime:F2} ms");

            return sb.ToString();
        }

        /// <summary>Get the top N most time-consuming operations.</summary>
        public IReadOnlyList<ProfileData> GetTopOperations(int count = 10)
        {
            return _profiles.Values.OrderByDescending(p => p.TotalMilliseconds).Take(count).ToList();
        }

        /// <summary>Get the top N most frequently called operations.</summary>
        public IReadOnlyList<ProfileData> GetMostFrequent(int count = 10)
        {
            return _profiles.Values.OrderByDescending(p => p.CallCount).Take(count).ToList();
        }
    }

    // =========================================================================
    // LawMathUtils — mathematical utility functions
    // =========================================================================

    /// <summary>Mathematical utility functions for law computations.</summary>
    public static class LawMathUtils
    {
        /// <summary>Clamp a value between min and max.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float value, float min, float max) =>
            MathF.Max(min, MathF.Min(max, value));

        /// <summary>Linear interpolation between a and b.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Lerp(float a, float b, float t) => a + t * (b - a);

        /// <summary>Inverse linear interpolation: find t such that Lerp(a, b, t) = value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float InverseLerp(float a, float b, float value)
        {
            if (MathF.Abs(b - a) < float.Epsilon)
                return 0f;
            return (value - a) / (b - a);
        }

        /// <summary>Smooth step (Hermite interpolation).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothStep(float edge0, float edge1, float x)
        {
            float t = Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        /// <summary>Smoother step (Ken Perlin's version).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmootherStep(float edge0, float edge1, float x)
        {
            float t = Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * t * (t * (t * 6f - 15f) + 10f);
        }

        /// <summary>Remap a value from one range to another.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax)
        {
            float t = InverseLerp(fromMin, fromMax, value);
            return Lerp(toMin, toMax, t);
        }

        /// <summary>Compute the Euclidean distance between two 3D points.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Distance3D(float x1, float y1, float z1, float x2, float y2, float z2)
        {
            float dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
            return MathF.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>Compute the dot product of two 3D vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot3D(float ax, float ay, float az, float bx, float by, float bz) =>
            ax * bx + ay * by + az * bz;

        /// <summary>Compute the cross product Z component of two 2D vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cross2D(float ax, float ay, float bx, float by) =>
            ax * by - ay * bx;

        /// <summary>Normalize a 3D vector.</summary>
        public static (float X, float Y, float Z) Normalize3D(float x, float y, float z)
        {
            float len = MathF.Sqrt(x * x + y * y + z * z);
            if (len < float.Epsilon)
                return (0f, 0f, 0f);
            return (x / len, y / len, z / len);
        }

        /// <summary>Compute the magnitude of a 3D vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Magnitude3D(float x, float y, float z) =>
            MathF.Sqrt(x * x + y * y + z * z);

        /// <summary>Compute the Laplacian of a scalar field at a point using 7-point stencil.</summary>
        public static float Laplacian7Point(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX, sy = field.SizeY, sz = field.SizeZ;
            float c = field[x, y, z];
            float left = x > 0 ? field[x - 1, y, z] : c;
            float right = x < sx - 1 ? field[x + 1, y, z] : c;
            float bottom = y > 0 ? field[x, y - 1, z] : c;
            float top = y < sy - 1 ? field[x, y + 1, z] : c;
            float back = z > 0 ? field[x, y, z - 1] : c;
            float front = z < sz - 1 ? field[x, y, z + 1] : c;
            return (left + right + bottom + top + back + front - 6f * c) / (dx * dx);
        }

        /// <summary>Compute the gradient X component using central differences.</summary>
        public static float GradientX(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX;
            float left = x > 0 ? field[x - 1, y, z] : field[0, y, z];
            float right = x < sx - 1 ? field[x + 1, y, z] : field[sx - 1, y, z];
            return (right - left) / (2f * dx);
        }

        /// <summary>Compute the gradient Y component using central differences.</summary>
        public static float GradientY(FieldGrid field, int x, int y, int z, float dy)
        {
            int sy = field.SizeY;
            float bottom = y > 0 ? field[x, y - 1, z] : field[x, 0, z];
            float top = y < sy - 1 ? field[x, y + 1, z] : field[x, sy - 1, z];
            return (top - bottom) / (2f * dy);
        }

        /// <summary>Compute the gradient Z component using central differences.</summary>
        public static float GradientZ(FieldGrid field, int x, int y, int z, float dz)
        {
            int sz = field.SizeZ;
            float back = z > 0 ? field[x, y, z - 1] : field[x, y, 0];
            float front = z < sz - 1 ? field[x, y, z + 1] : field[x, y, sz - 1];
            return (front - back) / (2f * dz);
        }

        /// <summary>Compute the divergence of a vector field at a point.</summary>
        public static float Divergence(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientX(vx, x, y, z, dx) + GradientY(vy, x, y, z, dx) + GradientZ(vz, x, y, z, dx);
        }

        /// <summary>Compute the curl X component at a point.</summary>
        public static float CurlX(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientY(vz, x, y, z, dx) - GradientZ(vy, x, y, z, dx);
        }

        /// <summary>Compute the curl Y component at a point.</summary>
        public static float CurlY(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientZ(vx, x, y, z, dx) - GradientX(vz, x, y, z, dx);
        }

        /// <summary>Compute the curl Z component at a point.</summary>
        public static float CurlZ(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientX(vy, x, y, z, dx) - GradientY(vx, x, y, z, dx);
        }

        /// <summary>Compute the total kinetic energy of a velocity field.</summary>
        public static float KineticEnergy(FieldGrid vx, FieldGrid vy, FieldGrid vz, float density)
        {
            float energy = 0f;
            int total = vx.TotalCells;
            for (int i = 0; i < total; i++)
            {
                float v2 = vx.Data[i] * vx.Data[i] + vy.Data[i] * vy.Data[i] + vz.Data[i] * vz.Data[i];
                energy += 0.5f * density * v2;
            }
            return energy;
        }

        /// <summary>Compute the total thermal energy of a temperature field.</summary>
        public static float ThermalEnergy(FieldGrid temperature, float specificHeat, float density)
        {
            float energy = 0f;
            int total = temperature.TotalCells;
            for (int i = 0; i < total; i++)
                energy += specificHeat * density * temperature.Data[i];
            return energy;
        }

        /// <summary>Compute the L2 norm of a field.</summary>
        public static float L2Norm(FieldGrid field)
        {
            double sum = 0;
            for (int i = 0; i < field.TotalCells; i++)
                sum += (double)field.Data[i] * field.Data[i];
            return MathF.Sqrt((float)(sum / field.TotalCells));
        }

        /// <summary>Compute the maximum absolute value in a field.</summary>
        public static float InfinityNorm(FieldGrid field)
        {
            float max = 0f;
            for (int i = 0; i < field.TotalCells; i++)
            {
                float abs = MathF.Abs(field.Data[i]);
                if (abs > max)
                    max = abs;
            }
            return max;
        }

        /// <summary>Compute the field integral (sum * volume element).</summary>
        public static float Integral(FieldGrid field, float dx)
        {
            float volume = dx * dx * dx;
            float sum = 0f;
            for (int i = 0; i < field.TotalCells; i++)
                sum += field.Data[i];
            return sum * volume;
        }

        /// <summary>Compute the field average over interior cells only.</summary>
        public static float InteriorAverage(FieldGrid field)
        {
            float sum = 0f;
            int count = 0;
            for (int z = 1; z < field.SizeZ - 1; z++)
                for (int y = 1; y < field.SizeY - 1; y++)
                    for (int x = 1; x < field.SizeX - 1; x++)
                    {
                        sum += field[x, y, z];
                        count++;
                    }
            return count > 0 ? sum / count : 0f;
        }

        /// <summary>Compute the maximum gradient magnitude in a field.</summary>
        public static float MaxGradientMagnitude(FieldGrid field, float dx)
        {
            float maxGrad = 0f;
            for (int z = 1; z < field.SizeZ - 1; z++)
                for (int y = 1; y < field.SizeY - 1; y++)
                    for (int x = 1; x < field.SizeX - 1; x++)
                    {
                        float gx = GradientX(field, x, y, z, dx);
                        float gy = GradientY(field, x, y, z, dx);
                        float gz = GradientZ(field, x, y, z, dx);
                        float mag = MathF.Sqrt(gx * gx + gy * gy + gz * gz);
                        if (mag > maxGrad)
                            maxGrad = mag;
                    }
            return maxGrad;
        }
    }

    // =========================================================================
    // LawPhysicsUtils — physics-specific utility functions
    // =========================================================================

    /// <summary>Physics-specific utility functions for law computations.</summary>
    public static class LawPhysicsUtils
    {
        /// <summary>Compute the speed of sound for an ideal gas.</summary>
        public static float SpeedOfSound(float temperature, float gamma = 1.4f, float R = 287.058f)
        {
            return MathF.Sqrt(gamma * R * temperature);
        }

        /// <summary>Compute the Mach number.</summary>
        public static float MachNumber(float velocity, float soundSpeed)
        {
            return MathF.Abs(soundSpeed) > float.Epsilon ? MathF.Abs(velocity) / soundSpeed : 0f;
        }

        /// <summary>Compute the Reynolds number.</summary>
        public static float ReynoldsNumber(float density, float velocity, float length, float viscosity)
        {
            return MathF.Abs(viscosity) > float.Epsilon ? density * MathF.Abs(velocity) * length / viscosity : float.MaxValue;
        }

        /// <summary>Compute the Prandtl number.</summary>
        public static float PrandtlNumber(float viscosity, float specificHeat, float thermalConductivity)
        {
            return MathF.Abs(thermalConductivity) > float.Epsilon ? viscosity * specificHeat / thermalConductivity : 0f;
        }

        /// <summary>Compute the Nusselt number from Rayleigh and Prandtl numbers.</summary>
        public static float NusseltNumber(float rayleigh, float prandtl, string correlation = "churchill")
        {
            return correlation.ToLowerInvariant() switch
            {
                "churchill" => 0.68f + 0.670f * MathF.Pow(rayleigh * prandtl, 0.25f) /
                    MathF.Pow(1f + MathF.Pow(0.492f / prandtl, 9f / 16f), 4f / 9f),
                "rayleigh_benard" => MathF.Pow(rayleigh * prandtl, 0.25f) * 0.069f,
                _ => MathF.Pow(rayleigh, 0.25f)
            };
        }

        /// <summary>Compute the Biot number.</summary>
        public static float BiotNumber(float heatTransferCoeff, float length, float thermalConductivity)
        {
            return MathF.Abs(thermalConductivity) > float.Epsilon ? heatTransferCoeff * length / thermalConductivity : float.MaxValue;
        }

        /// <summary>Compute the Fourier number.</summary>
        public static float FourierNumber(float diffusivity, float time, float length)
        {
            return MathF.Abs(length * length) > float.Epsilon ? diffusivity * time / (length * length) : 0f;
        }

        /// <summary>Compute the Peclet number.</summary>
        public static float PecletNumber(float velocity, float length, float diffusivity)
        {
            return MathF.Abs(diffusivity) > float.Epsilon ? velocity * length / diffusivity : float.MaxValue;
        }

        /// <summary>Compute the Grashof number.</summary>
        public static float GrashofNumber(float beta, float deltaT, float length, float gravity, float viscosity, float kinematicViscosity)
        {
            return MathF.Abs(kinematicViscosity * kinematicViscosity) > float.Epsilon
                ? beta * deltaT * length * length * length * gravity / (kinematicViscosity * kinematicViscosity)
                : 0f;
        }

        /// <summary>Compute the Rayleigh number.</summary>
        public static float RayleighNumber(float grashof, float prandtl) => grashof * prandtl;

        /// <summary>Compute the Strouhal number.</summary>
        public static float StrouhalNumber(float frequency, float length, float velocity)
        {
            return MathF.Abs(velocity) > float.Epsilon ? frequency * length / velocity : 0f;
        }

        /// <summary>Compute the Froude number.</summary>
        public static float FroudeNumber(float velocity, float gravity, float length)
        {
            return MathF.Abs(gravity * length) > float.Epsilon ? velocity / MathF.Sqrt(gravity * length) : float.MaxValue;
        }

        /// <summary>Compute the Weber number.</summary>
        public static float WeberNumber(float density, float velocity, float length, float surfaceTension)
        {
            return MathF.Abs(surfaceTension) > float.Epsilon ? density * velocity * velocity * length / surfaceTension : float.MaxValue;
        }

        /// <summary>Compute the Knudsen number.</summary>
        public static float KnudsenNumber(float meanFreePath, float characteristicLength)
        {
            return MathF.Abs(characteristicLength) > float.Epsilon ? meanFreePath / characteristicLength : float.MaxValue;
        }

        /// <summary>Compute the Mach cone half-angle.</summary>
        public static float MachConeAngle(float machNumber)
        {
            return machNumber > 1f ? MathF.Asin(1f / machNumber) : MathF.PI / 2f;
        }

        /// <summary>Compute the Doppler shift for a moving source.</summary>
        public static float DopplerShift(float sourceFreq, float sourceVelocity, float observerVelocity, float soundSpeed)
        {
            float denominator = soundSpeed - sourceVelocity;
            return MathF.Abs(denominator) > float.Epsilon
                ? sourceFreq * (soundSpeed + observerVelocity) / denominator
                : sourceFreq;
        }

        /// <summary>Compute the adiabatic temperature lapse rate.</summary>
        public static float AdiabaticLapseRate(float gravity, float specificHeat) =>
            MathF.Abs(specificHeat) > float.Epsilon ? gravity / specificHeat : 0f;

        /// <summary>Compute the saturation vapor pressure (Magnus formula).</summary>
        public static float SaturationVaporPressure(float temperatureCelsius)
        {
            return 610.78f * MathF.Exp(17.27f * temperatureCelsius / (temperatureCelsius + 237.3f));
        }

        /// <summary>Compute the heat transfer rate using Newton's law of cooling.</summary>
        public static float NewtonsCooling(float surfaceTemp, float fluidTemp, float heatTransferCoeff, float area)
        {
            return heatTransferCoeff * area * (surfaceTemp - fluidTemp);
        }

        /// <summary>Compute the Stefan-Boltzmann radiative heat flux.</summary>
        public static float StefanBoltzmannFlux(float temperature, float emissivity = 1f)
        {
            const float sigma = 5.670374419e-8f;
            return emissivity * sigma * MathF.Pow(temperature, 4);
        }

        /// <summary>Compute the Planck distribution at a given wavelength and temperature.</summary>
        public static float PlanckDistribution(float wavelength, float temperature)
        {
            const float h = 6.62607015e-34f;
            const float c = 299792458f;
            const float kB = 1.380649e-23f;
            float numerator = 2f * h * c * c / MathF.Pow(wavelength, 5);
            float exponent = h * c / (wavelength * kB * temperature);
            if (exponent > 100f)
                return 0f;
            return numerator / (MathF.Exp(exponent) - 1f);
        }

        /// <summary>Compute the gravitational potential energy.</summary>
        public static float GravitationalPotentialEnergy(float mass, float height, float g = 9.81f) => mass * g * height;

        /// <summary>Compute the kinetic energy of a particle.</summary>
        public static float KineticEnergy(float mass, float velocity) => 0.5f * mass * velocity * velocity;

        /// <summary>Compute the orbital velocity for a circular orbit.</summary>
        public static float OrbitalVelocity(float centralMass, float radius, float G = 6.674e-11f)
        {
            return MathF.Abs(radius) > float.Epsilon ? MathF.Sqrt(G * centralMass / radius) : 0f;
        }

        /// <summary>Compute the escape velocity.</summary>
        public static float EscapeVelocity(float mass, float radius, float G = 6.674e-11f)
        {
            return MathF.Abs(radius) > float.Epsilon ? MathF.Sqrt(2f * G * mass / radius) : 0f;
        }

        /// <summary>Compute the Schwarzschild radius.</summary>
        public static float SchwarzschildRadius(float mass, float c = 299792458f, float G = 6.674e-11f)
        {
            return 2f * G * mass / (c * c);
        }
    }

    // =========================================================================
    // LawDiagnostics — diagnostic and error reporting utilities
    // =========================================================================

    /// <summary>Severity level for diagnostic messages.</summary>
    public enum DiagnosticSeverity
    {
        Info, Warning, Error, Critical
    }

    /// <summary>A diagnostic message from the compiler.</summary>
    public sealed class DiagnosticMessage
    {
        public DiagnosticSeverity Severity { get; init; }
        public string Code { get; init; } = "";
        public string Message { get; init; } = "";
        public int Line { get; init; }
        public int Column { get; init; }
        public string? Expression { get; init; }
        public string? Suggestion { get; init; }
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;

        public override string ToString()
        {
            string loc = Line > 0 ? $" at line {Line}:{Column}" : "";
            string sug = Suggestion != null ? $" (suggestion: {Suggestion})" : "";
            return $"[{Severity}] {Code}: {Message}{loc}{sug}";
        }
    }

    /// <summary>Collects and manages diagnostic messages during compilation.</summary>
    public sealed class LawDiagnostics
    {
        private readonly List<DiagnosticMessage> _messages = new();
        private readonly object _lock = new();

        public int ErrorCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Error || m.Severity == DiagnosticSeverity.Critical); }
        }

        public int WarningCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Warning); }
        }

        public int InfoCount
        {
            get { lock (_lock) return _messages.Count(m => m.Severity == DiagnosticSeverity.Info); }
        }

        public bool HasErrors => ErrorCount > 0;

        public IReadOnlyList<DiagnosticMessage> Messages
        {
            get { lock (_lock) return _messages.ToList(); }
        }

        public void Report(DiagnosticSeverity severity, string code, string message, int line = 0, int column = 0, string? expression = null, string? suggestion = null)
        {
            lock (_lock)
            {
                _messages.Add(new DiagnosticMessage
                {
                    Severity = severity,
                    Code = code,
                    Message = message,
                    Line = line,
                    Column = column,
                    Expression = expression,
                    Suggestion = suggestion
                });
            }
        }

        public void Info(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Info, code, message, suggestion: suggestion);

        public void Warn(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Warning, code, message, suggestion: suggestion);

        public void Error(string code, string message, int line = 0, int column = 0, string? suggestion = null) =>
            Report(DiagnosticSeverity.Error, code, message, line, column, suggestion: suggestion);

        public void Critical(string code, string message, string? suggestion = null) =>
            Report(DiagnosticSeverity.Critical, code, message, suggestion: suggestion);

        public void Clear() { lock (_lock) { _messages.Clear(); } }

        public IReadOnlyList<DiagnosticMessage> GetBySeverity(DiagnosticSeverity severity)
        {
            lock (_lock)
                return _messages.Where(m => m.Severity == severity).ToList();
        }

        public IReadOnlyList<DiagnosticMessage> GetByCode(string code)
        {
            lock (_lock)
                return _messages.Where(m => m.Code == code).ToList();
        }

        public string FormatReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"=== Diagnostics Report ({_messages.Count} messages) ===");
            sb.AppendLine($"Errors: {ErrorCount}, Warnings: {WarningCount}, Info: {InfoCount}");
            sb.AppendLine();

            foreach (var msg in _messages.OrderByDescending(m => m.Severity))
            {
                sb.AppendLine(msg.ToString());
            }

            return sb.ToString();
        }

        public string FormatErrorsOnly()
        {
            var sb = new StringBuilder();
            var errors = GetBySeverity(DiagnosticSeverity.Error).Concat(GetBySeverity(DiagnosticSeverity.Critical));
            foreach (var err in errors)
                sb.AppendLine(err.ToString());
            return sb.ToString();
        }

        public static LawDiagnostics FromCompilationResult(CompilationResult result)
        {
            var diag = new LawDiagnostics();
            if (!result.Success)
            {
                foreach (var error in result.Errors)
                    diag.Error("COMP001", error);
            }
            foreach (var warning in result.Warnings)
                diag.Warn("COMP002", warning);
            return diag;
        }

        public static LawDiagnostics FromValidationResult(ValidationResult result)
        {
            var diag = new LawDiagnostics();
            if (!result.IsValid)
            {
                foreach (var error in result.Errors)
                    diag.Error("VAL001", error);
            }
            foreach (var warning in result.Warnings)
                diag.Warn("VAL002", warning);
            if (!result.DimensionallyConsistent)
                diag.Error("VAL003", "Expression is not dimensionally consistent");
            return diag;
        }
    }
}
