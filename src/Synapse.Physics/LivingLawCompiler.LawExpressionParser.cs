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
}
