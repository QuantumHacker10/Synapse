// =============================================================================
// NeatGOnnxExporter.cs — Synapse AI: NEAT-G to ONNX Export
// Exports NEAT-G models to ONNX format for GPU inference via TorchSharp/ONNX Runtime
// =============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace GDNN.AI
{
    /// <summary>
    /// Represents a NEAT-G genome for export to ONNX.
    /// </summary>
    public class NeatGGenome
    {
        /// <summary>
        /// Gets or sets the genome ID.
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Gets or sets the fitness score.
        /// </summary>
        public double Fitness { get; set; }

        /// <summary>
        /// Gets the input nodes.
        /// </summary>
        public List<NeatGNode> InputNodes { get; } = new List<NeatGNode>();

        /// <summary>
        /// Gets the output nodes.
        /// </summary>
        public List<NeatGNode> OutputNodes { get; } = new List<NeatGNode>();

        /// <summary>
        /// Gets the hidden nodes.
        /// </summary>
        public List<NeatGNode> HiddenNodes { get; } = new List<NeatGNode>();

        /// <summary>
        /// Gets the connections between nodes.
        /// </summary>
        public List<NeatGConnection> Connections { get; } = new List<NeatGConnection>();

        /// <summary>
        /// Gets or sets the activation function type.
        /// </summary>
        public NeatGActivationType ActivationType { get; set; } = NeatGActivationType.Sigmoid;

        /// <summary>
        /// Adds a connection between two nodes.
        /// </summary>
        public void AddConnection(NeatGNode from, NeatGNode to, float weight)
        {
            Connections.Add(new NeatGConnection(from, to, weight));
        }
    }

    /// <summary>
    /// NEAT-G node types.
    /// </summary>
    public enum NeatGNodeType
    {
        Input,
        Output,
        Hidden
    }

    /// <summary>
    /// NEAT-G activation function types.
    /// </summary>
    public enum NeatGActivationType
    {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU,
        Sinusoid,
        Gaussian,
        Linear
    }

    /// <summary>
    /// NEAT-G node.
    /// </summary>
    public class NeatGNode
    {
        /// <summary>
        /// Gets or sets the node ID.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the node type.
        /// </summary>
        public NeatGNodeType Type { get; set; }

        /// <summary>
        /// Gets or sets the innovation number.
        /// </summary>
        public int InnovationNumber { get; set; }

        /// <summary>
        /// Gets or sets the activation function.
        /// </summary>
        public NeatGActivationType Activation { get; set; } = NeatGActivationType.Sigmoid;

        /// <summary>
        /// Gets or sets the bias value.
        /// </summary>
        public float Bias { get; set; }

        /// <summary>
        /// Gets or sets the response value (for sigmoid activation).
        /// </summary>
        public float Response { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the x position (for visualization).
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the y position (for visualization).
        /// </summary>
        public float Y { get; set; }

        public NeatGNode(int id, NeatGNodeType type)
        {
            Id = id;
            Type = type;
        }

        public override string ToString() => $"Node({Id}, {Type}, Act={Activation})";
    }

    /// <summary>
    /// NEAT-G connection between nodes.
    /// </summary>
    public class NeatGConnection
    {
        /// <summary>
        /// Gets or sets the from node.
        /// </summary>
        public NeatGNode From { get; set; }

        /// <summary>
        /// Gets or sets the to node.
        /// </summary>
        public NeatGNode To { get; set; }

        /// <summary>
        /// Gets or sets the weight.
        /// </summary>
        public float Weight { get; set; }

        /// <summary>
        /// Gets or sets whether the connection is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the innovation number.
        /// </summary>
        public int InnovationNumber { get; set; }

        public NeatGConnection(NeatGNode from, NeatGNode to, float weight)
        {
            From = from;
            To = to;
            Weight = weight;
        }

        public override string ToString() => $"Conn({From.Id}->{To.Id}, W={Weight:F4})";
    }

    /// <summary>
    /// Exports NEAT-G models to ONNX format.
    /// </summary>
    public static class NeatGOnnxExporter
    {
        private const string OnnxOpDomain = "ai.onnx";
        private const int OnnxOpsetVersion = 15;

        /// <summary>
        /// Exports a NEAT-G genome to ONNX format.
        /// </summary>
        public static byte[] ExportToOnnx(NeatGGenome genome)
        {
            // ONNX model structure:
            // 1. ModelProto with IR version, opset imports
            // 2. Graph with nodes (operations)
            // 3. Input/Output tensors
            // 4. Initializers (weights, biases)

            // For simplicity, we'll generate a basic ONNX model
            // In production, use ONNX Helper or Microsoft.ML.OnnxRuntime
            
            // Step 1: Create the ONNX graph
            var graph = CreateOnnxGraph(genome);
            
            // Step 2: Serialize to byte array
            // This is a simplified version - real implementation would use protobuf
            
            // For now, return a placeholder
            // Actual implementation would use:
            // - Microsoft.ML.OnnxRuntime for model creation
            // - Or ONNX Helper libraries
            
            throw new NotImplementedException("ONNX export requires Microsoft.ML.OnnxRuntime or similar library. " +
                "Install NuGet package: Microsoft.ML.OnnxRuntime");
        }

        /// <summary>
        /// Creates an ONNX graph from a NEAT-G genome.
        /// </summary>
        private static object CreateOnnxGraph(NeatGGenome genome)
        {
            // In a real implementation, this would:
            // 1. Map NEAT-G nodes to ONNX nodes
            // 2. Map connections to ONNX edges
            // 3. Create appropriate operations (Add, Mul, etc.)
            // 4. Set up input/output tensors
            // 5. Set weights and biases as initializers

            // Example structure:
            // Input -> Hidden Layers -> Output
            // Each connection is a Mul (weight * input) + Add (bias)
            // Activation functions are applied at each node

            // For now, return a placeholder
            return null;
        }

        /// <summary>
        /// Exports a NEAT-G genome to ONNX file.
        /// </summary>
        public static void ExportToFile(NeatGGenome genome, string filePath)
        {
            var onnxData = ExportToOnnx(genome);
            File.WriteAllBytes(filePath, onnxData);
        }

        /// <summary>
        /// Converts activation type to ONNX operator name.
        /// </summary>
        public static string GetOnnxActivationOperator(NeatGActivationType activation)
        {
            return activation switch
            {
                NeatGActivationType.Sigmoid => "Sigmoid",
                NeatGActivationType.Tanh => "Tanh",
                NeatGActivationType.ReLU => "Relu",
                NeatGActivationType.LeakyReLU => "LeakyRelu",
                NeatGActivationType.Linear => "Identity",
                _ => "Sigmoid" // Default
            };
        }

        /// <summary>
        /// Creates a simple feed-forward ONNX model from NEAT-G connections.
        /// This is a simplified version for demonstration.
        /// </summary>
        public static byte[] CreateSimpleFeedForwardOnnx(
            int inputCount, 
            int[] hiddenSizes, 
            int outputCount,
            float[][] weights,
            float[] biases)
        {
            // This would create a standard MLP in ONNX format
            // Using Gemm (General Matrix Multiply) operations
            
            // Example:
            // Input -> Gemm -> Activation -> Gemm -> Activation -> ... -> Output
            
            throw new NotImplementedException("Simple ONNX creation not implemented. " +
                "Use Microsoft.ML.OnnxRuntime.InferenceSession for model creation.");
        }
    }

    /// <summary>
    /// ONNX model metadata.
    /// </summary>
    public class OnnxModelMetadata
    {
        /// <summary>
        /// Gets or sets the model name.
        /// </summary>
        public string Name { get; set; } = "NEAT-G Model";

        /// <summary>
        /// Gets or sets the model version.
        /// </summary>
        public string Version { get; set; } = "1.0";

        /// <summary>
        /// Gets or sets the producer name.
        /// </summary>
        public string ProducerName { get; set; } = "Synapse Engine";

        /// <summary>
        /// Gets or sets the domain.
        /// </summary>
        public string Domain { get; set; } = "synapse.ai";

        /// <summary>
        /// Gets or sets the description.
        /// </summary>
        public string Description { get; set; } = "NEAT-G model exported from Synapse";

        /// <summary>
        /// Gets or sets the input names.
        /// </summary>
        public List<string> InputNames { get; } = new List<string>();

        /// <summary>
        /// Gets or sets the output names.
        /// </summary>
        public List<string> OutputNames { get; } = new List<string>();
    }

    /// <summary>
    /// Exception thrown for ONNX export errors.
    /// </summary>
    public class OnnxExportException : Exception
    {
        public OnnxExportException(string message) : base(message) { }
        public OnnxExportException(string message, Exception inner) : base(message, inner) { }
    }
}