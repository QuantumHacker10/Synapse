using FluentAssertions;
using GDNN.Core.Neurons;
using System.Numerics;
using Xunit;

namespace Synapse.Tests.AI;

public class NeuronTypesTests
{
    [Fact]
    public void SdfPrimitiveKernel_Compute_ShouldReturnValidDistance()
    {
        var neuron = new SdfPrimitiveKernel();
        var input = NeuronInput.Default with { Position = new Vector3(1.0f, 0.0f, 0.0f) };

        var result = neuron.Compute(input);

        float.IsNaN(result.Value).Should().BeFalse();
    }

    [Fact]
    public void CurvatureModulatorKernel_ParameterCount_ShouldBeTen()
    {
        var neuron = new CurvatureModulatorKernel();

        neuron.GetParameterCount().Should().Be(10);
    }

    [Fact]
    public void MarchingCubeKernel_ParameterCount_ShouldBePositive()
    {
        var neuron = new MarchingCubeKernel();

        neuron.GetParameterCount().Should().BeGreaterThan(0);
    }
}
