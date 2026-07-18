using System.Collections.Immutable;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Core.NEAT;
using Xunit;

namespace Synapse.Tests.AI;

public class IrradianceFitnessTests
{
    [Fact]
    public void FitnessComponent_IrradianceError_IsRecognized()
    {
        var values = System.Enum.GetValues<FitnessComponent>();
        values.Should().Contain(FitnessComponent.IrradianceError);
    }

    [Fact]
    public async Task ComputeIrradianceError_ReturnsFiniteValue()
    {
        var config = EvolutionConfig.CreateDefault();
        config.EnableIrradianceFitness = true;

        var manager = new GenomePopulationManager(config, new System.Random(42));
        var genome = manager.CreateRandomGenome(inputCount: 3, outputCount: 2, generation: 0);

        var context = new EvaluationContext
        {
            InputData = ImmutableArray.Create(0.5, 0.3, 0.8),
            TargetOutput = ImmutableArray.Create(0.4, 0.6),
            ComponentWeights = ImmutableDictionary<FitnessComponent, double>.Empty
                .Add(FitnessComponent.IrradianceError, 1.0)
        }.ApplyEvolutionConfig(config);

        var evaluator = new FitnessEvaluator(context);
        var evaluated = await evaluator.EvaluateAsync(genome, context, CancellationToken.None);

        evaluated.FitnessComponents.Should().ContainKey(FitnessComponent.IrradianceError);
        var irradianceScore = evaluated.FitnessComponents[FitnessComponent.IrradianceError];
        double.IsFinite(irradianceScore).Should().BeTrue();
        irradianceScore.Should().BeInRange(0.0, 1.0);
    }
}
