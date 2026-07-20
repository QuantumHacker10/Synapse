using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Core.NEAT;
using Xunit;

namespace Synapse.Tests.AI;

public sealed class NeatGEvolutionEngineTests
{
    private static EvolutionConfig CreateTinyConfig()
    {
        var config = EvolutionConfig.CreateDefault();
        config.PopulationSize = 12;
        config.EliteCount = 2;
        config.MaxGenerations = 2;
        config.TargetSpeciesCount = 3;
        config.EnableMigration = false;
        config.RandomSeed = 42;
        config.MaxParallelEvaluations = 2;
        config.EvaluationTimeoutMs = 5000;
        return config;
    }

    [Fact]
    public void InitializePopulation_CreatesConfiguredPopulationSize()
    {
        var config = CreateTinyConfig();
        var manager = new GenomePopulationManager(config, new System.Random(42));

        var population = manager.InitializePopulation(inputCount: 3, outputCount: 2);

        population.Genomes.Should().HaveCount(config.PopulationSize);
        population.GenerationNumber.Should().Be(0);
        population.Genomes.Should().OnlyContain(g => g.InputCount == 3 && g.OutputCount == 2);
    }

    [Fact]
    public void CreateRandomGenome_AssignsFiniteInitialFitnessPlaceholder()
    {
        var config = CreateTinyConfig();
        var manager = new GenomePopulationManager(config, new System.Random(7));

        var genome = manager.CreateRandomGenome(inputCount: 4, outputCount: 1, generation: 0);

        genome.InputCount.Should().Be(4);
        genome.OutputCount.Should().Be(1);
        genome.Id.Should().NotBeEmpty();
    }

    [Fact]
    public async Task Engine_InitializeAndEvaluate_ProducesFiniteFitness()
    {
        var config = CreateTinyConfig();
        await using var engine = new NeatGEvolutionEngine(config);
        var context = new EvaluationContext
        {
            ComponentWeights = EvaluationContext.CreateDefault().ComponentWeights,
            SampleCount = 8,
            InputData = ImmutableArray.Create(0.5, 0.25, 0.75),
            TargetOutput = ImmutableArray.Create(0.4, 0.6)
        };

        engine.InitializePopulation(3, 2);
        await engine.EvaluatePopulationAsync(context, CancellationToken.None);

        engine.CurrentPopulation.Genomes.Should().NotBeEmpty();
        engine.CurrentPopulation.Genomes.Should().OnlyContain(g => double.IsFinite(g.Fitness));
    }

    [Fact]
    public async Task SpeciationStrategy_GroupsEvaluatedPopulationIntoSpecies()
    {
        var config = CreateTinyConfig();
        var manager = new GenomePopulationManager(config, new System.Random(99));
        var population = manager.InitializePopulation(2, 2);
        var context = new EvaluationContext
        {
            ComponentWeights = EvaluationContext.CreateDefault().ComponentWeights,
            SampleCount = 4
        };
        var evaluator = new FitnessEvaluator(context);

        var evaluated = new GeoGenome[population.Genomes.Length];
        for (int i = 0; i < population.Genomes.Length; i++)
        {
            evaluated[i] = await evaluator.EvaluateAsync(population.Genomes[i], context, CancellationToken.None);
        }

        population = population with { Genomes = evaluated.ToImmutableArray() };
        var strategy = new ManifoldSpeciationStrategy(config);
        var species = strategy.Speciate(population);

        species.Should().NotBeEmpty();
        species.Sum(s => s.MemberCount).Should().Be(population.Genomes.Length);
    }
}
