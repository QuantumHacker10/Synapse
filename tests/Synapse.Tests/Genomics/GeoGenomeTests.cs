using FluentAssertions;
using GDNN.Core.Genome;
using Xunit;

namespace Synapse.Tests.Genomics;

public class GeoGenomeTests
{
    [Fact]
    public void GeoGenome_Create_ShouldHaveDefaultFitness()
    {
        var genome = GeoGenome.CreateEmpty();

        genome.Metadata.FitnessScore.Should().Be(0.0);
    }

    [Fact]
    public void GeoGenome_Mutate_ShouldReturnNewGenome()
    {
        var genome = NeuronGeneFactory.CreateRandomGenome(seed: 42);
        var original = genome.Clone();

        var mutated = GenomeUtilities.Mutate(genome, rate: 0.5f, count: 5, seed: 42);

        mutated.Should().NotBe(original);
    }

    [Fact]
    public void GeoGenome_Crossover_ShouldReturnChild()
    {
        var parent1 = NeuronGeneFactory.CreateRandomGenome(seed: 1);
        var parent2 = NeuronGeneFactory.CreateRandomGenome(seed: 2);

        var child = GenomeUtilities.Crossover(parent1, parent2, rate: 0.5f, seed: 42);

        child.Should().NotBe(parent1);
        child.Should().NotBe(parent2);
        child.Metadata.ParentIds.Should().HaveCount(2);
    }
}
