using System.Collections.Generic;

namespace GeneticAlgorithm
{
    public interface IFitnessFunction
    {
        double GetFitness(IEnumerable<double> values);
    }
}