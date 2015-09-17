using GeneticAlgorithm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GATestApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            var fitness = new ShortestPathFitness();

            var ga = new GeneticAlgorithm.GeneticAlgorithm(0.8, 0.05, 100, 1000, 4);
            ga.FitnessFunction = fitness;

            ga.Go();

            double[] best = new double[4];
            var bestFitness = 0.0;
            ga.GetBest(out best, out bestFitness);

            Console.ReadKey();

        }
    }
}
