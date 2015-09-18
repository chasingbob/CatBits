using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace FFNNTestApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            //var activation = new SigmoidActivation();
            //var ffnn = new FeedforwardNeuralNetwork(activation);
            //ffnn.NumberOfInputValues = 2;
            //ffnn.NumberOfHiddenNeurons = 2;
            //ffnn.NumberOfOutputValues = 1;
            //ffnn.Reset();

            //ffnn.SetInputs(new double[] { 0, 1,});
            //var result = ffnn.Calculate();

            //result.ToList().ForEach(r => Console.WriteLine(r));

            //Console.ReadKey();

            var fitness = new BooleanOrFitness();

            var ga = new GeneticAlgorithm.GeneticAlgorithm(0.8, 0.05, 100, 100, 4);
            ga.FitnessFunction = fitness;

            ga.Go();

            double[] best = new double[4];
            var bestFitness = 0.0;
            ga.GetBest(out best, out bestFitness);

            Console.ReadKey();
        }
    }
}
