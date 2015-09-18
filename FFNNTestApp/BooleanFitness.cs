using GeneticAlgorithm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace FFNNTestApp
{
    public class BooleanOrFitness : IFitnessFunction
    {
        public double GetFitness(IEnumerable<double> values)
        {

            var activation = new SigmoidActivation();
            var network = new FeedforwardNeuralNetwork(activation);

            var total = 0.0;
            //  OR

            network.SetInputs(new double[] { 0.0, 0.0 });
            var result = network.Calculate();
            var answer = result[0];
            total += 1 - answer;


            network.SetInputs(new double[] { 1.0, 0.0 });
            result = network.Calculate();
            answer = result[0];
            total += answer;

            network.SetInputs(new double[] { 0.0, 1.0 });
            result = network.Calculate();
            answer = result[0];
            total += answer;
            
            network.SetInputs(new double[] { 1.0, 1.0 });
            result = network.Calculate();
            answer = result[0];
            total += answer;

            return total;
        }
    }
}
