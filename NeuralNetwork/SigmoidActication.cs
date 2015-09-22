using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Algorithms.NeuralNetwork;

namespace NeuralNetwork
{
    public class SigmoidActivation : IActivation
    {
        public static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public double Compute(IEnumerable<double> inputs, IEnumerable<double> weights)
        {
            return Sigmoid(computeSigmoid(inputs, weights)); 
        }
    }
}
