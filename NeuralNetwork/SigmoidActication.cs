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
        public double Compute(IEnumerable<double> inputs, IEnumerable<double> weights)
        {
            return computeSigmoid(inputs, weights); 
        }
    }
}
