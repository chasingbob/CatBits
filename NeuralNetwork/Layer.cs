using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Layer
    {
        public IActivation Activation { get; set; }
        public IEnumerable<Neuron> Neurons { get; set; }
    }
}
