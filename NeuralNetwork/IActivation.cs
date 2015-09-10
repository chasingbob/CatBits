using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public interface IActivation
    {
        double Compute(IEnumerable<double> inputs, IEnumerable<double> weights);
    }
}
