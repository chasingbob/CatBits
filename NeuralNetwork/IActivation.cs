using System.Collections.Generic;

namespace NeuralNetwork
{
    public interface IActivation
    {
        double Compute(IEnumerable<double> inputs, IEnumerable<double> weights);
    }
}
