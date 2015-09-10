using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FFNNTestApp
{
    public class Program
    {
        static void Main(string[] args)
        {
            var activation = new NeuralNetwork.SigmoidActication();
            var ffnn = new NeuralNetwork.FeedforwardNeuralNetwork(activation);
            ffnn.Inputs = new double[] { 0.1, 0.9, -1.0 };
            var result = ffnn.Calculate();

            result.ToList().ForEach(r => Console.WriteLine(r));

            result = ffnn.Calculate();

            result.ToList().ForEach(r => Console.WriteLine(r));

            Console.ReadKey();
        }
    }
}
