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
            var activation = new NeuralNetwork.SigmoidActivation();
            var ffnn = new NeuralNetwork.FeedforwardNeuralNetwork(activation);
            ffnn.NumberOfInputValues = 2;
            ffnn.NumberOfHiddenNeurons = 2;
            ffnn.NumberOfOutputValues = 1;
            ffnn.Reset();

            ffnn.SetInputs(new double[] { 0, 1,});
            var result = ffnn.Calculate();

            result.ToList().ForEach(r => Console.WriteLine(r));

            result = ffnn.Calculate();

            result.ToList().ForEach(r => Console.WriteLine(r));

            Console.ReadKey();
        }
    }
}
