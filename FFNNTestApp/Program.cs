using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using System.Drawing;
using System.IO;


namespace FFNNTestApp
{
    public class Program
    {

        static void Main()
        {
            //  Is it an apple?
            var posTrain = @"C:\Temp\NN\train_pos";
            var posTest = @"C:\Temp\NN\test_pos";
            var negTrain = @"C:\Temp\NN\train_neg";
            var negTest = @"C:\Temp\NN\test_neg";
            var pearsTest = @"C:\Temp\NN\pears_test";

            var network = new FeedForwardNeuralNetwork(625, 1, 1, 30, -1);

            var valTrainPos = GetValues(posTrain);
            var valTrainNeg = GetValues(negTrain);
            var valTestPos = GetValues(posTest);
            var valTestNeg = GetValues(negTest);
            var valPears = GetValues(pearsTest);

            for (int i=0; i<10000; i++)
            {
                foreach (var p in valTrainPos)
                {
                    network.Train(p.Item2, new float[] { 1.0f }, 0.02f);
                }
                foreach (var p in valTrainNeg)
                {
                    network.Train(p.Item2, new float[] { 0.0f }, 0.02f);
                }
            }


            foreach (var p in valTestPos)
            {
                var result = network.Calculate(p.Item2);
                Console.WriteLine($"Pos: {result[0]}");
            }

            Console.WriteLine("\r\n");

            foreach (var p in valTestNeg)
            {
                var result = network.Calculate(p.Item2);
                Console.WriteLine($"Neg: {result[0]}");
            }

            Console.WriteLine("\r\n");

            foreach (var p in valPears)
            {
                var result = network.Calculate(p.Item2);
                Console.WriteLine($"Pear (Not apple): {result[0]}");
            }
        }

        private static List<Tuple<int, float[]>> GetValues(string posTrain)
        {
            var values = new List<Tuple<int, float[]>>();
            var count = 0;
            foreach (var f in Directory.GetFiles(posTrain))
            {
                var img = (Bitmap)Image.FromFile(f);
                var list = new List<float>();

                for (var x = 0; x < 25; x++)
                {
                    for (var y = 0; y < 25; y++)
                    {
                        var p = (float)((float)(img.GetPixel(x, y).R + img.GetPixel(x, y).G + img.GetPixel(x, y).B) / 3) / 255.0f;
                        list.Add(p);
                    }
                }
                values.Add(new Tuple<int, float[]>(count, list.ToArray()));

                count++;
            }

            return values;
        }

     
           
    }
}
