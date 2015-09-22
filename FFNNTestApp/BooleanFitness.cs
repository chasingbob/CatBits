﻿using GeneticAlgorithm;
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
            var total = 0.0;
            //  OR

            var network = new FeedForwardNeuralNetwork(2, 2, 1);
            network.SetWeights(values.ToArray());
            

            var result = network.Calculate(new double[] { 0.0, 0.0 }) ;
            var answer = result[0];
            total += 1 - answer;


            result = network.Calculate(new double[] { 1.0, 0.0 });
            answer = result[0];
            total += answer;

            result = network.Calculate(new double[] { 0.0, 1.0 });
            answer = result[0];
            total += answer;
            
            result = network.Calculate(new double[] { 1.0, 1.0 });
            answer = result[0];
            total += answer;

            return total;
        }
    }
}
