using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static System.Math;
using static Algorithms.NeuralNetwork;

namespace AlgorithmTests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DissimilarVectors_ThrowsException()
        {
            computeSigmoid(inputs: new[] { 1, 2.0, 3 }, weights: new[] { 0.1 });
        }

        [TestMethod]
        public void Compute_IsSigmoidal()
        {
            var acceptedError = 0.05;
            var randomInput = new Random().NextDouble();

            var computed = computeSigmoid(new[] { randomInput }, new[] { 0.99 });

            var error = Abs(Tanh(randomInput) - computed);

            Assert.IsTrue(error < acceptedError);
        }
    }
}
