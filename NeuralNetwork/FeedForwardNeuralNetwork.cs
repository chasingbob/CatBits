using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class FeedForwardNeuralNetwork
    {
        private Random rnd;

        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;
        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;

        public FeedForwardNeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.rnd = new Random(0); // for InitializeWeights() and Shuffle()

            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];
            this.InitializeWeights();
        } // ctor

        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        // ----------------------------------------------------------------------------------------

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases,
            // h-o weights, h-o biases
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) +
              numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        private void InitializeWeights()
        {
            // initialize weights and biases to small random values
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) +
              numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }

        public double[] GetWeights()
        {
            // returns the current set of wweights, presumably after training
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) +
              numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] Calculate(double[] inputs)
        {
            return ComputeOutputs(inputs);
        }

        // --------------------------------------------------------------------------------

        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");

            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

            for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)   // apply activation
                this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput];
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs

        private static double HyperTan(double x)
        {
            if (x < -20.0)
                return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0)
                return 1.0;
            else
                return
             Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes so scale doesn't have to be re-computed 
            double max = oSums[0];      // determine max output sum
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        // ---------------------------------------------------------------------------------

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainData) // training stopping condition
        {
            // average squared error per training item
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }

        // ---------------------------------------------------------------------------------

        public void Train(double[][] trainData, int maxEpochs,
          double learnRate, double momentum)
        {
            // train using back-prop
            // back-prop specific arrays
            double[] oGrads = new double[numOutput]; // output gradients
            double[] hGrads = new double[numHidden];

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            double[] oPrevBiasesDelta = new double[numOutput];

            // train a back-prop style NN classifier using learning rate and momentum
            int epoch = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEpochs)
            {
                Shuffle(sequence); // visit each training data in random order
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs 

                    // 1. compute output gradients
                    for (int i = 0; i < numOutput; ++i)
                    {
                        // derivative for softmax = (1 - y) * y (same as log-sigmoid)
                        double derivative = (1 - outputs[i]) * outputs[i];
                        // 'mean squared error version' includes (1-y)(y) derivative
                        oGrads[i] = derivative * (tValues[i] - outputs[i]);
                    }

                    // 2. compute hidden gradients
                    for (int i = 0; i < numHidden; ++i)
                    {
                        // derivative of tanh = (1 - y) * (1 + y)
                        double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                        double sum = 0.0;
                        for (int j = 0; j < numOutput; ++j) // sum of numOutput terms
                        {
                            double x = oGrads[j] * hoWeights[i][j];
                            sum += x;
                        }
                        hGrads[i] = derivative * sum;
                    }

                    // 3a. update hidden weights (gradients must be computed
                    // right-to-left but weights can be updated in any order)
                    for (int i = 0; i < numInput; ++i) // 0..2 (3)
                    {
                        for (int j = 0; j < numHidden; ++j) // 0..3 (4)
                        {
                            double delta = learnRate * hGrads[j] * inputs[i];
                            ihWeights[i][j] += delta; // update. note '+' 
                                                      // now add momentum using previous delta.
                            ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                            ihPrevWeightsDelta[i][j] = delta; // save the delta for momentum 
                        }
                    }

                    // 3b. update hidden biases
                    for (int i = 0; i < numHidden; ++i)
                    {
                        double delta = learnRate * hGrads[i]; // 1.0 is constant input for bias
                        hBiases[i] += delta;
                        hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
                        hPrevBiasesDelta[i] = delta; // don't forget to save the delta
                    }

                    // 4. update hidden-output weights
                    for (int i = 0; i < numHidden; ++i)
                    {
                        for (int j = 0; j < numOutput; ++j)
                        {
                            double delta = learnRate * oGrads[j] * hOutputs[i];
                            hoWeights[i][j] += delta;
                            hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
                            hoPrevWeightsDelta[i][j] = delta; // save
                        }
                    }

                    // 4b. update output biases
                    for (int i = 0; i < numOutput; ++i)
                    {
                        double delta = learnRate * oGrads[i] * 1.0;
                        oBiases[i] += delta;
                        oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
                        oPrevBiasesDelta[i] = delta; // save
                    }

                } // each training item
                ++epoch;
            }
        } // Train2

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // get x-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly.
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
    } // class NeuralNetwork
}
