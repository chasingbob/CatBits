using System;
using System.Linq;
using CatBits.Common;

namespace NeuralNetwork
{
    public class FeedForwardNeuralNetwork
    {
        private Layer[] layers;
        public int NrOfOutputs { get; private set; }
        public int NrOfNeuronsPerHiddenLayer { get; private set; }
        public int NrOfHiddenLayers { get; private set; }
        public int NrOfInputs { get; private set; }
        private float bias;

        public FeedForwardNeuralNetwork(int nrOfInputs, int nrOfOutputs, int nrHiddenLayers, int nrOfNeuronsPerHiddenLayer, float bias, float[] weights = null)
        {
            this.bias = bias;

            this.NrOfInputs = nrOfInputs;
            this.NrOfHiddenLayers = nrHiddenLayers;
            this.NrOfOutputs = nrOfOutputs;
            this.NrOfNeuronsPerHiddenLayer = nrOfNeuronsPerHiddenLayer;

            layers = new Layer[nrHiddenLayers + 1];

            layers[0] = new Layer(nrOfInputs, nrOfNeuronsPerHiddenLayer);
            for (int i = 1; i < nrHiddenLayers; i++)
                layers[i] = new Layer(nrOfNeuronsPerHiddenLayer, nrOfNeuronsPerHiddenLayer);

            // last layer is the output layer
            layers[nrHiddenLayers] = new Layer(nrOfNeuronsPerHiddenLayer, nrOfOutputs);


            if (weights == null)
                weights = GetRandomWeights(weights);

            SetWeights(weights);
        }

        private float[] GetRandomWeights(float[] weights)
        {
            weights = GetWeights();
            Random rnd = RandomManager.Instance.Random;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (float)rnd.NextDouble() * 2 - 1;
            }
            return weights;
        }

        public void SetWeights(float[] weights)
        {
            int w = 0;
            foreach (var layer in layers)
            {
                foreach (var n in layer.Neurons)
                {
                    for (int i = 0; i < n.NeuronWeights.Length; i++)
                    {
                        n.NeuronWeights[i] = weights[w];
                        w++;
                    }
                }
            }
        }

        public float[] GetWeights()
        {
            return layers.SelectMany(l => l.Neurons).SelectMany(l => l.NeuronWeights).ToArray();
        }

        public float[] Calculate(float[] input)
        {
            float[] currentValues = input;

            foreach (var layer in layers)
            {
                float[] output = new float[layer.Neurons.Length];
                for (int i = 0; i < layer.Neurons.Length; i++)
                {
                    float sum = layer.Neurons[i].Sum(bias, currentValues);

                    float outputValue;
                    if (sum >= 10) outputValue = 1;
                    else if (sum < -10) outputValue = 0;
                    else
                        outputValue = ActivationFunction(sum); //Sigmoid(sum);


                    output[i] = outputValue;
                }
                currentValues = output;
            }

            return currentValues;
        }

        public void Train(float[] input, float[] expectedOutput, float learningRate)
        {

            //int inputIndex = 0;
            int outputIndex = NrOfHiddenLayers + 1;

            float[][] values = CalculateActivationPerNeuron(input);

            float[][] deltas = new float[NrOfHiddenLayers + 1][];

            deltas[layers.Length - 1] = new float[NrOfOutputs];
            var lastLayer = deltas[layers.Length - 1];
            var outputValues = values[outputIndex];
            for (int i = 0; i < NrOfOutputs; i++)
                lastLayer[i] = expectedOutput[i] - outputValues[i];

            for (int layerIdx = layers.Length - 2; layerIdx >= 0; layerIdx--)
            {
                var layer = layers[layerIdx];
                deltas[layerIdx] = new float[layer.Neurons.Length];

                var nextLayer = layers[layerIdx + 1];
                var nrOfNeuronsInNextLayer = nextLayer.Neurons.Length;
                var deltasOfNextLayer = deltas[layerIdx + 1];
                var deltasOfCurrentLayer = deltas[layerIdx];
                var nrOfNeuronsInLayer = layer.Neurons.Length;
                for (int j = 0; j < nrOfNeuronsInLayer; j++)
                {
                    float deltaSum = 0;
                    for (int k = 0; k < nrOfNeuronsInNextLayer; k++)
                        deltaSum += nextLayer.Neurons[k].NeuronWeights[j] * deltasOfNextLayer[k];

                    deltasOfCurrentLayer[j] = deltaSum;
                }
            }

            // apply weights
            for (int layerIdx = 0; layerIdx < layers.Length; layerIdx++)
            {
                var layer = layers[layerIdx];
                var valuesOfNextLayer = values[layerIdx + 1];
                // value array from inputs of current layer
                float[] valuesOfInputForLayer = values[layerIdx];
                var deltasOfCurrentLayer = deltas[layerIdx];
                var nrOfNeuronsInLayer = layer.Neurons.Length;
                for (int neuronIdx = 0; neuronIdx < nrOfNeuronsInLayer; neuronIdx++)
                {
                    // dSigmoid = value * (1 - value) (partial derivatives components of the gradient)
                    // value = the output of the current neuron
                    // dValue = the derivative of the value
                    var valueOfNextLayerOfNeuron = valuesOfNextLayer[neuronIdx];
                    float dValue = valueOfNextLayerOfNeuron * (1 - valueOfNextLayerOfNeuron);

                    // delta of current neuron
                    float delta = deltasOfCurrentLayer[neuronIdx];

                    var neuron = layer.Neurons[neuronIdx];
                    var nrOfWeightsOfNeuron = neuron.NeuronWeights.Length;
                    for (int k = 0; k < nrOfWeightsOfNeuron; k++)
                    {
                        // http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
                        float newWeight = neuron.NeuronWeights[k] + learningRate * delta * dValue * valuesOfInputForLayer[k];
                        //if (newWeight > 1) newWeight = 1;
                        //if (newWeight < -1) newWeight = -1;
                        neuron.NeuronWeights[k] = newWeight;
                    }
                }
            }
        }

        private float[][] CalculateActivationPerNeuron(float[] input)
        {
            float[][] values = new float[1 + 1 + NrOfHiddenLayers][];

            float[] currentValues = input;
            values[0] = currentValues;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                float[] output = new float[layer.Neurons.Length];
                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    float sum = layer.Neurons[j].Sum(bias, currentValues);

                    float outputValue;
                    if (sum >= 10) outputValue = 1;
                    else if (sum < -10) outputValue = 0;
                    else outputValue = ActivationFunction(sum);  //Sigmoid(sum);

                    output[j] = outputValue;
                }
                currentValues = output;
                values[i + 1] = currentValues;
            }

            return values;
        }
      
        public virtual float ActivationFunction(float value)
        {
            return Activation.Sigmoid(value);
        }

        
    }
}
