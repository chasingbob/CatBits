namespace NeuralNetwork
{
    public struct Neuron
    {

        private float[] neuronWeights;

        public float[] NeuronWeights { get { return neuronWeights; } }

        public Neuron(int nrOfInputs)
        {
            neuronWeights = new float[nrOfInputs];
        }

        public float Sum(float bias, float[] input)
        {
            float sum = 0;
            for (int i = 0; i < input.Length; i++)
                sum += input[i] * neuronWeights[i];

            sum += bias * neuronWeights[neuronWeights.Length - 1];

            return sum;
        }
    }
}
