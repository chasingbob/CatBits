namespace NeuralNetwork
{
    public struct Neuron
    {
        public float[] NeuronWeights { get; private set; }

        public Neuron(int nrOfInputs)
        {
            NeuronWeights = new float[nrOfInputs];
        }

        public float Sum(float bias, float[] input)
        {
            float sum = 0;
            for (int i = 0; i < input.Length; i++)
                sum += input[i] * NeuronWeights[i];

            sum += bias * NeuronWeights[NeuronWeights.Length - 1];

            return sum;
        }
    }
}
