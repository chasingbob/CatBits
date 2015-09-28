namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public Neuron[] Neurons { get; private set; }
        public NeuralLayer(int nrOfNeuronsInPreviousLayer, int nrOfNeurons)
        {
            Neurons = new Neuron[nrOfNeurons];
            for (int i = 0; i < nrOfNeurons; i++)
                Neurons[i] = new Neuron(nrOfNeuronsInPreviousLayer);
        }
    }
}
