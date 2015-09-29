namespace NeuralNetwork
{
    public class Layer
    {
        public Neuron[] Neurons { get; private set; }
        public Layer(int nrOfNeuronsInPreviousLayer, int nrOfNeurons)
        {
            Neurons = new Neuron[nrOfNeurons];
            for (int i = 0; i < nrOfNeurons; i++)
                Neurons[i] = new Neuron(nrOfNeuronsInPreviousLayer);
        }
    }
}
