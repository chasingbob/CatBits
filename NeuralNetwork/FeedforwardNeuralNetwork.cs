using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace NeuralNetwork
{
    public class FeedforwardNeuralNetwork
    {
        public FeedforwardNeuralNetwork(IActivation _activation)
        {
            Activation = _activation;
        }

        public void Reset()
        {
            Vs = new double[(NumberOfInputValues) * NumberOfHiddenNeurons];
            Ws = new double[(NumberOfHiddenNeurons + 1) * NumberOfOutputValues];
            H = new double[NumberOfHiddenNeurons + 1];
            H[NumberOfHiddenNeurons] = -1;
            Outputs = new double[NumberOfOutputValues];
            Inputs = new double[NumberOfInputValues];

            for (int i=0; i<Vs.Count(); i++)
                Vs[i] = random.NextDouble();

            for (int i = 0; i < Ws.Count(); i++)
                Ws[i] = random.NextDouble();
        }

        IActivation Activation { get; }

        Random random = new Random(DateTime.Now.Millisecond);
        //  Inputs 
        public double[] Inputs;

        public void SetInputs(double[] inputs)
        {
            Inputs = inputs;
        }

        public void SetWeights(double[] weights)
        {
            for (int i=0; i<NumberOfInputValues * NumberOfHiddenNeurons; i++)
            {
                Vs[i] = weights[i];
            }

            var count = 0;
            for (int i=NumberOfInputValues * NumberOfHiddenNeurons; i < weights.Count(); i++)
            {
                Ws[count++] = weights[i];
            }
        }


        //  Hidden 
        public double[] Vs;
        public double[] H;


        //  Outputs
        public double[] Ws;
        public double[] Outputs;

        public int NumberOfInputValues { get; set; }
        public int NumberOfHiddenNeurons { get; set; }
        public int NumberOfOutputValues { get; set; }


        public double[] Calculate()
        {
            for (int i = 0; i < NumberOfHiddenNeurons; i++)
            {
                var weightRange = Vs.ToList().GetRange((NumberOfInputValues) * i, (NumberOfInputValues));
                H[i] = Activation.Compute(new List<double>(Inputs), weightRange);
            }

            for (int i = 0; i < NumberOfOutputValues; i++)
            {
                var weightRange = Ws.ToList().GetRange((NumberOfHiddenNeurons+1)*i, (NumberOfHiddenNeurons + 1));
                Outputs[i] = Activation.Compute(new List<double>(H), weightRange );
            }

            return Outputs.ToArray();
            
        }

        public void Save(string filename)
        {
            throw new NotImplementedException();
        }

        public void Load(string filename)
        {
            throw new NotImplementedException();
        }

    }
}
