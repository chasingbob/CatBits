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
            InitWeights();
        }

        private void InitWeights()
        {
            for (int i=0; i<9; i++)
                Vs[i] = random.NextDouble();

            for (int i = 0; i < 8; i++)
                Ws[i] = random.NextDouble();
        }

        IActivation Activation { get; }

        Random random = new Random(DateTime.Now.Millisecond);
        //  Inputs 
        public double[] Inputs = new double[3];


        //  Hidden 
        public double[] Vs = new double[(2+1) * 3];
        public double[] H = new double[4];


        //  Outputs
        public double[] Ws = new double[(3+1) * 2];
        public double[] Outputs = new double[2];


        public double[] Calculate()
        {
            for (int i=0; i<3; i++)
                H[i] = Activation.Compute(new List<double>(Inputs), new List<Double>() { Vs[i * 3], Vs[i * 3 + 1], Vs[i * 3 + 2] });

            H[3] = -1;

            for (int i = 0; i < 2; i++)
                Outputs[i] = Activation.Compute(new List<double>(H), new List<double>() {Ws[i *2], Ws[i*2 +1], Ws[i*2 + 2], Ws[i*2 + 3] });

            return Outputs.ToArray();
            
        }

    }
}
