using System;
using System.Linq;

namespace GeneticAlgorithm
{
    public class Genome
    {
        public Genome()
        {
        }

        public Genome(int length)
        {
            Length = length;
            Genes = new double[length];
            CreateGenes();
        }
        public Genome(int length, bool createGenes)
        {
            Length = length;
            Genes = new double[length];
            if (createGenes)
                CreateGenes();
        }

        public Genome(ref double[] genes)
        {
            Length = genes.Length;
            Genes = new double[Length];
            Array.Copy(genes, Genes,Length);
        }

        public Genome DeepCopy()
        {
            Genome g = new Genome(Length, false);
            Array.Copy(Genes, g.Genes, Length);
            return g;
        }

        private void CreateGenes()
        {
            for (int i = 0; i < Genes.Length; i++)
                Genes[i] = random.NextDouble() ;
        }

        public void Crossover(ref Genome genome2, out Genome child1, out Genome child2)
        {
            int pos = (int)(random.NextDouble() * (double)Length);
            child1 = new Genome(Length, false);
            child2 = new Genome(Length, false);
            for (int i = 0; i < Length; i++)
            {
                if (i < pos)
                {
                    child1.Genes[i] = Genes[i];
                    child2.Genes[i] = genome2.Genes[i];
                }
                else
                {
                    child1.Genes[i] = genome2.Genes[i];
                    child2.Genes[i] = Genes[i];
                }
            }
        }

        public void Mutate()
        {
            for (int pos = 0; pos < Length; pos++)
            {
                if (random.NextDouble() < MutationRate)
                    Genes[pos] = (Genes[pos] + random.NextDouble())  / 2.0;
            }
        }

        public double[] Genes { get; set; }

        public void Output()
        {
            Genes.ToList().ForEach(_ => Console.WriteLine(_));
        }

        public void GetValues(ref double[] values)
        {
            for (int i = 0; i < Length; i++)
                values[i] = Genes[i];
        }

        static Random random = new Random();

        public double Fitness { get; set; }

        public static double MutationRate { get; set; }

        public int Length { get; private set; }
    }
}

