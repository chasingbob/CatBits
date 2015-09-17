﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeneticAlgorithm
{
	public class GeneticAlgorithm
	{
		/// <summary>
		/// Default constructor sets mutation rate to 5%, crossover to 80%, population to 100,
		/// and generations to 2000.
		/// </summary>
		public GeneticAlgorithm()
		{
			InitialValues();
			MutationRate = 0.05;
			CrossoverRate = 0.80;
			PopulationSize = 100;
			GenerationSize = 2000;
			StrFitness = "";
		}

		public GeneticAlgorithm(double crossoverRate,
				  double mutationRate,
				  int populationSize,
				  int generationSize,
				  int genomeSize)
		{
			InitialValues();
			MutationRate = mutationRate;
			CrossoverRate = crossoverRate;
			PopulationSize = populationSize;
			GenerationSize = generationSize;
			GenomeSize = genomeSize;
			StrFitness = "";
		}

		public GeneticAlgorithm(int genomeSize)
		{
			InitialValues();
			GenomeSize = genomeSize;
		}


		public void InitialValues()
		{
			Elitism = false;
		}


		/// <summary>
		/// Method which starts the GA executing.
		/// </summary>
		public void Go()
		{
		   
			if (GenomeSize == 0)
				throw new IndexOutOfRangeException("Genome size not set");

			FitnessTable = new List<double>();
			CurrentGeneration = new List<Genome>(GenerationSize);
			NextGeneration = new List<Genome>(GenerationSize);
			Genome.MutationRate = MutationRate;


			CreateGenomes();
			RankPopulation();


			for (int i = 0; i < GenerationSize; i++)
			{
				CreateNextGeneration();
				double fitness = RankPopulation();

				if (i % 100 == 0)
				{
					Console.WriteLine("Generation " + i + ", Best Fitness: " + fitness);
				}
			}

		}

		/// <summary>
		/// After ranking all the genomes by fitness, use a 'roulette wheel' selection
		/// method.  This allocates a large probability of selection to those with the 
		/// highest fitness.
		/// </summary>
		/// <returns>Random individual biased towards highest fitness</returns>
		private int RouletteSelection()
		{
			double randomFitness = random.NextDouble() * TotalFitness;
			int idx = -1;
			int mid;
			int first = 0;
			int last = PopulationSize - 1;
			mid = (last - first) / 2;

			//  ArrayList's BinarySearch is for exact values only
			//  so do this by hand.
			while (idx == -1 && first <= last)
			{
				if (randomFitness < FitnessTable[mid])
				{
					last = mid;
				}
				else if (randomFitness > FitnessTable[mid])
				{
					first = mid;
				}
				mid = (first + last) / 2;
				//  lies between i and i+1
				if ((last - first) == 1)
					idx = last;
			}
			return idx;
		}

		/// <summary>
		/// Rank population and sort in order of fitness.
		/// </summary>
		private double RankPopulation()
		{
			TotalFitness = 0.0;
			foreach (Genome g in CurrentGeneration)
			{
				g.Fitness = FitnessFunction.GetFitness(g.Genes);
				TotalFitness += g.Fitness;
			}
			CurrentGeneration.Sort(delegate (Genome x, Genome y)
			{
				return Comparer<double>.Default.Compare(y.Fitness, x.Fitness);
			});

			//  now sorted in order of fitness.
			double fitness = 0.0;
			FitnessTable.Clear();
			foreach (Genome t in CurrentGeneration)
			{
				fitness += t.Fitness;
				FitnessTable.Add(t.Fitness);
			}

			return FitnessTable[FitnessTable.Count - 1];
		}

		/// <summary>
		/// Create the *initial* genomes by repeated calling the supplied fitness function
		/// </summary>
		private void CreateGenomes()
		{
			for (int i = 0; i < PopulationSize; i++)
			{
				Genome g = new Genome(GenomeSize);
				CurrentGeneration.Add(g);
			}
		}

		private void CreateNextGeneration()
		{
			NextGeneration.Clear();
			Genome g = null;
			if (Elitism)
				g = CurrentGeneration[PopulationSize - 1].DeepCopy();

			for (int i = 0; i < PopulationSize; i += 2)
			{
				int pidx1 = RouletteSelection();
				int pidx2 = RouletteSelection();
				Genome parent1, parent2, child1, child2;
				parent1 = CurrentGeneration[pidx1];
				parent2 = CurrentGeneration[pidx2];

				if (random.NextDouble() < CrossoverRate)
				{
					parent1.Crossover(ref parent2, out child1, out child2);
				}
				else
				{
					child1 = parent1;
					child2 = parent2;
				}
				child1.Mutate();
				child2.Mutate();

				NextGeneration.Add(child1);
				NextGeneration.Add(child2);
			}
			if (Elitism && g != null)
				NextGeneration[0] = g;

			CurrentGeneration.Clear();
			foreach (Genome ge in NextGeneration)
				CurrentGeneration.Add(ge);
		}


		public double MutationRate { get; set; }
		public double CrossoverRate { get; set; }
		public int PopulationSize { get; set; }
		public int GenerationSize { get; set; }
		public int GenomeSize { get; set; }
		public double TotalFitness { get; set; }
		public string StrFitness { get; set; }
		public bool Elitism { get; set; }

		public List<Genome> CurrentGeneration;
		public List<Genome> NextGeneration;
		public List<double> FitnessTable;

		static Random random = new Random();


		public IFitnessFunction FitnessFunction { get; set; }
		

		public void GetBest(out double[] values, out double fitness)
		{
			Genome g = CurrentGeneration[PopulationSize - 1];
			values = new double[g.Length];
			g.GetValues(ref values);
			fitness = g.Fitness;
		}

		public void GetWorst(out double[] values, out double fitness)
		{
			GetNthGenome(0, out values, out fitness);
		}

		public void GetNthGenome(int n, out double[] values, out double fitness)
		{
			if (n < 0 || n > PopulationSize - 1)
				throw new ArgumentOutOfRangeException("n too large, or too small");
			Genome g = CurrentGeneration[n];
			values = new double[g.Length];
			g.GetValues(ref values);
			fitness = g.Fitness;
		}
	}

}
