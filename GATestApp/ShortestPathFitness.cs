using GeneticAlgorithm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GATestApp
{
    public class ShortestPathFitness : IFitnessFunction
    {
        public double GetFitness(IEnumerable<double> values)
        {
            var array = values.ToArray();
            var X1 = array[0] * 10;
            var Y1 = array[1] * 10;
            var X2 = array[2] * 10;
            var Y2 = array[3] * 10;

            var BeginX = 0;
            var BeginY = 0;

            var EndX = 10;
            var EndY = 10;

            return GetEuclideanDistance(BeginX,BeginY,X1, Y1) + 
                GetEuclideanDistance(X1,Y1,X2,Y2) +
                GetEuclideanDistance(X2,Y2,EndX,EndY);
        }

        public static double GetEuclideanDistance(double X1, double Y1, double X2, double Y2)
        {
            return Math.Sqrt(Math.Pow(X2 - X1, 2) + Math.Pow(Y2 - Y1, 2));
        }
    }
}
