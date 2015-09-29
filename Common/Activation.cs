using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CatBits.Common
{
    public class Activation
    {
        public static float Sigmoid(float input)
        {
            //if (input > 5)
            //    return 1;
            //if (input < -5)
            //    return 0;

            return (float)(1 / (1 + Exp(-input)));
        }

        public static double Exp(double val)
        {
            long tmp = (long)(1512775 * val + (1072693248 - 60801));
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }
    }
}
