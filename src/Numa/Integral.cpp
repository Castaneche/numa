#include "Integral.h"

namespace numa {
	namespace integral {
		double simpson(const std::vector<double>& y, const double& step)
		{
			double result = 0.0;
			double sum = 0.0;

			//Even
			for (unsigned int i = 2; i < y.size() - 1; i += 2)
			{
				sum += y[i];
			}
			sum *= 2.0;
			result += sum;

			//Odd
			sum = 0.0;
			for (unsigned int i = 1; i < y.size(); i += 2)
			{
				sum += y[i];
			}
			sum *= 4.0;
			result += sum;

			result += (y[0] + y[y.size() - 1]); //f(a) + f(b)
			result *= (step / 3.0);
			return result;
		}
	}
}