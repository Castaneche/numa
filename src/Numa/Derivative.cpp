#include "Derivative.h"


namespace numa {
	namespace derivative {

		double threepts(const std::vector<double>& x, const std::vector<double>& y, const unsigned int& xindex)
		{
			double result = 0;

			if (0 < xindex && xindex < (x.size() - 1)) {
				double stepL = x[xindex] - x[(xindex - 1)];
				double stepR = x[(xindex + 1)] - x[xindex];

				result = (y[(xindex + 1)] - y[(xindex - 1)]) / (stepL + stepR);
			}
			else if (xindex == 0) //Left border
			{
				double step = x[(xindex + 1)] - x[xindex];
				result = (y[(xindex + 1)] - y[xindex]) / step;
			}
			else if (xindex == (x.size() - 1)) { //Right border
				double step = x[xindex] - x[(xindex - 1)];
				result = (y[xindex] - y[(xindex - 1)]) / step;
			}
			return result;
		}

		std::vector<double> threepts(const std::vector<double>& x, const std::vector<double>& y)
		{
			std::vector<double> result(x.size(), 0);
			for (unsigned int i = 0; i < x.size(); i++) {
				result[i] = derivative::threepts(x, y, i);
			}
			return result;
		}

		double fivepts(const std::vector<double>& x, const std::vector<double>& y, const unsigned int& xindex)
		{
			double result = 0;
			if (1 < xindex && xindex < (x.size() - 2)) {
				double stepL = x[xindex] - x[(xindex - 1)];
				double stepR = x[(xindex + 1)] - x[xindex];

				result = (y[(xindex - 2)] + 8.0 * y[(xindex + 1)] - 8.0 * y[(xindex - 1)] - y[(xindex + 2)]) / (6 * (stepL + stepR));
			}
			else if (1 >= xindex || xindex >= (x.size() - 2)) { //Borders
				result = derivative::threepts(x, y, xindex);
			}

			return result;
		}

		std::vector<double> fivepts(const std::vector<double>& x, const std::vector<double>& y)
		{
			std::vector<double> result(x.size(), 0);
			for (unsigned int i = 0; i < x.size(); i++) {
				result[i] = derivative::fivepts(x, y, i);
			}
			return result;
		}

	}
}

