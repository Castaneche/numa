#include "Common.h"

namespace numa {
	std::vector<double> linspace(double start, double stop, int N)
	{
		if (start > stop) { //swap values if start is greater than stop
			double tmp = stop;
			stop = start;
			start = tmp;
		}

		std::vector<double> arr;
		double step = (stop - start) / (double)N;
		for (double x = start; x < stop; x += step)
			arr.push_back(x);

		return arr;
	}
	std::vector<double> arange(double start, double stop, double step)
	{
		if (start > stop) { //swap values if start is greater than stop
			double tmp = stop;
			stop = start;
			start = tmp;
		}

		std::vector<double> arr;
		for (double x = start; x < stop; x += step)
			arr.push_back(x);

		return arr;
	}
	std::vector<double> arange(double start, int N, double step)
	{
		std::vector<double> arr;
		for (unsigned int i = 0; i < N; i++) {
			double x = start + i * step;
			arr.push_back(x);
		}

		return arr;
	}
	std::vector<double> genarray(double start, double end, int N, std::function<double(double)> mathfunction)
	{
		std::vector<double> x = linspace(start, end, N);
		std::vector<double> y;
		for (auto& i : x)
			y.push_back(mathfunction(i));

		return y;
	}
	std::vector<double> genarray(double start, double end, double step, std::function<double(double)> mathfunction)
	{
		std::vector<double> x = arange(start, end, step);
		std::vector<double> y;
		for (auto& i : x)
			y.push_back(mathfunction(i));

		return y;
	}
	std::vector<double> genarray(const std::vector<double>& xdata, std::function<double(double)> mathfunction)
	{
		std::vector<double> y;
		for (auto& i : xdata)
			y.push_back(mathfunction(i));

		return y;
	}
}