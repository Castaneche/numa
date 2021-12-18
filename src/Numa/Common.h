#pragma once

#include <vector>
#include <functional>

namespace numa {

	//Generate an array of values spaced evenly
	std::vector<double> linspace(double start, double stop, int N);
	//Generate an array of values spaced evenly
	std::vector<double> arange(double start, double stop, double step);
	//Generate an array of data based on the mathematical function
	std::vector<double> genarray(double start, double end, int N, std::function<double(double)> mathfunction);
	std::vector<double> genarray(double start, double end, double step, std::function<double(double)> mathfunction);
	std::vector<double> genarray(const std::vector<double>& xdata, std::function<double(double)> mathfunction);

}