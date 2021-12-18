#pragma once

#include "Common.h"

#include <vector>
#include <memory>

namespace numa {
    namespace fft {

		enum class Window { None, Hamming, Hann, Blackmann };

		//Perform fft on data. 
		//WARNING : input data will be modified
		Data spectrum(const std::vector<double>& xdata, const std::vector<double>& ydata, Window windowFunction = Window::None);

		//Window functions 
		// https://matthieu-brucher.developpez.com/tutoriels/algo/fft/#LII-B
		// https://en.wikipedia.org/wiki/Window_function#Blackman_window
		void applyHann(std::vector<double>& arr);
		void applyHamming(std::vector<double>& arr);
		void applyBlackmann(std::vector<double>& arr);
    }
}