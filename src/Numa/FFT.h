#pragma once

#include <vector>
#include <memory>

namespace numa {
    namespace fft {

		//Perform fft on data. 
		//WARNING : input data will be modified
		void spectrum(std::vector<double>& xdata, std::vector<double>& ydata, Window windowFunction = Window::None);

		//Window functions 
		// https://matthieu-brucher.developpez.com/tutoriels/algo/fft/#LII-B
		// https://en.wikipedia.org/wiki/Window_function#Blackman_window

		enum class Window { None, Hamming, Hann, Blackmann };

		void applyHann(std::vector<double>& arr);
		void applyHamming(std::vector<double>& arr);
		void applyBlackmann(std::vector<double>& arr);
    }
}