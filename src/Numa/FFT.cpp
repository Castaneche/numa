#include "FFT.h"

#include <string>
#include <cmath>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>


namespace numa {
	namespace fft {

		Data spectrum(const std::vector<double>& xdata, const std::vector<double>& ydata, Window windowFunction)
		{
			Data data;
			data.y = ydata;

			double step = std::abs(xdata[0] - xdata[1]); //Temporary : will be replaced by  _curve->GetStep();
			int n = xdata.size();
			double nby2 = std::floor(n / 2.0); // n/2
			double fs = 1.0 / step; //signal frequency
			double df = fs / double(n); //frequency step

			//Generate spectrum x axis
			data.x = arange(0.0, nby2, df);			

			//Apply window function
			if (windowFunction == Window::Hamming) { // Hamming
				applyHamming(data.y);
			}
			else if (windowFunction == Window::Hann) { //Hann
				applyHann(data.y);
			}
			else if (windowFunction == Window::Blackmann) { //Blackmann
				applyBlackmann(data.y);
			}

			gsl_fft_real_wavetable* real;
			gsl_fft_halfcomplex_wavetable* hc;
			gsl_fft_real_workspace* work;

			work = gsl_fft_real_workspace_alloc(n);
			real = gsl_fft_real_wavetable_alloc(n);
			hc = gsl_fft_halfcomplex_wavetable_alloc(n);

			//Compute fft
			gsl_fft_real_transform(&data.y[0], 1, n, real, work);

			//The real transform outputs halfcomplex
			//The real part is extracted using i*2 as index
			for (int i = 0; i < nby2; i++) {
				data.y[i] = std::abs(data.y[i * 2]) / double(n);
			}
			//Now the array is half filled with real coefficient, we don't need the other half of the array. 
			data.y.erase(data.y.begin() + nby2, data.y.end());

			gsl_fft_real_wavetable_free(real);
			gsl_fft_halfcomplex_wavetable_free(hc);
			gsl_fft_real_workspace_free(work);

			return data;
		}



		void applyHann(std::vector<double>& arr)
		{
			double N = arr.size();
			for (unsigned int i = 0; i < N; i++) {
				arr[i] *= 0.5 * (1.0 - std::cos((2 * M_PI * i) / (N - 1)));
			}
		}

		void applyHamming(std::vector<double>& arr)
		{
			double N = arr.size();
			for (unsigned int i = 0; i < N; i++) {
				arr[i] *= 0.53836 - 0.46164 * std::cos((2 * M_PI * i) / (N - 1));
			}
		}

		void applyBlackmann(std::vector<double>& arr)
		{
			double N = arr.size();
			for (unsigned int i = 0; i < N; i++) {
				arr[i] *= 0.42659 - 0.49656 * std::cos((2 * M_PI * i) / (N - 1)) + 0.076849 * std::cos((4 * M_PI * i) / (N - 1));
			}
		}
	}
}

