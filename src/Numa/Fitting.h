#pragma once

#include "Common.h"

#include <string>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace numa {
    namespace fit {
        
        //Model : y(x) = c0 + x * c1
        void linear(const std::vector<double>& x, const std::vector<double>& y, double* c0, double* c1, std::string& result);

        //Model : y(x) = c0 + x * c1 + x^2 * c2
        void polynomial(const std::vector<double>& x, const std::vector<double>& y, double& c0, double& c1, double& c2, std::string& result);

        //Model : y(t) =  A * exp(-lambda * t) + b
        void exponential(const std::vector<double>& x, const std::vector<double>& y, double& b, double& lambda, double& A);

        //Model : y(t) = A * sin(omega * t + phi)
        void sinus(const std::vector<double>& x, const std::vector<double>& y, double& phi, double& omega, double& A);
    }
}