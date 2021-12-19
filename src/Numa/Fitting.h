#pragma once

#include "Common.h"

#include <string>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace numa {
    namespace fit {
        
        //Model : y(x) = c0 * x + c1
        //Return vector of adjusted parameters {c0, c1}
        std::vector<double> linear(const std::vector<double>& x, const std::vector<double>& y, std::string& result);

        //Model : y(x) = c0 * x^2 + c1 * x + c3
        //Return vector of adjusted parameters {c0, c1, c2}
        std::vector<double> polynomial(const std::vector<double>& x, const std::vector<double>& y, std::string& result);

        /*Model: y(t) = A * exp(-lambda * t) + b
        * Parameter's order: { A, lambda, b }
        * Return: vector of the 3 adjusted parameters
        * */
        std::vector<double> exponential(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double> init_params = { 1.0, 1.0, 0.0});

        /*Model : y(t) = A * sin(omega * t + phi)
        * Parameter's order: { A, omega, phi }
        * Return: vector of the 3 adjusted parameters
        * */
        std::vector<double> sinus(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double> init_params = { 1.0, 1.0, 0.0 });
    }
}