#pragma once

#include "Common.h"

#include <string>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#include <vector>
#include <array>
#include <cassert>
#include <functional>
#include <sstream>
#include <iomanip>

#include "Fitting_internal.h"

namespace numa {

    struct NonLinearVerbose {
        std::string name;
        std::string trsname;
        size_t niter;
        size_t nevalf;
        size_t nevaldf;
        int info;
        double chisq0;
        double chisq;
        double dof;
        std::vector<double> vars;
        std::vector<double> errs;
        std::string status;

        void clear() {
            name = "";
            trsname = "";
            niter = 0;
            nevalf = 0;
            nevaldf = 0;
            info = 0;
            chisq0 = 0;
            chisq = 0;
            dof = 0;
            vars.clear();
            errs.clear();
            status = "";
        }

        std::string to_string() {
            std::stringstream ss;
            ss << std::setprecision(16) << std::fixed
                << "summary from method" << name << "/" << trsname << std::endl
                << "number of iterations: " << niter << std::endl
                << "function evaluations : " << nevalf << std::endl
                << "Jacobian evaluations: " << nevaldf << std::endl
                << "reason for stopping: " << ((info == 1) ? "small step size" : "small gradient") << std::endl
                << "initial |f(x)| = " << chisq0 << std::endl
                << "final | f(x) | = " << chisq << std::endl;

            for (unsigned int i = 0; i < vars.size(); i++)
                ss << "variable " << i << " = " << vars[i] << std::endl;

            ss << "status = " << status << std::endl;

            return ss.str();
        }
    };


    class Fitter {
    public:
        Fitter();
        ~Fitter();

        /**
        * Performs a linear fit on dataset (x,y)
        * Model : Y = c0 + c1*X
        * 
        * @param x data
        * @param y data, must be same size as x
        * @return vector of adjusted parameters { c0, c1 }
        */
        std::vector<double> linear(const std::vector<double>& x, const std::vector<double>& y);
        /**
        * Performs a linear fit on dataset (x,y)
        * Model : Y = c0 + c1*X
        *
        * @param x vector of x data
        * @param y vector of y data, must be same size as x
        * @param err vector of error (standard deviation) on y values, must be same size as y
        * @return vector of adjusted parameters { c0, c1 }
        */
        std::vector<double> linear(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err);

        /**
        * Performs a linear fit without constant term on dataset (x,y)
        * Model : Y = c*X
        *
        * @param x data
        * @param y data, must be same size as x
        * @return the adjusted parameter c
        */
        double linear_mul(const std::vector<double>& x, const std::vector<double>& y);
        /**
        * Performs a linear fit without constant term on dataset (x,y)
        * Model : Y = c*X
        * 
        * @param x vector of x data
        * @param y vector of y data, must be same size as x
        * @param err vector of error (standard deviation) on y values, must be same size as y
        * @return the adjusted parameter c
        */
        double linear_mul(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err);


        /**
        * Performs a polynomial fit of degree (n-1)
        * Model : c0 + c1*X + c2*X^2 + ... + cn*X^(n-1)
        *
        * @param n is automatically determined by the number of initial parameters you fed into the function
        * @param x vector of x data
        * @param y vector of y data, must be same size as x
        * @param initial_params array of initial values of the parameters
        * @param ignore_params array of boolean, ignore_params[i] = true means we don't fit the i'th parameter
        * @return vector of adjusted parameters { c0, c1, c2, ..., cn}
        */
        template<auto n>
        std::vector<double> polynomial(const std::vector<double>& x, const std::vector<double>& y, const std::array<double, n>& initial_params, const std::array<bool, n>& ignore_params);
            
        /**
        * Performs a polynomial fit of degree (n-1)
        * Model : c0 + c1*X + c2*X^2 + ... + cn*X^(n-1)
        *
        * @param n is automatically determined by the number of initial parameters you fed into the function
        * @param x vector of x data
        * @param y vector of y data, must be same size as x
        * @param err vector of error (standard deviation) on y values, must be same size as y
        * @param initial_params array of initial values of the parameters
        * @param ignore_params array of boolean, ignore_params[i] = true means we don't fit the i'th parameter
        * @return vector of adjusted parameters { c0, c1, c2, ..., cn}
        */
        template<auto n>
        std::vector<double> polynomial(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err, const std::array<double, n>& initial_params, const std::array<bool, n>& ignore_params);

        /**
        * Performs a non-linear least-squares fit.
        *
        * @param f a function of type double (double x, double c1, double c2, ..., double cn)
        * where  c1, ..., cn are the coefficients to be fitted.
        * @param initial_params intial guess for the parameters. The size of the array must to
        * be equal to the number of coefficients to be fitted.
        * @param x the idependent data.
        * @param y the dependent data, must to have the same size as x.
        * @return std::vector<double> with the computed coefficients
        */
        template<typename Callable, auto n>
        std::vector<double> non_linear_fit(Callable f, const std::array<double, n>& initial_params, const std::vector<double>& x, const std::vector<double>& y);

        template<typename Callable, auto n>
        std::vector<double> non_linear_fit(Callable f, const std::array<double, n>& initial_params, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>* err);


        std::string GetOutput();

    private:
        /* Linear internal functions */

        //Transform array of standard deviation errors to array of weights for fitting functions of the gsl library
        std::vector<double> GetWeightsFromErrors(const std::vector<double>& err);

        //Contains output string formated for linear fitting
        std::string GetLinearOuputString(const std::vector<double>& vars, const double& cov00, const double& cov01, const double& cov11, const double& sumsq, const double& correlation);
        std::string GetLinearMulOuputString(const double& var, const double& cov11, const double& sumsq, const double& correlation);

        //Contains output string formated for polynomial fitting
        std::string GetPolynomialOuputString(const std::vector<double>& variables);

    private:
        /* Non Linear internal functions */

        /*
        Following code is a slightly modified version of this repo https://github.com/Eleobert/gsl-curve-fit
        His work is under GPL-3.0 licence
        */
        // For information about non-linear least-squares fit with gsl
        // see https://www.gnu.org/software/gsl/doc/html/nls.html


        std::vector<double> internal_solve_system(gsl_vector* initial_params, gsl_multifit_nlinear_fdf* fdf, gsl_multifit_nlinear_parameters* params, const std::vector<double>* err = nullptr);

        template<typename C1>
        std::vector<double> curve_fit_impl(func_f_type f, func_df_type df, func_fvv_type fvv, gsl_vector* initial_params, fit_data<C1>& fd);

    private:
        /* private helpers */

        //Reset members, used at the start of every fitting functions
        void ResetState();
    private:
        /* Members */
        std::string _output;

        NonLinearVerbose _nonLinearVerbose;
    };


    template<auto n>
    inline std::vector<double> Fitter::polynomial(const std::vector<double>& x, const std::vector<double>& y, const std::array<double, n>& initial_params, const std::array<bool, n>& ignore_params)
    {
        assert(x.size() == y.size());

        //variables
        int N;
        double chisq;
        gsl_matrix* X, * cov;
        gsl_vector* Y, * W, * C;

        unsigned int p = n;

        //init
        N = x.size();
        X = gsl_matrix_alloc(N, p);
        Y = gsl_vector_alloc(N);
        C = gsl_vector_alloc(p);
        cov = gsl_matrix_alloc(p, p);

        //fill matrices and vectors
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < p; k++) {
                double xx = 1.0;
                for (int j = 0; j < k; j++)
                    xx *= x[i];

                double f = 0.0;
                if (ignore_params[k] == false)
                    f = xx;
                else
                    f = 0.0; // xx * initial_params[k]; Don't know what to put in here 
                gsl_matrix_set(X, i, k, f);
            }

            gsl_vector_set(Y, i, y[i]);
        }

        //run fitting algorithm
        gsl_multifit_linear_workspace* work
            = gsl_multifit_linear_alloc(N, p);
        int r = gsl_multifit_linear(X, Y, C, cov, &chisq, work);
        gsl_multifit_linear_free(work);

        std::vector<double> variables(p);

        //extract params
        for (unsigned int i = 0; i < p; i++) {
            if (ignore_params[i] == false)
                variables[i] = gsl_vector_get(C, i);
            else
                variables[i] = initial_params[i];
        }

        _output = GetPolynomialOuputString(variables);

        //clean
        gsl_matrix_free(X);
        gsl_vector_free(Y);
        gsl_vector_free(C);
        gsl_matrix_free(cov);

        return variables;
    }

    template<auto n>
    inline std::vector<double> Fitter::polynomial(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err, const std::array<double, n>& initial_params, const std::array<bool, n>& ignore_params)
    {
        assert(x.size() == y.size());
        assert(y.size() == err.size());

        //variables
        int N;
        double chisq;
        gsl_matrix* X, * cov;
        gsl_vector* Y, * W, * C;

        unsigned int p = n;

        //init
        N = x.size();
        X = gsl_matrix_alloc(N, p);
        Y = gsl_vector_alloc(N);
        W = gsl_vector_alloc(N);
        C = gsl_vector_alloc(p);
        cov = gsl_matrix_alloc(p, p);

        //fill matrices and vectors
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < p; k++) {
                double xx = 1.0;
                for (int j = 0; j < k; j++)
                    xx *= x[i];

                double f = 0.0;
                if (ignore_params[k] == false)
                    f = xx;
                else
                    f = 0.0; // xx * initial_params[k]; Don't know what to put in here 
                gsl_matrix_set(X, i, k, f);
            }

            gsl_vector_set(Y, i, y[i]);

            //Convert standard deviation errors to weights
            if (err[i] == 0)
                gsl_vector_set(W, i, 1.0);
            else
                gsl_vector_set(W, i, 1.0 / (err[i] * err[i]));
        }

        //run fitting algorithm
        gsl_multifit_linear_workspace* work
            = gsl_multifit_linear_alloc(N, p);
        int r = gsl_multifit_wlinear(X, W, Y, C, cov, &chisq, work);
        gsl_multifit_linear_free(work);

        std::vector<double> variables(p);

        //extract params
        for (unsigned int i = 0; i < p; i++) {
            if (ignore_params[i] == false)
                variables[i] = gsl_vector_get(C, i);
            else
                variables[i] = initial_params[i];
        }

        _output = GetPolynomialOuputString(variables);

        /*printf("[ %+.5e, %+.5e, %+.5e  \n",
            COV(0, 0), COV(0, 1), COV(0, 2));
        printf("  %+.5e, %+.5e, %+.5e  \n",
            COV(1, 0), COV(1, 1), COV(1, 2));
        printf("  %+.5e, %+.5e, %+.5e ]\n",
            COV(2, 0), COV(2, 1), COV(2, 2));
        printf("# chisq = %g\n", chisq);*/

        //clean
        gsl_matrix_free(X);
        gsl_vector_free(Y);
        gsl_vector_free(W);
        gsl_vector_free(C);
        gsl_matrix_free(cov);

        return variables;
    }


    template<typename C1>
    std::vector<double> Fitter::curve_fit_impl(func_f_type f, func_df_type df, func_fvv_type fvv, gsl_vector* initial_params, fit_data<C1>& fd)
    {
        assert(fd.t.size() == fd.y.size());

        auto fdf = gsl_multifit_nlinear_fdf();
        auto fdf_params = gsl_multifit_nlinear_default_parameters();

        fdf.f = f;
        fdf.df = df;
        fdf.fvv = fvv;
        fdf.n = fd.t.size();
        fdf.p = initial_params->size;
        fdf.params = &fd;

        // "This selects the Levenberg-Marquardt algorithm with geodesic acceleration."
        fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;
        return internal_solve_system(initial_params, &fdf, &fdf_params, fd.err);
    }


    template<typename Callable, auto n>
    inline std::vector<double> Fitter::non_linear_fit(Callable f, const std::array<double, n>& initial_params, const std::vector<double>& x, const std::vector<double>& y)
    {
        // We can't pass lambdas without convert to std::function.
        //constexpr auto n = decltype(n_params(std::function(f)))::n_args - 1;
        //assert(initial_params.size() == n);

        auto params = internal_make_gsl_vector_ptr(initial_params);
        auto fd = fit_data<Callable>{ x, y, nullptr, f };
        return Fitter::curve_fit_impl(internal_f<decltype(fd), n>, nullptr, nullptr, params, fd);
    }

    template<typename Callable, auto n>
    inline std::vector<double> Fitter::non_linear_fit(Callable f, const std::array<double, n>& initial_params, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>* err)
    {
        auto params = internal_make_gsl_vector_ptr(initial_params);
        auto fd = fit_data<Callable>{ x, y, err, f };
        return Fitter::curve_fit_impl(internal_f<decltype(fd), n>, nullptr, nullptr, params, fd);
    }

}


