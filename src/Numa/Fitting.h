#pragma once

#include "Common.h"

#include <string>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>

#include <vector>
#include <array>
#include <cassert>
#include <functional>

namespace numa {
    namespace fit {

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
                std::string result = "";
                result += "summary from method" + name + "/" + trsname + '\n';
                result += "number of iterations: " + std::to_string(niter) + '\n';
                result += "function evaluations: " + std::to_string(nevalf) + '\n';
                result += "Jacobian evaluations: " + std::to_string(nevaldf) + '\n';
                result += "reason for stopping: " + (info == 1) ? "small step size" : "small gradient";
                result += '\n';
                result += "initial |f(x)| = " + std::to_string(chisq0) + '\n';
                result += "final | f(x) | = " + std::to_string(chisq) + '\n';
                
                for (unsigned int i = 0; i < vars.size(); i++)
                    result += "variable " + std::to_string(i) + " = " + std::to_string(vars[i]) + '\n';

                result += "status = " + status + '\n';

                return result;
            }
        };
        
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
        std::vector<double> exponential(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double> init_params = { 1.0, 1.0, 0.0}, NonLinearVerbose* verbose = nullptr);

        /*Model : y(t) = A * sin(omega * t + phi)
        * Parameter's order: { A, omega, phi }
        * Return: vector of the 3 adjusted parameters
        * */
        std::vector<double> sinus(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double> init_params = { 1.0, 1.0, 0.0 }, NonLinearVerbose* verbose = nullptr);





        /*
        Following code is a slightly modified version of this repo https://github.com/Eleobert/gsl-curve-fit
        His work is under GPL-3.0 licence
        */


        // For information about non-linear least-squares fit with gsl
        // see https://www.gnu.org/software/gsl/doc/html/nls.html

        template <class R, class... ARGS>
        struct function_ripper {
            static constexpr size_t n_args = sizeof...(ARGS);
        };

        /**
         * This function returns the number of parameters of a given function.This
         * overload is to be used specialy with lambdas.
         */
        template <class R, class... ARGS>
        auto constexpr n_params(std::function<R(ARGS...)>)
        {
            return function_ripper<R, ARGS...>();
        }

        /**
         * This function returns the number of parameters of a given function.
         */
        template <class R, class... ARGS>
        auto constexpr n_params(R(ARGS...))
        {
            return function_ripper<R, ARGS...>();
        }

        template <typename F, size_t... Is>
        auto gen_tuple_impl(F func, std::index_sequence<Is...>)
        {
            return std::make_tuple(func(Is)...);
        }

        template <size_t N, typename F>
        auto gen_tuple(F func)
        {
            return gen_tuple_impl(func, std::make_index_sequence<N>{});
        }

        template<typename C1>
        struct fit_data
        {
            const std::vector<double>& t;
            const std::vector<double>& y;
            // the actual function to be fitted
            C1 f;
        };


        template<typename FitData, int n_params>
        int internal_f(const gsl_vector* x, void* params, gsl_vector* f)
        {
            auto* d = static_cast<FitData*>(params);
            // Convert the parameter values from gsl_vector (in x) into std::tuple
            auto init_args = [x](int index)
            {
                return gsl_vector_get(x, index);
            };
            auto parameters = gen_tuple<n_params>(init_args);

            // Calculate the error for each...
            for (size_t i = 0; i < d->t.size(); ++i)
            {
                double ti = d->t[i];
                double yi = d->y[i];
                auto func = [ti, &d](auto ...xs)
                {
                    // call the actual function to be fitted
                    return d->f(ti, xs...);
                };
                auto y = std::apply(func, parameters);
                gsl_vector_set(f, i, yi - y);
            }
            return GSL_SUCCESS;
        }

        using func_f_type = int (*) (const gsl_vector*, void*, gsl_vector*);
        using func_df_type = int (*) (const gsl_vector*, void*, gsl_matrix*);
        using func_fvv_type = int (*) (const gsl_vector*, const gsl_vector*, void*, gsl_vector*);

        template<auto n>
        gsl_vector* internal_make_gsl_vector_ptr(const std::array<double, n>& vec) {
            auto* result = gsl_vector_alloc(n);
            int i = 0;
            for (const auto e : vec)
            {
                gsl_vector_set(result, i, e);
                i++;
            }
            return result;
        }

        auto internal_solve_system(gsl_vector* initial_params, gsl_multifit_nlinear_fdf* fdf,
            gsl_multifit_nlinear_parameters* params,
            NonLinearVerbose* verbose = nullptr)-> std::vector<double>;

        template<typename C1>
        std::vector<double> curve_fit_impl(func_f_type f, func_df_type df, func_fvv_type fvv, gsl_vector* initial_params, fit_data<C1>& fd, NonLinearVerbose* verbose = nullptr)
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
            return internal_solve_system(initial_params, &fdf, &fdf_params, verbose);
        }


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
        std::vector<double> curve_fit(Callable f, const std::array<double, n>& initial_params, const std::vector<double>& x, const std::vector<double>& y, NonLinearVerbose* verbose = nullptr)
        {
            // We can't pass lambdas without convert to std::function.
            //constexpr auto n = decltype(n_params(std::function(f)))::n_args - 1;
            //assert(initial_params.size() == n);

            auto params = internal_make_gsl_vector_ptr(initial_params);
            auto fd = fit_data<Callable>{ x, y, f };
            return curve_fit_impl(internal_f<decltype(fd), n>, nullptr, nullptr, params, fd, verbose);
        }

    }
}