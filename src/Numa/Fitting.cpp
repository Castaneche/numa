#include "Fitting.h"

#include <assert.h>

#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#include <iostream>

namespace numa {

	Fitter::Fitter() {

	}
	Fitter::~Fitter() {

	}

	std::vector<double> Fitter::GetWeightsFromErrors(const std::vector<double>& err) {
		std::vector<double> weights;
		for (double e : err) {
			if (e != 0)
				weights.push_back(1.0 / (e * e));
			else
				weights.push_back(1.0);
		}
		return weights;
	}

	std::string Fitter::GetLinearOuputString(const double& cov00, const double& cov01, const double& cov11, const double& sumsq, const double& correlation) {
		return 
			" cov00 : " + std::to_string(cov00) + "\n"
			+ " cov01 : " + std::to_string(cov01) + "\n"
			+ " cov11 : " + std::to_string(cov11) + "\n"
			+ " sumsq : " + std::to_string(sumsq) + "\n"
			+ " correlation : " + std::to_string(correlation) + "\n";
	}
	std::string Fitter::GetLinearMulOuputString(const double& cov11, const double& sumsq, const double& correlation) {
		return 
			" cov11 : " + std::to_string(cov11) + "\n"
			+ " sumsq : " + std::to_string(sumsq) + "\n"
			+ " correlation : " + std::to_string(correlation) + "\n";
	}

	std::vector<double> Fitter::linear(const std::vector<double>& x, const std::vector<double>& y)
	{
		std::vector<double> variables(2);

		double cov00, cov01, cov11, sumsq;
		int r = gsl_fit_linear(&x[0], 1, &y[0], 1, x.size(), &variables[0], &variables[1], &cov00, &cov01, &cov11, &sumsq);
		double correlation = gsl_stats_correlation(&x[0], 1, &y[0], 1, x.size());
		_output
			= " Y = " + std::to_string(variables[0]) + " + " + std::to_string(variables[1]) + " X\n"
			+ GetLinearOuputString(cov00, cov01, cov11, sumsq, correlation);

		return variables;
	}

	std::vector<double> Fitter::linear(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err)
	{
		std::vector<double> variables(2);

		std::vector<double> weights = GetWeightsFromErrors(err);

		double cov00, cov01, cov11, sumsq;
		int r = gsl_fit_wlinear(&x[0], 1, &weights[0], 1, &y[0], 1, x.size(), &variables[0], &variables[1], &cov00, &cov01, &cov11, &sumsq);
		double correlation = gsl_stats_correlation(&x[0], 1, &y[0], 1, x.size());
		_output
			= " Y = " + std::to_string(variables[0]) + " + " + std::to_string(variables[1]) + " X\n"
			+ GetLinearOuputString(cov00, cov01, cov11, sumsq, correlation);

		return variables;
	}

	double Fitter::linear_mul(const std::vector<double>& x, const std::vector<double>& y)
	{
		double var = 0.0;

		double cov00, cov01, cov11, sumsq;
		int r = gsl_fit_mul(&x[0], 1, &y[0], 1, x.size(), &var, &cov11, &sumsq);
		double correlation = gsl_stats_correlation(&x[0], 1, &y[0], 1, x.size());
		_output
			= " Y = " + std::to_string(var) + " X\n"
			+ GetLinearMulOuputString(cov11, sumsq, correlation);

		return var;
	}
	double Fitter::linear_mul(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& err)
	{
		double var = 0.0;

		std::vector<double> weights = GetWeightsFromErrors(err);

		double cov00, cov01, cov11, sumsq;
		int r = gsl_fit_wmul(&x[0], 1, &weights[0], 1, &y[0], 1, x.size(), &var, &cov11, &sumsq);
		double correlation = gsl_stats_correlation(&x[0], 1, &y[0], 1, x.size());
		_output
			= " Y = " + std::to_string(var) + " X\n"
			+ " cov11 : " + std::to_string(cov11) + "\n"
			+ " sumsq : " + std::to_string(sumsq) + "\n"
			+ " correlation : " + std::to_string(correlation) + "\n";

		return var;
	}

	std::string Fitter::GetPolynomialOuputString(const std::vector<double>& variables) {
		std::string out = "Y = " + std::to_string(variables[0]) + " + " + std::to_string(variables[1]) + " X";
		for (unsigned int i = 2; i < variables.size(); i++)
			out += " + " + std::to_string(variables[i]) + " X^" + std::to_string(i);
		return out;
	}


	

	std::vector<double> Fitter::internal_solve_system(gsl_vector* initial_params, gsl_multifit_nlinear_fdf* fdf,
		gsl_multifit_nlinear_parameters* params, const std::vector<double>* err)
	{
		// This specifies a trust region method
		const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
		const size_t max_iter = 200;
		const double xtol = 1.0e-8;
		const double gtol = 1.0e-8;
		const double ftol = 1.0e-8;
		double chisq0, chisq;
		gsl_vector* f;

		auto* work = gsl_multifit_nlinear_alloc(T, params, fdf->n, fdf->p);
		int info;

		// initialize solver
		if (err == nullptr){
			gsl_multifit_nlinear_init(initial_params, fdf, work);
		}
		else {
			std::vector<double> weights = GetWeightsFromErrors(*err);
			gsl_vector_view wts = gsl_vector_view_array(&weights[0], weights.size());
			gsl_multifit_nlinear_winit(initial_params, &wts.vector, fdf, work);
		}

		/* compute initial cost function */
		f = gsl_multifit_nlinear_residual(work);
		gsl_blas_ddot(f, f, &chisq0);

		//iterate until convergence
		int status = gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, nullptr, nullptr, &info, work);

		// result will be stored here
		gsl_vector* y = gsl_multifit_nlinear_position(work);
		auto result = std::vector<double>(initial_params->size);

		for (int i = 0; i < result.size(); i++)
		{
			result[i] = gsl_vector_get(y, i);
		}

		auto niter = gsl_multifit_nlinear_niter(work);
		auto nfev = fdf->nevalf;
		auto njev = fdf->nevaldf;
		auto naev = fdf->nevalfvv;

		/* covariance matrix*/
		gsl_matrix* J = gsl_multifit_nlinear_jac(work);
		gsl_matrix* covar = gsl_matrix_alloc(result.size(), result.size());
		gsl_multifit_nlinear_covar(J, 0.0, covar);

		/* compute final cost */
		gsl_blas_ddot(f, f, &chisq);

#define FIT(i) gsl_vector_get(y, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

		_nonLinearVerbose.clear();

		_nonLinearVerbose.name = gsl_multifit_nlinear_name(work);
		_nonLinearVerbose.trsname = gsl_multifit_nlinear_trs_name(work);
		_nonLinearVerbose.niter = gsl_multifit_nlinear_niter(work);
		_nonLinearVerbose.nevalf = fdf->nevalf;
		_nonLinearVerbose.nevaldf = fdf->nevaldf;
		_nonLinearVerbose.info = info;
		_nonLinearVerbose.chisq0 = chisq0;
		_nonLinearVerbose.chisq = chisq;

		//double dof = n - result.size();
		//verbose->dof = dof;

		//double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

		for (unsigned int k = 0; k < result.size(); k++) {
			_nonLinearVerbose.vars.push_back(result[k]);
			//verbose->errs.push_back(c * ERR(k));
		}

		_nonLinearVerbose.status = gsl_strerror(status);
		_output = _nonLinearVerbose.to_string();

		// nfev - number of function evaluations
		// njev - number of Jacobian evaluations
		// naev - number of f_vv evaluations
		//logger::debug("curve fitted after ", niter, " iterations {nfev = ", nfev, "} {njev = ", njev, "} {naev = ", naev, "}");

		gsl_multifit_nlinear_free(work);
		gsl_vector_free(initial_params);
		return result;
	}



	std::string Fitter::GetOutput() {
		return _output;
	}


}//numa namespace

