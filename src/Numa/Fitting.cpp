#include "Fitting.h"

#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

namespace numa {

	namespace fit {

		static struct data {
			size_t n;
			double* t;
			double* y;
		};

		static int exp_f(const gsl_vector* x, void* data, gsl_vector* f)
		{
			size_t n = ((struct data*)data)->n;
			double* t = ((struct data*)data)->t;
			double* y = ((struct data*)data)->y;

			double A = gsl_vector_get(x, 0);
			double lambda = gsl_vector_get(x, 1);
			double b = gsl_vector_get(x, 2);

			size_t i;

			for (i = 0; i < n; i++)
			{
				/* Model Yi = A * exp(-lambda * t_i) + b */
				double Yi = A * exp(-lambda * t[i]) + b;
				gsl_vector_set(f, i, Yi - y[i]);
			}

			return GSL_SUCCESS;
		}

		static int	exp_df(const gsl_vector* x, void* data, gsl_matrix* J)
		{
			size_t n = ((struct data*)data)->n;
			double* t = ((struct data*)data)->t;

			double A = gsl_vector_get(x, 0);
			double lambda = gsl_vector_get(x, 1);

			size_t i;

			for (i = 0; i < n; i++)
			{
				/* Jacobian matrix J(i,j) = dfi / dxj, */
				/* where fi = (Yi - yi)/sigma[i],      */
				/*       Yi = A * exp(-lambda * t_i) + b  */
				/* and the xj are the parameters (A,lambda,b) */
				double e = exp(-lambda * t[i]);
				gsl_matrix_set(J, i, 0, e);
				gsl_matrix_set(J, i, 1, -t[i] * A * e);
				gsl_matrix_set(J, i, 2, 1.0);
			}

			return GSL_SUCCESS;
		}


		static int sin_f(const gsl_vector* x, void* data, gsl_vector* f)
		{
			size_t n = ((struct data*)data)->n;
			double* t = ((struct data*)data)->t;
			double* y = ((struct data*)data)->y;

			double A = gsl_vector_get(x, 0);
			double omega = gsl_vector_get(x, 1);
			double phi = gsl_vector_get(x, 2);

			size_t i;

			for (i = 0; i < n; i++)
			{
				/* Model Yi = A * sin(omega * t_i + phi) */
				double Yi = A * sin(omega * t[i] + phi);
				gsl_vector_set(f, i, Yi - y[i]);
			}

			return GSL_SUCCESS;
		}

		static int sin_df(const gsl_vector* x, void* data, gsl_matrix* J)
		{
			size_t n = ((struct data*)data)->n;
			double* t = ((struct data*)data)->t;

			double A = gsl_vector_get(x, 0);
			double omega = gsl_vector_get(x, 1);
			double phi = gsl_vector_get(x, 2);

			size_t i;

			for (i = 0; i < n; i++)
			{
				/* Jacobian matrix J(i,j) = dfi / dxj, */
				/* where fi = (Yi - yi)/sigma[i],      */
				/*       Yi = A * sin(omega * t_i + phi)  */
				/* and the xj are the parameters (A,omega,phi) */
				gsl_matrix_set(J, i, 0, sin(omega * t[i] + phi));
				gsl_matrix_set(J, i, 1, A * t[i] * cos(omega * t[i] + phi));
				gsl_matrix_set(J, i, 2, A * cos(omega * t[i] + phi));
			}

			return GSL_SUCCESS;
		}


		// ==============================================================================

		static void callback(const size_t iter, void* params, const gsl_multifit_nlinear_workspace* w)
		{
			gsl_vector* f = gsl_multifit_nlinear_residual(w);
			gsl_vector* x = gsl_multifit_nlinear_position(w);
			double rcond;

			/* compute reciprocal condition number of J(x) */
			gsl_multifit_nlinear_rcond(&rcond, w);

			fprintf(stderr, "iter %2zu: A = %.4f, lambda = %.4f, b = %.4f, cond(J) = %8.4f, |f(x)| = %.4f\n",
				iter,
				gsl_vector_get(x, 0),
				gsl_vector_get(x, 1),
				gsl_vector_get(x, 2),
				1.0 / rcond,
				gsl_blas_dnrm2(f));
		}


		// ==============================================================================


		/*int linear(const double* x, const size_t xstride, const double* y, const size_t ystride, const size_t n, double* c0, double* c1, double* cov00, double* cov01, double* cov11, double* sumsq, std::string& result)
		{
			int r = gsl_fit_linear(x, xstride, y, ystride, n, c0, c1, cov00, cov01, cov11, sumsq);
			double correlation = gsl_stats_correlation(x, xstride, y, ystride, n);
			result
				= " Y = " + std::to_string(*c0) + " + " + std::to_string(*c1) + " X\n"
				+ " cov00 : " + std::to_string(*cov00) + "\n"
				+ " cov01 : " + std::to_string(*cov01) + "\n"
				+ " cov11 : " + std::to_string(*cov11) + "\n"
				+ " sumsq : " + std::to_string(*sumsq) + "\n"
				+ " correlation : " + std::to_string(correlation) + "\n";

			return r;
		}*/

		int linear(std::vector<double>& x, std::vector<double>& y, double* c0, double* c1, std::string& result)
		{
			double cov00, cov01, cov11, sumsq;
			int r = gsl_fit_linear(&x[0], 1, &y[0], 1, x.size(), c0, c1, &cov00, &cov01, &cov11, &sumsq);
			double correlation = gsl_stats_correlation(&x[0], 1, &y[0], 1, x.size());
			result
				= " Y = " + std::to_string(*c0) + " + " + std::to_string(*c1) + " X\n"
				+ " cov00 : " + std::to_string(cov00) + "\n"
				+ " cov01 : " + std::to_string(cov01) + "\n"
				+ " cov11 : " + std::to_string(cov11) + "\n"
				+ " sumsq : " + std::to_string(sumsq) + "\n"
				+ " correlation : " + std::to_string(correlation) + "\n";

			return r;
		}

		int polynomial(std::vector<double>& x, std::vector<double>& y, double& c0, double& c1, double& c2, std::string& result)
		{
			//variables
			int n;
			double chisq;
			gsl_matrix* X, * cov;
			gsl_vector* Y, * W, * C;

			//init
			n = x.size();
			X = gsl_matrix_alloc(n, 3);
			Y = gsl_vector_alloc(n);
			C = gsl_vector_alloc(3);
			cov = gsl_matrix_alloc(3, 3);

			//fill matrices and vectors
			for (int i = 0; i < n; i++)
			{
				gsl_matrix_set(X, i, 0, 1.0);
				gsl_matrix_set(X, i, 1, x[i]);
				gsl_matrix_set(X, i, 2, x[i] * x[i]);

				gsl_vector_set(Y, i, y[i]);
				//gsl_vector_set(w, i, 1.0 / (ei * ei));
			}

			//run fitting algorithm
			gsl_multifit_linear_workspace* work
				= gsl_multifit_linear_alloc(n, 3);
			int r = gsl_multifit_linear(X, Y, C, cov, &chisq, work);
			gsl_multifit_linear_free(work);

			//extract params
			c0 = gsl_vector_get(C, 0);
			c1 = gsl_vector_get(C, 1);
			c2 = gsl_vector_get(C, 2);

			result = " Y = " + std::to_string(c0) + " + " + std::to_string(c1) + " X + " + std::to_string(c2) + " X^2\n";


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
			//gsl_vector_free(w);
			gsl_vector_free(C);
			gsl_matrix_free(cov);

			return r;
		}

		int exponential(std::vector<double>& x, std::vector<double>& y, double& b, double& lambda, double& A)
		{
			const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
			gsl_multifit_nlinear_workspace* w;
			gsl_multifit_nlinear_fdf fdf;
			gsl_multifit_nlinear_parameters fdf_params =
				gsl_multifit_nlinear_default_parameters();
			const size_t n = y.size();
			const size_t p = 3;

			gsl_vector* f;
			gsl_matrix* J;
			gsl_matrix* covar = gsl_matrix_alloc(p, p);
			std::vector<double> weights;
			weights.resize(n);
			struct data d = { n, x.data(), y.data() };
			double x_init[3] = { 1.0, 1.0, 0.0 }; /* starting values */
			gsl_vector_view xx = gsl_vector_view_array(x_init, p);
			gsl_vector_view wts = gsl_vector_view_array(weights.data(), n);
			gsl_rng* r;
			double chisq, chisq0;
			int status, info;
			size_t i;

			const double xtol = 1e-8;
			const double gtol = 1e-8;
			const double ftol = 0.0;

			gsl_rng_env_setup();
			r = gsl_rng_alloc(gsl_rng_default);

			/* define the function to be minimized */
			fdf.f = exp_f;
			fdf.df = exp_df;   /* set to NULL for finite-difference Jacobian */
			fdf.fvv = NULL;     /* not using geodesic acceleration */
			fdf.n = n;
			fdf.p = p;
			fdf.params = &d;

			/* this is the data to be fitted */
			for (i = 0; i < n; i++)
			{
				double si = 0.1 * y[i];
				weights[i] = 1.0 / (si * si);
			};

			/* allocate workspace with default parameters */
			w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

			/* initialize solver with starting point and weights */
			gsl_multifit_nlinear_winit(&xx.vector, &wts.vector, &fdf, w);

			/* compute initial cost function */
			f = gsl_multifit_nlinear_residual(w);
			gsl_blas_ddot(f, f, &chisq0);

			/* solve the system with a maximum of 100 iterations */
			status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol,
				callback, NULL, &info, w);

			/* compute covariance of best fit parameters */
			J = gsl_multifit_nlinear_jac(w);
			gsl_multifit_nlinear_covar(J, 0.0, covar);

			/* compute final cost */
			gsl_blas_ddot(f, f, &chisq);

#define FIT(i) gsl_vector_get(w->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

			fprintf(stderr, "summary from method '%s/%s'\n",
				gsl_multifit_nlinear_name(w),
				gsl_multifit_nlinear_trs_name(w));
			fprintf(stderr, "number of iterations: %zu\n",
				gsl_multifit_nlinear_niter(w));
			fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
			fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
			fprintf(stderr, "reason for stopping: %s\n",
				(info == 1) ? "small step size" : "small gradient");
			fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
			fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));

			{
				double dof = n - p;
				double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

				fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

				fprintf(stderr, "A      = %.5f +/- %.5f\n", FIT(0), c * ERR(0));
				fprintf(stderr, "lambda = %.5f +/- %.5f\n", FIT(1), c * ERR(1));
				fprintf(stderr, "b      = %.5f +/- %.5f\n", FIT(2), c * ERR(2));
			}

			fprintf(stderr, "status = %s\n", gsl_strerror(status));

			A = FIT(0);
			lambda = FIT(1);
			b = FIT(2);

			gsl_multifit_nlinear_free(w);
			gsl_matrix_free(covar);
			gsl_rng_free(r);

			return 0;
		}


		int sinus(std::vector<double>& x, std::vector<double>& y, double& phi, double& omega, double& A)
		{
			const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
			gsl_multifit_nlinear_workspace* w;
			gsl_multifit_nlinear_fdf fdf;
			gsl_multifit_nlinear_parameters fdf_params =
				gsl_multifit_nlinear_default_parameters();
			const size_t n = y.size();
			const size_t p = 3;

			gsl_vector* f;
			gsl_matrix* J;
			gsl_matrix* covar = gsl_matrix_alloc(p, p);
			std::vector<double> weights;
			weights.resize(n);
			struct data d = { n, x.data(), y.data() };
			double x_init[3] = { 1.0, 2 * M_PI * 10, 0.0 }; /* starting values */
			gsl_vector_view xx = gsl_vector_view_array(x_init, p);
			gsl_vector_view wts = gsl_vector_view_array(weights.data(), n);
			gsl_rng* r;
			double chisq, chisq0;
			int status, info;
			size_t i;

			const double xtol = 1e-8;
			const double gtol = 1e-8;
			const double ftol = 0.0;

			gsl_rng_env_setup();
			r = gsl_rng_alloc(gsl_rng_default);

			/* define the function to be minimized */
			fdf.f = sin_f;
			fdf.df = sin_df;   /* set to NULL for finite-difference Jacobian */
			fdf.fvv = NULL;     /* not using geodesic acceleration */
			fdf.n = n;
			fdf.p = p;
			fdf.params = &d;

			/* this is the data to be fitted */
			for (i = 0; i < n; i++)
			{
				double si = 0.1 * y[i];
				weights[i] = 1.0 / (si * si);
			};

			/* allocate workspace with default parameters */
			w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

			/* initialize solver with starting point and weights */
			gsl_multifit_nlinear_winit(&xx.vector, &wts.vector, &fdf, w);

			/* compute initial cost function */
			f = gsl_multifit_nlinear_residual(w);
			gsl_blas_ddot(f, f, &chisq0);

			/* solve the system with a maximum of 100 iterations */
			status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol,
				callback, NULL, &info, w);

			/* compute covariance of best fit parameters */
			J = gsl_multifit_nlinear_jac(w);
			gsl_multifit_nlinear_covar(J, 0.0, covar);

			/* compute final cost */
			gsl_blas_ddot(f, f, &chisq);

#define FIT(i) gsl_vector_get(w->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

			fprintf(stderr, "summary from method '%s/%s'\n",
				gsl_multifit_nlinear_name(w),
				gsl_multifit_nlinear_trs_name(w));
			fprintf(stderr, "number of iterations: %zu\n",
				gsl_multifit_nlinear_niter(w));
			fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
			fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
			fprintf(stderr, "reason for stopping: %s\n",
				(info == 1) ? "small step size" : "small gradient");
			fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
			fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));

			{
				double dof = n - p;
				double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

				fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

				fprintf(stderr, "A      = %.5f +/- %.5f\n", FIT(0), c * ERR(0));
				fprintf(stderr, "omega = %.5f +/- %.5f\n", FIT(1), c * ERR(1));
				fprintf(stderr, "phi      = %.5f +/- %.5f\n", FIT(2), c * ERR(2));
			}

			fprintf(stderr, "status = %s\n", gsl_strerror(status));

			A = FIT(0);
			omega = FIT(1);
			phi = FIT(2);

			gsl_multifit_nlinear_free(w);
			gsl_matrix_free(covar);
			gsl_rng_free(r);

			return 0;
		}


} //fit namespace
}//numa namespace

