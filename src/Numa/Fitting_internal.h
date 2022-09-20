#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <functional>
#include <vector>

using func_f_type = int (*) (const gsl_vector*, void*, gsl_vector*);
using func_df_type = int (*) (const gsl_vector*, void*, gsl_matrix*);
using func_fvv_type = int (*) (const gsl_vector*, const gsl_vector*, void*, gsl_vector*);

template <class R, class... ARGS>
struct function_ripper {
    static constexpr size_t n_args = sizeof...(ARGS);
};
template<typename C1>
struct fit_data
{
    const std::vector<double>& t;
    const std::vector<double>& y;
    const std::vector<double>* err;
    // the actual function to be fitted
    C1 f;
};

template <class R, class... ARGS>
inline auto constexpr n_params(std::function<R(ARGS...)>)
{
    return function_ripper<R, ARGS...>();
}

template <class R, class... ARGS>
inline auto constexpr n_params(R(ARGS...))
{
    return function_ripper<R, ARGS...>();
}

template <typename F, size_t... Is>
inline auto gen_tuple_impl(F func, std::index_sequence<Is...>)
{
    return std::make_tuple(func(Is)...);
}

template <size_t N, typename F>
inline auto gen_tuple(F func)
{
    return gen_tuple_impl(func, std::make_index_sequence<N>{});
}

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

template<auto n>
inline gsl_vector* internal_make_gsl_vector_ptr(const std::array<double, n>& vec) {
    auto* result = gsl_vector_alloc(n);
    int i = 0;
    for (const auto e : vec)
    {
        gsl_vector_set(result, i, e);
        i++;
    }
    return result;
}