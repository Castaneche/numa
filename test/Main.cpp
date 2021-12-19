#include <iostream>

#include "Numa/Numa.h"

int main(void)
{   
    auto xexp = numa::linspace(0, 1.0, 100);
    auto yexp = numa::genarray(xexp, [](double i) {
        return 1.0 * exp(- 0.2 * i);
        });

    auto result = numa::fit::exponential(xexp, yexp);

    std::cout << "1 : " << result[0] << "\nlambda : " << result[1] << "\nb : " << result[2] << std::endl << std::endl << std::endl;

    auto xsin = numa::linspace(0, 1.0, 10000);
    auto ysin = numa::genarray(xsin, [](double i) {
        return 20.0 * sin(7.54 * i + 1.24);
        });

    result = numa::fit::sinus(xsin, ysin, { 1, 10, 0});

    std::cout << "A : " << result[0] << "\nomega : " << result[1] << "\nphi : " << result[2] << std::endl << std::endl << std::endl;


    return 0;
}
