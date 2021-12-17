#pragma once

#include <vector>

namespace numa {
    namespace integral {
        //integration Simpson's 1/3 rule
        double simpson(const std::vector<double>& y, const double& step);
    }
}