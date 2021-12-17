#pragma once

#include <vector>

namespace numa {
    namespace derivative {
        //3 points derivatives
        double threepts(const std::vector<double>& x, const std::vector<double>& y, const unsigned int& xindex);
        std::vector<double> threepts(const std::vector<double>& x, const std::vector<double>& y);

        //5 points derivatives
        double fivepts(const std::vector<double>& x, const std::vector<double>& y, const unsigned int& xindex);
        std::vector<double> fivepts(const std::vector<double>& x, const std::vector<double>& y);
    }
}