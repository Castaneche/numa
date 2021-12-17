     
target_sources(
    numa PUBLIC
    Numa.h
    Fitting.h
    Derivative.h
    Integral.h
    FFT.h

    PRIVATE
    Fitting.cpp
    Derivative.cpp
    Integral.cpp
    FFT.cpp
    )
