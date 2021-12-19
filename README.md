# Numa - C++ wrapper for numerical analysis functions of the Gnu Scientific Library
It consists of a set of c++ source files that provide an easier way of using some of the numerical analysis functions of the gsl library.

## Overview
List of the features :

- **Common** : utility functions such as linspace, arange ...
- **Fitting** using predefined models : linear, polynomial, exponential, sinus.
- **Discrete Derivative** : 3 or 5 points.
- **Discrete Integral** : simpson.
- **Fast Fourier Transform** : compute the frequency spectrum of input data.

## Build
The library is meant to be build using CMake either by using :

- a subfolder and calling ``` add_subdirectory()```
- or FetchContent  
  ```
  FetchContent_Declare(numa GIT_REPOSITORY "https://github.com/castaneche/numa")
  FetchContent_MakeAvailable(numa) 
  ``` 
