cmake_minimum_required (VERSION 3.16.3)

project ("numa") 

#Find the gsl library
message(STATUS "[GSL] Finding package...")
SET(GSL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gsl-2.5.0")
find_package(GSL REQUIRED)
message(STATUS "[GSL] Done.")

#Create numa library
add_library(numa)
add_subdirectory(src/Numa) #add sources files
target_include_directories(numa PUBLIC src) #allows #include "Numa/xxx.h"
target_link_libraries(numa PUBLIC GSL::gsl GSL::gslcblas) #link gsl to numa

#Build tests
add_subdirectory(test)