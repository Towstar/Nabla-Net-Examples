cmake_minimum_required(VERSION 3.21)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(NablaNetExamples LANGUAGES CXX)
endif()

option(NABLANET_EXAMPLES_BUILD_DATASET_EXAMPLES
    "Build the FordA and Bank Marketing data-backed examples"
    OFF
)
option(NABLANET_EXAMPLES_BUILD_FASHION_MNIST
    "Build the Fashion-MNIST SVD feature example"
    OFF
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/NablaNetExampleSupport.cmake")

add_subdirectory(xor-bce)
add_subdirectory(margin-loss-robustness)

if(NABLANET_EXAMPLES_BUILD_DATASET_EXAMPLES)
    add_subdirectory(automotive-time-series-classification)
    add_subdirectory(bank-marketing)
endif()

if(NABLANET_EXAMPLES_BUILD_FASHION_MNIST)
    add_subdirectory(fashion-mnist-classification)
endif()
