cmake_minimum_required(VERSION 3.21)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(NablaNetFashionMnistExample LANGUAGES CXX)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/NablaNetExampleSupport.cmake")

find_package(ZLIB QUIET)

if(ZLIB_FOUND)
    set(NABLANET_FASHION_ZLIB_TARGET ZLIB::ZLIB)
else()
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()

    # The raw dataset is stored as gzip-compressed IDX files. Fetch zlib only
    # when the host does not already provide it, and build it statically so the
    # example executable has no runtime DLL dependency.
    set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        nablanet_zlib
        URL https://zlib.net/fossils/zlib-1.3.2.tar.gz
        URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
    )
    FetchContent_MakeAvailable(nablanet_zlib)
    set(NABLANET_FASHION_ZLIB_TARGET ZLIB::ZLIBSTATIC)
endif()

add_nablanet_example(
    nablanet_fashion_mnist_classification
    fashion-mnist-classification.cpp
)
target_sources(
    nablanet_fashion_mnist_classification
    PRIVATE
        fashion_mnist_idx.cpp
        fashion_mnist_split.cpp
        fashion_mnist_svd.cpp
        fashion_mnist_dataset.cpp
        fashion_mnist_metrics.cpp
)
target_compile_definitions(
    nablanet_fashion_mnist_classification
    PRIVATE NABLANET_FASHION_DEFAULT_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data/fashion"
)
target_link_libraries(
    nablanet_fashion_mnist_classification
    PRIVATE
        ${NABLANET_FASHION_ZLIB_TARGET}
)

find_package(Eigen3 3.4 QUIET NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()

    # Eigen is header-only. Pin the 3.4.1 source archive to its exact release
    # commit and suppress Eigen's optional tests, demos, and documentation.
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        nablanet_eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/d71c30c47858effcbd39967097a2d99ee48db464/eigen-d71c30c47858effcbd39967097a2d99ee48db464.tar.gz
    )
    FetchContent_MakeAvailable(nablanet_eigen)
endif()

target_link_libraries(
    nablanet_fashion_mnist_classification
    PRIVATE Eigen3::Eigen
)
