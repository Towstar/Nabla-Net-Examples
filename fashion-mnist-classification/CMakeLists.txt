cmake_minimum_required(VERSION 3.21)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(NablaNetFashionMnistExample LANGUAGES CXX)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/NablaNetExampleSupport.cmake")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs)

add_nablanet_example(
    nablanet_fashion_mnist_classification
    fashion-mnist-classification.cpp
)
target_include_directories(
    nablanet_fashion_mnist_classification
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(nablanet_fashion_mnist_classification PRIVATE ${OpenCV_LIBS})
