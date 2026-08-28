cmake_minimum_required(VERSION 3.21)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(NablaNetBankMarketingExample LANGUAGES CXX)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/NablaNetExampleSupport.cmake")
add_nablanet_example(nablanet_bank_marketing bank-marketing.cpp)
