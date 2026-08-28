include_guard(GLOBAL)

# Each example can be configured independently. When the library target does
# not already exist, use the adjacent source checkout and keep its tests off.
if(NOT TARGET NablaNet::NablaNet)
    if(NOT DEFINED BUILD_TESTING)
        set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree.")
    endif()

    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../../lib"
        "${CMAKE_BINARY_DIR}/nablanet"
    )
endif()

function(add_nablanet_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_link_libraries(${target_name} PRIVATE NablaNet::NablaNet)
endfunction()
