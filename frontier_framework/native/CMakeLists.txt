cmake_minimum_required(VERSION 3.16)
project(frontier_native LANGUAGES C CXX)

add_library(frontier_simulation
    simulated_environment.c
    simulated_environment.cpp
)

target_include_directories(frontier_simulation
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(frontier_simulation PUBLIC c_std_99 cxx_std_11)

include(CTest)
if(BUILD_TESTING)
    add_executable(frontier_native_test test_simulated_environment.cpp)
    target_link_libraries(frontier_native_test PRIVATE frontier_simulation)
    add_test(NAME frontier_native_test COMMAND frontier_native_test)
endif()
