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

add_library(frontier_neural_brain
    neural_brain/src/types.cpp
    neural_brain/src/neural_cell.cpp
    neural_brain/src/brain_graph.cpp
    neural_brain/src/gnn_layer.cpp
    neural_brain/src/memory_layer.cpp
    neural_brain/src/feature_encoder.cpp
    neural_brain/src/tree_models.cpp
    neural_brain/src/ensemble.cpp
    neural_brain/src/learning_controller.cpp
)

target_include_directories(frontier_neural_brain
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/neural_brain/include
)
target_compile_features(frontier_neural_brain PUBLIC cxx_std_20)

include(CTest)
if(BUILD_TESTING)
    add_executable(frontier_native_test test_simulated_environment.cpp)
    target_link_libraries(frontier_native_test PRIVATE
        frontier_simulation
        frontier_neural_brain
    )
    add_test(NAME frontier_native_test COMMAND frontier_native_test)
endif()
