cmake_minimum_required(VERSION 3.18)
project(cnn_libtorch_checks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(check_libtorch_inference check_libtorch_inference.cpp)
target_link_libraries(check_libtorch_inference "${TORCH_LIBRARIES}")
set_property(TARGET check_libtorch_inference PROPERTY CXX_STANDARD 17)

add_library(cnn_libtorch_bridge SHARED cnn_libtorch_bridge.cpp)
target_link_libraries(cnn_libtorch_bridge "${TORCH_LIBRARIES}")
set_property(TARGET cnn_libtorch_bridge PROPERTY CXX_STANDARD 17)
