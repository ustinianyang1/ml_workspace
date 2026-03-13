cmake_minimum_required(VERSION 3.18)

project(perceptron VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PERCEPTRON_BUILD_PYTHON "Build Python bindings with pybind11" ON)

add_library(perceptron_core STATIC
    src/perceptron.cpp
)

target_include_directories(perceptron_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(perceptron_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(perceptron_core PRIVATE /W4)
else()
    target_compile_options(perceptron_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PERCEPTRON_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    
    # 自动获取当前 Python 环境中 pybind11 的路径
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(perceptron_cpp src/bindings.cpp)

    target_link_libraries(perceptron_cpp PRIVATE perceptron_core)

    target_include_directories(perceptron_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # 强制将所有模式的编译产物直接输出到 python 目录下
    set_target_properties(perceptron_cpp PROPERTIES
        OUTPUT_NAME "perceptron_cpp"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
endif()

message(STATUS "Configuring perceptron project")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Python bindings: ${PERCEPTRON_BUILD_PYTHON}")
