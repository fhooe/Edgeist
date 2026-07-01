cmake_minimum_required(VERSION 3.21)

project(edgeist_V2 VERSION 2.0.0 LANGUAGES CXX DESCRIPTION "Robust embedded neural-network runtime and tooling")

option(EDGEIST_BUILD_TESTS "Build Edgeist unit and validation tests" ON)
option(EDGEIST_BUILD_EXAMPLES "Build desktop and embedded examples" ON)
option(EDGEIST_ENABLE_TRAINING "Compile training runtime support" ON)
option(EDGEIST_ENABLE_EXCEPTIONS "Enable C++ exceptions in examples/tools" ON)
option(EDGEIST_ENABLE_ASAN "Enable AddressSanitizer for host builds" OFF)
option(EDGEIST_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for host builds" OFF)
option(EDGEIST_DETERMINISTIC_MATH "Favor deterministic math settings over fastest math" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EdgeistCompilerOptions)

add_subdirectory(runtime)

if(EDGEIST_BUILD_EXAMPLES)
    add_subdirectory(examples/desktop)
    add_subdirectory(examples/embedded)
    add_subdirectory(examples/mnist_training)
endif()

if(EDGEIST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
