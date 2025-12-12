cmake_minimum_required(VERSION 3.21)

set(TARGET_NAME "edgeist-desktop")
set(ENABLE_TESTS_${TARGET_NAME} FALSE CACHE BOOL "Build unit tests (default off).")
if(ENABLE_TESTS_${TARGET_NAME})
    include(CTest)
endif()

add_target(${TARGET_NAME} "EXECUTABLE")
