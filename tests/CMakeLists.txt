find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(edgeist_runtime_tests
    test_runtime.cpp
)

target_link_libraries(edgeist_runtime_tests PRIVATE edgeist::runtime)
target_include_directories(edgeist_runtime_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(edgeist_runtime_tests PRIVATE
    EDGEIST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    EDGEIST_PYTHON_GOLDEN_MODEL="${PROJECT_SOURCE_DIR}/tools/python_golden_model.py"
)
edgeist_apply_common_options(edgeist_runtime_tests)

add_test(NAME edgeist_runtime_tests COMMAND edgeist_runtime_tests)
add_test(NAME edgeist_python_golden_model COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/python_golden_model.py --case all)
