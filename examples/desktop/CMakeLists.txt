add_executable(edgeist-desktop src/main.cpp)
target_link_libraries(edgeist-desktop PRIVATE edgeist::runtime)
target_include_directories(edgeist-desktop PRIVATE include)
edgeist_apply_common_options(edgeist-desktop)
