add_executable(edgeist-embedded-skeleton main.cpp)
target_link_libraries(edgeist-embedded-skeleton PRIVATE edgeist::runtime)
edgeist_apply_common_options(edgeist-embedded-skeleton)
