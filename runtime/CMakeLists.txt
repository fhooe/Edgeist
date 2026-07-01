add_library(edgeist_runtime STATIC
    src/layers/adaptive_avg_pool1d_layer.cpp
    src/layers/adaptive_avg_pool2d_layer.cpp
    src/layers/batch_norm_layer.cpp
    src/layers/common.cpp
    src/layers/conv2d_layer.cpp
    src/layers/cross_entropy_loss.cpp
    src/layers/dropout_layer.cpp
    src/layers/flatten_layer.cpp
    src/layers/linear_layer.cpp
    src/layers/max_pool2d_layer.cpp
    src/layers/relu_layer.cpp
    src/layers/softmax_layer.cpp
    src/memory.cpp
    src/model_format.cpp
    src/model_runtime.cpp
    src/numeric.cpp
    src/optimizers.cpp
    src/storage.cpp
)

add_library(edgeist::runtime ALIAS edgeist_runtime)

target_include_directories(edgeist_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(edgeist_runtime PUBLIC
    EDGEIST_ENABLE_TRAINING=$<BOOL:${EDGEIST_ENABLE_TRAINING}>
    EDGEIST_DETERMINISTIC_MATH=$<BOOL:${EDGEIST_DETERMINISTIC_MATH}>
)

edgeist_apply_common_options(edgeist_runtime)

install(TARGETS edgeist_runtime EXPORT EdgeistRuntimeTargets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT EdgeistRuntimeTargets NAMESPACE edgeist:: DESTINATION lib/cmake/EdgeistRuntime)
