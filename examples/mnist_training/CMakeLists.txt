add_executable(edgeist-mnist-training main_example.cpp)
target_link_libraries(edgeist-mnist-training PRIVATE edgeist::runtime)
edgeist_apply_common_options(edgeist-mnist-training)

add_custom_command(
    TARGET edgeist-mnist-training
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/model.hex
            $<TARGET_FILE_DIR:edgeist-mnist-training>/model.hex
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/model_trainable.hex
            $<TARGET_FILE_DIR:edgeist-mnist-training>/model_trainable.hex
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/mnist_train_all_random.bin
            $<TARGET_FILE_DIR:edgeist-mnist-training>/mnist_train_all_random.bin
)

if(EDGEIST_BUILD_TESTS)
    enable_testing()
    add_test(
        NAME edgeist_mnist_training_smoke
        COMMAND edgeist-mnist-training 5
        WORKING_DIRECTORY $<TARGET_FILE_DIR:edgeist-mnist-training>
    )
endif()
