/**
 * @file
 * @brief Entrypoint
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "loss_function.h"
#include "mnist_loader.h"
#include "nmcm.h"
#include "optimizer_data_types.h"

using namespace Edgeist;

namespace {
// constants
constexpr int NUM_OUTPUTS = 10;
constexpr int TRAINING_SIZE = 60000;
constexpr int BATCH_SIZE = 64;
constexpr int TRAINING_EPOCHS = 1;
constexpr float LEARNING_RATE = 0.001F;

// Loads file into a buffer
auto loadFileToBuffer(const std::string& filename, std::streamsize& size) -> char*
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    if (!file.read(buffer, size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        delete[] buffer;
        return nullptr;
    }
    return buffer;
}

// execute a forwardPass and calculate accuracy & loss
auto evaluateModel(Model<float>& myModel, const std::vector<MNISTImage>& images, SoftmaxCrossEntropyLoss<float>& lossFn, const std::string& label) -> void
{
    float output[NUM_OUTPUTS] = { 0.0f };
    float expected[NUM_OUTPUTS] = { 0.0f };

    uint32_t correct = 0;
    float totalLoss = 0.0f;

    for (const auto& img : images) {
        const float* input = reinterpret_cast<const float*>(img.data);
        myModel.inferenceSram(input, output);

        expected[img.label] = 1.0f;

        // prediction & loss
        int maxIdx = 0;
        float maxVal = output[0];
        for (int i = 0; i < NUM_OUTPUTS; ++i) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }
        totalLoss += lossFn.compute(output, expected, NUM_OUTPUTS);

        if (maxIdx == img.label) {
            ++correct;
        }

        expected[img.label] = 0.0f;
    }

    float accuracy = static_cast<float>(correct) / images.size();
    std::cout << "Accuracy " << label << ": " << accuracy * 100 << "%" << std::endl;
    std::cout << "Loss " << label << ": " << totalLoss / images.size() << std::endl
              << std::endl;
}
} // namespace

auto main() -> int
{
    std::streamsize sizeFixed = 0;
    std::streamsize sizeTrainable = 0;
    char* modelFixed = loadFileToBuffer("model.hex", sizeFixed);
    char* modelTrainable = loadFileToBuffer("model_trainable.hex", sizeTrainable);

    if (modelFixed == nullptr || modelTrainable == nullptr) {
        return EXIT_FAILURE;
    }

    Model<float> myModel(modelFixed, modelTrainable, OptimizerID::SGD, LEARNING_RATE);
    myModel.init();

    auto trainImages = std::vector<MNISTImage>();
    auto testImages = std::vector<MNISTImage>();
    try {
        trainImages = loadMNISTBatch("mnist_train_all_random.bin");
        testImages = loadMNISTBatch("mnist_test_all_random.bin");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load MNIST-Data: " << e.what() << std::endl;
        return 1;
    }

    SoftmaxCrossEntropyLoss<float> loss;

    evaluateModel(myModel, testImages, loss, "before training");

    float expected[NUM_OUTPUTS] = { 0.0F };
    std::cout << "=== Training started ===" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < TRAINING_EPOCHS; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << " started..." << std::endl;

        for (int batch = 0; batch < TRAINING_SIZE / BATCH_SIZE; ++batch) {
            myModel.initGradients();

            for (int i = 0; i < BATCH_SIZE; ++i) {
                const MNISTImage& img = trainImages[batch * BATCH_SIZE + i];
                const auto* input = reinterpret_cast<const float*>(img.data);

                expected[img.label] = 1.0f;
                myModel.train(input, expected, loss);
                expected[img.label] = 0.0f;
            }

            myModel.update(BATCH_SIZE);
            myModel.deleteGradients();
        }

        std::cout << "Epoch " << epoch + 1 << " finished." << std::endl;
        evaluateModel(myModel, testImages, loss, "after epoch: " + std::to_string(epoch + 1));
    }
    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> const duration = end - start;
    std::cout << "Program runtime: " << duration.count() << " seconds for " << TRAINING_EPOCHS << "epochs";

    if (modelFixed) {
        delete[] modelFixed;
    }
    if (modelTrainable) {
        delete[] modelTrainable;
    }

    std::cout << "Debug Segfault" << std::endl;

    return 0;
}
