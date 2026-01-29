/**
 * @file
 * @brief Entrypoint
 */

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "desktop_application/config-Edgeist-Edgeist.h"
#include "loss_function.h"
#include "mnist_loader.h"
#include "nmcm.h"
#include "optimizer_data_types.h"

namespace {
// constants
constexpr int NUM_OUTPUTS = 10;
constexpr int TRAINING_SIZE = 60000;
constexpr int BATCH_SIZE = 64;
constexpr int TRAINING_EPOCHS = 1;
constexpr float LEARNING_RATE = 0.001F;

/**
 * @brief Loads the content of an entire file
 * @param filename The file to load
 * @return The content of the entire file
 */
auto loadFileToBuffer(const std::string& filename) -> std::vector<uint8_t>
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open file: {}", filename));
    }
    const auto size = std::filesystem::file_size(filename);

    std::vector<uint8_t> buffer(size);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error(std::format("Failed to read file: {}", filename));
    }
    return buffer;
}
// execute a forwardPass and calculate accuracy & loss
auto evaluateModel(Edgeist::Model<float>& myModel, const std::vector<Edgeist::MNISTImage>& images, Edgeist::SoftmaxCrossEntropyLoss<float>& lossFn, const std::string& label) -> void
{
    float output[NUM_OUTPUTS] = { 0.0F };
    float expected[NUM_OUTPUTS] = { 0.0F };

    uint32_t correct = 0;
    float totalLoss = 0.0F;

    for (const auto& img : images) {
        const auto* input = reinterpret_cast<const float*>(img.data);
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

        expected[img.label] = 0.0F;
    }

    const float accuracy = static_cast<float>(correct) / images.size();
    std::cout << "Accuracy " << label << ": " << accuracy * 100 << "%\n";
    std::cout << "Loss " << label << ": " << totalLoss / images.size() << '\n'
              << std::endl;
}
} // namespace

auto main(int argc, char** argv) -> int
{
    namespace Po = boost::program_options;

    auto desc = Po::options_description("allowed options");
    std::vector<uint8_t> modelFixed;
    std::vector<uint8_t> modelTrainable;

    try {
        auto vMap = Po::variables_map();
        auto modelPath = std::string();
        auto modelTrainablePath = std::string();
        auto trainImagesPath = std::string();
        auto testImagesPath = std::string();

        desc.add_options()("help,h", "display help message");
        desc.add_options()("version,V", "display application version");
        desc.add_options()("model,m", Po::value(&modelPath)->default_value("model.hex"), "path to the model file");
        desc.add_options()("trainable,t", Po::value(&modelTrainablePath)->default_value("model_trainable.hex"), "path to the trainable file");
        desc.add_options()("train_images,i", Po::value(&trainImagesPath)->default_value("mnist_train_all_random.bin"), "path to training images file");
        desc.add_options()("test_images,e", Po::value(&testImagesPath)->default_value("mnist_test_all_random.bin"), "path to the test images file");

        Po::store(Po::parse_command_line(argc, argv, desc), vMap);
        Po::notify(vMap);

        if (vMap.contains("help")) {
            std::cout << desc << '\n';
            return EXIT_SUCCESS;
        }

        if (vMap.contains("version")) {
            std::cout << std::format("version {}.{}.{}\n", Edgeist::Edgeist::VERSION_MAJOR, Edgeist::Edgeist::VERSION_MINOR, Edgeist::Edgeist::VERSION_PATCH);
            return EXIT_SUCCESS;
        }

        modelFixed = loadFileToBuffer(modelPath);
        modelTrainable = loadFileToBuffer(modelTrainablePath);
        auto trainImages = Edgeist::loadMNISTBatch(trainImagesPath);
        auto testImages = Edgeist::loadMNISTBatch(testImagesPath);

        auto myModel = Edgeist::Model<float>(modelFixed.data(), modelTrainable.data(), Edgeist::OptimizerID::SGD, LEARNING_RATE);
        auto loss = Edgeist::SoftmaxCrossEntropyLoss<float>();

        myModel.init();
        evaluateModel(myModel, testImages, loss, "before training");

        float expected[NUM_OUTPUTS] = { 0.0F };
        std::cout << "=== Training started ===\n";

        auto start = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < TRAINING_EPOCHS; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << " started...\n";

            for (int batch = 0; batch < TRAINING_SIZE / BATCH_SIZE; ++batch) {
                myModel.initGradients();

                for (int i = 0; i < BATCH_SIZE; ++i) {
                    const auto& [data, label] = trainImages.at((batch * BATCH_SIZE) + i);
                    const auto* input = reinterpret_cast<const float*>(data);

                    expected[label] = 1.0F;
                    myModel.train(input, expected, loss);
                    expected[label] = 0.0F;
                }

                myModel.update(BATCH_SIZE);
                myModel.deleteGradients();
            }

            std::cout << "Epoch " << epoch + 1 << " finished.\n";
            evaluateModel(myModel, testImages, loss, "after epoch: " + std::to_string(epoch + 1));
        }
        const auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> const duration = end - start;
        std::cout << "Program runtime: " << duration.count() << " seconds for " << TRAINING_EPOCHS << "epochs";

        return EXIT_SUCCESS;
    } catch (const Po::error& error) {
        std::cerr << error.what() << "\n";
        std::cerr << desc << "\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
    } catch (...) {
        std::cerr << "unknown error occurred\n";
    }

    return EXIT_FAILURE;
}
