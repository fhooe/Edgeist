/**
 * @brief Entrypoint
 **/

#include "loss_function.h"
#include "mnist_loader.h"
#include "nmcm.h"
#include "optimizer_data_types.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Edgeist;

namespace {
// constants
constexpr int NUM_OUTPUTS = 10;
constexpr int TRAINING_SIZE = 60000;
constexpr int BATCH_SIZE = 64;
constexpr int TRAINING_EPOCHS = 1;
constexpr float LEARNING_RATE = 0.001F;
// === Konstanten ===

// === Hilfsfunktionen ===

// Lädt den Inhalt einer Datei in einen Puffer
char* loadFileToBuffer(const std::string& filename, std::streamsize& size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Fehler beim öffnen der Datei: " << filename << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    if (!file.read(buffer, size)) {
        std::cerr << "Fehler beim Lesen der Datei: " << filename << std::endl;
        delete[] buffer;
        return nullptr;
    }
    return buffer;
}

// F�hrt eine Forwardpass durch und berechnet Accuracy und Loss
void evaluateModel(model<float>& myModel, const std::vector<MnistImage>& images, SoftmaxCrossEntropyLoss<float>& lossFn, const std::string& label)
{
    float output[NUM_OUTPUTS] = { 0.0f };
    float expected[NUM_OUTPUTS] = { 0.0f };

    uint32_t correct = 0;
    float totalLoss = 0.0f;

    for (const auto& img : images) {
        const float* input = reinterpret_cast<const float*>(img.data);
        myModel.inferenceSram(input, output);

        expected[img.label] = 1.0f;

        // Prediction & Loss
        int maxIdx = 0;
        float maxVal = output[0];
        for (int i = 0; i < NUM_OUTPUTS; ++i) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }
        totalLoss += lossFn.compute(output, expected, NUM_OUTPUTS);

        if (maxIdx == img.label)
            ++correct;

        expected[img.label] = 0.0f;
    }

    float accuracy = static_cast<float>(correct) / images.size();
    std::cout << "Accuracy " << label << ": " << accuracy * 100 << "%" << std::endl;
    std::cout << "Loss " << label << ": " << totalLoss / images.size() << std::endl
              << std::endl;
}
} // namespace

int main()
{
    // === Modell laden ===
    std::streamsize sizeFixed = 0, sizeTrainable = 0;
    char* modelFixed = loadFileToBuffer("model.hex", sizeFixed);
    char* modelTrainable = loadFileToBuffer("model_trainable.hex", sizeTrainable);

    if (!modelFixed || !modelTrainable) {
        // Modell konte nicht geladen werden
        return 1;
    }

    model<float> myModel(modelFixed, modelTrainable, SGD, LEARNING_RATE);
    myModel.init();

    // === MNIST Daten laden ===
    std::vector<MnistImage> trainImages, testImages;
    try {
        trainImages = load_mnist_batch("mnist_train_all_random.bin");
        testImages = load_mnist_batch("mnist_test_all_random.bin");
    } catch (const std::exception& e) {
        std::cerr << "Fehler beim Laden der MNIST-Daten: " << e.what() << std::endl;
        return 1;
    }

    // Loss Funktion definieren
    SoftmaxCrossEntropyLoss<float> loss;

    // === Test vor dem Training ===
    evaluateModel(myModel, testImages, loss, "vor training");

    // === Training ===
    float expected[NUM_OUTPUTS] = { 0.0f };
    std::cout << "=== Training gestartet ===" << std::endl;
    // Startzeitpunkt erfassen

    auto start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < TRAINING_EPOCHS; ++epoch) {
        std::cout << "Epoche " << epoch + 1 << " gestartet..." << std::endl;

        for (int batch = 0; batch < TRAINING_SIZE / BATCH_SIZE; ++batch) {
            myModel.initGradients();

            for (int i = 0; i < BATCH_SIZE; ++i) {
                const MnistImage& img = trainImages[batch * BATCH_SIZE + i];
                const float* input = reinterpret_cast<const float*>(img.data);

                expected[img.label] = 1.0f;
                myModel.train(input, expected, loss);
                expected[img.label] = 0.0f;
            }

            myModel.update(BATCH_SIZE);
            myModel.deleteGradients();
        }

        std::cout << "Epoche " << epoch + 1 << " abgeschlossen." << std::endl;
        evaluateModel(myModel, testImages, loss, "nach Epoche: " + std::to_string(epoch + 1));
    }
    // Endzeitpunkt erfassen
    auto end = std::chrono::steady_clock::now();

    // Dauer berechnen
    std::chrono::duration<double> duration = end - start;

    // Dauer in Sekunden ausgeben
    std::cout << "Programmlaufzeit: " << duration.count() << " Sekunden for " << TRAINING_EPOCHS << "Epochen";

    // === Aufräumen ===
    if (modelFixed) {
        delete[] modelFixed;
    }

    if (modelTrainable) {
        delete[] modelTrainable;
    }

    std::cout << "Debug Segfault" << std::endl;

    return 0;
}
