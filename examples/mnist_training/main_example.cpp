#include "edgeist/edgeist.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace ei = edgeist;

[[noreturn]] void die(const char* message)
{
    std::cerr << message << '\n';
    std::exit(1);
}

std::vector<std::byte> load(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        die(path);
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

void check(ei::Status status)
{
    if (!status.ok()) {
        std::cerr << status.message << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc > 2) {
        die("usage: edgeist-mnist-training [epochs]");
    }

    std::size_t epochs = 1;
    if (argc == 2) {
        char* end = nullptr;
        epochs = static_cast<std::size_t>(std::strtoull(argv[1], &end, 10));
        if (argv[1][0] == '-' || *end != '\0' || epochs == 0) {
            die("epochs must be greater than 0");
        }
    }

    auto model = load("model.hex");
    auto trainable = load("model_trainable.hex");

    std::array<std::byte, 128 * 1024> scratch {};
    ei::ScratchArena arena { ei::ByteSpan(scratch) };
    ei::SramStorage storage { ei::ByteSpan(trainable) };

    ei::TrainingConfig config {};
    config.mode = ei::RuntimeMode::FullTraining;
    config.learning_rate = 0.01F;

    ei::ModelRuntime runtime;
    check(runtime.init(ei::ModelView { ei::ConstByteSpan(model), ei::ConstByteSpan(trainable) }, storage, arena, config));

    std::ifstream mnist("mnist_train_all_random.bin", std::ios::binary);
    if (!mnist) {
        die("mnist_train_all_random.bin");
    }

    float loss = 0.0F;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        loss = 0.0F;
        std::array<float, 784> image {};
        std::array<float, 10> target {};
        std::uint8_t label = 0;

        mnist.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size() * sizeof(float)));
        mnist.read(reinterpret_cast<char*>(&label), 1);
        if (!mnist || label >= target.size()) {
            die("bad mnist_train_all_random.bin");
        }
        target[label] = 1.0F;

        check(runtime.begin_micro_batch());
        check(runtime.train_sample(ei::ConstSpan<float>(image), ei::ConstSpan<float>(target), &loss));
        check(runtime.apply_updates(1));
    }

    std::ofstream out("model_trainable_after.hex", std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(trainable.data()), static_cast<std::streamsize>(trainable.size()));

    std::cout << "loss: " << loss << '\n';
}
