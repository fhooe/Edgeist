#include "edgeist/edgeist.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

auto load_file(const std::string& path) -> std::vector<std::byte>
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return data;
}

void print_usage(const char* executable)
{
    std::cout << "Usage: " << executable << " --model model.nmcf --trainable model_trainable.bin [options]\n"
              << "Options:\n"
              << "  --train                         Use last-layer training strategy instead of inference only\n"
              << "  --inference-type int8|fp16|fp32  Inference dtype; must match exported parameter encoding\n"
              << "  --training-type int8|fp16|fp32   Training dtype; must match exported trainable parameter encoding\n"
              << "  --help                          Show this help\n";
}

bool parse_type_argument(const std::string& value, edgeist::NumericDataType& out)
{
    const auto status = edgeist::parse_numeric_data_type(value, out);
    if (!status.ok()) {
        std::cerr << "invalid data type '" << value << "': " << status.message << "\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::string model_path;
        std::string trainable_path;
        bool train_mode = false;
        edgeist::NumericDataType inference_type = edgeist::NumericDataType::Float32;
        edgeist::NumericDataType training_type = edgeist::NumericDataType::Float32;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
                model_path = argv[++i];
            } else if ((arg == "--trainable" || arg == "-t") && i + 1 < argc) {
                trainable_path = argv[++i];
            } else if (arg == "--train") {
                train_mode = true;
            } else if (arg == "--inference-type" && i + 1 < argc) {
                if (!parse_type_argument(argv[++i], inference_type)) {
                    return 2;
                }
            } else if (arg == "--training-type" && i + 1 < argc) {
                if (!parse_type_argument(argv[++i], training_type)) {
                    return 2;
                }
            } else {
                std::cerr << "unknown or incomplete argument: " << arg << "\n";
                print_usage(argv[0]);
                return 2;
            }
        }

        if (model_path.empty() || trainable_path.empty()) {
            print_usage(argv[0]);
            return 2;
        }

        auto model = load_file(model_path);
        auto trainable = load_file(trainable_path);
        edgeist::ModelInfo info;
        auto status = edgeist::validate_model(edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable), &info);
        if (!status.ok()) {
            std::cerr << "model validation failed: " << edgeist::to_string(status.code) << ": " << status.message << "\n";
            return 1;
        }

        edgeist::TrainingConfig config;
        config.mode = train_mode ? edgeist::RuntimeMode::LastLayerTraining : edgeist::RuntimeMode::InferenceOnly;
        config.optimizer = edgeist::OptimizerKind::Sgd;
        config.learning_rate = 0.001F;
        config.inference_data_type = inference_type;
        config.training_data_type = training_type;
        edgeist::MemoryReport report;
        status = edgeist::MemoryPlanner::estimate(info, config, report);
        if (!status.ok()) {
            std::cerr << "memory planning failed: " << status.message << "\n";
            return 1;
        }

        std::cout << "Model validated successfully.\n"
                  << "Layers: " << info.layers.size() << "\n"
                  << "Model bytes: " << info.model_bytes << "\n"
                  << "Trainable bytes: " << info.trainable_bytes << "\n"
                  << "Inference type: " << edgeist::numeric_data_type_name(config.inference_data_type) << "\n"
                  << "Training type: " << edgeist::numeric_data_type_name(config.training_data_type) << "\n"
                  << "Memory report:\n" << report.to_json() << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
