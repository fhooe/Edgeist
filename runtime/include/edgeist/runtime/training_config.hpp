#pragma once

#include <cstdint>

#include "edgeist/runtime/numeric.hpp"

namespace edgeist {

enum class RuntimeMode : std::uint8_t {
    InferenceOnly = 0,
    FullTraining,
    LastLayerTraining,
    FrozenLayerTraining,
};

enum class OptimizerKind : std::uint8_t {
    Sgd = 0,
    Momentum,
    Adam,
};

enum class StorageKind : std::uint8_t {
    Sram = 0,
    Flash,
    External,
};

struct TrainingConfig {
    RuntimeMode mode { RuntimeMode::InferenceOnly };
    OptimizerKind optimizer { OptimizerKind::Sgd };
    StorageKind trainable_storage { StorageKind::Sram };
    NumericDataType inference_data_type { NumericDataType::Float32 };
    NumericDataType training_data_type { NumericDataType::Float32 };
    QuantizationConfig quantization {};
    float learning_rate { 0.001F };
    float momentum { 0.9F };
    float beta1 { 0.9F };
    float beta2 { 0.999F };
    float epsilon { 1.0e-8F };
    std::uint32_t micro_batch_size { 1 };
    std::uint32_t frozen_prefix_layers { 0 };
    std::uint32_t deterministic_seed { 0xC0FFEEU };
    bool deterministic_math { true };
    bool allow_init_allocation { true };
    // Softmax, CrossEntropy, and optimizer state stay float32 even when lower
    // precision is selected. This mirrors common embedded training practice:
    // quantize inference-heavy tensors while keeping numerically sensitive math stable.
    bool keep_loss_and_optimizer_fp32 { true };
};

} // namespace edgeist
