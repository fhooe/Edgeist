#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "edgeist/runtime/memory.hpp"
#include "edgeist/runtime/model_format.hpp"
#include "edgeist/runtime/optimizers.hpp"
#include "edgeist/runtime/storage.hpp"
#include "edgeist/runtime/training_config.hpp"

namespace edgeist {

struct RuntimeStats {
    std::uint32_t trained_samples { 0 };
    float last_loss { 0.0F };
    MemoryReport memory {};
};

struct ActivationTraceBuffer {
    // values stores decoded float32 layer outputs for diagnostics only. Runtime
    // activations remain in the selected typed scratch buffers.
    Span<float> values {};
    Span<std::uint32_t> offsets {};
    std::uint32_t layer_count { 0 };
    std::uint32_t used_values { 0 };
};

class ModelRuntime {
public:
    ModelRuntime() = default;

    [[nodiscard]] auto init(ModelView view, StorageBackend& trainable_storage, ScratchArena& arena,
        TrainingConfig config) -> Status;

    [[nodiscard]] auto inference(ConstSpan<float> input, Span<float> output) noexcept -> Status;
    [[nodiscard]] auto inference_with_trace(ConstSpan<float> input, Span<float> output, ActivationTraceBuffer& trace) noexcept -> Status;

    [[nodiscard]] auto begin_micro_batch() noexcept -> Status;
    [[nodiscard]] auto train_sample(ConstSpan<float> input, ConstSpan<float> expected_output, float* loss_out = nullptr) noexcept -> Status;
    [[nodiscard]] auto apply_updates(std::uint32_t accumulated_samples) noexcept -> Status;

    [[nodiscard]] auto stats() const noexcept -> const RuntimeStats& { return stats_; }
    [[nodiscard]] auto memory_report() const noexcept -> const MemoryReport& { return stats_.memory; }

private:
    struct ParameterConstViews {
        ConstTypedTensorView weights {};
        ConstTypedTensorView bias {};
    };

    struct ParameterMutableViews {
        MutableTypedTensorView weights {};
        MutableTypedTensorView bias {};
    };

    [[nodiscard]] auto should_train_layer(std::size_t layer_index) const noexcept -> bool;
    [[nodiscard]] auto forward_impl(ConstSpan<float> input, Span<float> output, bool training, ActivationTraceBuffer* trace) noexcept -> Status;
    [[nodiscard]] auto layer_param_const_views(const LayerInfo& layer, ParameterConstViews& views) const noexcept -> Status;
    [[nodiscard]] auto layer_param_mutable_views(const LayerInfo& layer, ParameterMutableViews& views) noexcept -> Status;
    [[nodiscard]] auto require_layer_encoding(const LayerInfo& layer, NumericDataType execution_type) const noexcept -> Status;
    [[nodiscard]] auto typed_view_for_bytes(ByteSpan bytes, NumericDataType type, std::size_t elements, float scale) noexcept -> MutableTypedTensorView;
    [[nodiscard]] auto typed_view_for_bytes(ConstByteSpan bytes, NumericDataType type, std::size_t elements, float scale) const noexcept -> ConstTypedTensorView;
    [[nodiscard]] auto parameter_scale_for_weights() const noexcept -> float { return config_.quantization.weight_scale; }
    [[nodiscard]] auto parameter_scale_for_bias() const noexcept -> float { return config_.quantization.bias_scale; }
    [[nodiscard]] auto apply_typed_optimizer(MutableTypedTensorView params, ConstSpan<float> grads, OptimizerStateView state,
        float learning_scale) noexcept -> Status;

    ModelView view_ {};
    StorageBackend* trainable_storage_ { nullptr };
    ScratchArena* arena_ { nullptr };
    TrainingConfig config_ {};
    RuntimeStats stats_ {};

    ByteSpan activation_a_ {};
    ByteSpan activation_b_ {};
    ByteSpan saved_activations_ {};
    Span<float> grad_a_ {};
    Span<float> grad_b_ {};
    Span<float> gradients_ {};
    Span<float> opt_state_1_ {};
    Span<float> opt_state_2_ {};
    Span<std::uint32_t> argmax_ {};
    Span<std::uint8_t> dropout_masks_ {};

    std::vector<std::size_t> saved_offsets_;      // byte offsets into saved_activations_
    std::vector<std::size_t> gradient_offsets_;   // float element offsets into gradients_
    std::vector<std::size_t> optimizer_offsets_;  // float element offsets into optimizer state
    std::vector<std::size_t> argmax_offsets_;      // uint32 element offsets into retained max-pool indices
    std::vector<std::size_t> dropout_offsets_;     // byte offsets into retained dropout masks
    std::uint32_t rng_state_ { 0 };
    bool initialized_ { false };
};

} // namespace edgeist
