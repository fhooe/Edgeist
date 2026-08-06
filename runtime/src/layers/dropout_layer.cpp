#include "edgeist/runtime/layers/dropout_layer.hpp"

#include <cmath>

namespace edgeist::layers {

namespace {
[[nodiscard]] auto validate_dropout_rate(float dropout_rate) noexcept -> Status
{
    if (!std::isfinite(dropout_rate) || dropout_rate < 0.0F || dropout_rate >= 1.0F) {
        return make_status(ErrorCode::InvalidArgument, "dropout rate must be in [0, 1)");
    }
    return Status::success();
}

[[nodiscard]] auto next_random(std::uint32_t& state) noexcept -> float
{
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8U) * (1.0F / 16777216.0F);
}
} // namespace

auto DropoutLayer::forward_inference(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    if (input.elements != output.elements) {
        return make_status(ErrorCode::ShapeMismatch, "dropout input/output element counts must match");
    }
    if (input.type == output.type && input.scale == output.scale) {
        return copy_typed_tensor(input, output);
    }
    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, value));
    }
    return Status::success();
}

auto DropoutLayer::forward_training(ConstTypedTensorView input, MutableTypedTensorView output, Span<std::uint8_t> mask,
    float dropout_rate, std::uint32_t& rng_state) noexcept -> Status
{
    if (input.elements != output.elements) {
        return make_status(ErrorCode::ShapeMismatch, "dropout input/output element counts must match");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "dropout training input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "dropout training output invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(mask.size(), input.elements, "dropout mask too small"));
    EDGEIST_RETURN_IF_ERROR(validate_dropout_rate(dropout_rate));

    const float scale = 1.0F / (1.0F - dropout_rate);
    for (std::size_t i = 0; i < input.elements; ++i) {
        const bool keep = next_random(rng_state) >= dropout_rate;
        mask[i] = keep ? std::uint8_t { 1 } : std::uint8_t { 0 };
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, keep ? value * scale : 0.0F));
    }
    return Status::success();
}

auto DropoutLayer::backward(ConstSpan<float> grad_output, ConstSpan<std::uint8_t> mask, Span<float> grad_input,
    float dropout_rate) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_dropout_rate(dropout_rate));
    EDGEIST_RETURN_IF_ERROR(require_span_size(mask.size(), grad_output.size(), "dropout mask too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), grad_output.size(), "dropout grad_input too small"));
    if ((grad_output.data() == nullptr || mask.data() == nullptr || grad_input.data() == nullptr) && !grad_output.empty()) {
        return make_status(ErrorCode::NullPointer, "dropout backward span is null");
    }

    const float scale = 1.0F / (1.0F - dropout_rate);
    for (std::size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = mask[i] != 0U ? grad_output[i] * scale : 0.0F;
    }
    return Status::success();
}

} // namespace edgeist::layers
