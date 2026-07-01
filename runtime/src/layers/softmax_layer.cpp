#include "edgeist/runtime/layers/softmax_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace edgeist::layers {

auto SoftmaxLayer::forward(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    if (input.elements != output.elements) {
        return make_status(ErrorCode::ShapeMismatch, "softmax input/output element counts must match");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "softmax input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "softmax output tensor invalid"));
    if (input.elements == 0U) {
        return make_status(ErrorCode::InvalidArgument, "softmax input is empty");
    }

    float max_value = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        max_value = std::max(max_value, value);
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        sum += std::exp(value - max_value);
    }
    if (!std::isfinite(sum) || sum <= 0.0F) {
        return make_status(ErrorCode::NumericError, "softmax normalization failed");
    }

    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, std::exp(value - max_value) / sum));
    }
    return Status::success();
}

auto SoftmaxLayer::backward(ConstSpan<float> softmax_output, ConstSpan<float> grad_output, Span<float> grad_input) noexcept -> Status
{
    if (softmax_output.data() == nullptr || grad_output.data() == nullptr || grad_input.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "softmax backward span is null");
    }
    if (grad_output.size() < softmax_output.size() || grad_input.size() < softmax_output.size()) {
        return make_status(ErrorCode::BufferTooSmall, "softmax backward span too small");
    }
    float dot = 0.0F;
    for (std::size_t i = 0; i < softmax_output.size(); ++i) {
        dot += softmax_output[i] * grad_output[i];
    }
    for (std::size_t i = 0; i < softmax_output.size(); ++i) {
        grad_input[i] = softmax_output[i] * (grad_output[i] - dot);
    }
    return Status::success();
}

} // namespace edgeist::layers
