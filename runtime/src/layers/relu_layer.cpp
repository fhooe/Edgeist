#include "edgeist/runtime/layers/relu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace edgeist::layers {

auto ReLULayer::forward(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    if (output.elements != input.elements) {
        return make_status(ErrorCode::ShapeMismatch, "relu input/output element counts must match");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "relu input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "relu output tensor invalid"));

    if (input.type == NumericDataType::Int8 && output.type == NumericDataType::Int8 && input.scale == output.scale) {
        const auto* in = reinterpret_cast<const std::int8_t*>(input.bytes.data());
        auto* out = reinterpret_cast<std::int8_t*>(output.bytes.data());
        for (std::size_t i = 0; i < input.elements; ++i) {
            out[i] = static_cast<std::int8_t>(std::max<int>(0, in[i]));
        }
        return Status::success();
    }

    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, value > 0.0F ? value : 0.0F));
    }
    return Status::success();
}

auto ReLULayer::backward(ConstTypedTensorView saved_input, ConstSpan<float> grad_output, Span<float> grad_input) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(saved_input, "relu backward saved input invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), saved_input.elements, "relu grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), saved_input.elements, "relu grad_input too small"));
    for (std::size_t i = 0; i < saved_input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(saved_input, i, value));
        grad_input[i] = value > 0.0F ? grad_output[i] : 0.0F;
    }
    return Status::success();
}

} // namespace edgeist::layers
