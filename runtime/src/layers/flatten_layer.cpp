#include "edgeist/runtime/layers/flatten_layer.hpp"

#include <algorithm>

namespace edgeist::layers {

auto FlattenLayer::forward(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    if (input.type == output.type && input.elements == output.elements && input.scale == output.scale) {
        return copy_typed_tensor(input, output);
    }
    if (input.elements != output.elements) {
        return make_status(ErrorCode::ShapeMismatch, "flatten input/output element counts must match");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "flatten input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "flatten output tensor invalid"));
    for (std::size_t i = 0; i < input.elements; ++i) {
        float value = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, value));
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, value));
    }
    return Status::success();
}

auto FlattenLayer::backward(ConstSpan<float> grad_output, Span<float> grad_input) noexcept -> Status
{
    if (grad_input.size() < grad_output.size()) {
        return make_status(ErrorCode::BufferTooSmall, "flatten grad_input too small");
    }
    std::copy(grad_output.begin(), grad_output.end(), grad_input.begin());
    return Status::success();
}

} // namespace edgeist::layers
