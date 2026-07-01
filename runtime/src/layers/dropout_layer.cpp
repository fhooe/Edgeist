#include "edgeist/runtime/layers/dropout_layer.hpp"

namespace edgeist::layers {

auto DropoutLayer::forward_inference(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    // In inference mode dropout is identity. Training-time stochastic dropout is
    // intentionally not used by the deterministic on-device training path.
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

} // namespace edgeist::layers
