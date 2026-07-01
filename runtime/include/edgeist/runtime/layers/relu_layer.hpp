#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class ReLULayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status;
    [[nodiscard]] static auto backward(ConstTypedTensorView saved_input, ConstSpan<float> grad_output, Span<float> grad_input) noexcept -> Status;
};

} // namespace edgeist::layers
