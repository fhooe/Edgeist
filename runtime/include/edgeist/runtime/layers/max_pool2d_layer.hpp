#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class MaxPool2DLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, MutableTypedTensorView output, Span<std::uint32_t> argmax,
        const Pool2DParams& params) noexcept -> Status;
    [[nodiscard]] static auto backward(ConstSpan<float> grad_output, ConstSpan<std::uint32_t> argmax, Span<float> grad_input,
        const Pool2DParams& params) noexcept -> Status;
};

} // namespace edgeist::layers
