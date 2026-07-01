#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class Conv2DLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, ConstTypedTensorView weights, ConstTypedTensorView bias,
        MutableTypedTensorView output, const Conv2DParams& params) noexcept -> Status;

    [[nodiscard]] static auto backward(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView weights,
        Span<float> grad_input, Span<float> grad_weights, Span<float> grad_bias,
        const Conv2DParams& params, bool accumulate) noexcept -> Status;
};

} // namespace edgeist::layers
