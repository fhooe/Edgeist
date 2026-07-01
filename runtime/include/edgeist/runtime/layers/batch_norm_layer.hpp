#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class BatchNormLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, ConstTypedTensorView gamma, ConstTypedTensorView beta,
        ConstTypedTensorView mean, ConstTypedTensorView variance, MutableTypedTensorView output, float epsilon) noexcept -> Status;

    [[nodiscard]] static auto backward_affine(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView gamma,
        ConstTypedTensorView mean, ConstTypedTensorView variance, Span<float> grad_input, Span<float> grad_gamma,
        Span<float> grad_beta, float epsilon, bool accumulate) noexcept -> Status;
};

} // namespace edgeist::layers
