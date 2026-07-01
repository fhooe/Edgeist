#pragma once

#include <cstdint>

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class LinearLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, ConstTypedTensorView weights, ConstTypedTensorView bias,
        MutableTypedTensorView output, std::uint32_t input_size, std::uint32_t output_size) noexcept -> Status;

    [[nodiscard]] static auto backward(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView weights,
        Span<float> grad_input, Span<float> grad_weights, Span<float> grad_bias,
        std::uint32_t input_size, std::uint32_t output_size, bool accumulate) noexcept -> Status;
};

} // namespace edgeist::layers
