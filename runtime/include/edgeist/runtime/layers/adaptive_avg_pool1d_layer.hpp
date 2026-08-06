#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class AdaptiveAvgPool1DLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, MutableTypedTensorView output, std::uint32_t channels,
        std::uint32_t input_w, std::uint32_t output_w) noexcept -> Status;
    [[nodiscard]] static auto backward(ConstSpan<float> grad_output, Span<float> grad_input, std::uint32_t channels,
        std::uint32_t input_w, std::uint32_t output_w) noexcept -> Status;
};

} // namespace edgeist::layers
