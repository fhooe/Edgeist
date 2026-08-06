#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class DropoutLayer final {
public:
    [[nodiscard]] static auto forward_inference(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status;
    [[nodiscard]] static auto forward_training(ConstTypedTensorView input, MutableTypedTensorView output, Span<std::uint8_t> mask,
        float dropout_rate, std::uint32_t& rng_state) noexcept -> Status;
    [[nodiscard]] static auto backward(ConstSpan<float> grad_output, ConstSpan<std::uint8_t> mask, Span<float> grad_input,
        float dropout_rate) noexcept -> Status;
};

} // namespace edgeist::layers
