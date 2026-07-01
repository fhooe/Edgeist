#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class AdaptiveAvgPool2DLayer final {
public:
    [[nodiscard]] static auto forward(ConstTypedTensorView input, MutableTypedTensorView output, std::uint32_t channels,
        std::uint32_t input_h, std::uint32_t input_w, std::uint32_t output_h, std::uint32_t output_w) noexcept -> Status;
};

} // namespace edgeist::layers
