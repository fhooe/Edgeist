#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class DropoutLayer final {
public:
    [[nodiscard]] static auto forward_inference(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status;
};

} // namespace edgeist::layers
