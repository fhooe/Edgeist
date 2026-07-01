#pragma once

#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

class CrossEntropyLoss final {
public:
    [[nodiscard]] static auto loss(ConstSpan<float> probabilities, ConstSpan<float> target, float& loss, float epsilon = 1.0e-7F) noexcept -> Status;
    [[nodiscard]] static auto gradient(ConstSpan<float> probabilities, ConstSpan<float> target, Span<float> grad_logits) noexcept -> Status;
};

} // namespace edgeist::layers
