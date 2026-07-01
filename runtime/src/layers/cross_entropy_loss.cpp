#include "edgeist/runtime/layers/cross_entropy_loss.hpp"

#include <algorithm>
#include <cmath>

namespace edgeist::layers {

auto CrossEntropyLoss::loss(ConstSpan<float> probabilities, ConstSpan<float> target, float& loss, float epsilon) noexcept -> Status
{
    if (probabilities.data() == nullptr || target.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "cross entropy span is null");
    }
    if (target.size() < probabilities.size()) {
        return make_status(ErrorCode::BufferTooSmall, "cross entropy target too small");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_status(ErrorCode::InvalidArgument, "cross entropy epsilon must be positive");
    }
    loss = 0.0F;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const float p = std::clamp(probabilities[i], epsilon, 1.0F);
        loss -= target[i] * std::log(p);
    }
    if (!std::isfinite(loss)) {
        return make_status(ErrorCode::NumericError, "cross entropy produced non-finite loss");
    }
    return Status::success();
}

auto CrossEntropyLoss::gradient(ConstSpan<float> probabilities, ConstSpan<float> target, Span<float> grad_logits) noexcept -> Status
{
    if (probabilities.data() == nullptr || target.data() == nullptr || grad_logits.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "cross entropy gradient span is null");
    }
    if (target.size() < probabilities.size() || grad_logits.size() < probabilities.size()) {
        return make_status(ErrorCode::BufferTooSmall, "cross entropy gradient span too small");
    }
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        grad_logits[i] = probabilities[i] - target[i];
    }
    return Status::success();
}


} // namespace edgeist::layers
