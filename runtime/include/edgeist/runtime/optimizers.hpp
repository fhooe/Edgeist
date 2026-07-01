#pragma once

#include <cstddef>
#include <cstdint>

#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"
#include "edgeist/runtime/training_config.hpp"

namespace edgeist {

struct OptimizerStateView {
    Span<float> momentum_or_m {};
    Span<float> velocity_or_v {};
    std::uint32_t timestep { 1 };
};

[[nodiscard]] auto optimizer_state_elements(OptimizerKind kind, std::size_t parameter_count) noexcept -> std::size_t;

[[nodiscard]] auto apply_optimizer(OptimizerKind kind, Span<float> parameters, ConstSpan<float> gradients,
    OptimizerStateView state, const TrainingConfig& config, float scale = 1.0F) noexcept -> Status;

[[nodiscard]] auto apply_sgd(Span<float> parameters, ConstSpan<float> gradients, float learning_rate, float scale = 1.0F) noexcept -> Status;
[[nodiscard]] auto apply_momentum(Span<float> parameters, ConstSpan<float> gradients, Span<float> velocity,
    float learning_rate, float momentum, float scale = 1.0F) noexcept -> Status;
[[nodiscard]] auto apply_adam(Span<float> parameters, ConstSpan<float> gradients, Span<float> m, Span<float> v,
    float learning_rate, float beta1, float beta2, float epsilon, std::uint32_t timestep, float scale = 1.0F) noexcept -> Status;

} // namespace edgeist
