#include "edgeist/runtime/optimizers.hpp"

#include <cmath>

namespace edgeist {

auto optimizer_state_elements(OptimizerKind kind, std::size_t parameter_count) noexcept -> std::size_t
{
    switch (kind) {
    case OptimizerKind::Sgd:
        return 0;
    case OptimizerKind::Momentum:
        return parameter_count;
    case OptimizerKind::Adam:
        return parameter_count * 2U;
    }
    return 0;
}

auto apply_sgd(Span<float> parameters, ConstSpan<float> gradients, float learning_rate, float scale) noexcept -> Status
{
    if (parameters.size() != gradients.size()) {
        return make_status(ErrorCode::ShapeMismatch, "SGD parameter/gradient size mismatch");
    }
    if (parameters.data() == nullptr || gradients.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "SGD span is null");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] -= learning_rate * gradients[i] * scale;
    }
    return Status::success();
}

auto apply_momentum(Span<float> parameters, ConstSpan<float> gradients, Span<float> velocity,
    float learning_rate, float momentum, float scale) noexcept -> Status
{
    if (parameters.size() != gradients.size() || parameters.size() != velocity.size()) {
        return make_status(ErrorCode::ShapeMismatch, "Momentum optimizer size mismatch");
    }
    if (parameters.data() == nullptr || gradients.data() == nullptr || velocity.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "Momentum optimizer span is null");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        velocity[i] = momentum * velocity[i] + gradients[i] * scale;
        parameters[i] -= learning_rate * velocity[i];
    }
    return Status::success();
}

auto apply_adam(Span<float> parameters, ConstSpan<float> gradients, Span<float> m, Span<float> v,
    float learning_rate, float beta1, float beta2, float epsilon, std::uint32_t timestep, float scale) noexcept -> Status
{
    if (parameters.size() != gradients.size() || parameters.size() != m.size() || parameters.size() != v.size()) {
        return make_status(ErrorCode::ShapeMismatch, "Adam optimizer size mismatch");
    }
    if (parameters.data() == nullptr || gradients.data() == nullptr || m.data() == nullptr || v.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "Adam optimizer span is null");
    }
    const auto t = static_cast<float>(timestep == 0 ? 1U : timestep);
    const float beta1_corr = 1.0F - std::pow(beta1, t);
    const float beta2_corr = 1.0F - std::pow(beta2, t);
    if (beta1_corr == 0.0F || beta2_corr == 0.0F) {
        return make_status(ErrorCode::NumericError, "Adam bias correction underflow");
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const float g = gradients[i] * scale;
        m[i] = beta1 * m[i] + (1.0F - beta1) * g;
        v[i] = beta2 * v[i] + (1.0F - beta2) * g * g;
        const float m_hat = m[i] / beta1_corr;
        const float v_hat = v[i] / beta2_corr;
        parameters[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    return Status::success();
}

auto apply_optimizer(OptimizerKind kind, Span<float> parameters, ConstSpan<float> gradients,
    OptimizerStateView state, const TrainingConfig& config, float scale) noexcept -> Status
{
    switch (kind) {
    case OptimizerKind::Sgd:
        return apply_sgd(parameters, gradients, config.learning_rate, scale);
    case OptimizerKind::Momentum:
        return apply_momentum(parameters, gradients, state.momentum_or_m, config.learning_rate, config.momentum, scale);
    case OptimizerKind::Adam:
        return apply_adam(parameters, gradients, state.momentum_or_m, state.velocity_or_v,
            config.learning_rate, config.beta1, config.beta2, config.epsilon, state.timestep, scale);
    }
    return make_status(ErrorCode::InvalidArgument, "unknown optimizer kind");
}

} // namespace edgeist
