#include "edgeist/runtime/layers/batch_norm_layer.hpp"

#include <algorithm>
#include <cmath>

namespace edgeist::layers {

auto BatchNormLayer::forward(ConstTypedTensorView input, ConstTypedTensorView gamma, ConstTypedTensorView beta,
    ConstTypedTensorView mean, ConstTypedTensorView variance, MutableTypedTensorView output, float epsilon) noexcept -> Status
{
    if (input.elements == 0U || output.elements != input.elements) {
        return make_status(ErrorCode::ShapeMismatch, "batchnorm input/output element counts invalid");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "batchnorm input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "batchnorm output invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(gamma, "batchnorm gamma invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(beta, "batchnorm beta invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(mean, "batchnorm mean invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(variance, "batchnorm variance invalid"));
    if (gamma.elements == 0U || beta.elements < gamma.elements || mean.elements < gamma.elements || variance.elements < gamma.elements) {
        return make_status(ErrorCode::ShapeMismatch, "batchnorm parameter element counts invalid");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_status(ErrorCode::InvalidArgument, "batchnorm epsilon must be positive");
    }

    const auto spatial = input.elements / gamma.elements;
    if (spatial == 0U || spatial * gamma.elements != input.elements) {
        return make_status(ErrorCode::ShapeMismatch, "batchnorm input elements must be divisible by channels");
    }

    for (std::size_t c = 0; c < gamma.elements; ++c) {
        float g = 0.0F;
        float b = 0.0F;
        float m = 0.0F;
        float v = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(gamma, c, g));
        EDGEIST_RETURN_IF_ERROR(read_typed_value(beta, c, b));
        EDGEIST_RETURN_IF_ERROR(read_typed_value(mean, c, m));
        EDGEIST_RETURN_IF_ERROR(read_typed_value(variance, c, v));
        const float inv = 1.0F / std::sqrt(v + epsilon);
        for (std::size_t s = 0; s < spatial; ++s) {
            const auto idx = c * spatial + s;
            float x = 0.0F;
            EDGEIST_RETURN_IF_ERROR(read_typed_value(input, idx, x));
            EDGEIST_RETURN_IF_ERROR(write_typed_value(output, idx, (x - m) * inv * g + b));
        }
    }
    return Status::success();
}

auto BatchNormLayer::backward_affine(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView gamma,
    ConstTypedTensorView mean, ConstTypedTensorView variance, Span<float> grad_input, Span<float> grad_gamma,
    Span<float> grad_beta, float epsilon, bool accumulate, bool compute_parameter_gradients) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "batchnorm backward input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(gamma, "batchnorm backward gamma invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(mean, "batchnorm backward mean invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(variance, "batchnorm backward variance invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), input.elements, "batchnorm grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input.elements, "batchnorm grad_input too small"));
    if (compute_parameter_gradients) {
        EDGEIST_RETURN_IF_ERROR(require_span_size(grad_gamma.size(), gamma.elements, "batchnorm grad_gamma too small"));
        EDGEIST_RETURN_IF_ERROR(require_span_size(grad_beta.size(), gamma.elements, "batchnorm grad_beta too small"));
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return make_status(ErrorCode::InvalidArgument, "batchnorm epsilon must be positive");
    }
    const auto spatial = input.elements / gamma.elements;
    if (spatial == 0U || spatial * gamma.elements != input.elements) {
        return make_status(ErrorCode::ShapeMismatch, "batchnorm input elements must be divisible by channels");
    }
    if (compute_parameter_gradients && !accumulate) {
        std::fill(grad_gamma.begin(), grad_gamma.begin() + gamma.elements, 0.0F);
        std::fill(grad_beta.begin(), grad_beta.begin() + gamma.elements, 0.0F);
    }
    for (std::size_t c = 0; c < gamma.elements; ++c) {
        float g = 0.0F;
        float m = 0.0F;
        float v = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(gamma, c, g));
        EDGEIST_RETURN_IF_ERROR(read_typed_value(mean, c, m));
        EDGEIST_RETURN_IF_ERROR(read_typed_value(variance, c, v));
        const float inv = 1.0F / std::sqrt(v + epsilon);
        for (std::size_t s = 0; s < spatial; ++s) {
            const auto idx = c * spatial + s;
            const float go = grad_output[idx];
            grad_input[idx] = go * g * inv;
            if (compute_parameter_gradients) {
                float x = 0.0F;
                EDGEIST_RETURN_IF_ERROR(read_typed_value(input, idx, x));
                grad_gamma[c] += go * (x - m) * inv;
                grad_beta[c] += go;
            }
        }
    }
    return Status::success();
}

} // namespace edgeist::layers
