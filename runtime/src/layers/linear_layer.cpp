#include "edgeist/runtime/layers/linear_layer.hpp"

#include <algorithm>
#include <cstdint>

namespace edgeist::layers {

namespace {
[[nodiscard]] auto all_int8(ConstTypedTensorView input, ConstTypedTensorView weights, MutableTypedTensorView output) noexcept -> bool
{
    return input.type == NumericDataType::Int8 && weights.type == NumericDataType::Int8 && output.type == NumericDataType::Int8;
}
} // namespace

auto LinearLayer::forward(ConstTypedTensorView input, ConstTypedTensorView weights, ConstTypedTensorView bias,
    MutableTypedTensorView output, std::uint32_t input_size, std::uint32_t output_size) noexcept -> Status
{
    input.elements = input_size;
    weights.elements = static_cast<std::size_t>(input_size) * output_size;
    bias.elements = output_size;
    output.elements = output_size;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "linear input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(weights, "linear weight tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(bias, "linear bias tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "linear output tensor invalid"));

    if (all_int8(input, weights, output)) {
        const auto* x = reinterpret_cast<const std::int8_t*>(input.bytes.data());
        const auto* w = reinterpret_cast<const std::int8_t*>(weights.bytes.data());
        for (std::uint32_t out = 0; out < output_size; ++out) {
            std::int32_t acc_q = 0;
            const auto row = static_cast<std::size_t>(out) * input_size;
            for (std::uint32_t in = 0; in < input_size; ++in) {
                acc_q += static_cast<std::int32_t>(x[in]) * static_cast<std::int32_t>(w[row + in]);
            }
            float bias_value = 0.0F;
            EDGEIST_RETURN_IF_ERROR(read_typed_value(bias, out, bias_value));
            const float acc = static_cast<float>(acc_q) * input.scale * weights.scale + bias_value;
            EDGEIST_RETURN_IF_ERROR(write_typed_value(output, out, acc));
        }
        return Status::success();
    }

    for (std::uint32_t out = 0; out < output_size; ++out) {
        float acc = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(bias, out, acc));
        const auto row = static_cast<std::size_t>(out) * input_size;
        for (std::uint32_t in = 0; in < input_size; ++in) {
            float x = 0.0F;
            float w = 0.0F;
            EDGEIST_RETURN_IF_ERROR(read_typed_value(input, in, x));
            EDGEIST_RETURN_IF_ERROR(read_typed_value(weights, row + in, w));
            acc += x * w;
        }
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, out, acc));
    }
    return Status::success();
}

auto LinearLayer::backward(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView weights,
    Span<float> grad_input, Span<float> grad_weights, Span<float> grad_bias,
    std::uint32_t input_size, std::uint32_t output_size, bool accumulate) noexcept -> Status
{
    input.elements = input_size;
    weights.elements = static_cast<std::size_t>(input_size) * output_size;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "linear backward input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(weights, "linear backward weight tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), output_size, "linear backward grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input_size, "linear backward grad_input too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_weights.size(), static_cast<std::size_t>(input_size) * output_size, "linear backward grad_weights too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_bias.size(), output_size, "linear backward grad_bias too small"));

    std::fill(grad_input.begin(), grad_input.begin() + input_size, 0.0F);
    if (!accumulate) {
        std::fill(grad_weights.begin(), grad_weights.begin() + static_cast<std::size_t>(input_size) * output_size, 0.0F);
        std::fill(grad_bias.begin(), grad_bias.begin() + output_size, 0.0F);
    }

    for (std::uint32_t out = 0; out < output_size; ++out) {
        const float go = grad_output[out];
        grad_bias[out] += go;
        const auto row = static_cast<std::size_t>(out) * input_size;
        for (std::uint32_t in = 0; in < input_size; ++in) {
            float x = 0.0F;
            float w = 0.0F;
            EDGEIST_RETURN_IF_ERROR(read_typed_value(input, in, x));
            EDGEIST_RETURN_IF_ERROR(read_typed_value(weights, row + in, w));
            grad_weights[row + in] += go * x;
            grad_input[in] += go * w;
        }
    }
    return Status::success();
}

} // namespace edgeist::layers
