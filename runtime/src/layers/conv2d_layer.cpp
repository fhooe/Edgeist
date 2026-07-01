#include "edgeist/runtime/layers/conv2d_layer.hpp"

#include <algorithm>
#include <cstdint>

namespace edgeist::layers {

namespace {
[[nodiscard]] auto validate_conv_params(const Conv2DParams& p) noexcept -> Status
{
    if (p.groups == 0U || p.channels_in % p.groups != 0U || p.channels_out % p.groups != 0U) {
        return make_status(ErrorCode::ShapeMismatch, "conv2d groups do not divide channel counts");
    }
    if (p.kernel_h == 0U || p.kernel_w == 0U || p.stride_h == 0U || p.stride_w == 0U || p.dilation_h == 0U || p.dilation_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "conv2d kernel/stride/dilation must be non-zero");
    }
    return Status::success();
}
} // namespace

auto Conv2DLayer::forward(ConstTypedTensorView input, ConstTypedTensorView weights, ConstTypedTensorView bias,
    MutableTypedTensorView output, const Conv2DParams& p) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_conv_params(p));
    const auto input_needed = static_cast<std::size_t>(p.channels_in) * p.input_h * p.input_w;
    const auto output_needed = static_cast<std::size_t>(p.channels_out) * p.output_h * p.output_w;
    const auto channels_per_group = p.channels_in / p.groups;
    const auto weights_needed = static_cast<std::size_t>(p.channels_out) * channels_per_group * p.kernel_h * p.kernel_w;
    input.elements = input_needed;
    output.elements = output_needed;
    weights.elements = weights_needed;
    bias.elements = p.channels_out;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "conv2d input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(weights, "conv2d weight tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(bias, "conv2d bias tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "conv2d output tensor invalid"));

    const auto out_per_group = p.channels_out / p.groups;
    const bool int8_fast = input.type == NumericDataType::Int8 && weights.type == NumericDataType::Int8 && output.type == NumericDataType::Int8;
    const auto* x8 = int8_fast ? reinterpret_cast<const std::int8_t*>(input.bytes.data()) : nullptr;
    const auto* w8 = int8_fast ? reinterpret_cast<const std::int8_t*>(weights.bytes.data()) : nullptr;

    for (std::uint32_t g = 0; g < p.groups; ++g) {
        for (std::uint32_t ocg = 0; ocg < out_per_group; ++ocg) {
            const auto oc = g * out_per_group + ocg;
            for (std::uint32_t oy = 0; oy < p.output_h; ++oy) {
                for (std::uint32_t ox = 0; ox < p.output_w; ++ox) {
                    float bias_value = 0.0F;
                    EDGEIST_RETURN_IF_ERROR(read_typed_value(bias, oc, bias_value));
                    float acc = bias_value;
                    std::int32_t acc_q = 0;
                    for (std::uint32_t icg = 0; icg < channels_per_group; ++icg) {
                        const auto ic = g * channels_per_group + icg;
                        for (std::uint32_t ky = 0; ky < p.kernel_h; ++ky) {
                            for (std::uint32_t kx = 0; kx < p.kernel_w; ++kx) {
                                const auto in_y = static_cast<int>(oy * p.stride_h + ky * p.dilation_h) - static_cast<int>(p.pad_h);
                                const auto in_x = static_cast<int>(ox * p.stride_w + kx * p.dilation_w) - static_cast<int>(p.pad_w);
                                if (in_y < 0 || in_x < 0 || in_y >= static_cast<int>(p.input_h) || in_x >= static_cast<int>(p.input_w)) {
                                    continue;
                                }
                                const auto ii = input_index(ic, static_cast<std::uint32_t>(in_y), static_cast<std::uint32_t>(in_x), p.input_h, p.input_w);
                                const auto wi = conv_weight_index(oc, icg, ky, kx, channels_per_group, p.kernel_h, p.kernel_w);
                                if (int8_fast) {
                                    acc_q += static_cast<std::int32_t>(x8[ii]) * static_cast<std::int32_t>(w8[wi]);
                                } else {
                                    float xv = 0.0F;
                                    float wv = 0.0F;
                                    EDGEIST_RETURN_IF_ERROR(read_typed_value(input, ii, xv));
                                    EDGEIST_RETURN_IF_ERROR(read_typed_value(weights, wi, wv));
                                    acc += xv * wv;
                                }
                            }
                        }
                    }
                    if (int8_fast) {
                        acc += static_cast<float>(acc_q) * input.scale * weights.scale;
                    }
                    EDGEIST_RETURN_IF_ERROR(write_typed_value(output, input_index(oc, oy, ox, p.output_h, p.output_w), acc));
                }
            }
        }
    }
    return Status::success();
}

auto Conv2DLayer::backward(ConstTypedTensorView input, ConstSpan<float> grad_output, ConstTypedTensorView weights,
    Span<float> grad_input, Span<float> grad_weights, Span<float> grad_bias,
    const Conv2DParams& p, bool accumulate) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_conv_params(p));
    const auto input_needed = static_cast<std::size_t>(p.channels_in) * p.input_h * p.input_w;
    const auto output_needed = static_cast<std::size_t>(p.channels_out) * p.output_h * p.output_w;
    const auto channels_per_group = p.channels_in / p.groups;
    const auto weights_needed = static_cast<std::size_t>(p.channels_out) * channels_per_group * p.kernel_h * p.kernel_w;
    input.elements = input_needed;
    weights.elements = weights_needed;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "conv2d backward input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(weights, "conv2d backward weight tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), output_needed, "conv2d backward grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input_needed, "conv2d backward grad_input too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_weights.size(), weights_needed, "conv2d backward grad_weights too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_bias.size(), p.channels_out, "conv2d backward grad_bias too small"));

    std::fill(grad_input.begin(), grad_input.begin() + input_needed, 0.0F);
    if (!accumulate) {
        std::fill(grad_weights.begin(), grad_weights.begin() + weights_needed, 0.0F);
        std::fill(grad_bias.begin(), grad_bias.begin() + p.channels_out, 0.0F);
    }

    const auto out_per_group = p.channels_out / p.groups;
    for (std::uint32_t g = 0; g < p.groups; ++g) {
        for (std::uint32_t ocg = 0; ocg < out_per_group; ++ocg) {
            const auto oc = g * out_per_group + ocg;
            for (std::uint32_t oy = 0; oy < p.output_h; ++oy) {
                for (std::uint32_t ox = 0; ox < p.output_w; ++ox) {
                    const float go = grad_output[input_index(oc, oy, ox, p.output_h, p.output_w)];
                    grad_bias[oc] += go;
                    for (std::uint32_t icg = 0; icg < channels_per_group; ++icg) {
                        const auto ic = g * channels_per_group + icg;
                        for (std::uint32_t ky = 0; ky < p.kernel_h; ++ky) {
                            for (std::uint32_t kx = 0; kx < p.kernel_w; ++kx) {
                                const auto in_y = static_cast<int>(oy * p.stride_h + ky * p.dilation_h) - static_cast<int>(p.pad_h);
                                const auto in_x = static_cast<int>(ox * p.stride_w + kx * p.dilation_w) - static_cast<int>(p.pad_w);
                                if (in_y < 0 || in_x < 0 || in_y >= static_cast<int>(p.input_h) || in_x >= static_cast<int>(p.input_w)) {
                                    continue;
                                }
                                const auto ii = input_index(ic, static_cast<std::uint32_t>(in_y), static_cast<std::uint32_t>(in_x), p.input_h, p.input_w);
                                const auto wi = conv_weight_index(oc, icg, ky, kx, channels_per_group, p.kernel_h, p.kernel_w);
                                float xv = 0.0F;
                                float wv = 0.0F;
                                EDGEIST_RETURN_IF_ERROR(read_typed_value(input, ii, xv));
                                EDGEIST_RETURN_IF_ERROR(read_typed_value(weights, wi, wv));
                                grad_weights[wi] += go * xv;
                                grad_input[ii] += go * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    return Status::success();
}

} // namespace edgeist::layers
