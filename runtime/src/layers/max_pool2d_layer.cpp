#include "edgeist/runtime/layers/max_pool2d_layer.hpp"

#include <algorithm>
#include <limits>

namespace edgeist::layers {

auto MaxPool2DLayer::forward(ConstTypedTensorView input, MutableTypedTensorView output, Span<std::uint32_t> argmax,
    const Pool2DParams& p) noexcept -> Status
{
    const auto input_needed = static_cast<std::size_t>(p.channels) * p.input_h * p.input_w;
    const auto output_needed = static_cast<std::size_t>(p.channels) * p.output_h * p.output_w;
    input.elements = input_needed;
    output.elements = output_needed;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "maxpool input tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "maxpool output tensor invalid"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(argmax.size(), output_needed, "maxpool argmax too small"));
    if (p.kernel_h == 0U || p.kernel_w == 0U || p.stride_h == 0U || p.stride_w == 0U || p.dilation_h == 0U || p.dilation_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "maxpool kernel, stride, and dilation must be non-zero");
    }

    for (std::uint32_t c = 0; c < p.channels; ++c) {
        for (std::uint32_t oy = 0; oy < p.output_h; ++oy) {
            for (std::uint32_t ox = 0; ox < p.output_w; ++ox) {
                float best = -std::numeric_limits<float>::infinity();
                std::uint32_t best_index = 0;
                for (std::uint32_t ky = 0; ky < p.kernel_h; ++ky) {
                    for (std::uint32_t kx = 0; kx < p.kernel_w; ++kx) {
                        const auto in_y = static_cast<int>(oy * p.stride_h + ky * p.dilation_h) - static_cast<int>(p.pad_h);
                        const auto in_x = static_cast<int>(ox * p.stride_w + kx * p.dilation_w) - static_cast<int>(p.pad_w);
                        if (in_y < 0 || in_x < 0 || in_y >= static_cast<int>(p.input_h) || in_x >= static_cast<int>(p.input_w)) {
                            continue;
                        }
                        const auto idx = input_index(c, static_cast<std::uint32_t>(in_y), static_cast<std::uint32_t>(in_x), p.input_h, p.input_w);
                        float value = 0.0F;
                        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, idx, value));
                        if (value > best) {
                            best = value;
                            best_index = static_cast<std::uint32_t>(idx);
                        }
                    }
                }
                const auto out_idx = input_index(c, oy, ox, p.output_h, p.output_w);
                EDGEIST_RETURN_IF_ERROR(write_typed_value(output, out_idx, best));
                argmax[out_idx] = best_index;
            }
        }
    }
    return Status::success();
}

auto MaxPool2DLayer::backward(ConstSpan<float> grad_output, ConstSpan<std::uint32_t> argmax, Span<float> grad_input,
    const Pool2DParams& p) noexcept -> Status
{
    const auto input_needed = static_cast<std::size_t>(p.channels) * p.input_h * p.input_w;
    const auto output_needed = static_cast<std::size_t>(p.channels) * p.output_h * p.output_w;
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), output_needed, "maxpool grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(argmax.size(), output_needed, "maxpool argmax too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input_needed, "maxpool grad_input too small"));
    std::fill(grad_input.begin(), grad_input.begin() + input_needed, 0.0F);
    for (std::size_t i = 0; i < output_needed; ++i) {
        if (argmax[i] >= input_needed) {
            return make_status(ErrorCode::OutOfBounds, "maxpool argmax index outside input");
        }
        grad_input[argmax[i]] += grad_output[i];
    }
    return Status::success();
}

} // namespace edgeist::layers
