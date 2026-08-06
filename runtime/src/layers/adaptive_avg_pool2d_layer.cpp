#include "edgeist/runtime/layers/adaptive_avg_pool2d_layer.hpp"

#include <algorithm>
#include <cstdint>

namespace edgeist::layers {

namespace {
[[nodiscard]] auto start_index(std::uint32_t out_idx, std::uint32_t out_size, std::uint32_t in_size) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(out_idx) * in_size) / out_size);
}

[[nodiscard]] auto end_index(std::uint32_t out_idx, std::uint32_t out_size, std::uint32_t in_size) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>(((static_cast<std::uint64_t>(out_idx) + 1U) * in_size + out_size - 1U) / out_size);
}
} // namespace

auto AdaptiveAvgPool2DLayer::forward(ConstTypedTensorView input, MutableTypedTensorView output, std::uint32_t channels,
    std::uint32_t input_h, std::uint32_t input_w, std::uint32_t output_h, std::uint32_t output_w) noexcept -> Status
{
    input.elements = static_cast<std::size_t>(channels) * input_h * input_w;
    output.elements = static_cast<std::size_t>(channels) * output_h * output_w;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "adaptive avg pool2d input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "adaptive avg pool2d output invalid"));
    if (channels == 0U || input_h == 0U || input_w == 0U || output_h == 0U || output_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool2d dimensions must be non-zero");
    }
    for (std::uint32_t c = 0; c < channels; ++c) {
        for (std::uint32_t oy = 0; oy < output_h; ++oy) {
            const auto y0 = start_index(oy, output_h, input_h);
            const auto y1 = end_index(oy, output_h, input_h);
            for (std::uint32_t ox = 0; ox < output_w; ++ox) {
                const auto x0 = start_index(ox, output_w, input_w);
                const auto x1 = end_index(ox, output_w, input_w);
                float sum = 0.0F;
                for (std::uint32_t iy = y0; iy < y1; ++iy) {
                    for (std::uint32_t ix = x0; ix < x1; ++ix) {
                        float value = 0.0F;
                        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, input_index(c, iy, ix, input_h, input_w), value));
                        sum += value;
                    }
                }
                const auto count = static_cast<float>((y1 - y0) * (x1 - x0));
                EDGEIST_RETURN_IF_ERROR(write_typed_value(output, input_index(c, oy, ox, output_h, output_w), sum / count));
            }
        }
    }
    return Status::success();
}

auto AdaptiveAvgPool2DLayer::backward(ConstSpan<float> grad_output, Span<float> grad_input, std::uint32_t channels,
    std::uint32_t input_h, std::uint32_t input_w, std::uint32_t output_h, std::uint32_t output_w) noexcept -> Status
{
    if (channels == 0U || input_h == 0U || input_w == 0U || output_h == 0U || output_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool2d dimensions must be non-zero");
    }
    const auto input_needed = static_cast<std::size_t>(channels) * input_h * input_w;
    const auto output_needed = static_cast<std::size_t>(channels) * output_h * output_w;
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), output_needed, "adaptive avg pool2d grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input_needed, "adaptive avg pool2d grad_input too small"));
    std::fill(grad_input.begin(), grad_input.begin() + input_needed, 0.0F);

    for (std::uint32_t c = 0; c < channels; ++c) {
        for (std::uint32_t oy = 0; oy < output_h; ++oy) {
            const auto y0 = start_index(oy, output_h, input_h);
            const auto y1 = end_index(oy, output_h, input_h);
            for (std::uint32_t ox = 0; ox < output_w; ++ox) {
                const auto x0 = start_index(ox, output_w, input_w);
                const auto x1 = end_index(ox, output_w, input_w);
                const auto count = static_cast<float>((y1 - y0) * (x1 - x0));
                const float contribution = grad_output[input_index(c, oy, ox, output_h, output_w)] / count;
                for (std::uint32_t iy = y0; iy < y1; ++iy) {
                    for (std::uint32_t ix = x0; ix < x1; ++ix) {
                        grad_input[input_index(c, iy, ix, input_h, input_w)] += contribution;
                    }
                }
            }
        }
    }
    return Status::success();
}

} // namespace edgeist::layers
