#include "edgeist/runtime/layers/adaptive_avg_pool1d_layer.hpp"

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

auto AdaptiveAvgPool1DLayer::forward(ConstTypedTensorView input, MutableTypedTensorView output, std::uint32_t channels,
    std::uint32_t input_w, std::uint32_t output_w) noexcept -> Status
{
    input.elements = static_cast<std::size_t>(channels) * input_w;
    output.elements = static_cast<std::size_t>(channels) * output_w;
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "adaptive avg pool1d input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "adaptive avg pool1d output invalid"));
    if (channels == 0U || input_w == 0U || output_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool1d dimensions must be non-zero");
    }
    for (std::uint32_t c = 0; c < channels; ++c) {
        for (std::uint32_t ow = 0; ow < output_w; ++ow) {
            const auto start = start_index(ow, output_w, input_w);
            const auto end = end_index(ow, output_w, input_w);
            float sum = 0.0F;
            for (std::uint32_t iw = start; iw < end; ++iw) {
                float value = 0.0F;
                EDGEIST_RETURN_IF_ERROR(read_typed_value(input, static_cast<std::size_t>(c) * input_w + iw, value));
                sum += value;
            }
            EDGEIST_RETURN_IF_ERROR(write_typed_value(output, static_cast<std::size_t>(c) * output_w + ow, sum / static_cast<float>(end - start)));
        }
    }
    return Status::success();
}

auto AdaptiveAvgPool1DLayer::backward(ConstSpan<float> grad_output, Span<float> grad_input, std::uint32_t channels,
    std::uint32_t input_w, std::uint32_t output_w) noexcept -> Status
{
    if (channels == 0U || input_w == 0U || output_w == 0U) {
        return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool1d dimensions must be non-zero");
    }
    const auto input_needed = static_cast<std::size_t>(channels) * input_w;
    const auto output_needed = static_cast<std::size_t>(channels) * output_w;
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_output.size(), output_needed, "adaptive avg pool1d grad_output too small"));
    EDGEIST_RETURN_IF_ERROR(require_span_size(grad_input.size(), input_needed, "adaptive avg pool1d grad_input too small"));
    std::fill(grad_input.begin(), grad_input.begin() + input_needed, 0.0F);

    for (std::uint32_t c = 0; c < channels; ++c) {
        for (std::uint32_t ow = 0; ow < output_w; ++ow) {
            const auto start = start_index(ow, output_w, input_w);
            const auto end = end_index(ow, output_w, input_w);
            const float contribution = grad_output[static_cast<std::size_t>(c) * output_w + ow]
                / static_cast<float>(end - start);
            for (std::uint32_t iw = start; iw < end; ++iw) {
                grad_input[static_cast<std::size_t>(c) * input_w + iw] += contribution;
            }
        }
    }
    return Status::success();
}

} // namespace edgeist::layers
