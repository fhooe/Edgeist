#include "edgeist/runtime/layers/adaptive_avg_pool1d_layer.hpp"

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

} // namespace edgeist::layers
