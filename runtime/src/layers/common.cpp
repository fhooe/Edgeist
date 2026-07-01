#include "edgeist/runtime/layers/common.hpp"

namespace edgeist::layers {

auto require_span_size(std::size_t actual, std::size_t needed, const char* message) noexcept -> Status
{
    if (actual < needed) {
        return make_status(ErrorCode::BufferTooSmall, message);
    }
    return Status::success();
}

auto input_index(std::uint32_t c, std::uint32_t y, std::uint32_t x, std::uint32_t height, std::uint32_t width) noexcept -> std::size_t
{
    return (static_cast<std::size_t>(c) * height + y) * width + x;
}

auto conv_weight_index(std::uint32_t oc, std::uint32_t ic_per_group, std::uint32_t ky, std::uint32_t kx,
    std::uint32_t channels_per_group, std::uint32_t kernel_h, std::uint32_t kernel_w) noexcept -> std::size_t
{
    return ((static_cast<std::size_t>(oc) * channels_per_group + ic_per_group) * kernel_h + ky) * kernel_w + kx;
}

} // namespace edgeist::layers
