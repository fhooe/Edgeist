#pragma once

#include <cstdint>

#include "edgeist/runtime/numeric.hpp"
#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"

namespace edgeist::layers {

struct Conv2DParams {
    std::uint32_t channels_in { 1 };
    std::uint32_t channels_out { 1 };
    std::uint32_t input_h { 1 };
    std::uint32_t input_w { 1 };
    std::uint32_t output_h { 1 };
    std::uint32_t output_w { 1 };
    std::uint32_t kernel_h { 1 };
    std::uint32_t kernel_w { 1 };
    std::uint32_t pad_h { 0 };
    std::uint32_t pad_w { 0 };
    std::uint32_t stride_h { 1 };
    std::uint32_t stride_w { 1 };
    std::uint32_t dilation_h { 1 };
    std::uint32_t dilation_w { 1 };
    std::uint32_t groups { 1 };
};

struct Pool2DParams {
    std::uint32_t channels { 1 };
    std::uint32_t input_h { 1 };
    std::uint32_t input_w { 1 };
    std::uint32_t output_h { 1 };
    std::uint32_t output_w { 1 };
    std::uint32_t kernel_h { 1 };
    std::uint32_t kernel_w { 1 };
    std::uint32_t pad_h { 0 };
    std::uint32_t pad_w { 0 };
    std::uint32_t stride_h { 1 };
    std::uint32_t stride_w { 1 };
};

[[nodiscard]] auto require_span_size(std::size_t actual, std::size_t needed, const char* message) noexcept -> Status;
[[nodiscard]] auto input_index(std::uint32_t c, std::uint32_t y, std::uint32_t x, std::uint32_t height, std::uint32_t width) noexcept -> std::size_t;
[[nodiscard]] auto conv_weight_index(std::uint32_t oc, std::uint32_t ic_per_group, std::uint32_t ky, std::uint32_t kx,
    std::uint32_t channels_per_group, std::uint32_t kernel_h, std::uint32_t kernel_w) noexcept -> std::size_t;

} // namespace edgeist::layers
