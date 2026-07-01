#pragma once

#include <cstddef>
#include <cstdint>

#include "edgeist/runtime/checked_math.hpp"
#include "edgeist/runtime/status.hpp"

namespace edgeist {

struct TensorShape {
    std::uint32_t channels { 1 };
    std::uint32_t height { 1 };
    std::uint32_t width { 1 };

    [[nodiscard]] auto elements(std::size_t& out) const noexcept -> Status
    {
        if (channels == 0 || height == 0 || width == 0) {
            return make_status(ErrorCode::ShapeMismatch, "tensor dimensions must be non-zero");
        }
        std::size_t tmp = 0;
        if (!checked_mul(static_cast<std::size_t>(channels), static_cast<std::size_t>(height), tmp)
            || !checked_mul(tmp, static_cast<std::size_t>(width), out)) {
            return make_status(ErrorCode::OutOfBounds, "tensor element count overflow");
        }
        return Status::success();
    }
};

[[nodiscard]] inline auto tensor_elements_or_zero(const TensorShape& shape) noexcept -> std::size_t
{
    std::size_t out = 0;
    return shape.elements(out).ok() ? out : 0U;
}

} // namespace edgeist
