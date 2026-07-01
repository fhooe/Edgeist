#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace edgeist {

[[nodiscard]] constexpr auto checked_add(std::size_t a, std::size_t b, std::size_t& out) noexcept -> bool
{
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

[[nodiscard]] constexpr auto checked_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept -> bool
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] constexpr auto checked_add_mul(std::size_t base, std::size_t count, std::size_t element_size, std::size_t& out) noexcept -> bool
{
    std::size_t bytes = 0;
    if (!checked_mul(count, element_size, bytes)) {
        return false;
    }
    return checked_add(base, bytes, out);
}

[[nodiscard]] constexpr auto align_up(std::size_t value, std::size_t alignment) noexcept -> std::size_t
{
    if (alignment == 0) {
        return value;
    }
    const auto mask = alignment - 1U;
    return (value + mask) & ~mask;
}

} // namespace edgeist
