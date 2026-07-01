#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace edgeist {

template <typename T>
class Span {
public:
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr Span() noexcept = default;
    constexpr Span(T* data, std::size_t size) noexcept
        : data_(data)
        , size_(size)
    {
    }

    template <std::size_t N>
    constexpr Span(T (&arr)[N]) noexcept
        : data_(arr)
        , size_(N)
    {
    }

    template <std::size_t N>
    constexpr Span(std::array<std::remove_const_t<T>, N>& arr) noexcept
        : data_(arr.data())
        , size_(N)
    {
    }

    template <std::size_t N>
    constexpr Span(const std::array<std::remove_const_t<T>, N>& arr) noexcept
        requires std::is_const_v<T>
        : data_(arr.data())
        , size_(N)
    {
    }

    constexpr Span(std::vector<std::remove_const_t<T>>& vec) noexcept
        requires(!std::is_const_v<T>)
        : data_(vec.data())
        , size_(vec.size())
    {
    }

    constexpr Span(const std::vector<std::remove_const_t<T>>& vec) noexcept
        requires std::is_const_v<T>
        : data_(vec.data())
        , size_(vec.size())
    {
    }

    [[nodiscard]] constexpr auto data() const noexcept -> T* { return data_; }
    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return size_; }
    [[nodiscard]] constexpr auto empty() const noexcept -> bool { return size_ == 0; }
    [[nodiscard]] constexpr auto size_bytes() const noexcept -> std::size_t { return size_ * sizeof(T); }

    [[nodiscard]] constexpr auto begin() const noexcept -> iterator { return data_; }
    [[nodiscard]] constexpr auto end() const noexcept -> iterator { return data_ + size_; }

    [[nodiscard]] constexpr auto operator[](std::size_t index) const noexcept -> T& { return data_[index]; }

    [[nodiscard]] constexpr auto subspan(std::size_t offset, std::size_t count) const noexcept -> Span<T>
    {
        if (offset > size_) {
            return {};
        }
        const auto remaining = size_ - offset;
        return Span<T>(data_ + offset, count > remaining ? remaining : count);
    }

private:
    T* data_ { nullptr };
    std::size_t size_ { 0 };
};

using ByteSpan = Span<std::byte>;
using ConstByteSpan = Span<const std::byte>;

template <typename T>
using ConstSpan = Span<const T>;

} // namespace edgeist
