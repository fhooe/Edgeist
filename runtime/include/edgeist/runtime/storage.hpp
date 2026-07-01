#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"

namespace edgeist {

class StorageBackend {
public:
    virtual ~StorageBackend() = default;
    [[nodiscard]] virtual auto size() const noexcept -> std::size_t = 0;
    [[nodiscard]] virtual auto read(std::size_t offset, ByteSpan destination) const noexcept -> Status = 0;
    [[nodiscard]] virtual auto write(std::size_t offset, ConstByteSpan source) noexcept -> Status = 0;
    [[nodiscard]] virtual auto writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte* = 0;
    [[nodiscard]] virtual auto readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte* = 0;
};

class SramStorage final : public StorageBackend {
public:
    explicit SramStorage(ByteSpan buffer) noexcept
        : buffer_(buffer)
    {
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t override { return buffer_.size(); }
    [[nodiscard]] auto read(std::size_t offset, ByteSpan destination) const noexcept -> Status override;
    [[nodiscard]] auto write(std::size_t offset, ConstByteSpan source) noexcept -> Status override;
    [[nodiscard]] auto writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte* override;
    [[nodiscard]] auto readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte* override;

private:
    ByteSpan buffer_ {};
};

class FlashStorage final : public StorageBackend {
public:
    explicit FlashStorage(ConstByteSpan image) noexcept
        : image_(image)
    {
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t override { return image_.size(); }
    [[nodiscard]] auto read(std::size_t offset, ByteSpan destination) const noexcept -> Status override;
    [[nodiscard]] auto write(std::size_t offset, ConstByteSpan source) noexcept -> Status override;
    [[nodiscard]] auto writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte* override;
    [[nodiscard]] auto readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte* override;

private:
    ConstByteSpan image_ {};
};

class ExternalMemoryStorage final : public StorageBackend {
public:
    explicit ExternalMemoryStorage(std::vector<std::byte> backing = {})
        : backing_(std::move(backing))
    {
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t override { return backing_.size(); }
    [[nodiscard]] auto read(std::size_t offset, ByteSpan destination) const noexcept -> Status override;
    [[nodiscard]] auto write(std::size_t offset, ConstByteSpan source) noexcept -> Status override;
    [[nodiscard]] auto writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte* override;
    [[nodiscard]] auto readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte* override;
    auto resize(std::size_t bytes) -> void { backing_.resize(bytes); }

private:
    std::vector<std::byte> backing_;
};

[[nodiscard]] auto copy_to_storage(StorageBackend& storage, std::size_t offset, ConstByteSpan source) noexcept -> Status;

} // namespace edgeist
