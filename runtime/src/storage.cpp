#include "edgeist/runtime/storage.hpp"

#include <algorithm>
#include <cstring>

namespace edgeist {

namespace {
[[nodiscard]] auto range_valid(std::size_t size, std::size_t offset, std::size_t bytes) noexcept -> bool
{
    return offset <= size && bytes <= (size - offset);
}
} // namespace

auto SramStorage::read(std::size_t offset, ByteSpan destination) const noexcept -> Status
{
    if (destination.size() == 0U) {
        return Status::success();
    }
    if (destination.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "destination buffer is null");
    }
    if (!range_valid(buffer_.size(), offset, destination.size())) {
        return make_status(ErrorCode::OutOfBounds, "SRAM read outside storage range");
    }
    std::memcpy(destination.data(), buffer_.data() + offset, destination.size());
    return Status::success();
}

auto SramStorage::write(std::size_t offset, ConstByteSpan source) noexcept -> Status
{
    if (source.size() == 0U) {
        return Status::success();
    }
    if (source.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "source buffer is null");
    }
    if (!range_valid(buffer_.size(), offset, source.size())) {
        return make_status(ErrorCode::OutOfBounds, "SRAM write outside storage range");
    }
    std::memcpy(buffer_.data() + offset, source.data(), source.size());
    return Status::success();
}

auto SramStorage::writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte*
{
    if (!range_valid(buffer_.size(), offset, bytes)) {
        return nullptr;
    }
    return buffer_.data() + offset;
}

auto SramStorage::readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte*
{
    if (!range_valid(buffer_.size(), offset, bytes)) {
        return nullptr;
    }
    return buffer_.data() + offset;
}

auto FlashStorage::read(std::size_t offset, ByteSpan destination) const noexcept -> Status
{
    if (destination.size() == 0U) {
        return Status::success();
    }
    if (destination.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "destination buffer is null");
    }
    if (!range_valid(image_.size(), offset, destination.size())) {
        return make_status(ErrorCode::OutOfBounds, "Flash read outside storage range");
    }
    std::memcpy(destination.data(), image_.data() + offset, destination.size());
    return Status::success();
}

auto FlashStorage::write(std::size_t, ConstByteSpan) noexcept -> Status
{
    return make_status(ErrorCode::StorageError, "FlashStorage is read-only; use a flash driver or staging backend for updates");
}

auto FlashStorage::writable_data(std::size_t, std::size_t) noexcept -> std::byte*
{
    return nullptr;
}

auto FlashStorage::readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte*
{
    if (!range_valid(image_.size(), offset, bytes)) {
        return nullptr;
    }
    return image_.data() + offset;
}

auto ExternalMemoryStorage::read(std::size_t offset, ByteSpan destination) const noexcept -> Status
{
    if (destination.size() == 0U) {
        return Status::success();
    }
    if (destination.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "destination buffer is null");
    }
    if (!range_valid(backing_.size(), offset, destination.size())) {
        return make_status(ErrorCode::OutOfBounds, "external memory read outside storage range");
    }
    std::memcpy(destination.data(), backing_.data() + offset, destination.size());
    return Status::success();
}

auto ExternalMemoryStorage::write(std::size_t offset, ConstByteSpan source) noexcept -> Status
{
    if (source.size() == 0U) {
        return Status::success();
    }
    if (source.data() == nullptr) {
        return make_status(ErrorCode::NullPointer, "source buffer is null");
    }
    if (!range_valid(backing_.size(), offset, source.size())) {
        return make_status(ErrorCode::OutOfBounds, "external memory write outside storage range");
    }
    std::memcpy(backing_.data() + offset, source.data(), source.size());
    return Status::success();
}

auto ExternalMemoryStorage::writable_data(std::size_t offset, std::size_t bytes) noexcept -> std::byte*
{
    if (!range_valid(backing_.size(), offset, bytes)) {
        return nullptr;
    }
    return backing_.data() + offset;
}

auto ExternalMemoryStorage::readable_data(std::size_t offset, std::size_t bytes) const noexcept -> const std::byte*
{
    if (!range_valid(backing_.size(), offset, bytes)) {
        return nullptr;
    }
    return backing_.data() + offset;
}

auto copy_to_storage(StorageBackend& storage, std::size_t offset, ConstByteSpan source) noexcept -> Status
{
    return storage.write(offset, source);
}

} // namespace edgeist
