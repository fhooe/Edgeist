#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "edgeist/runtime/model_format.hpp"
#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"
#include "edgeist/runtime/training_config.hpp"

namespace edgeist {

class ScratchArena {
public:
    explicit ScratchArena(ByteSpan buffer) noexcept
        : buffer_(buffer)
    {
    }

    auto reset() noexcept -> void { offset_ = 0; peak_ = 0; }

    template <typename T>
    [[nodiscard]] auto allocate(std::size_t count, T*& out, std::size_t alignment = alignof(T)) noexcept -> Status
    {
        std::size_t bytes = 0;
        if (!checked_mul(count, sizeof(T), bytes)) {
            return make_status(ErrorCode::OutOfBounds, "arena allocation size overflow");
        }
        const auto aligned = align_up(offset_, alignment);
        std::size_t end = 0;
        if (!checked_add(aligned, bytes, end) || end > buffer_.size()) {
            out = nullptr;
            return make_status(ErrorCode::OutOfMemory, "scratch arena exhausted");
        }
        out = reinterpret_cast<T*>(buffer_.data() + aligned);
        offset_ = end;
        if (offset_ > peak_) {
            peak_ = offset_;
        }
        return Status::success();
    }

    [[nodiscard]] auto allocate_bytes(std::size_t bytes, std::byte*& out, std::size_t alignment = alignof(std::max_align_t)) noexcept -> Status;
    [[nodiscard]] auto capacity() const noexcept -> std::size_t { return buffer_.size(); }
    [[nodiscard]] auto used() const noexcept -> std::size_t { return offset_; }
    [[nodiscard]] auto peak() const noexcept -> std::size_t { return peak_; }

private:
    ByteSpan buffer_ {};
    std::size_t offset_ { 0 };
    std::size_t peak_ { 0 };
};

struct MemoryReport {
    std::size_t model_flash_bytes { 0 };
    std::size_t trainable_storage_bytes { 0 };
    std::size_t persistent_sram_bytes { 0 };
    std::size_t activation_scratch_bytes { 0 };
    std::size_t gradient_bytes { 0 };
    std::size_t optimizer_state_bytes { 0 };
    std::size_t temporary_bytes { 0 };
    std::size_t total_peak_sram_bytes { 0 };
    std::size_t inference_element_bytes { 4 };
    std::size_t training_element_bytes { 4 };

    [[nodiscard]] auto to_json() const -> std::string;
};

class MemoryPlanner {
public:
    [[nodiscard]] static auto estimate(const ModelInfo& info, const TrainingConfig& config, MemoryReport& report) noexcept -> Status;
};

} // namespace edgeist
