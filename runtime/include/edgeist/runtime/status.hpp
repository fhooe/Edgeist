#pragma once

#include <cstdint>
#include <string_view>

namespace edgeist {

enum class ErrorCode : std::uint8_t {
    Ok = 0,
    InvalidArgument,
    NullPointer,
    OutOfMemory,
    InvalidModel,
    UnsupportedLayer,
    UnsupportedDataEncoding,
    ShapeMismatch,
    BufferTooSmall,
    OutOfBounds,
    StorageError,
    DisabledFeature,
    NotInitialized,
    NumericError,
};

struct Status {
    ErrorCode code { ErrorCode::Ok };
    std::string_view message { "OK" };

    [[nodiscard]] constexpr auto ok() const noexcept -> bool { return code == ErrorCode::Ok; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] static constexpr auto success() noexcept -> Status { return {}; }
};

[[nodiscard]] constexpr auto make_status(ErrorCode code, std::string_view message) noexcept -> Status
{
    return Status { code, message };
}

[[nodiscard]] constexpr auto to_string(ErrorCode code) noexcept -> std::string_view
{
    switch (code) {
    case ErrorCode::Ok:
        return "Ok";
    case ErrorCode::InvalidArgument:
        return "InvalidArgument";
    case ErrorCode::NullPointer:
        return "NullPointer";
    case ErrorCode::OutOfMemory:
        return "OutOfMemory";
    case ErrorCode::InvalidModel:
        return "InvalidModel";
    case ErrorCode::UnsupportedLayer:
        return "UnsupportedLayer";
    case ErrorCode::UnsupportedDataEncoding:
        return "UnsupportedDataEncoding";
    case ErrorCode::ShapeMismatch:
        return "ShapeMismatch";
    case ErrorCode::BufferTooSmall:
        return "BufferTooSmall";
    case ErrorCode::OutOfBounds:
        return "OutOfBounds";
    case ErrorCode::StorageError:
        return "StorageError";
    case ErrorCode::DisabledFeature:
        return "DisabledFeature";
    case ErrorCode::NotInitialized:
        return "NotInitialized";
    case ErrorCode::NumericError:
        return "NumericError";
    }
    return "Unknown";
}

#define EDGEIST_RETURN_IF_ERROR(expr)       \
    do {                                    \
        const ::edgeist::Status _st = (expr); \
        if (!_st.ok()) {                    \
            return _st;                     \
        }                                   \
    } while (false)

} // namespace edgeist
