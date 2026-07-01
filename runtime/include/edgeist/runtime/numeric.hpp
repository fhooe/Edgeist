#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"

namespace edgeist {

enum class NumericDataType : std::uint8_t {
    Int8 = 0,
    Float16 = 1,
    Float32 = 2,
};

struct QuantizationConfig {
    // Symmetric quantization scales used for int8 tensors. Bias is intentionally
    // separate because exported quantized models may store bias with a different
    // scale than activations or weights.
    float activation_scale { 1.0F / 64.0F };
    float weight_scale { 1.0F / 64.0F };
    float bias_scale { 1.0F / 64.0F };
    float gradient_scale { 1.0F / 64.0F };
};

struct ConstTypedTensorView {
    ConstByteSpan bytes {};
    NumericDataType type { NumericDataType::Float32 };
    std::size_t elements { 0 };
    float scale { 1.0F };
};

struct MutableTypedTensorView {
    ByteSpan bytes {};
    NumericDataType type { NumericDataType::Float32 };
    std::size_t elements { 0 };
    float scale { 1.0F };
};

[[nodiscard]] constexpr auto numeric_data_type_name(NumericDataType type) noexcept -> std::string_view
{
    switch (type) {
    case NumericDataType::Int8:
        return "int8";
    case NumericDataType::Float16:
        return "fp16";
    case NumericDataType::Float32:
        return "fp32";
    }
    return "unknown";
}

[[nodiscard]] constexpr auto numeric_data_type_bytes(NumericDataType type) noexcept -> std::size_t
{
    switch (type) {
    case NumericDataType::Int8:
        return 1U;
    case NumericDataType::Float16:
        return 2U;
    case NumericDataType::Float32:
        return 4U;
    }
    return 0U;
}

[[nodiscard]] constexpr auto typed_tensor_bytes(std::size_t elements, NumericDataType type) noexcept -> std::size_t
{
    return elements * numeric_data_type_bytes(type);
}

[[nodiscard]] auto parse_numeric_data_type(std::string_view name, NumericDataType& out) noexcept -> Status;
[[nodiscard]] auto float32_to_float16_bits(float value) noexcept -> std::uint16_t;
[[nodiscard]] auto float16_bits_to_float32(std::uint16_t bits) noexcept -> float;
[[nodiscard]] auto quantize_int8(float value, float scale) noexcept -> std::int8_t;
[[nodiscard]] auto dequantize_int8(std::int8_t value, float scale) noexcept -> float;

[[nodiscard]] auto validate_typed_tensor(ConstTypedTensorView tensor, const char* name) noexcept -> Status;
[[nodiscard]] auto validate_typed_tensor(MutableTypedTensorView tensor, const char* name) noexcept -> Status;
[[nodiscard]] auto read_typed_value(ConstTypedTensorView tensor, std::size_t index, float& out) noexcept -> Status;
[[nodiscard]] auto write_typed_value(MutableTypedTensorView tensor, std::size_t index, float value) noexcept -> Status;
[[nodiscard]] auto encode_float_span(ConstSpan<float> input, MutableTypedTensorView output) noexcept -> Status;
[[nodiscard]] auto decode_to_float_span(ConstTypedTensorView input, Span<float> output) noexcept -> Status;
[[nodiscard]] auto copy_typed_tensor(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status;

// Kept for optimizer/gradient emulation and public tests. Layer forward paths no
// longer use this to simulate lower precision on float32 buffers.

} // namespace edgeist
