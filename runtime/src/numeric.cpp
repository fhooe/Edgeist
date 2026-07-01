#include "edgeist/runtime/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace edgeist {

namespace {
[[nodiscard]] auto bit_cast_u32(float value) noexcept -> std::uint32_t
{
    std::uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

[[nodiscard]] auto bit_cast_float(std::uint32_t value) noexcept -> float
{
    float out = 0.0F;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

[[nodiscard]] auto span_has_bytes(ConstByteSpan bytes, std::size_t needed) noexcept -> bool
{
    return bytes.size() >= needed && (needed == 0U || bytes.data() != nullptr);
}

[[nodiscard]] auto span_has_bytes(ByteSpan bytes, std::size_t needed) noexcept -> bool
{
    return bytes.size() >= needed && (needed == 0U || bytes.data() != nullptr);
}
} // namespace

auto parse_numeric_data_type(std::string_view name, NumericDataType& out) noexcept -> Status
{
    if (name == "fp32" || name == "float32" || name == "f32") {
        out = NumericDataType::Float32;
        return Status::success();
    }
    if (name == "fp16" || name == "float16" || name == "half" || name == "f16") {
        out = NumericDataType::Float16;
        return Status::success();
    }
    if (name == "int8" || name == "i8" || name == "qint8") {
        out = NumericDataType::Int8;
        return Status::success();
    }
    return make_status(ErrorCode::InvalidArgument, "unknown numeric data type; use int8, fp16, or fp32");
}

auto float32_to_float16_bits(float value) noexcept -> std::uint16_t
{
    const std::uint32_t f = bit_cast_u32(value);
    const std::uint32_t sign = (f >> 16U) & 0x8000U;
    const std::uint32_t exponent = (f >> 23U) & 0xFFU;
    const std::uint32_t mantissa = f & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0U) {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
        return static_cast<std::uint16_t>(sign | 0x7E00U);
    }

    const int new_exponent = static_cast<int>(exponent) - 127 + 15;
    if (new_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (new_exponent <= 0) {
        if (new_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t subnormal = mantissa | 0x800000U;
        const int shift = 14 - new_exponent;
        std::uint32_t half_mantissa = subnormal >> static_cast<std::uint32_t>(shift);
        const std::uint32_t round_bit = (subnormal >> static_cast<std::uint32_t>(shift - 1)) & 1U;
        half_mantissa += round_bit;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_exponent = static_cast<std::uint32_t>(new_exponent) << 10U;
    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t round_bit = (mantissa >> 12U) & 1U;
    half_mantissa += round_bit;
    if (half_mantissa == 0x400U) {
        half_mantissa = 0U;
        half_exponent += 0x400U;
        if (half_exponent >= 0x7C00U) {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<std::uint16_t>(sign | half_exponent | half_mantissa);
}

auto float16_bits_to_float32(std::uint16_t bits) noexcept -> float
{
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa = bits & 0x03FFU;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            return bit_cast_float(sign);
        }
        exponent = 1U;
        while ((mantissa & 0x0400U) == 0U) {
            mantissa <<= 1U;
            --exponent;
        }
        mantissa &= 0x03FFU;
    } else if (exponent == 0x1FU) {
        return bit_cast_float(sign | 0x7F800000U | (mantissa << 13U));
    }

    const std::uint32_t out_exponent = (exponent + (127U - 15U)) << 23U;
    const std::uint32_t out_mantissa = mantissa << 13U;
    return bit_cast_float(sign | out_exponent | out_mantissa);
}

auto quantize_int8(float value, float scale) noexcept -> std::int8_t
{
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0F) {
        return 0;
    }
    const float quantized = std::round(value / scale);
    const float clamped = std::clamp(quantized, -128.0F, 127.0F);
    return static_cast<std::int8_t>(clamped);
}

auto dequantize_int8(std::int8_t value, float scale) noexcept -> float
{
    if (!std::isfinite(scale) || scale <= 0.0F) {
        return 0.0F;
    }
    return static_cast<float>(value) * scale;
}

auto validate_typed_tensor(ConstTypedTensorView tensor, const char* name) noexcept -> Status
{
    const auto elem_bytes = numeric_data_type_bytes(tensor.type);
    if (elem_bytes == 0U) {
        return make_status(ErrorCode::UnsupportedDataEncoding, name);
    }
    const auto needed = tensor.elements * elem_bytes;
    if (!span_has_bytes(tensor.bytes, needed)) {
        return make_status(ErrorCode::BufferTooSmall, name);
    }
    if (tensor.type == NumericDataType::Int8 && (!std::isfinite(tensor.scale) || tensor.scale <= 0.0F)) {
        return make_status(ErrorCode::NumericError, "int8 tensor scale must be finite and positive");
    }
    return Status::success();
}

auto validate_typed_tensor(MutableTypedTensorView tensor, const char* name) noexcept -> Status
{
    const auto elem_bytes = numeric_data_type_bytes(tensor.type);
    if (elem_bytes == 0U) {
        return make_status(ErrorCode::UnsupportedDataEncoding, name);
    }
    const auto needed = tensor.elements * elem_bytes;
    if (!span_has_bytes(tensor.bytes, needed)) {
        return make_status(ErrorCode::BufferTooSmall, name);
    }
    if (tensor.type == NumericDataType::Int8 && (!std::isfinite(tensor.scale) || tensor.scale <= 0.0F)) {
        return make_status(ErrorCode::NumericError, "int8 tensor scale must be finite and positive");
    }
    return Status::success();
}

auto read_typed_value(ConstTypedTensorView tensor, std::size_t index, float& out) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(tensor, "typed tensor read outside storage"));
    if (index >= tensor.elements) {
        return make_status(ErrorCode::OutOfBounds, "typed tensor read index outside element count");
    }
    switch (tensor.type) {
    case NumericDataType::Float32: {
        float value = 0.0F;
        std::memcpy(&value, tensor.bytes.data() + index * sizeof(float), sizeof(value));
        out = value;
        return Status::success();
    }
    case NumericDataType::Float16: {
        std::uint16_t value = 0;
        std::memcpy(&value, tensor.bytes.data() + index * sizeof(std::uint16_t), sizeof(value));
        out = float16_bits_to_float32(value);
        return Status::success();
    }
    case NumericDataType::Int8: {
        const auto* data = reinterpret_cast<const std::int8_t*>(tensor.bytes.data());
        out = dequantize_int8(data[index], tensor.scale);
        return Status::success();
    }
    }
    return make_status(ErrorCode::UnsupportedDataEncoding, "unsupported typed tensor read type");
}

auto write_typed_value(MutableTypedTensorView tensor, std::size_t index, float value) noexcept -> Status
{
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(tensor, "typed tensor write outside storage"));
    if (index >= tensor.elements) {
        return make_status(ErrorCode::OutOfBounds, "typed tensor write index outside element count");
    }
    switch (tensor.type) {
    case NumericDataType::Float32:
        std::memcpy(tensor.bytes.data() + index * sizeof(float), &value, sizeof(value));
        return Status::success();
    case NumericDataType::Float16: {
        const std::uint16_t value16 = float32_to_float16_bits(value);
        std::memcpy(tensor.bytes.data() + index * sizeof(std::uint16_t), &value16, sizeof(value16));
        return Status::success();
    }
    case NumericDataType::Int8: {
        auto* data = reinterpret_cast<std::int8_t*>(tensor.bytes.data());
        data[index] = quantize_int8(value, tensor.scale);
        return Status::success();
    }
    }
    return make_status(ErrorCode::UnsupportedDataEncoding, "unsupported typed tensor write type");
}

auto encode_float_span(ConstSpan<float> input, MutableTypedTensorView output) noexcept -> Status
{
    if (input.size() < output.elements) {
        return make_status(ErrorCode::BufferTooSmall, "input float span too small for typed encode");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "typed encode output span too small"));
    for (std::size_t i = 0; i < output.elements; ++i) {
        EDGEIST_RETURN_IF_ERROR(write_typed_value(output, i, input[i]));
    }
    return Status::success();
}

auto decode_to_float_span(ConstTypedTensorView input, Span<float> output) noexcept -> Status
{
    if (output.size() < input.elements) {
        return make_status(ErrorCode::BufferTooSmall, "output float span too small for typed decode");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "typed decode input span too small"));
    for (std::size_t i = 0; i < input.elements; ++i) {
        EDGEIST_RETURN_IF_ERROR(read_typed_value(input, i, output[i]));
    }
    return Status::success();
}

auto copy_typed_tensor(ConstTypedTensorView input, MutableTypedTensorView output) noexcept -> Status
{
    if (input.type != output.type || input.elements != output.elements) {
        return make_status(ErrorCode::InvalidArgument, "typed tensor copy requires identical type and element count");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(input, "typed tensor copy input invalid"));
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(output, "typed tensor copy output invalid"));
    const auto bytes = typed_tensor_bytes(input.elements, input.type);
    if (bytes != 0U) {
        std::memcpy(output.bytes.data(), input.bytes.data(), bytes);
    }
    return Status::success();
}

} // namespace edgeist
