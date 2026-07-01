#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "edgeist/runtime/numeric.hpp"
#include "edgeist/runtime/span.hpp"
#include "edgeist/runtime/status.hpp"
#include "edgeist/runtime/tensor.hpp"

namespace edgeist {

inline constexpr std::uint32_t kNmcfMagic = 0x46434D4EU;
inline constexpr std::uint16_t kMaxLayerCount = 256;

enum class LayerId : std::uint8_t {
    Linear = 1,
    Flatten = 2,
    Conv2d = 3,
    Dropout = 4,
    MaxPool2d = 5,
    ReLU = 6,
    Softmax = 7,
    AdaptiveAvgPool1d = 8,
    AdaptiveAvgPool2d = 9,
    BatchNorm1d = 10,
    BatchNorm2d = 11,
};

enum class DataEncodingId : std::uint8_t {
    Float32 = 0,
    Float16 = 1,
    Float64 = 2,
    Int32 = 3,
    Int64 = 4,
    Int8 = 5,
};


[[nodiscard]] constexpr auto data_encoding_bytes(DataEncodingId encoding) noexcept -> std::size_t
{
    switch (encoding) {
    case DataEncodingId::Float32:
        return 4U;
    case DataEncodingId::Float16:
        return 2U;
    case DataEncodingId::Float64:
        return 8U;
    case DataEncodingId::Int32:
        return 4U;
    case DataEncodingId::Int64:
        return 8U;
    case DataEncodingId::Int8:
        return 1U;
    }
    return 0U;
}

[[nodiscard]] constexpr auto is_supported_runtime_encoding(DataEncodingId encoding) noexcept -> bool
{
    return encoding == DataEncodingId::Float32 || encoding == DataEncodingId::Float16 || encoding == DataEncodingId::Int8;
}

[[nodiscard]] constexpr auto data_encoding_to_numeric_type(DataEncodingId encoding) noexcept -> NumericDataType
{
    switch (encoding) {
    case DataEncodingId::Int8:
        return NumericDataType::Int8;
    case DataEncodingId::Float16:
        return NumericDataType::Float16;
    case DataEncodingId::Float32:
    case DataEncodingId::Float64:
    case DataEncodingId::Int32:
    case DataEncodingId::Int64:
        return NumericDataType::Float32;
    }
    return NumericDataType::Float32;
}

[[nodiscard]] constexpr auto numeric_type_to_data_encoding(NumericDataType type) noexcept -> DataEncodingId
{
    switch (type) {
    case NumericDataType::Int8:
        return DataEncodingId::Int8;
    case NumericDataType::Float16:
        return DataEncodingId::Float16;
    case NumericDataType::Float32:
        return DataEncodingId::Float32;
    }
    return DataEncodingId::Float32;
}

[[nodiscard]] constexpr auto layer_name(LayerId id) noexcept -> std::string_view
{
    switch (id) {
    case LayerId::Linear:
        return "Linear";
    case LayerId::Flatten:
        return "Flatten";
    case LayerId::Conv2d:
        return "Conv2D";
    case LayerId::Dropout:
        return "Dropout";
    case LayerId::MaxPool2d:
        return "MaxPool2D";
    case LayerId::ReLU:
        return "ReLU";
    case LayerId::Softmax:
        return "Softmax";
    case LayerId::AdaptiveAvgPool1d:
        return "AdaptiveAvgPool1D";
    case LayerId::AdaptiveAvgPool2d:
        return "AdaptiveAvgPool2D";
    case LayerId::BatchNorm1d:
        return "BatchNorm1D";
    case LayerId::BatchNorm2d:
        return "BatchNorm2D";
    }
    return "Unknown";
}

#pragma pack(push, 1)
struct ModelHeaderPrefix {
    std::uint16_t headerSize;
    std::uint32_t magicNumber;
    std::array<std::uint8_t, 8> version;
    std::uint16_t layerNrs;
    std::uint16_t channelsIn;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionInputY;
    std::uint16_t channelsOut;
    std::uint32_t dimensionOutputX;
    std::uint32_t dimensionOutputY;
};

struct LayerPrefix {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
};

struct ParamOffsets {
    std::uint32_t pruningMaskOffset;
    std::uint32_t weightsMaskOffset;
    std::uint32_t weightsTrainableOffset;
    std::uint32_t weightsFrozenOffset;
    std::uint32_t biasMaskOffset;
    std::uint32_t biasTrainableOffset;
    std::uint32_t biasFrozenOffset;
};

struct LinearHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
    std::uint8_t dataEncoding;
    std::uint32_t weightsAmountFrozen;
    std::uint32_t weightsAmountTrainable;
    std::uint16_t biasAmountFrozen;
    std::uint32_t biasAmountTrainable;
    ParamOffsets offsets;
};

struct Conv2dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionInputY;
    std::uint32_t dimensionOutputX;
    std::uint32_t dimensionOutputY;
    std::uint16_t channelsIn;
    std::uint16_t channelsOut;
    std::array<std::uint8_t, 2> kernelSize;
    std::array<std::uint8_t, 2> padding;
    std::array<std::uint8_t, 2> stride;
    std::array<std::uint8_t, 2> dilation;
    std::uint8_t groups;
    std::uint8_t dataEncoding;
    std::uint32_t weightsAmountFrozen;
    std::uint32_t weightsAmountTrainable;
    std::uint16_t biasAmountFrozen;
    std::uint32_t biasAmountTrainable;
    ParamOffsets offsets;
};

struct MaxPool2dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionInputY;
    std::uint32_t dimensionOutputX;
    std::uint32_t dimensionOutputY;
    std::uint16_t channelsIn;
    std::uint16_t channelsOut;
    std::uint8_t kernelSize;
    std::uint8_t padding;
    std::uint8_t stride;
    std::uint8_t dilation;
};

struct UnaryHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
};

struct BatchNorm1dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
    std::uint8_t dataEncoding;
    std::uint32_t weightsAmountFrozen;
    std::uint32_t weightsAmountTrainable;
    std::uint16_t biasAmountFrozen;
    std::uint32_t biasAmountTrainable;
    ParamOffsets offsets;
};

struct BatchNorm2dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
    std::uint16_t channelsIn;
    std::uint16_t channelsOut;
    std::uint8_t dataEncoding;
    std::uint32_t weightsAmountFrozen;
    std::uint32_t weightsAmountTrainable;
    std::uint16_t biasAmountFrozen;
    std::uint32_t biasAmountTrainable;
    ParamOffsets offsets;
};

struct AdaptiveAvgPool1dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
    std::uint16_t channelsIn;
    std::uint16_t channelsOut;
};

struct AdaptiveAvgPool2dHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionInputY;
    std::uint32_t dimensionOutputX;
    std::uint32_t dimensionOutputY;
    std::uint16_t channelsIn;
    std::uint16_t channelsOut;
};

struct DropoutHeader {
    std::uint8_t id;
    std::uint8_t layerNr;
    std::uint8_t predecessorNr;
    std::uint32_t dimensionInputX;
    std::uint32_t dimensionOutputX;
    float dropoutRate;
};
#pragma pack(pop)

struct LayerInfo {
    LayerId id { LayerId::Linear };
    std::uint8_t layer_nr { 0 };
    std::uint8_t predecessor_count { 0 };
    std::size_t file_offset { 0 };
    std::size_t next_file_offset { 0 };
    std::size_t header_bytes { 0 };
    TensorShape input {};
    TensorShape output {};
    DataEncodingId data_encoding { DataEncodingId::Float32 };
    std::uint32_t weights_frozen { 0 };
    std::uint32_t weights_trainable { 0 };
    std::uint32_t bias_frozen { 0 };
    std::uint32_t bias_trainable { 0 };
    ParamOffsets offsets {};
    std::array<std::uint32_t, 4> kernel { 1, 1, 0, 1 }; // kH/kW, padding, stride for compact symmetric Conv metadata.
    std::array<std::uint32_t, 2> dilation { 1, 1 };
    std::uint32_t groups { 1 };
    float dropout_rate { 0.0F };

    [[nodiscard]] auto input_elements(std::size_t& out) const noexcept -> Status { return input.elements(out); }
    [[nodiscard]] auto output_elements(std::size_t& out) const noexcept -> Status { return output.elements(out); }
    [[nodiscard]] auto has_trainable_params() const noexcept -> bool { return (weights_trainable + bias_trainable) > 0U; }
};

struct ModelInfo {
    ModelHeaderPrefix header {};
    TensorShape input {};
    TensorShape output {};
    std::vector<LayerInfo> layers;
    std::size_t model_bytes { 0 };
    std::size_t trainable_bytes { 0 };
};

class ModelView {
public:
    ModelView() = default;
    ModelView(ConstByteSpan model, ConstByteSpan trainable) noexcept;

    [[nodiscard]] auto validate() -> Status;
    [[nodiscard]] auto info() const noexcept -> const ModelInfo& { return info_; }
    [[nodiscard]] auto model_bytes() const noexcept -> ConstByteSpan { return model_; }
    [[nodiscard]] auto trainable_bytes() const noexcept -> ConstByteSpan { return trainable_; }

    template <typename T>
    [[nodiscard]] auto layer_header(const LayerInfo& layer) const noexcept -> const T*
    {
        if (layer.file_offset > model_.size() || sizeof(T) > model_.size() - layer.file_offset) {
            return nullptr;
        }
        return reinterpret_cast<const T*>(model_.data() + layer.file_offset);
    }

private:
    ConstByteSpan model_ {};
    ConstByteSpan trainable_ {};
    ModelInfo info_ {};
};

[[nodiscard]] auto validate_model(ConstByteSpan model, ConstByteSpan trainable, ModelInfo* out = nullptr) -> Status;

} // namespace edgeist
