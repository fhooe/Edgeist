#include "edgeist/runtime/model_format.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace edgeist {

namespace {

template <typename T>
[[nodiscard]] auto read_struct(ConstByteSpan bytes, std::size_t offset, T& out) noexcept -> Status
{
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        return make_status(ErrorCode::InvalidModel, "binary model is truncated while reading a structure");
    }
    std::memcpy(&out, bytes.data() + offset, sizeof(T));
    return Status::success();
}

[[nodiscard]] auto read_u32(ConstByteSpan bytes, std::size_t offset, std::uint32_t& out) noexcept -> Status
{
    return read_struct(bytes, offset, out);
}

[[nodiscard]] auto known_layer_id(std::uint8_t raw, LayerId& id) noexcept -> bool
{
    switch (static_cast<LayerId>(raw)) {
    case LayerId::Linear:
    case LayerId::Flatten:
    case LayerId::Conv2d:
    case LayerId::Dropout:
    case LayerId::MaxPool2d:
    case LayerId::ReLU:
    case LayerId::Softmax:
    case LayerId::AdaptiveAvgPool1d:
    case LayerId::AdaptiveAvgPool2d:
    case LayerId::BatchNorm1d:
    case LayerId::BatchNorm2d:
        id = static_cast<LayerId>(raw);
        return true;
    }
    return false;
}

[[nodiscard]] auto validate_nonzero(std::uint32_t value, const char* message) noexcept -> Status
{
    if (value == 0U) {
        return make_status(ErrorCode::ShapeMismatch, message);
    }
    return Status::success();
}

[[nodiscard]] auto range_in_bytes(std::size_t total, std::size_t offset, std::size_t count, std::size_t element_size, const char* message) noexcept -> Status
{
    if (count == 0U) {
        return Status::success();
    }
    std::size_t end = 0;
    if (!checked_add_mul(offset, count, element_size, end) || end > total) {
        return make_status(ErrorCode::OutOfBounds, message);
    }
    return Status::success();
}

[[nodiscard]] auto validate_param_offsets(const LayerInfo& info, std::size_t model_size, std::size_t trainable_size) noexcept -> Status
{
    const auto element_size = data_encoding_bytes(info.data_encoding);
    if (element_size == 0U || !is_supported_runtime_encoding(info.data_encoding)) {
        return make_status(ErrorCode::UnsupportedDataEncoding, "layer parameter encoding must be int8, fp16, or fp32");
    }
    EDGEIST_RETURN_IF_ERROR(range_in_bytes(trainable_size, info.offsets.weightsTrainableOffset, info.weights_trainable, element_size, "trainable weight block outside trainable storage"));
    EDGEIST_RETURN_IF_ERROR(range_in_bytes(trainable_size, info.offsets.biasTrainableOffset, info.bias_trainable, element_size, "trainable bias block outside trainable storage"));
    const auto layer_bytes = info.next_file_offset - info.file_offset;
    EDGEIST_RETURN_IF_ERROR(range_in_bytes(layer_bytes, info.offsets.weightsFrozenOffset, info.weights_frozen, element_size, "frozen weight block outside layer blob"));
    EDGEIST_RETURN_IF_ERROR(range_in_bytes(layer_bytes, info.offsets.biasFrozenOffset, info.bias_frozen, element_size, "frozen bias block outside layer blob"));
    if (info.offsets.pruningMaskOffset != 0U && info.offsets.pruningMaskOffset >= layer_bytes) {
        return make_status(ErrorCode::OutOfBounds, "pruning mask offset outside layer blob");
    }
    if (info.offsets.weightsMaskOffset != 0U && info.offsets.weightsMaskOffset >= layer_bytes) {
        return make_status(ErrorCode::OutOfBounds, "weights mask offset outside layer blob");
    }
    if (info.offsets.biasMaskOffset != 0U && info.offsets.biasMaskOffset >= layer_bytes) {
        return make_status(ErrorCode::OutOfBounds, "bias mask offset outside layer blob");
    }
    (void)model_size;
    return Status::success();
}

[[nodiscard]] auto validate_predecessors(ConstByteSpan model, const LayerInfo& info) noexcept -> Status
{
    std::size_t pred_bytes = 0;
    if (!checked_mul(static_cast<std::size_t>(info.predecessor_count), sizeof(std::uint16_t), pred_bytes)) {
        return make_status(ErrorCode::OutOfBounds, "predecessor table size overflow");
    }
    std::size_t end = 0;
    if (!checked_add(info.file_offset + info.header_bytes, pred_bytes, end) || end > info.next_file_offset || end > model.size()) {
        return make_status(ErrorCode::InvalidModel, "layer predecessor table is truncated");
    }
    for (std::size_t i = 0; i < info.predecessor_count; ++i) {
        std::uint16_t pred = 0;
        EDGEIST_RETURN_IF_ERROR(read_struct(model, info.file_offset + info.header_bytes + i * sizeof(std::uint16_t), pred));
        if (info.layer_nr != 0U && pred >= info.layer_nr) {
            return make_status(ErrorCode::InvalidModel, "layer predecessor points to current or future layer");
        }
    }
    return Status::success();
}

[[nodiscard]] auto parse_unary(ConstByteSpan model, const std::vector<std::uint32_t>& offsets, std::size_t index, LayerInfo& info) noexcept -> Status
{
    UnaryHeader header {};
    EDGEIST_RETURN_IF_ERROR(read_struct(model, offsets[index], header));
    info.header_bytes = sizeof(UnaryHeader);
    info.layer_nr = header.layerNr;
    info.predecessor_count = header.predecessorNr;
    info.input = TensorShape { 1, 1, header.dimensionInputX };
    info.output = TensorShape { 1, 1, header.dimensionOutputX };
    EDGEIST_RETURN_IF_ERROR(validate_nonzero(header.dimensionInputX, "unary layer input dimension is zero"));
    EDGEIST_RETURN_IF_ERROR(validate_nonzero(header.dimensionOutputX, "unary layer output dimension is zero"));
    if (info.id == LayerId::ReLU || info.id == LayerId::Softmax || info.id == LayerId::Flatten) {
        if (header.dimensionInputX != header.dimensionOutputX) {
            return make_status(ErrorCode::ShapeMismatch, "unary layer input/output dimensions must match");
        }
    }
    return Status::success();
}

} // namespace

ModelView::ModelView(ConstByteSpan model, ConstByteSpan trainable) noexcept
    : model_(model)
    , trainable_(trainable)
{
}

auto ModelView::validate() -> Status
{
    info_ = {};
    if (model_.data() == nullptr || model_.size() == 0U) {
        return make_status(ErrorCode::NullPointer, "model buffer is null or empty");
    }
    if (model_.size() < sizeof(ModelHeaderPrefix)) {
        return make_status(ErrorCode::InvalidModel, "model is shorter than the fixed header");
    }

    EDGEIST_RETURN_IF_ERROR(read_struct(model_, 0, info_.header));
    info_.model_bytes = model_.size();
    info_.trainable_bytes = trainable_.size();

    if (info_.header.magicNumber != kNmcfMagic) {
        return make_status(ErrorCode::InvalidModel, "invalid NMCF magic number");
    }
    if (info_.header.layerNrs == 0U || info_.header.layerNrs > kMaxLayerCount) {
        return make_status(ErrorCode::InvalidModel, "invalid layer count");
    }
    const std::size_t layer_count = info_.header.layerNrs;
    std::size_t min_header_size = 0;
    if (!checked_add(sizeof(ModelHeaderPrefix), layer_count * sizeof(std::uint32_t), min_header_size)) {
        return make_status(ErrorCode::OutOfBounds, "header size overflow");
    }
    if (info_.header.headerSize < min_header_size || info_.header.headerSize > model_.size()) {
        return make_status(ErrorCode::InvalidModel, "headerSize is outside valid range");
    }
    info_.input = TensorShape { info_.header.channelsIn, info_.header.dimensionInputY, info_.header.dimensionInputX };
    info_.output = TensorShape { info_.header.channelsOut, info_.header.dimensionOutputY, info_.header.dimensionOutputX };
    std::size_t ignored = 0;
    EDGEIST_RETURN_IF_ERROR(info_.input.elements(ignored));
    EDGEIST_RETURN_IF_ERROR(info_.output.elements(ignored));

    std::vector<std::uint32_t> offsets(layer_count);
    for (std::size_t i = 0; i < layer_count; ++i) {
        EDGEIST_RETURN_IF_ERROR(read_u32(model_, sizeof(ModelHeaderPrefix) + i * sizeof(std::uint32_t), offsets[i]));
        if (offsets[i] < info_.header.headerSize || offsets[i] >= model_.size()) {
            return make_status(ErrorCode::InvalidModel, "layer offset outside model image");
        }
        if (i > 0 && offsets[i] <= offsets[i - 1]) {
            return make_status(ErrorCode::InvalidModel, "layer offsets must be strictly increasing");
        }
    }

    info_.layers.reserve(layer_count);
    for (std::size_t i = 0; i < layer_count; ++i) {
        LayerPrefix prefix {};
        EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], prefix));
        LayerInfo info {};
        if (!known_layer_id(prefix.id, info.id)) {
            return make_status(ErrorCode::UnsupportedLayer, "unsupported layer id in model image");
        }
        info.file_offset = offsets[i];
        info.next_file_offset = (i + 1U < layer_count) ? offsets[i + 1U] : model_.size();
        if (info.next_file_offset <= info.file_offset) {
            return make_status(ErrorCode::InvalidModel, "invalid layer byte range");
        }
        const auto layer_bytes = info.next_file_offset - info.file_offset;

        switch (info.id) {
        case LayerId::Linear: {
            LinearHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(LinearHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { 1, 1, header.dimensionInputX };
            info.output = TensorShape { 1, 1, header.dimensionOutputX };
            info.data_encoding = static_cast<DataEncodingId>(header.dataEncoding);
            info.weights_frozen = header.weightsAmountFrozen;
            info.weights_trainable = header.weightsAmountTrainable;
            info.bias_frozen = header.biasAmountFrozen;
            info.bias_trainable = header.biasAmountTrainable;
            info.offsets = header.offsets;
            EDGEIST_RETURN_IF_ERROR(validate_nonzero(header.dimensionInputX, "linear input dimension is zero"));
            EDGEIST_RETURN_IF_ERROR(validate_nonzero(header.dimensionOutputX, "linear output dimension is zero"));
            if (!is_supported_runtime_encoding(info.data_encoding)) {
                return make_status(ErrorCode::UnsupportedDataEncoding, "linear parameter encoding must be int8, fp16, or fp32");
            }
            const auto expected_w = static_cast<std::size_t>(header.dimensionInputX) * header.dimensionOutputX;
            if (expected_w != static_cast<std::size_t>(info.weights_frozen) + info.weights_trainable) {
                return make_status(ErrorCode::ShapeMismatch, "linear weight count does not match layer dimensions");
            }
            if (header.dimensionOutputX != static_cast<std::size_t>(info.bias_frozen) + info.bias_trainable) {
                return make_status(ErrorCode::ShapeMismatch, "linear bias count does not match output dimension");
            }
            EDGEIST_RETURN_IF_ERROR(validate_param_offsets(info, model_.size(), trainable_.size()));
            break;
        }
        case LayerId::Conv2d: {
            Conv2dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(Conv2dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { header.channelsIn, header.dimensionInputY, header.dimensionInputX };
            info.output = TensorShape { header.channelsOut, header.dimensionOutputY, header.dimensionOutputX };
            info.data_encoding = static_cast<DataEncodingId>(header.dataEncoding);
            info.weights_frozen = header.weightsAmountFrozen;
            info.weights_trainable = header.weightsAmountTrainable;
            info.bias_frozen = header.biasAmountFrozen;
            info.bias_trainable = header.biasAmountTrainable;
            info.offsets = header.offsets;
            info.kernel = { header.kernelSize[0], header.kernelSize[1], header.padding[0], header.stride[0] };
            info.dilation = { header.dilation[0], header.dilation[1] };
            info.groups = header.groups == 0U ? 1U : header.groups;
            EDGEIST_RETURN_IF_ERROR(info.input.elements(ignored));
            EDGEIST_RETURN_IF_ERROR(info.output.elements(ignored));
            if (!is_supported_runtime_encoding(info.data_encoding)) {
                return make_status(ErrorCode::UnsupportedDataEncoding, "conv parameter encoding must be int8, fp16, or fp32");
            }
            if (info.groups == 0U || header.channelsIn % info.groups != 0U || header.channelsOut % info.groups != 0U) {
                return make_status(ErrorCode::ShapeMismatch, "conv groups do not divide channel counts");
            }
            const auto expected_w = static_cast<std::size_t>(header.channelsOut) * (header.channelsIn / info.groups)
                * header.kernelSize[0] * header.kernelSize[1];
            if (expected_w != static_cast<std::size_t>(info.weights_frozen) + info.weights_trainable) {
                return make_status(ErrorCode::ShapeMismatch, "conv weight count does not match layer dimensions");
            }
            if (header.channelsOut != static_cast<std::size_t>(info.bias_frozen) + info.bias_trainable) {
                return make_status(ErrorCode::ShapeMismatch, "conv bias count does not match output channels");
            }
            EDGEIST_RETURN_IF_ERROR(validate_param_offsets(info, model_.size(), trainable_.size()));
            break;
        }
        case LayerId::MaxPool2d: {
            MaxPool2dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(MaxPool2dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { header.channelsIn, header.dimensionInputY, header.dimensionInputX };
            info.output = TensorShape { header.channelsOut, header.dimensionOutputY, header.dimensionOutputX };
            info.kernel = { header.kernelSize, header.kernelSize, header.padding, header.stride == 0U ? 1U : header.stride };
            info.dilation = { header.dilation == 0U ? 1U : header.dilation, header.dilation == 0U ? 1U : header.dilation };
            EDGEIST_RETURN_IF_ERROR(info.input.elements(ignored));
            EDGEIST_RETURN_IF_ERROR(info.output.elements(ignored));
            if (header.channelsIn != header.channelsOut) {
                return make_status(ErrorCode::ShapeMismatch, "max pool input/output channels must match");
            }
            break;
        }
        case LayerId::ReLU:
        case LayerId::Softmax:
        case LayerId::Flatten:
            EDGEIST_RETURN_IF_ERROR(parse_unary(model_, offsets, i, info));
            break;
        case LayerId::BatchNorm1d: {
            BatchNorm1dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(BatchNorm1dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { 1, 1, header.dimensionInputX };
            info.output = TensorShape { 1, 1, header.dimensionOutputX };
            info.data_encoding = static_cast<DataEncodingId>(header.dataEncoding);
            info.weights_frozen = header.weightsAmountFrozen;
            info.weights_trainable = header.weightsAmountTrainable;
            info.bias_frozen = header.biasAmountFrozen;
            info.bias_trainable = header.biasAmountTrainable;
            info.offsets = header.offsets;
            if (!is_supported_runtime_encoding(info.data_encoding)) {
                return make_status(ErrorCode::UnsupportedDataEncoding, "batchnorm parameter encoding must be int8, fp16, or fp32");
            }
            EDGEIST_RETURN_IF_ERROR(validate_param_offsets(info, model_.size(), trainable_.size()));
            break;
        }
        case LayerId::BatchNorm2d: {
            BatchNorm2dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(BatchNorm2dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { header.channelsIn, 1, header.dimensionInputX / std::max<std::uint16_t>(header.channelsIn, 1) };
            info.output = TensorShape { header.channelsOut, 1, header.dimensionOutputX / std::max<std::uint16_t>(header.channelsOut, 1) };
            info.data_encoding = static_cast<DataEncodingId>(header.dataEncoding);
            info.weights_frozen = header.weightsAmountFrozen;
            info.weights_trainable = header.weightsAmountTrainable;
            info.bias_frozen = header.biasAmountFrozen;
            info.bias_trainable = header.biasAmountTrainable;
            info.offsets = header.offsets;
            if (header.channelsIn == 0U || header.channelsOut == 0U || header.dimensionInputX % header.channelsIn != 0U || header.dimensionOutputX % header.channelsOut != 0U) {
                return make_status(ErrorCode::ShapeMismatch, "batchnorm2d flattened dimensions are not divisible by channels");
            }
            if (!is_supported_runtime_encoding(info.data_encoding)) {
                return make_status(ErrorCode::UnsupportedDataEncoding, "batchnorm parameter encoding must be int8, fp16, or fp32");
            }
            EDGEIST_RETURN_IF_ERROR(validate_param_offsets(info, model_.size(), trainable_.size()));
            break;
        }
        case LayerId::AdaptiveAvgPool1d: {
            AdaptiveAvgPool1dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(AdaptiveAvgPool1dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { header.channelsIn, 1, header.dimensionInputX };
            info.output = TensorShape { header.channelsOut, 1, header.dimensionOutputX };
            EDGEIST_RETURN_IF_ERROR(info.input.elements(ignored));
            EDGEIST_RETURN_IF_ERROR(info.output.elements(ignored));
            if (header.channelsIn != header.channelsOut) {
                return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool1d channels must match");
            }
            break;
        }
        case LayerId::AdaptiveAvgPool2d: {
            AdaptiveAvgPool2dHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(AdaptiveAvgPool2dHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { header.channelsIn, header.dimensionInputY, header.dimensionInputX };
            info.output = TensorShape { header.channelsOut, header.dimensionOutputY, header.dimensionOutputX };
            EDGEIST_RETURN_IF_ERROR(info.input.elements(ignored));
            EDGEIST_RETURN_IF_ERROR(info.output.elements(ignored));
            if (header.channelsIn != header.channelsOut) {
                return make_status(ErrorCode::ShapeMismatch, "adaptive avg pool2d channels must match");
            }
            break;
        }
        case LayerId::Dropout: {
            DropoutHeader header {};
            EDGEIST_RETURN_IF_ERROR(read_struct(model_, offsets[i], header));
            info.header_bytes = sizeof(DropoutHeader);
            info.layer_nr = header.layerNr;
            info.predecessor_count = header.predecessorNr;
            info.input = TensorShape { 1, 1, header.dimensionInputX };
            info.output = TensorShape { 1, 1, header.dimensionOutputX };
            info.dropout_rate = header.dropoutRate;
            if (header.dimensionInputX != header.dimensionOutputX || header.dropoutRate < 0.0F || header.dropoutRate >= 1.0F) {
                return make_status(ErrorCode::ShapeMismatch, "invalid dropout dimensions or rate");
            }
            break;
        }
        }

        if (info.header_bytes > layer_bytes) {
            return make_status(ErrorCode::InvalidModel, "layer header exceeds its byte range");
        }
        if (info.layer_nr != i) {
            return make_status(ErrorCode::InvalidModel, "layerNr does not match offset table order");
        }
        EDGEIST_RETURN_IF_ERROR(validate_predecessors(model_, info));
        info_.layers.push_back(info);
    }

    for (std::size_t i = 1; i < info_.layers.size(); ++i) {
        std::size_t prev_out = 0;
        std::size_t this_in = 0;
        EDGEIST_RETURN_IF_ERROR(info_.layers[i - 1].output_elements(prev_out));
        EDGEIST_RETURN_IF_ERROR(info_.layers[i].input_elements(this_in));
        if (prev_out != this_in) {
            return make_status(ErrorCode::ShapeMismatch, "adjacent layer tensor sizes do not match");
        }
    }

    return Status::success();
}

auto validate_model(ConstByteSpan model, ConstByteSpan trainable, ModelInfo* out) -> Status
{
    ModelView view(model, trainable);
    EDGEIST_RETURN_IF_ERROR(view.validate());
    if (out != nullptr) {
        *out = view.info();
    }
    return Status::success();
}

} // namespace edgeist
