#include "edgeist/runtime/model_runtime.hpp"

#include "edgeist/runtime/layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace edgeist {

namespace {
constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

[[nodiscard]] auto const_bytes_from_mutable(ByteSpan bytes) noexcept -> ConstByteSpan
{
    return ConstByteSpan(bytes.data(), bytes.size());
}

[[nodiscard]] auto byte_span_from_storage(StorageBackend& storage, std::uint32_t offset_bytes, std::size_t bytes) noexcept -> ByteSpan
{
    auto* ptr = storage.writable_data(offset_bytes, bytes);
    return ptr == nullptr ? ByteSpan() : ByteSpan(ptr, bytes);
}

[[nodiscard]] auto const_byte_span_from_storage(const StorageBackend& storage, std::uint32_t offset_bytes, std::size_t bytes) noexcept -> ConstByteSpan
{
    const auto* ptr = storage.readable_data(offset_bytes, bytes);
    return ptr == nullptr ? ConstByteSpan() : ConstByteSpan(ptr, bytes);
}

[[nodiscard]] auto layer_elements(const LayerInfo& layer, bool output) noexcept -> std::size_t
{
    std::size_t elems = 0;
    const auto status = output ? layer.output_elements(elems) : layer.input_elements(elems);
    return status.ok() ? elems : 0U;
}

[[nodiscard]] auto trainable_parameter_count(const LayerInfo& layer) noexcept -> std::size_t
{
    return static_cast<std::size_t>(layer.weights_trainable) + layer.bias_trainable;
}

[[nodiscard]] auto supports_runtime_training(LayerId id) noexcept -> bool
{
    return id == LayerId::Linear || id == LayerId::ReLU || id == LayerId::Softmax || id == LayerId::Flatten;
}

[[nodiscard]] auto span_bytes_for_elements(std::size_t elements, NumericDataType type) noexcept -> std::size_t
{
    return elements * numeric_data_type_bytes(type);
}

#if EDGEIST_ENABLE_TRAINING
[[nodiscard]] auto apply_gradient_precision(Span<float> values, NumericDataType type, float int8_scale) noexcept -> Status
{
    if (values.data() == nullptr && values.size() != 0U) {
        return make_status(ErrorCode::NullPointer, "gradient precision span is null");
    }
    if (type == NumericDataType::Float32) {
        return Status::success();
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (type == NumericDataType::Float16) {
            values[i] = float16_bits_to_float32(float32_to_float16_bits(values[i]));
        } else {
            values[i] = dequantize_int8(quantize_int8(values[i], int8_scale), int8_scale);
        }
    }
    return Status::success();
}
#endif

} // namespace

auto ModelRuntime::init(ModelView view, StorageBackend& trainable_storage, ScratchArena& arena,
    TrainingConfig config) -> Status
{
#if !EDGEIST_ENABLE_TRAINING
    if (config.mode != RuntimeMode::InferenceOnly) {
        return make_status(ErrorCode::DisabledFeature, "training was disabled at compile time");
    }
#endif
    EDGEIST_RETURN_IF_ERROR(view.validate());
    view_ = std::move(view);
    trainable_storage_ = &trainable_storage;
    arena_ = &arena;
    config_ = config;
    initialized_ = false;
    stats_ = {};
    arena_->reset();

    if (config_.mode != RuntimeMode::InferenceOnly && config_.micro_batch_size == 0U) {
        return make_status(ErrorCode::InvalidArgument, "micro_batch_size must be non-zero for training");
    }
    if (trainable_storage_->size() < view_.info().trainable_bytes) {
        return make_status(ErrorCode::StorageError, "trainable storage is smaller than validated trainable image");
    }

    for (const auto& layer : view_.info().layers) {
        if (layer.has_trainable_params() && !is_supported_runtime_encoding(layer.data_encoding)) {
            return make_status(ErrorCode::UnsupportedDataEncoding, "runtime parameter encoding must be int8, fp16, or fp32");
        }
        if (layer.has_trainable_params() || layer.weights_frozen != 0U || layer.bias_frozen != 0U) {
            EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, config_.inference_data_type));
            if (config_.mode != RuntimeMode::InferenceOnly) {
                EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, config_.training_data_type));
            }
        }
    }

    EDGEIST_RETURN_IF_ERROR(MemoryPlanner::estimate(view_.info(), config_, stats_.memory));

    std::size_t max_activation_elements = 0;
    std::size_t saved_total_bytes = 0;
    std::size_t gradient_total = 0;
    std::size_t optimizer_total_per_state = 0;
    std::size_t max_argmax_elements = 0;

    saved_offsets_.assign(view_.info().layers.size(), npos);
    gradient_offsets_.assign(view_.info().layers.size(), npos);
    optimizer_offsets_.assign(view_.info().layers.size(), npos);

    for (std::size_t i = 0; i < view_.info().layers.size(); ++i) {
        const auto& layer = view_.info().layers[i];
        max_activation_elements = std::max(max_activation_elements, layer_elements(layer, false));
        max_activation_elements = std::max(max_activation_elements, layer_elements(layer, true));
        if (layer.id == LayerId::MaxPool2d) {
            max_argmax_elements = std::max(max_argmax_elements, layer_elements(layer, true));
        }
        if (config_.mode != RuntimeMode::InferenceOnly) {
            saved_offsets_[i] = saved_total_bytes;
            saved_total_bytes += span_bytes_for_elements(layer_elements(layer, false), config_.training_data_type);
            if (should_train_layer(i) && layer.has_trainable_params()) {
                if (!supports_runtime_training(layer.id)) {
                    return make_status(ErrorCode::UnsupportedLayer, "runtime training currently supports Linear/ReLU/Flatten/Softmax graphs");
                }
                if (layer.weights_frozen != 0U || layer.bias_frozen != 0U) {
                    return make_status(ErrorCode::UnsupportedLayer, "runtime training currently requires fully trainable parameter blocks");
                }
                EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, config_.training_data_type));
                gradient_offsets_[i] = gradient_total;
                optimizer_offsets_[i] = optimizer_total_per_state;
                gradient_total += trainable_parameter_count(layer);
                optimizer_total_per_state += trainable_parameter_count(layer);
            }
        }
    }

    if (config_.mode == RuntimeMode::LastLayerTraining) {
        gradient_total = 0;
        optimizer_total_per_state = 0;
        std::fill(gradient_offsets_.begin(), gradient_offsets_.end(), npos);
        std::fill(optimizer_offsets_.begin(), optimizer_offsets_.end(), npos);
        for (std::size_t idx = view_.info().layers.size(); idx-- > 0;) {
            const auto& layer = view_.info().layers[idx];
            if (layer.has_trainable_params()) {
                if (layer.weights_frozen != 0U || layer.bias_frozen != 0U) {
                    return make_status(ErrorCode::UnsupportedLayer, "last-layer training requires a fully trainable last parameter layer");
                }
                EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, config_.training_data_type));
                gradient_offsets_[idx] = 0;
                optimizer_offsets_[idx] = 0;
                gradient_total = trainable_parameter_count(layer);
                optimizer_total_per_state = trainable_parameter_count(layer);
                break;
            }
        }
    }

    const std::size_t activation_element_bytes = config_.mode == RuntimeMode::InferenceOnly
        ? numeric_data_type_bytes(config_.inference_data_type)
        : std::max(numeric_data_type_bytes(config_.inference_data_type), numeric_data_type_bytes(config_.training_data_type));
    const std::size_t activation_bytes = max_activation_elements * activation_element_bytes;
    std::byte* a = nullptr;
    std::byte* b = nullptr;
    EDGEIST_RETURN_IF_ERROR(arena_->allocate_bytes(activation_bytes, a));
    EDGEIST_RETURN_IF_ERROR(arena_->allocate_bytes(activation_bytes, b));
    activation_a_ = ByteSpan(a, activation_bytes);
    activation_b_ = ByteSpan(b, activation_bytes);

    if (config_.mode != RuntimeMode::InferenceOnly) {
        float* ga = nullptr;
        float* gb = nullptr;
        std::byte* saved = nullptr;
        float* grads = nullptr;
        float* opt1 = nullptr;
        float* opt2 = nullptr;
        std::uint32_t* argmax = nullptr;
        EDGEIST_RETURN_IF_ERROR(arena_->allocate(max_activation_elements, ga));
        EDGEIST_RETURN_IF_ERROR(arena_->allocate(max_activation_elements, gb));
        EDGEIST_RETURN_IF_ERROR(arena_->allocate_bytes(saved_total_bytes, saved));
        EDGEIST_RETURN_IF_ERROR(arena_->allocate(gradient_total, grads));
        if (config_.optimizer == OptimizerKind::Momentum || config_.optimizer == OptimizerKind::Adam) {
            EDGEIST_RETURN_IF_ERROR(arena_->allocate(optimizer_total_per_state, opt1));
            std::fill(opt1, opt1 + optimizer_total_per_state, 0.0F);
        }
        if (config_.optimizer == OptimizerKind::Adam) {
            EDGEIST_RETURN_IF_ERROR(arena_->allocate(optimizer_total_per_state, opt2));
            std::fill(opt2, opt2 + optimizer_total_per_state, 0.0F);
        }
        EDGEIST_RETURN_IF_ERROR(arena_->allocate(max_argmax_elements, argmax));
        grad_a_ = Span<float>(ga, max_activation_elements);
        grad_b_ = Span<float>(gb, max_activation_elements);
        saved_activations_ = ByteSpan(saved, saved_total_bytes);
        gradients_ = Span<float>(grads, gradient_total);
        opt_state_1_ = Span<float>(opt1, (config_.optimizer == OptimizerKind::Momentum || config_.optimizer == OptimizerKind::Adam) ? optimizer_total_per_state : 0U);
        opt_state_2_ = Span<float>(opt2, config_.optimizer == OptimizerKind::Adam ? optimizer_total_per_state : 0U);
        argmax_ = Span<std::uint32_t>(argmax, max_argmax_elements);
        std::fill(gradients_.begin(), gradients_.end(), 0.0F);
    } else if (max_argmax_elements > 0U) {
        std::uint32_t* argmax = nullptr;
        EDGEIST_RETURN_IF_ERROR(arena_->allocate(max_argmax_elements, argmax));
        argmax_ = Span<std::uint32_t>(argmax, max_argmax_elements);
    }

    initialized_ = true;
    return Status::success();
}

auto ModelRuntime::should_train_layer(std::size_t layer_index) const noexcept -> bool
{
    if (config_.mode == RuntimeMode::InferenceOnly) {
        return false;
    }
    if (config_.mode == RuntimeMode::FullTraining) {
        return true;
    }
    if (config_.mode == RuntimeMode::FrozenLayerTraining) {
        return layer_index >= config_.frozen_prefix_layers;
    }
    if (config_.mode == RuntimeMode::LastLayerTraining) {
        for (std::size_t idx = view_.info().layers.size(); idx-- > 0;) {
            if (view_.info().layers[idx].has_trainable_params()) {
                return idx == layer_index;
            }
        }
    }
    return false;
}

auto ModelRuntime::typed_view_for_bytes(ByteSpan bytes, NumericDataType type, std::size_t elements, float scale) noexcept -> MutableTypedTensorView
{
    return MutableTypedTensorView { bytes.subspan(0, typed_tensor_bytes(elements, type)), type, elements, scale };
}

auto ModelRuntime::typed_view_for_bytes(ConstByteSpan bytes, NumericDataType type, std::size_t elements, float scale) const noexcept -> ConstTypedTensorView
{
    return ConstTypedTensorView { bytes.subspan(0, typed_tensor_bytes(elements, type)), type, elements, scale };
}

auto ModelRuntime::require_layer_encoding(const LayerInfo& layer, NumericDataType execution_type) const noexcept -> Status
{
    if (!layer.has_trainable_params()) {
        return Status::success();
    }
    if (data_encoding_to_numeric_type(layer.data_encoding) != execution_type) {
        return make_status(ErrorCode::UnsupportedDataEncoding,
            "selected runtime data type must match the exported parameter encoding; export weights as int8/fp16/fp32 instead of casting float32 at runtime");
    }
    return Status::success();
}

auto ModelRuntime::layer_param_const_views(const LayerInfo& layer, ParameterConstViews& views) const noexcept -> Status
{
    if (trainable_storage_ == nullptr) {
        return make_status(ErrorCode::NotInitialized, "runtime has no trainable storage");
    }
    if (layer.weights_frozen != 0U || layer.bias_frozen != 0U) {
        return make_status(ErrorCode::UnsupportedLayer, "runtime execution currently requires contiguous trainable parameters");
    }
    const auto type = data_encoding_to_numeric_type(layer.data_encoding);
    const auto elem_bytes = data_encoding_bytes(layer.data_encoding);
    const auto w_bytes = static_cast<std::size_t>(layer.weights_trainable) * elem_bytes;
    const auto b_bytes = static_cast<std::size_t>(layer.bias_trainable) * elem_bytes;
    const auto weights = const_byte_span_from_storage(*trainable_storage_, layer.offsets.weightsTrainableOffset, w_bytes);
    const auto bias = const_byte_span_from_storage(*trainable_storage_, layer.offsets.biasTrainableOffset, b_bytes);
    if ((layer.weights_trainable > 0U && weights.data() == nullptr) || (layer.bias_trainable > 0U && bias.data() == nullptr)) {
        return make_status(ErrorCode::StorageError, "parameter span outside trainable storage");
    }
    views.weights = ConstTypedTensorView { weights, type, layer.weights_trainable, parameter_scale_for_weights() };
    views.bias = ConstTypedTensorView { bias, type, layer.bias_trainable, parameter_scale_for_bias() };
    return Status::success();
}

auto ModelRuntime::layer_param_mutable_views(const LayerInfo& layer, ParameterMutableViews& views) noexcept -> Status
{
    if (trainable_storage_ == nullptr) {
        return make_status(ErrorCode::NotInitialized, "runtime has no trainable storage");
    }
    if (layer.weights_frozen != 0U || layer.bias_frozen != 0U) {
        return make_status(ErrorCode::UnsupportedLayer, "runtime execution currently requires contiguous trainable parameters");
    }
    const auto type = data_encoding_to_numeric_type(layer.data_encoding);
    const auto elem_bytes = data_encoding_bytes(layer.data_encoding);
    const auto w_bytes = static_cast<std::size_t>(layer.weights_trainable) * elem_bytes;
    const auto b_bytes = static_cast<std::size_t>(layer.bias_trainable) * elem_bytes;
    const auto weights = byte_span_from_storage(*trainable_storage_, layer.offsets.weightsTrainableOffset, w_bytes);
    const auto bias = byte_span_from_storage(*trainable_storage_, layer.offsets.biasTrainableOffset, b_bytes);
    if ((layer.weights_trainable > 0U && weights.data() == nullptr) || (layer.bias_trainable > 0U && bias.data() == nullptr)) {
        return make_status(ErrorCode::StorageError, "parameter span outside trainable storage");
    }
    views.weights = MutableTypedTensorView { weights, type, layer.weights_trainable, parameter_scale_for_weights() };
    views.bias = MutableTypedTensorView { bias, type, layer.bias_trainable, parameter_scale_for_bias() };
    return Status::success();
}

auto ModelRuntime::forward_impl(ConstSpan<float> input, Span<float> output, bool training, ActivationTraceBuffer* trace) noexcept -> Status
{
    std::size_t model_input_elements = 0;
    std::size_t model_output_elements = 0;
    EDGEIST_RETURN_IF_ERROR(view_.info().input.elements(model_input_elements));
    EDGEIST_RETURN_IF_ERROR(view_.info().output.elements(model_output_elements));
    if (input.size() < model_input_elements || output.size() < model_output_elements) {
        return make_status(ErrorCode::BufferTooSmall, "model input or output span too small");
    }
    if (trace != nullptr) {
        if (trace->offsets.size() < view_.info().layers.size() + 1U) {
            return make_status(ErrorCode::BufferTooSmall, "activation trace offset span too small");
        }
        trace->layer_count = 0;
        trace->used_values = 0;
    }

    const NumericDataType execution_type = training ? config_.training_data_type : config_.inference_data_type;
    auto current_mut = typed_view_for_bytes(activation_a_, execution_type, model_input_elements, config_.quantization.activation_scale);
    EDGEIST_RETURN_IF_ERROR(encode_float_span(input, current_mut));
    ConstTypedTensorView current { const_bytes_from_mutable(current_mut.bytes), execution_type, model_input_elements, config_.quantization.activation_scale };
    ByteSpan* next_buffer = &activation_b_;

    for (std::size_t i = 0; i < view_.info().layers.size(); ++i) {
        const auto& layer = view_.info().layers[i];
        const auto in_size = layer_elements(layer, false);
        const auto out_size = layer_elements(layer, true);
        if (current.elements < in_size) {
            return make_status(ErrorCode::BufferTooSmall, "internal activation tensor too small");
        }
        auto out = typed_view_for_bytes(*next_buffer, execution_type, out_size, config_.quantization.activation_scale);
        EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(out, "internal output activation tensor too small"));

        if (training && saved_offsets_[i] != npos) {
            const auto bytes = typed_tensor_bytes(in_size, execution_type);
            if (saved_offsets_[i] > saved_activations_.size() || bytes > saved_activations_.size() - saved_offsets_[i]) {
                return make_status(ErrorCode::BufferTooSmall, "saved activation tensor too small");
            }
            std::memcpy(saved_activations_.data() + saved_offsets_[i], current.bytes.data(), bytes);
        }

        switch (layer.id) {
        case LayerId::Linear: {
            EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, execution_type));
            ParameterConstViews params;
            EDGEIST_RETURN_IF_ERROR(layer_param_const_views(layer, params));
            EDGEIST_RETURN_IF_ERROR(layers::LinearLayer::forward(current, params.weights, params.bias, out,
                layer.input.width, layer.output.width));
            break;
        }
        case LayerId::ReLU:
            EDGEIST_RETURN_IF_ERROR(layers::ReLULayer::forward(current, out));
            break;
        case LayerId::Softmax:
            EDGEIST_RETURN_IF_ERROR(layers::SoftmaxLayer::forward(current, out));
            break;
        case LayerId::Flatten:
            EDGEIST_RETURN_IF_ERROR(layers::FlattenLayer::forward(current, out));
            break;
        case LayerId::MaxPool2d: {
            layers::Pool2DParams params { layer.input.channels, layer.input.height, layer.input.width, layer.output.height,
                layer.output.width, layer.kernel[0], layer.kernel[1], layer.kernel[2], layer.kernel[2], layer.kernel[3], layer.kernel[3] };
            EDGEIST_RETURN_IF_ERROR(layers::MaxPool2DLayer::forward(current, out, Span<std::uint32_t>(argmax_.data(), out_size), params));
            break;
        }
        case LayerId::AdaptiveAvgPool1d:
            EDGEIST_RETURN_IF_ERROR(layers::AdaptiveAvgPool1DLayer::forward(current, out,
                layer.input.channels, layer.input.width, layer.output.width));
            break;
        case LayerId::AdaptiveAvgPool2d:
            EDGEIST_RETURN_IF_ERROR(layers::AdaptiveAvgPool2DLayer::forward(current, out,
                layer.input.channels, layer.input.height, layer.input.width, layer.output.height, layer.output.width));
            break;
        case LayerId::Conv2d: {
            EDGEIST_RETURN_IF_ERROR(require_layer_encoding(layer, execution_type));
            ParameterConstViews params_view;
            EDGEIST_RETURN_IF_ERROR(layer_param_const_views(layer, params_view));
            layers::Conv2DParams params { layer.input.channels, layer.output.channels, layer.input.height, layer.input.width,
                layer.output.height, layer.output.width, layer.kernel[0], layer.kernel[1], layer.kernel[2], layer.kernel[2],
                layer.kernel[3], layer.kernel[3], layer.dilation[0], layer.dilation[1], layer.groups };
            EDGEIST_RETURN_IF_ERROR(layers::Conv2DLayer::forward(current, params_view.weights, params_view.bias, out, params));
            break;
        }
        case LayerId::BatchNorm1d:
        case LayerId::BatchNorm2d:
        case LayerId::Dropout:
            EDGEIST_RETURN_IF_ERROR(layers::DropoutLayer::forward_inference(current, out));
            break;
        }

        if (trace != nullptr) {
            if (trace->used_values > trace->values.size() || out_size > trace->values.size() - trace->used_values) {
                return make_status(ErrorCode::BufferTooSmall, "activation trace value span too small");
            }
            trace->offsets[i] = trace->used_values;
            EDGEIST_RETURN_IF_ERROR(decode_to_float_span(ConstTypedTensorView { const_bytes_from_mutable(out.bytes), out.type, out.elements, out.scale },
                Span<float>(trace->values.data() + trace->used_values, out_size)));
            trace->used_values += static_cast<std::uint32_t>(out_size);
            trace->layer_count = static_cast<std::uint32_t>(i + 1U);
            trace->offsets[i + 1U] = trace->used_values;
        }

        current = ConstTypedTensorView { const_bytes_from_mutable(out.bytes), execution_type, out_size, config_.quantization.activation_scale };
        next_buffer = (next_buffer == &activation_a_) ? &activation_b_ : &activation_a_;
    }

    EDGEIST_RETURN_IF_ERROR(decode_to_float_span(current, output));
    return Status::success();
}

auto ModelRuntime::inference(ConstSpan<float> input, Span<float> output) noexcept -> Status
{
    if (!initialized_) {
        return make_status(ErrorCode::NotInitialized, "runtime not initialized");
    }
    return forward_impl(input, output, false, nullptr);
}

auto ModelRuntime::inference_with_trace(ConstSpan<float> input, Span<float> output, ActivationTraceBuffer& trace) noexcept -> Status
{
    if (!initialized_) {
        return make_status(ErrorCode::NotInitialized, "runtime not initialized");
    }
    return forward_impl(input, output, false, &trace);
}

auto ModelRuntime::begin_micro_batch() noexcept -> Status
{
#if !EDGEIST_ENABLE_TRAINING
    return make_status(ErrorCode::DisabledFeature, "training was disabled at compile time");
#endif
    if (!initialized_) {
        return make_status(ErrorCode::NotInitialized, "runtime not initialized");
    }
    if (config_.mode == RuntimeMode::InferenceOnly) {
        return make_status(ErrorCode::DisabledFeature, "training disabled by runtime mode");
    }
    std::fill(gradients_.begin(), gradients_.end(), 0.0F);
    return Status::success();
}

auto ModelRuntime::train_sample(ConstSpan<float> input, ConstSpan<float> expected_output, float* loss_out) noexcept -> Status
{
#if !EDGEIST_ENABLE_TRAINING
    (void)input;
    (void)expected_output;
    (void)loss_out;
    return make_status(ErrorCode::DisabledFeature, "training was disabled at compile time");
#else
    if (!initialized_) {
        return make_status(ErrorCode::NotInitialized, "runtime not initialized");
    }
    if (config_.mode == RuntimeMode::InferenceOnly) {
        return make_status(ErrorCode::DisabledFeature, "training disabled by runtime mode");
    }
    for (const auto& layer : view_.info().layers) {
        if (!supports_runtime_training(layer.id)) {
            return make_status(ErrorCode::UnsupportedLayer, "generic runtime training currently supports Linear/ReLU/Flatten/Softmax graphs");
        }
    }

    std::size_t out_elems = 0;
    EDGEIST_RETURN_IF_ERROR(view_.info().output.elements(out_elems));
    Span<float> prediction(grad_b_.data(), out_elems);
    EDGEIST_RETURN_IF_ERROR(forward_impl(input, prediction, true, nullptr));
    if (expected_output.size() < out_elems) {
        return make_status(ErrorCode::BufferTooSmall, "expected output too small");
    }
    float loss = 0.0F;
    EDGEIST_RETURN_IF_ERROR(layers::CrossEntropyLoss::loss(ConstSpan<float>(prediction.data(), out_elems), expected_output, loss));
    if (loss_out != nullptr) {
        *loss_out = loss;
    }
    stats_.last_loss = loss;
    EDGEIST_RETURN_IF_ERROR(layers::CrossEntropyLoss::gradient(ConstSpan<float>(prediction.data(), out_elems), expected_output,
        Span<float>(grad_a_.data(), out_elems)));
    EDGEIST_RETURN_IF_ERROR(apply_gradient_precision(Span<float>(grad_a_.data(), out_elems), config_.training_data_type, config_.quantization.gradient_scale));

    ConstSpan<float> current_grad(grad_a_.data(), out_elems);
    Span<float> next_grad = grad_b_;

    for (std::size_t idx = view_.info().layers.size(); idx-- > 0;) {
        const auto& layer = view_.info().layers[idx];
        const auto in_size = layer_elements(layer, false);
        const auto out_size = layer_elements(layer, true);
        const auto saved_bytes = saved_activations_.subspan(saved_offsets_[idx], typed_tensor_bytes(in_size, config_.training_data_type));
        ConstTypedTensorView saved { const_bytes_from_mutable(saved_bytes), config_.training_data_type, in_size, config_.quantization.activation_scale };
        Span<float> dst(next_grad.data(), in_size);

        switch (layer.id) {
        case LayerId::Softmax:
            std::copy(current_grad.begin(), current_grad.begin() + out_size, dst.begin());
            break;
        case LayerId::ReLU:
            EDGEIST_RETURN_IF_ERROR(layers::ReLULayer::backward(saved, current_grad, dst));
            break;
        case LayerId::Flatten:
            EDGEIST_RETURN_IF_ERROR(layers::FlattenLayer::backward(current_grad, dst));
            break;
        case LayerId::Linear: {
            ParameterConstViews params;
            EDGEIST_RETURN_IF_ERROR(layer_param_const_views(layer, params));
            const bool train_this_layer = should_train_layer(idx) && gradient_offsets_[idx] != npos;
            if (train_this_layer) {
                Span<float> gw(gradients_.data() + gradient_offsets_[idx], layer.weights_trainable);
                Span<float> gb(gradients_.data() + gradient_offsets_[idx] + layer.weights_trainable, layer.bias_trainable);
                EDGEIST_RETURN_IF_ERROR(layers::LinearLayer::backward(saved, current_grad, params.weights, dst, gw, gb,
                    layer.input.width, layer.output.width, true));
                EDGEIST_RETURN_IF_ERROR(apply_gradient_precision(gw, config_.training_data_type, config_.quantization.gradient_scale));
                EDGEIST_RETURN_IF_ERROR(apply_gradient_precision(gb, config_.training_data_type, config_.quantization.gradient_scale));
            } else {
                std::fill(dst.begin(), dst.begin() + in_size, 0.0F);
                for (std::uint32_t out = 0; out < layer.output.width; ++out) {
                    const auto row = static_cast<std::size_t>(out) * layer.input.width;
                    for (std::uint32_t in = 0; in < layer.input.width; ++in) {
                        float weight = 0.0F;
                        EDGEIST_RETURN_IF_ERROR(read_typed_value(params.weights, row + in, weight));
                        dst[in] += weight * current_grad[out];
                    }
                }
            }
            break;
        }
        default:
            return make_status(ErrorCode::UnsupportedLayer, "unsupported training layer encountered");
        }
        EDGEIST_RETURN_IF_ERROR(apply_gradient_precision(Span<float>(dst.data(), in_size), config_.training_data_type, config_.quantization.gradient_scale));
        current_grad = ConstSpan<float>(dst.data(), in_size);
        next_grad = (next_grad.data() == grad_a_.data()) ? grad_b_ : grad_a_;
    }

    ++stats_.trained_samples;
    return Status::success();
#endif
}

auto ModelRuntime::apply_typed_optimizer(MutableTypedTensorView params, ConstSpan<float> grads, OptimizerStateView state,
    float learning_scale) noexcept -> Status
{
    if (params.elements != grads.size()) {
        return make_status(ErrorCode::ShapeMismatch, "optimizer parameter/gradient size mismatch");
    }
    EDGEIST_RETURN_IF_ERROR(validate_typed_tensor(params, "optimizer typed parameter tensor invalid"));
    if (grads.data() == nullptr && grads.size() != 0U) {
        return make_status(ErrorCode::NullPointer, "optimizer gradients are null");
    }
    const auto t = static_cast<float>(state.timestep == 0 ? 1U : state.timestep);
    const float beta1_corr = 1.0F - std::pow(config_.beta1, t);
    const float beta2_corr = 1.0F - std::pow(config_.beta2, t);
    if (config_.optimizer == OptimizerKind::Adam && (beta1_corr == 0.0F || beta2_corr == 0.0F)) {
        return make_status(ErrorCode::NumericError, "Adam bias correction underflow");
    }

    for (std::size_t i = 0; i < params.elements; ++i) {
        float param = 0.0F;
        EDGEIST_RETURN_IF_ERROR(read_typed_value(ConstTypedTensorView { const_bytes_from_mutable(params.bytes), params.type, params.elements, params.scale }, i, param));
        const float grad = grads[i] * learning_scale;
        switch (config_.optimizer) {
        case OptimizerKind::Sgd:
            param -= config_.learning_rate * grad;
            break;
        case OptimizerKind::Momentum:
            if (state.momentum_or_m.size() < params.elements) {
                return make_status(ErrorCode::BufferTooSmall, "momentum state too small");
            }
            state.momentum_or_m[i] = config_.momentum * state.momentum_or_m[i] + grad;
            param -= config_.learning_rate * state.momentum_or_m[i];
            break;
        case OptimizerKind::Adam:
            if (state.momentum_or_m.size() < params.elements || state.velocity_or_v.size() < params.elements) {
                return make_status(ErrorCode::BufferTooSmall, "adam state too small");
            }
            state.momentum_or_m[i] = config_.beta1 * state.momentum_or_m[i] + (1.0F - config_.beta1) * grad;
            state.velocity_or_v[i] = config_.beta2 * state.velocity_or_v[i] + (1.0F - config_.beta2) * grad * grad;
            param -= config_.learning_rate * (state.momentum_or_m[i] / beta1_corr) / (std::sqrt(state.velocity_or_v[i] / beta2_corr) + config_.epsilon);
            break;
        }
        EDGEIST_RETURN_IF_ERROR(write_typed_value(params, i, param));
    }
    return Status::success();
}

auto ModelRuntime::apply_updates(std::uint32_t accumulated_samples) noexcept -> Status
{
#if !EDGEIST_ENABLE_TRAINING
    (void)accumulated_samples;
    return make_status(ErrorCode::DisabledFeature, "training was disabled at compile time");
#else
    if (!initialized_) {
        return make_status(ErrorCode::NotInitialized, "runtime not initialized");
    }
    if (config_.mode == RuntimeMode::InferenceOnly) {
        return make_status(ErrorCode::DisabledFeature, "training disabled by runtime mode");
    }
    if (accumulated_samples == 0U) {
        return make_status(ErrorCode::InvalidArgument, "cannot apply updates for zero samples");
    }
    const float scale = 1.0F / static_cast<float>(accumulated_samples);

    for (std::size_t i = 0; i < view_.info().layers.size(); ++i) {
        const auto& layer = view_.info().layers[i];
        if (!should_train_layer(i) || gradient_offsets_[i] == npos || !layer.has_trainable_params()) {
            continue;
        }
        ParameterMutableViews params;
        EDGEIST_RETURN_IF_ERROR(layer_param_mutable_views(layer, params));
        Span<float> gw(gradients_.data() + gradient_offsets_[i], layer.weights_trainable);
        Span<float> gb(gradients_.data() + gradient_offsets_[i] + layer.weights_trainable, layer.bias_trainable);
        const auto state_offset = optimizer_offsets_[i];
        OptimizerStateView state_w {};
        OptimizerStateView state_b {};
        state_w.timestep = std::max<std::uint32_t>(1U, stats_.trained_samples / accumulated_samples);
        state_b.timestep = state_w.timestep;
        if (config_.optimizer == OptimizerKind::Momentum) {
            state_w.momentum_or_m = Span<float>(opt_state_1_.data() + state_offset, layer.weights_trainable);
            state_b.momentum_or_m = Span<float>(opt_state_1_.data() + state_offset + layer.weights_trainable, layer.bias_trainable);
        } else if (config_.optimizer == OptimizerKind::Adam) {
            state_w.momentum_or_m = Span<float>(opt_state_1_.data() + state_offset, layer.weights_trainable);
            state_w.velocity_or_v = Span<float>(opt_state_2_.data() + state_offset, layer.weights_trainable);
            state_b.momentum_or_m = Span<float>(opt_state_1_.data() + state_offset + layer.weights_trainable, layer.bias_trainable);
            state_b.velocity_or_v = Span<float>(opt_state_2_.data() + state_offset + layer.weights_trainable, layer.bias_trainable);
        }
        EDGEIST_RETURN_IF_ERROR(apply_typed_optimizer(params.weights, ConstSpan<float>(gw.data(), gw.size()), state_w, scale));
        EDGEIST_RETURN_IF_ERROR(apply_typed_optimizer(params.bias, ConstSpan<float>(gb.data(), gb.size()), state_b, scale));
    }

    return Status::success();
#endif
}

} // namespace edgeist
