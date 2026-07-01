#include "edgeist/edgeist.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

template <std::size_t Capacity>
struct ByteImage {
    std::array<std::byte, Capacity> bytes {};
    std::size_t used { 0 };
};

template <std::size_t Capacity, typename T>
bool append_struct(ByteImage<Capacity>& image, const T& value)
{
    if (sizeof(T) > image.bytes.size() - image.used) {
        return false;
    }
    std::memcpy(image.bytes.data() + image.used, &value, sizeof(T));
    image.used += sizeof(T);
    return true;
}

template <std::size_t Capacity>
bool write_u32(ByteImage<Capacity>& image, std::size_t offset, std::uint32_t value)
{
    if (offset > image.bytes.size() || sizeof(value) > image.bytes.size() - offset) {
        return false;
    }
    std::memcpy(image.bytes.data() + offset, &value, sizeof(value));
    return true;
}

template <std::size_t Capacity>
bool append_float(ByteImage<Capacity>& image, float value)
{
    if (sizeof(float) > image.bytes.size() - image.used) {
        return false;
    }
    std::memcpy(image.bytes.data() + image.used, &value, sizeof(float));
    image.used += sizeof(float);
    return true;
}

ByteImage<512> make_demo_model_image()
{
    using namespace edgeist;
    constexpr std::uint16_t layer_count = 4;
    ByteImage<512> image {};

    ModelHeaderPrefix header {};
    header.headerSize = static_cast<std::uint16_t>(sizeof(ModelHeaderPrefix) + layer_count * sizeof(std::uint32_t));
    header.magicNumber = kNmcfMagic;
    header.version = { '0', '2', '.', '0', '0', '.', '0', '0' };
    header.layerNrs = layer_count;
    header.channelsIn = 1;
    header.dimensionInputX = 2;
    header.dimensionInputY = 1;
    header.channelsOut = 1;
    header.dimensionOutputX = 2;
    header.dimensionOutputY = 1;
    (void)append_struct(image, header);

    const std::size_t offset_table = image.used;
    image.used += layer_count * sizeof(std::uint32_t);

    std::array<std::uint32_t, layer_count> offsets {};
    offsets[0] = static_cast<std::uint32_t>(image.used);
    LinearHeader l0 {};
    l0.id = static_cast<std::uint8_t>(LayerId::Linear);
    l0.layerNr = 0;
    l0.dimensionInputX = 2;
    l0.dimensionOutputX = 3;
    l0.dataEncoding = static_cast<std::uint8_t>(DataEncodingId::Float32);
    l0.weightsAmountTrainable = 6;
    l0.biasAmountTrainable = 3;
    l0.offsets.weightsTrainableOffset = 0;
    l0.offsets.biasTrainableOffset = 6 * sizeof(float);
    (void)append_struct(image, l0);

    offsets[1] = static_cast<std::uint32_t>(image.used);
    UnaryHeader relu {};
    relu.id = static_cast<std::uint8_t>(LayerId::ReLU);
    relu.layerNr = 1;
    relu.dimensionInputX = 3;
    relu.dimensionOutputX = 3;
    (void)append_struct(image, relu);

    offsets[2] = static_cast<std::uint32_t>(image.used);
    LinearHeader l2 {};
    l2.id = static_cast<std::uint8_t>(LayerId::Linear);
    l2.layerNr = 2;
    l2.dimensionInputX = 3;
    l2.dimensionOutputX = 2;
    l2.dataEncoding = static_cast<std::uint8_t>(DataEncodingId::Float32);
    l2.weightsAmountTrainable = 6;
    l2.biasAmountTrainable = 2;
    l2.offsets.weightsTrainableOffset = (6 + 3) * sizeof(float);
    l2.offsets.biasTrainableOffset = (6 + 3 + 6) * sizeof(float);
    (void)append_struct(image, l2);

    offsets[3] = static_cast<std::uint32_t>(image.used);
    UnaryHeader softmax {};
    softmax.id = static_cast<std::uint8_t>(LayerId::Softmax);
    softmax.layerNr = 3;
    softmax.dimensionInputX = 2;
    softmax.dimensionOutputX = 2;
    (void)append_struct(image, softmax);

    for (std::size_t i = 0; i < offsets.size(); ++i) {
        (void)write_u32(image, offset_table + i * sizeof(std::uint32_t), offsets[i]);
    }
    return image;
}

ByteImage<17 * sizeof(float)> make_demo_trainable_image()
{
    ByteImage<17 * sizeof(float)> image {};
    const std::array<float, 17> values {
        0.2F, -0.1F, -0.3F, 0.4F, 0.1F, 0.2F, 0.0F, 0.1F, -0.1F,
        0.3F, -0.2F, 0.1F, -0.4F, 0.2F, 0.5F, 0.0F, 0.0F,
    };
    for (float value : values) {
        (void)append_float(image, value);
    }
    return image;
}

int print_status_if_error(edgeist::Status status, const char* step)
{
    if (status.ok()) {
        return 0;
    }
    std::printf("%s failed: %s: %.*s\n", step, edgeist::to_string(status.code).data(),
        static_cast<int>(status.message.size()), status.message.data());
    return 1;
}

} // namespace

int main()
{
    auto model = make_demo_model_image();
    auto trainable = make_demo_trainable_image();

    std::array<std::byte, 16 * 1024> scratch {};
    edgeist::ScratchArena arena { edgeist::ByteSpan(scratch) };
    edgeist::SramStorage trainable_storage { edgeist::ByteSpan(trainable.bytes.data(), trainable.used) };
    edgeist::ModelView view { edgeist::ConstByteSpan(model.bytes.data(), model.used), edgeist::ConstByteSpan(trainable.bytes.data(), trainable.used) };

    edgeist::TrainingConfig config {};
#if EDGEIST_ENABLE_TRAINING
    config.mode = edgeist::RuntimeMode::FullTraining;
#else
    config.mode = edgeist::RuntimeMode::InferenceOnly;
#endif
    config.optimizer = edgeist::OptimizerKind::Sgd;
    config.learning_rate = 0.05F;
    // The demo model exports float32 weights. Runtime precision must match the
    // exported parameter encoding; use int8/fp16 here only with a model image
    // exported in the same data type.
    config.inference_data_type = edgeist::NumericDataType::Float32;
    config.training_data_type = edgeist::NumericDataType::Float32;

    edgeist::ModelRuntime runtime;
    if (print_status_if_error(runtime.init(view, trainable_storage, arena, config), "runtime init") != 0) {
        return 1;
    }

    std::array<float, 2> input { 0.6F, -0.4F };
    std::array<float, 2> output {};
    std::array<float, 10> trace_values {};
    std::array<std::uint32_t, 5> trace_offsets {};
    edgeist::ActivationTraceBuffer trace { edgeist::Span<float>(trace_values), edgeist::Span<std::uint32_t>(trace_offsets) };

    if (print_status_if_error(runtime.inference_with_trace(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output), trace), "inference") != 0) {
        return 1;
    }
    std::printf("embedded demo output before training: %.6f %.6f\n", output[0], output[1]);
    std::printf("captured %u layer activation tensors using %u float slots\n", trace.layer_count, trace.used_values);

#if EDGEIST_ENABLE_TRAINING
    std::array<float, 2> target { 1.0F, 0.0F };
    float loss = 0.0F;
    if (print_status_if_error(runtime.begin_micro_batch(), "begin micro batch") != 0) {
        return 1;
    }
    if (print_status_if_error(runtime.train_sample(edgeist::ConstSpan<float>(input), edgeist::ConstSpan<float>(target), &loss), "train sample") != 0) {
        return 1;
    }
    if (print_status_if_error(runtime.apply_updates(1), "apply update") != 0) {
        return 1;
    }
    std::printf("embedded demo single-sample loss: %.6f\n", loss);
#else
    std::printf("training is disabled in this build\n");
#endif

    std::printf("scratch peak bytes: %zu\n", arena.peak());
    return 0;
}
