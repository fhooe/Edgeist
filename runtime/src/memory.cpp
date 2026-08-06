#include "edgeist/runtime/memory.hpp"
#include "edgeist/runtime/optimizers.hpp"

#include <sstream>

namespace edgeist {

auto ScratchArena::allocate_bytes(std::size_t bytes, std::byte*& out, std::size_t alignment) noexcept -> Status
{
    const auto aligned = align_up(offset_, alignment);
    std::size_t end = 0;
    if (!checked_add(aligned, bytes, end) || end > buffer_.size()) {
        out = nullptr;
        return make_status(ErrorCode::OutOfMemory, "scratch arena exhausted");
    }
    out = buffer_.data() + aligned;
    offset_ = end;
    if (offset_ > peak_) {
        peak_ = offset_;
    }
    return Status::success();
}

auto MemoryReport::to_json() const -> std::string
{
    std::ostringstream os;
    os << "{\n"
       << "  \"model_flash_bytes\": " << model_flash_bytes << ",\n"
       << "  \"trainable_storage_bytes\": " << trainable_storage_bytes << ",\n"
       << "  \"persistent_sram_bytes\": " << persistent_sram_bytes << ",\n"
       << "  \"activation_scratch_bytes\": " << activation_scratch_bytes << ",\n"
       << "  \"gradient_bytes\": " << gradient_bytes << ",\n"
       << "  \"optimizer_state_bytes\": " << optimizer_state_bytes << ",\n"
       << "  \"temporary_bytes\": " << temporary_bytes << ",\n"
       << "  \"total_peak_sram_bytes\": " << total_peak_sram_bytes << ",\n"
       << "  \"inference_element_bytes\": " << inference_element_bytes << ",\n"
       << "  \"training_element_bytes\": " << training_element_bytes << "\n"
       << "}";
    return os.str();
}

auto MemoryPlanner::estimate(const ModelInfo& info, const TrainingConfig& config, MemoryReport& report) noexcept -> Status
{
    report = {};
    report.model_flash_bytes = info.model_bytes;
    report.trainable_storage_bytes = info.trainable_bytes;
    report.inference_element_bytes = numeric_data_type_bytes(config.inference_data_type);
    report.training_element_bytes = numeric_data_type_bytes(config.training_data_type);

    std::size_t max_activation_elements = 0;
    std::size_t saved_activation_elements = 0;
    std::size_t gradient_parameter_elements = 0;
    std::size_t optimizer_parameter_elements = 0;
    std::size_t argmax_elements = 0;
    std::size_t dropout_mask_elements = 0;

    for (std::size_t i = 0; i < info.layers.size(); ++i) {
        const auto& layer = info.layers[i];
        std::size_t in_elems = 0;
        std::size_t out_elems = 0;
        EDGEIST_RETURN_IF_ERROR(layer.input_elements(in_elems));
        EDGEIST_RETURN_IF_ERROR(layer.output_elements(out_elems));
        if (in_elems > max_activation_elements) {
            max_activation_elements = in_elems;
        }
        if (out_elems > max_activation_elements) {
            max_activation_elements = out_elems;
        }
        if (config.mode != RuntimeMode::InferenceOnly) {
            saved_activation_elements += in_elems;
            const bool strategy_trains_layer =
                config.mode == RuntimeMode::FullTraining
                || (config.mode == RuntimeMode::FrozenLayerTraining && i >= config.frozen_prefix_layers)
                || (config.mode == RuntimeMode::LastLayerTraining && layer.has_trainable_params());
            if (strategy_trains_layer && layer.has_trainable_params()) {
                gradient_parameter_elements += layer.training_parameter_elements();
            }
            if (layer.id == LayerId::MaxPool2d) {
                argmax_elements += out_elems;
            }
            if (layer.id == LayerId::Dropout) {
                dropout_mask_elements += in_elems;
            }
        }
    }

    if (config.mode == RuntimeMode::LastLayerTraining) {
        gradient_parameter_elements = 0;
        for (auto it = info.layers.rbegin(); it != info.layers.rend(); ++it) {
            if (it->has_trainable_params()) {
                gradient_parameter_elements = it->training_parameter_elements();
                break;
            }
        }
    }

    optimizer_parameter_elements = optimizer_state_elements(config.optimizer, gradient_parameter_elements);

    const auto activation_element_bytes = config.mode == RuntimeMode::InferenceOnly
        ? report.inference_element_bytes
        : (report.inference_element_bytes > report.training_element_bytes ? report.inference_element_bytes : report.training_element_bytes);
    std::size_t two_activation_buffers = 0;
    if (!checked_mul(max_activation_elements, 2U * activation_element_bytes, two_activation_buffers)) {
        return make_status(ErrorCode::OutOfBounds, "activation memory estimate overflow");
    }
    report.activation_scratch_bytes = two_activation_buffers;

    if (config.mode != RuntimeMode::InferenceOnly) {
        if (!checked_mul(saved_activation_elements, report.training_element_bytes, report.persistent_sram_bytes)) {
            return make_status(ErrorCode::OutOfBounds, "saved activation memory estimate overflow");
        }
        if (!checked_add(report.persistent_sram_bytes, dropout_mask_elements, report.persistent_sram_bytes)) {
            return make_status(ErrorCode::OutOfBounds, "dropout mask memory estimate overflow");
        }
        if (!checked_mul(gradient_parameter_elements, sizeof(float), report.gradient_bytes)) {
            return make_status(ErrorCode::OutOfBounds, "gradient memory estimate overflow");
        }
        if (!checked_mul(optimizer_parameter_elements, sizeof(float), report.optimizer_state_bytes)) {
            return make_status(ErrorCode::OutOfBounds, "optimizer state memory estimate overflow");
        }
        std::size_t argmax_bytes = 0;
        if (!checked_mul(argmax_elements, sizeof(std::uint32_t), argmax_bytes)) {
            return make_status(ErrorCode::OutOfBounds, "argmax memory estimate overflow");
        }
        report.temporary_bytes += argmax_bytes;
        // Backward gradient propagation remains float32 for stable optimizer math.
        std::size_t grad_io = 0;
        if (!checked_mul(max_activation_elements, 2U * sizeof(float), grad_io)) {
            return make_status(ErrorCode::OutOfBounds, "gradient IO memory estimate overflow");
        }
        report.temporary_bytes += grad_io;
    }

    report.total_peak_sram_bytes = report.activation_scratch_bytes + report.persistent_sram_bytes
        + report.gradient_bytes + report.optimizer_state_bytes + report.temporary_bytes;
    return Status::success();
}

} // namespace edgeist
