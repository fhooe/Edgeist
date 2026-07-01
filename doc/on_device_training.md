# On-device training

## Strategy model

Training behavior is configured through `edgeist::TrainingConfig`.

```cpp
edgeist::TrainingConfig cfg;
cfg.mode = edgeist::RuntimeMode::LastLayerTraining;
cfg.optimizer = edgeist::OptimizerKind::Sgd;
cfg.micro_batch_size = 4;
cfg.learning_rate = 0.001F;
cfg.trainable_storage = edgeist::StorageKind::Sram;
```

Supported modes:

- **InferenceOnly:** no gradient buffers, no optimizer state, smallest RAM footprint.
- **FullTraining:** all trainable parameter layers are eligible for gradient accumulation.
- **LastLayerTraining:** only the last parameterized layer is updated.
- **FrozenLayerTraining:** layers before `frozen_prefix_layers` are excluded from updates.

## Optimizers

The runtime includes:

- SGD: no optimizer state;
- Momentum: one state tensor per trainable parameter;
- Adam: two state tensors per trainable parameter and a timestep.

Optimizer memory is included in `MemoryReport::optimizer_state_bytes`.

## Micro-batch accumulation

The runtime separates gradient accumulation from parameter updates:

```cpp
runtime.begin_micro_batch();
for (std::uint32_t sample = 0; sample < cfg.micro_batch_size; ++sample) {
    runtime.train_sample(input[sample], expected[sample]);
}
runtime.apply_updates(cfg.micro_batch_size);
```

This avoids holding a full batch in memory while producing the same averaged-gradient update as a host batch loop, within normal floating-point tolerance.

## Storage backends

Trainable weights are accessed through `StorageBackend`:

- `SramStorage`: direct mutable buffer for fastest training;
- `FlashStorage`: read-only image backend for inference or target flash drivers that stage writes elsewhere;
- `ExternalMemoryStorage`: host/portable implementation for optional external memory.

Embedded targets should implement a `StorageBackend` wrapper around their actual flash or external-memory driver. Writes should be page-aware and power-failure-safe.

## Runtime allocation policy

Training and inference calls do not allocate dynamically. The runtime may use C++ containers during `init` on host builds, but all tensor work buffers are carved from the caller-provided `ScratchArena` before execution.

## Currently integrated graph-level training path

The generic `ModelRuntime::train_sample` path supports Linear/ReLU/Flatten/Softmax graphs with fully trainable contiguous float32 parameter blocks. This covers the common last-layer and small-MLP adaptation cases used on constrained devices.

Layer kernels and validation are present for Conv2D, BatchNorm, pooling, dropout, and adaptive pooling. Full mixed-graph training for Conv2D/BatchNorm/pooling/dropout is staged behind the same layer-kernel interfaces and memory planner, but target deployments should validate these paths layer-by-layer before enabling full graph updates.

## Recommended deployment strategies

### Tiny SRAM budget

Use `InferenceOnly` or `LastLayerTraining`, SGD, and SRAM trainable storage for only the final classifier. Keep feature-extractor weights frozen in flash.

### Moderate SRAM budget

Use `FrozenLayerTraining` with a frozen convolutional stem and trainable dense head. Use micro-batch accumulation to reduce peak activation memory.

### High endurance concern

Avoid direct per-step flash writes. Accumulate updates in SRAM and checkpoint to flash periodically with wear leveling and a verified header/checksum.

### External memory available

Place optimizer state and rarely accessed trainable blocks in external memory. Keep current-layer activations and gradients in SRAM scratch.

## Determinism requirements

For reproducible target training:

- set `deterministic_seed`;
- document FPU mode and compiler flags;
- avoid fast-math and uncontrolled FMA contraction during validation;
- use identical batch ordering;
- compare layer outputs after every optimizer step during bring-up.

## edgeist_V2 full-network training validation

The test suite now includes a full Linear/ReLU/Linear/Softmax training case. The C++ runtime and the Python golden model execute the same three SGD update steps and compare losses, final trainable parameters, and final inference output. This validates the generic full-training path for contiguous trainable Linear/ReLU/Flatten/Softmax graphs.
