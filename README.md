# edgeist_V2

`edgeist_V2` is a production-oriented refactor of Edgeist for neural-network inference and constrained on-device training. It separates the reusable C++ runtime from examples, tests, Python tooling, and documentation.

## Repository layout

```text
runtime/                 Reusable C++20 runtime library
  include/edgeist/       Public API headers
  src/                   Runtime implementation
examples/desktop/        Host CLI example for validation and memory reports
examples/embedded/       Fixed-buffer embedded integration example
tests/                   C++ tests that execute the Python golden model
tools/                   Model validation, Python golden model, packaging helpers
nmcm/                    Python model tooling
```

## Devcontainer

Open the folder in VS Code and choose **Dev Containers: Reopen in Container**. The devcontainer is named `edgeist_V2` and opens the project under:

```bash
/workspaces/edgeist_V2
```

The devcontainer installs the host compiler, CMake, Ninja, Python tooling, ARM embedded tools, OpenOCD, and sanitizer-capable build support. Its `postCreateCommand` installs `nmcm/common` and configures the host-debug preset. The Python package version is static so the container also works from a ZIP checkout without `.git` metadata.

## Build and test

```bash
cmake --preset host-debug
cmake --build --preset host-debug
ctest --preset host-debug --output-on-failure
```

Run the wider validation matrix:

```bash
cmake --preset host-release
cmake --build --preset host-release
ctest --preset host-release --output-on-failure

cmake --preset asan
cmake --build --preset asan
ctest --preset asan --output-on-failure

cmake --preset ubsan
cmake --build --preset ubsan
ctest --preset ubsan --output-on-failure

cmake --preset inference-only
cmake --build --preset inference-only
```

## Python golden validation

The C++ test binary loads a Python-generated golden bundle and every test group compares against its own Python reference data. This keeps Python in the loop while avoiding repeated subprocess startup in small devcontainers. It compares:

- Softmax and CrossEntropy values
- Linear forward/backward/optimizer behavior
- Conv2D forward/backward behavior
- ReLU, pooling, and BatchNorm behavior
- fp16/int8/fp32 precision behavior
- binary model inference and activation trace values
- full-network training losses and updated weights

The golden model can also be run manually:

```bash
python3 tools/python_golden_model.py --case network_forward
python3 tools/python_golden_model.py --case full_training
python3 tools/python_golden_model.py --case all
```

## Desktop example

Validate a generated NMCF model and print memory usage:

```bash
./build/host-debug/examples/desktop/edgeist-desktop \
  --model path/to/model.hex \
  --trainable path/to/model_trainable.hex \
  --inference-type fp32
```

Use last-layer training mode and lower-precision execution policies. The selected runtime data type must match the model's exported parameter encoding; for example, use `--inference-type int8` only with an int8-encoded NMCF/trainable image.

```bash
./build/host-debug/examples/desktop/edgeist-desktop \
  --model path/to/int8_model.hex \
  --trainable path/to/int8_model_trainable.hex \
  --train \
  --inference-type int8 \
  --training-type int8
```

## Embedded example

The embedded example now builds and runs a small fixed-buffer model image, validates it, runs traced inference, performs one training update when training is enabled, and prints scratch peak usage.

```bash
./build/host-debug/examples/embedded/edgeist-embedded-skeleton
```

## Runtime API sketch

The runtime stores activations and trainable parameters in the selected type. It does not keep a float32 copy of int8/fp16 weights inside the layer kernels. Public input/output and activation traces are decoded to float32 at API boundaries for diagnostics.
Runtime initialization rejects models whose exported parameter encoding does not match the selected inference/training data type.


```cpp
#include "edgeist/edgeist.hpp"

std::array<std::byte, 8192> scratch_bytes{};
edgeist::ScratchArena arena{edgeist::ByteSpan(scratch_bytes)};

edgeist::ModelView view{edgeist::ConstByteSpan(model_image), edgeist::ConstByteSpan(trainable_image)};
edgeist::SramStorage trainable_storage{edgeist::ByteSpan(trainable_image)};

edgeist::TrainingConfig config{};
config.mode = edgeist::RuntimeMode::InferenceOnly;
config.inference_data_type = edgeist::NumericDataType::Int8;
config.training_data_type = edgeist::NumericDataType::Float32;

edgeist::ModelRuntime runtime;
auto status = runtime.init(view, trainable_storage, arena, config);
```

Capture activations without heap allocation:

```cpp
std::array<float, 128> trace_values{};
std::array<std::uint32_t, 16> trace_offsets{};
edgeist::ActivationTraceBuffer trace{
    edgeist::Span<float>(trace_values),
    edgeist::Span<std::uint32_t>(trace_offsets)
};
runtime.inference_with_trace(input, output, trace);
```

## Runtime configuration

`edgeist::TrainingConfig` supports:

- `RuntimeMode::InferenceOnly`
- `RuntimeMode::FullTraining`
- `RuntimeMode::LastLayerTraining`
- `RuntimeMode::FrozenLayerTraining`
- inference data type: `int8`, `fp16`, `fp32`
- training data type: `int8`, `fp16`, `fp32`
- deterministic int8 scales via `QuantizationConfig`
- micro-batch gradient accumulation via `micro_batch_size`
- optimizer selection: SGD, Momentum, Adam
- trainable storage selection: SRAM, Flash, External
- deterministic seed and deterministic math settings

Softmax, CrossEntropy, and optimizer state intentionally remain float32 by default to avoid unstable training on constrained devices.

## Model validation

All model images are checked before runtime initialization. Validation covers:

- magic number, header size, layer count, and version field presence
- layer offset table bounds and ordering
- known layer IDs and supported data encodings
- layer dimensions, tensor element counts, and adjacent tensor compatibility
- trainable and frozen parameter offsets and byte ranges
- predecessor table bounds
- malformed/truncated model inputs

## Packaging

Create a clean source archive:

```bash
python3 tools/package_refactored_repo.py --output edgeist_V2.zip
```

The package helper excludes build directories, `.git`, caches, bytecode, and compiled artifacts.

## Important docs

- [edgeist_V2 report](docs/EDGEIST_V2_REPORT.md)
- [Refactor report](docs/REFACTOR_REPORT.md)
- [Migration guide](docs/MIGRATION.md)
- [Numerical validation](docs/NUMERICAL_VALIDATION.md)
- [On-device training](docs/ON_DEVICE_TRAINING.md)
- [Memory strategies](docs/MEMORY_STRATEGIES.md)

## Current limitations

- Int8/fp16 model images are supported through the NMCF data-encoding field, but the current format still lacks explicit per-tensor/per-channel quantization metadata; default deterministic scales are used by the runtime configuration.
- Int8 training stores parameters/activations in int8 and quantizes gradients deterministically, while loss and optimizer state remain float32 for stability.
- Generic binary-model full training supports Linear/ReLU/Flatten/Softmax graphs with contiguous trainable parameter blocks.
- Conv2D and BatchNorm kernels are tested, but graph-level training for arbitrary Conv/BatchNorm networks remains future work.
