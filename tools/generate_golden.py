#!/usr/bin/env python3
"""Generate deterministic golden C++ reference values for Edgeist tests.

This script intentionally uses only the Python standard library so it can run in
minimal CI and embedded-toolchain containers.
"""

from __future__ import annotations

import math


def fmt(values: list[float]) -> str:
    rendered = []
    for value in values:
        text = f"{value:.10g}"
        if "e" not in text.lower() and "." not in text:
            text += ".0"
        rendered.append(text + "F")
    return ", ".join(rendered)


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def linear(x: list[float], w: list[float], b: list[float], input_size: int, output_size: int) -> list[float]:
    return [sum(x[j] * w[i * input_size + j] for j in range(input_size)) + b[i] for i in range(output_size)]


def conv2d() -> list[float]:
    inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    weights = [1.0, 0.0, 0.0, -1.0]
    bias = [0.5]
    out: list[float] = []
    for oy in range(2):
        for ox in range(2):
            acc = bias[0]
            for ky in range(2):
                for kx in range(2):
                    acc += inp[(oy + ky) * 3 + ox + kx] * weights[ky * 2 + kx]
            out.append(acc)
    return out


def batchnorm() -> list[float]:
    inp = [1.0, 2.0, 3.0, 4.0]
    gamma = [2.0, 3.0]
    beta = [0.5, -0.5]
    mean = [1.5, 3.5]
    variance = [0.25, 0.25]
    eps = 1e-5
    out: list[float] = []
    for c in range(2):
        inv = 1.0 / math.sqrt(variance[c] + eps)
        for s in range(2):
            idx = c * 2 + s
            out.append(((inp[idx] - mean[c]) * inv) * gamma[c] + beta[c])
    return out


def main() -> None:
    logits = [1.0, 2.0, 3.0]
    linear_input = [1.0, 2.0, 3.0]
    linear_weights = [0.1, -0.2, 0.3, 0.4, 0.5, -0.6]
    linear_bias = [0.01, -0.02]
    conv_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    conv_weights = [1.0, 0.0, 0.0, -1.0]
    conv_bias = [0.5]
    bn_input = [1.0, 2.0, 3.0, 4.0]
    bn_gamma = [2.0, 3.0]
    bn_beta = [0.5, -0.5]
    bn_mean = [1.5, 3.5]
    bn_var = [0.25, 0.25]

    print("#pragma once")
    print("\n#include <array>\n")
    print("namespace edgeist::testdata {\n")
    print(f"inline constexpr std::array<float, 3> kSoftmaxLogits {{ {fmt(logits)} }};")
    print(f"inline constexpr std::array<float, 3> kSoftmaxProbabilities {{ {fmt(softmax(logits))} }};\n")
    print(f"inline constexpr std::array<float, 3> kLinearInput {{ {fmt(linear_input)} }};")
    print(f"inline constexpr std::array<float, 6> kLinearWeights {{ {fmt(linear_weights)} }};")
    print(f"inline constexpr std::array<float, 2> kLinearBias {{ {fmt(linear_bias)} }};")
    print(f"inline constexpr std::array<float, 2> kLinearOutput {{ {fmt(linear(linear_input, linear_weights, linear_bias, 3, 2))} }};\n")
    print(f"inline constexpr std::array<float, 9> kConvInput {{ {fmt(conv_input)} }};")
    print(f"inline constexpr std::array<float, 4> kConvWeights {{ {fmt(conv_weights)} }};")
    print(f"inline constexpr std::array<float, 1> kConvBias {{ {fmt(conv_bias)} }};")
    print(f"inline constexpr std::array<float, 4> kConvOutput {{ {fmt(conv2d())} }};\n")
    print(f"inline constexpr std::array<float, 4> kBatchNormInput {{ {fmt(bn_input)} }};")
    print(f"inline constexpr std::array<float, 2> kBatchNormGamma {{ {fmt(bn_gamma)} }};")
    print(f"inline constexpr std::array<float, 2> kBatchNormBeta {{ {fmt(bn_beta)} }};")
    print(f"inline constexpr std::array<float, 2> kBatchNormMean {{ {fmt(bn_mean)} }};")
    print(f"inline constexpr std::array<float, 2> kBatchNormVariance {{ {fmt(bn_var)} }};")
    print(f"inline constexpr std::array<float, 4> kBatchNormOutput {{ {fmt(batchnorm())} }};\n")
    print("} // namespace edgeist::testdata")


if __name__ == "__main__":
    main()
