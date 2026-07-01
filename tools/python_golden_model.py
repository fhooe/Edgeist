#!/usr/bin/env python3
"""Deterministic Python golden model used by the C++ unit tests.

The script intentionally uses only the standard library so CTest can execute it
inside lightweight devcontainers and embedded CI images. The C++ test executable asks this script for all cases in one process and each
unit test compares its own results against the corresponding Python-generated
golden data. Batching avoids repeated Python process startup in small
devcontainers while preserving Python-in-the-loop validation for every test
case.
"""

from __future__ import annotations

import argparse
import math
import struct
from typing import Iterable

ACT_SCALE = 1.0 / 64.0
WEIGHT_SCALE = 1.0 / 64.0
GRAD_SCALE = 1.0 / 64.0
BIAS_SCALE = 1.0 / 64.0


def fp16(value: float) -> float:
    return float(struct.unpack("<e", struct.pack("<e", float(value)))[0])


def qint8_raw(value: float, scale: float) -> int:
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
        return 0
    q = round(value / scale)
    return int(max(-128, min(127, q)))


def int8(value: float, scale: float) -> float:
    return float(qint8_raw(value, scale)) * scale


def cast(value: float, dtype: str, scale: float) -> float:
    if dtype == "fp32":
        return float(value)
    if dtype == "fp16":
        return fp16(value)
    if dtype == "int8":
        return int8(value, scale)
    raise ValueError(dtype)


def emit(mapping: dict[str, Iterable[float] | float | int], prefix: str = "") -> None:
    for key, value in mapping.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, (float, int)):
            print(f"{full_key}={float(value):.10g}")
        else:
            print(f"{full_key}=" + ",".join(f"{float(v):.10g}" for v in value))


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def linear(x: list[float], w: list[float], b: list[float], input_size: int, output_size: int, dtype: str = "fp32") -> list[float]:
    if dtype == "int8":
        xq = [qint8_raw(v, ACT_SCALE) for v in x]
        wq = [qint8_raw(v, WEIGHT_SCALE) for v in w]
        bq = [qint8_raw(v, BIAS_SCALE) for v in b]
        out: list[float] = []
        for row in range(output_size):
            acc_q = 0
            for col in range(input_size):
                acc_q += xq[col] * wq[row * input_size + col]
            acc = float(acc_q) * ACT_SCALE * WEIGHT_SCALE + float(bq[row]) * BIAS_SCALE
            out.append(int8(acc, ACT_SCALE))
        return out
    if dtype == "fp16":
        xh = [fp16(v) for v in x]
        wh = [fp16(v) for v in w]
        bh = [fp16(v) for v in b]
        out = []
        for row in range(output_size):
            acc = bh[row]
            for col in range(input_size):
                acc += xh[col] * wh[row * input_size + col]
            out.append(fp16(acc))
        return out
    out = []
    for row in range(output_size):
        acc = b[row]
        for col in range(input_size):
            acc += x[col] * w[row * input_size + col]
        out.append(acc)
    return out


def relu(x: list[float], dtype: str = "fp32") -> list[float]:
    out = [v if v > 0.0 else 0.0 for v in x]
    return [cast(v, dtype, ACT_SCALE) for v in out] if dtype != "fp32" else out


def conv2d() -> tuple[list[float], list[float], list[float], list[float]]:
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

    grad_out = [1.0, 1.0, 1.0, 1.0]
    grad_input = [0.0] * 9
    grad_weights = [0.0] * 4
    grad_bias = [sum(grad_out)]
    idx = 0
    for oy in range(2):
        for ox in range(2):
            go = grad_out[idx]
            idx += 1
            for ky in range(2):
                for kx in range(2):
                    iidx = (oy + ky) * 3 + ox + kx
                    widx = ky * 2 + kx
                    grad_input[iidx] += weights[widx] * go
                    grad_weights[widx] += inp[iidx] * go
    return out, grad_input, grad_weights, grad_bias


def batchnorm() -> tuple[list[float], list[float], list[float], list[float]]:
    inp = [1.0, 2.0, 3.0, 4.0]
    gamma = [2.0, 3.0]
    beta = [0.5, -0.5]
    mean = [1.5, 3.5]
    variance = [0.25, 0.25]
    eps = 1e-5
    out: list[float] = []
    grad_in: list[float] = []
    grad_gamma = [0.0, 0.0]
    grad_beta = [0.0, 0.0]
    for c in range(2):
        inv = 1.0 / math.sqrt(variance[c] + eps)
        for s in range(2):
            idx = c * 2 + s
            centered = inp[idx] - mean[c]
            out.append(centered * inv * gamma[c] + beta[c])
            grad_gamma[c] += centered * inv
            grad_beta[c] += 1.0
            grad_in.append(gamma[c] * inv)
    return out, grad_in, grad_gamma, grad_beta


NETWORK_INITIAL = [
    0.2, -0.1, -0.3, 0.4, 0.1, 0.2, 0.0, 0.1, -0.1,
    0.3, -0.2, 0.1, -0.4, 0.2, 0.5, 0.0, 0.0,
]
NETWORK_X = [0.6, -0.4]
NETWORK_Y = [1.0, 0.0]
NETWORK_LR = 0.05


def unpack_network(params: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
    w1 = params[0:6]
    b1 = params[6:9]
    w2 = params[9:15]
    b2 = params[15:17]
    return w1, b1, w2, b2


def network_forward(params: list[float], dtype: str = "fp32") -> tuple[list[float], list[list[float]]]:
    w1, b1, w2, b2 = unpack_network(params)
    z1 = linear(NETWORK_X, w1, b1, 2, 3, dtype)
    a1 = relu(z1, dtype)
    logits = linear(a1, w2, b2, 3, 2, dtype)
    probs = softmax(logits)
    return probs, [z1, a1, logits, probs]


def network_train(params: list[float], steps: int = 3, dtype: str = "fp32") -> tuple[list[float], list[float], list[float]]:
    params = list(params)
    losses: list[float] = []
    for _ in range(steps):
        w1, b1, w2, b2 = unpack_network(params)
        z1 = linear(NETWORK_X, w1, b1, 2, 3, dtype)
        a1 = relu(z1, dtype)
        logits = linear(a1, w2, b2, 3, 2, dtype)
        probs = softmax(logits)
        losses.append(-math.log(max(probs[0], 1e-7)))

        dz2 = [probs[i] - NETWORK_Y[i] for i in range(2)]
        if dtype != "fp32":
            dz2 = [cast(v, dtype, GRAD_SCALE) for v in dz2]
        dw2 = [0.0] * 6
        db2 = dz2[:]
        da1 = [0.0] * 3
        for o in range(2):
            for i in range(3):
                dw2[o * 3 + i] += dz2[o] * a1[i]
                da1[i] += w2[o * 3 + i] * dz2[o]
        dz1 = [da1[i] if z1[i] > 0.0 else 0.0 for i in range(3)]
        if dtype != "fp32":
            dz1 = [cast(v, dtype, GRAD_SCALE) for v in dz1]
        dw1 = [0.0] * 6
        db1 = dz1[:]
        for o in range(3):
            for i in range(2):
                dw1[o * 2 + i] += dz1[o] * NETWORK_X[i]

        grads = dw1 + db1 + dw2 + db2
        params = [p - NETWORK_LR * g for p, g in zip(params, grads)]
    final_output, _ = network_forward(params, dtype)
    return losses, params, final_output


def case_softmax() -> dict[str, Iterable[float] | float]:
    logits = [1.0, 2.0, 3.0]
    probs = softmax(logits)
    target = [0.0, 0.0, 1.0]
    return {
        "probabilities": probs,
        "loss": -math.log(probs[2]),
        "gradient": [p - t for p, t in zip(probs, target)],
    }


def case_linear() -> dict[str, Iterable[float] | float]:
    x = [1.0, 2.0, 3.0]
    w = [0.1, -0.2, 0.3, 0.4, 0.5, -0.6]
    b = [0.01, -0.02]
    grad_out = [0.5, -1.0]
    grad_input = [0.0] * 3
    grad_weights = [0.0] * 6
    grad_bias = grad_out[:]
    for o in range(2):
        for i in range(3):
            grad_weights[o * 3 + i] = grad_out[o] * x[i]
            grad_input[i] += w[o * 3 + i] * grad_out[o]
    sgd_params = [1.0 - 0.1 * 0.2, 2.0 - 0.1 * -0.3]
    return {
        "output": linear(x, w, b, 3, 2),
        "grad_input": grad_input,
        "grad_weights": grad_weights,
        "grad_bias": grad_bias,
        "sgd_params": sgd_params,
        "adam_param": [0.9],
    }


def case_conv2d() -> dict[str, Iterable[float] | float]:
    out, gi, gw, gb = conv2d()
    return {"output": out, "grad_input": gi, "grad_weights": gw, "grad_bias": gb}


def case_relu_pool_batchnorm() -> dict[str, Iterable[float] | float]:
    relu_input = [-1.0, 0.0, 2.0, -3.0]
    pool_output = [5.0, 6.0, 8.0, 9.0]
    pool_grad = [0.0] * 9
    for idx, value in zip([4, 5, 7, 8], [1.0, 2.0, 3.0, 4.0]):
        pool_grad[idx] = value
    bn_out, bn_grad_in, bn_grad_gamma, bn_grad_beta = batchnorm()
    return {
        "relu_output": relu(relu_input),
        "relu_grad": [0.0, 0.0, 1.0, 0.0],
        "pool_output": pool_output,
        "pool_grad_input": pool_grad,
        "batchnorm_output": bn_out,
        "batchnorm_grad_input": bn_grad_in,
        "batchnorm_grad_gamma": bn_grad_gamma,
        "batchnorm_grad_beta": bn_grad_beta,
    }


def case_precision() -> dict[str, Iterable[float] | float]:
    x = [1.0, 2.0, 3.0]
    w = [0.1, -0.2, 0.3, 0.4, 0.5, -0.6]
    b = [0.01, -0.02]
    return {
        "fp16_linear_output": linear(x, w, b, 3, 2, "fp16"),
        "int8_linear_output": linear(x, w, b, 3, 2, "int8"),
        "fp16_cast": [fp16(0.33325195), fp16(-5.75)],
        "int8_cast": [int8(0.333, ACT_SCALE), int8(-9.0, ACT_SCALE)],
    }


def case_network_forward() -> dict[str, Iterable[float] | float]:
    output, activations = network_forward(NETWORK_INITIAL)
    return {
        "output": output,
        "activation_0": activations[0],
        "activation_1": activations[1],
        "activation_2": activations[2],
        "activation_3": activations[3],
    }


def case_full_training() -> dict[str, Iterable[float] | float]:
    losses, params, final_output = network_train(NETWORK_INITIAL, steps=3)
    return {
        "losses": losses,
        "final_trainable": params,
        "final_output": final_output,
    }


def case_memory() -> dict[str, Iterable[float] | float]:
    return {
        "training_has_more_gradient_bytes": [1.0],
        "fp32_bytes": [4.0],
        "fp16_bytes": [2.0],
        "int8_bytes": [1.0],
    }


CASES = {
    "softmax": case_softmax,
    "linear": case_linear,
    "conv2d": case_conv2d,
    "relu_pool_batchnorm": case_relu_pool_batchnorm,
    "precision": case_precision,
    "network_forward": case_network_forward,
    "full_training": case_full_training,
    "memory": case_memory,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES) + ["all"], required=True)
    args = parser.parse_args()
    if args.case == "all":
        for name in sorted(CASES):
            emit(CASES[name](), prefix=f"{name}.")
    else:
        emit(CASES[args.case]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
