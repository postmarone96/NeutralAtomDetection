"""The network in integer arithmetic: exactly what the RTL computes.

Every tensor is represented by int64 raw fixed-point values plus the precision
at which those values are held. A precision fixed<W,I,RND,SAT> means W-bit
signed integers carrying W-I fractional bits. Rescaling rounds to nearest with
ties to even, matching ieee.fixed_pkg fixed_round, and overflow clamps (AP_SAT).

A layer sums its products at their natural fraction, moves that sum onto the
accumulator's fraction, adds the bias and saturates once, then rounds to its
output precision.
"""

import numpy as np

from common import integer_bits_of, width_of


conv_blocks = (
    ("model_features_0", "model_features_1", "model_features_2"),
    ("model_features_3", "model_features_4", "model_features_5"),
    ("model_features_6", "model_features_7", "model_features_8"),
)


def precision_parts(precision):
    """Return (width, integer_bits) of a fixed<W,I,RND,SAT> precision."""
    return width_of(precision), integer_bits_of(precision)


def fractional_bits(precision):
    width, integer_bits = precision_parts(precision)
    return width - integer_bits


def saturate(raw_values, width):
    minimum = -(2 ** (width - 1))
    maximum = 2 ** (width - 1) - 1
    return np.clip(raw_values, minimum, maximum)


def quantise(values, precision):
    """Convert floats to raw fixed-point integers with ties-to-even rounding."""
    width, integer_bits = precision_parts(precision)
    fraction = width - integer_bits
    values = np.asarray(values, dtype=np.float64)
    raw_values = np.rint(values * 2.0 ** fraction)
    return saturate(raw_values, width).astype(np.int64)


def rescale(raw_values, source_fraction, target_fraction):
    """Move raw fixed-point integers using fixed_pkg's ties-to-even rounding."""
    shift = target_fraction - source_fraction
    if shift >= 0:
        return raw_values << shift
    shift = -shift
    quotient = raw_values >> shift
    remainder = raw_values - (quotient << shift)
    halfway = 1 << (shift - 1)
    round_up = (remainder > halfway) | (
        (remainder == halfway) & ((quotient & 1) != 0)
    )
    return quotient + round_up


def cast(raw_values, source_precision, target_precision):
    width, integer_bits = precision_parts(target_precision)
    target_fraction = width - integer_bits
    source_fraction = fractional_bits(source_precision)
    raw_values = rescale(raw_values, source_fraction, target_fraction)
    return saturate(raw_values, width)


def twos_complement(raw_values, width):
    """Return the unsigned bit patterns stored by a .mem file or RTL ROM."""
    mask = (1 << width) - 1
    return np.asarray(raw_values, dtype=np.int64) & mask


def layer_module(model, layer):
    """Return model.features[3] for 'model_features_3', for example."""
    group, index = layer.rsplit("_", 1)
    group = group.removeprefix("model_")
    return getattr(model, group)[int(index)]


def mac(input_values, weight_values, product_fraction, target_fraction, rows_per_chunk=4096):
    """Multiply and accumulate at the accumulator's fractional scale.

    When the accumulator is coarser than the products, each product is rounded
    before summation, matching a cast of every individual product.
    """
    if target_fraction >= product_fraction:
        sums = input_values @ weight_values.T
        return rescale(sums, product_fraction, target_fraction)

    sums = np.empty(
        (input_values.shape[0], weight_values.shape[0]),
        dtype=np.int64,
    )

    for start in range(0, input_values.shape[0], rows_per_chunk):
        end = start + rows_per_chunk
        chunk = input_values[start:end]
        products = chunk[:, None, :] * weight_values[None, :, :]
        products = rescale(products, product_fraction, target_fraction)
        sums[start:end] = products.sum(axis=2)
    return sums


def quantised_weights(model, layer, precisions):
    """Return layer weights as raw fixed-point integers in ROM order."""
    weights = layer_module(model, layer).weight.detach().cpu().numpy()
    weight_precision = precisions[(layer, "weight")]
    raw_weights = quantise(weights, weight_precision)
    return raw_weights.reshape(raw_weights.shape[0], -1)


def accumulate(input_values, input_precision, layer, model, precisions):
    """Compute the layer accumulator including bias and saturation."""
    weight_precision = precisions[(layer, "weight")]
    bias_precision = precisions[(layer, "bias")]
    accum_precision = precisions[(layer, "accum")]

    width, integer_bits = precision_parts(accum_precision)
    accum_fraction = width - integer_bits

    product_fraction = fractional_bits(input_precision) + fractional_bits(weight_precision)

    weights = quantised_weights(model, layer, precisions)
    sums = mac(input_values, weights, product_fraction, accum_fraction)

    bias = layer_module(model, layer).bias.detach().cpu().numpy()
    bias = quantise(bias, bias_precision)
    bias = rescale(bias, fractional_bits(bias_precision), accum_fraction)

    accum = sums + bias
    return saturate(accum, width)


def conv3x3(x, x_precision, layer, model, precisions):
    """Same-padded 3x3 convolution over every pixel."""
    n, channels, height, width = x.shape

    padded = np.pad(
        x,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (3, 3),
        axis=(2, 3),
    )

    patches = windows.transpose(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(
        n * height * width,
        channels * 9,
    )

    accum = accumulate(patches, x_precision, layer, model, precisions)
    accum = accum.reshape(n, height, width, -1)
    return accum.transpose(0, 3, 1, 2)


def dense(x, x_precision, layer, model, precisions):
    return accumulate(x, x_precision, layer, model, precisions)


def maxpool2(x):
    n, channels, height, width = x.shape
    rows = height // 2
    cols = width // 2
    blocks = x[:, :, : 2 * rows, : 2 * cols]
    blocks = blocks.reshape(n, channels, rows, 2, cols, 2)
    return blocks.max(axis=(3, 5))


def forward(model, precisions, crops, neighbor_features):
    """Return logit values and every intermediate fixed-point tensor."""
    stages = []

    def record_stage(name, values, stage_precision):
        stages.append((name, values, stage_precision))
        return values, stage_precision

    def dense_relu(x, x_precision, layer, relu_layer):
        accum_precision = precisions[(layer, "accum")]
        accum = dense(x, x_precision, layer, model, precisions)
        accum, _ = record_stage(f"{layer}_accum", accum, accum_precision)
        result_precision = precisions[(layer, "result")]
        x = cast(accum, accum_precision, result_precision)
        x, x_precision = record_stage(layer, x, result_precision)
        if relu_layer is None:
            return x, x_precision
        relu_precision = precisions[(relu_layer, "result")]
        x = np.maximum(x, 0)
        x = cast(x, x_precision, relu_precision)
        return record_stage(relu_layer, x, relu_precision)

    input_precision = precisions[("__input__", "result")]

    x = quantise(np.asarray(crops), input_precision)
    x, x_precision = record_stage("__input__", x, input_precision)

    for conv_layer, relu_layer, output_stage in conv_blocks:
        accum_precision = precisions[(conv_layer, "accum")]
        accum = conv3x3(x, x_precision, conv_layer, model, precisions)
        accum, _ = record_stage(f"{conv_layer}_accum", accum, accum_precision,)

        conv_precision = precisions[(conv_layer, "result")]
        x = cast(accum, accum_precision, conv_precision)
        x, x_precision = record_stage(conv_layer, x, conv_precision)
        relu_precision = precisions[(relu_layer, "result")]
        x = np.maximum(x, 0)
        x = cast(x, x_precision, relu_precision)
        x, x_precision = record_stage(relu_layer, x, relu_precision)

        if output_stage == "model_features_8":
            x = x.reshape(x.shape[0], -1)
        else:
            x = maxpool2(x)

        output_precision = precisions[(output_stage, "result")]
        x = cast(x, x_precision, output_precision)
        x, x_precision = record_stage(output_stage, x, output_precision)

    hidden, hidden_precision = dense_relu(x, x_precision, "model_classifier_0", "model_classifier_1",)

    neighbor_precision = precisions[("__input___neighbor", "result")]
    neighbors = quantise(np.asarray(neighbor_features), neighbor_precision)
    neighbors, _ = record_stage("__input___neighbor", neighbors, neighbor_precision)

    cat_precision = precisions[("cat", "result")]

    hidden = cast(hidden, hidden_precision, cat_precision,)
    neighbors = cast(neighbors, neighbor_precision, cat_precision)

    joined = np.concatenate([hidden, neighbors], axis=1)
    joined, _ = record_stage("cat", joined, cat_precision)

    second, second_precision = dense_relu(joined, cat_precision, "model_classifier_2", "model_classifier_3")

    logits, _ = dense_relu(second, second_precision, "model_classifier_4", None)

    return logits, stages
