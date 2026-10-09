"""Generate the fixed preprocessing geometry, arithmetic, and raw-frame vectors.

The network was trained with scalar mean/range normalization and
grid-neighbour crop means.  This generator freezes that contract into a small
VHDL package so the RTL does not contain real-valued arithmetic.
"""

from pathlib import Path

import h5py
import joblib
import numpy as np

import config
import fixed_point
from model import neighbor_table


rtl_dir = Path(__file__).resolve().parents[1]
src_dir = rtl_dir / "src"
vectors_dir = rtl_dir / "tb" / "vectors"
project_root = Path(__file__).resolve().parents[3]

frame_count = 8
affine_fraction_bits = 21


def _format_integer_vector(name, values, values_per_line=12):
    values = [str(int(value)) for value in values]
    chunks = [
        ", ".join(values[start:start + values_per_line])
        for start in range(0, len(values), values_per_line)
    ]
    body = (",\n" + " " * 8).join(chunks)
    return (
        f"    constant {name} : integer_vector(0 to {len(values) - 1}) := (\n"
        f"        {body}\n"
        "    );"
    )


def _crop_frame_addresses(frame_shape, sites, crop_size):
    frame_height, frame_width = (int(value) for value in frame_shape)
    half = crop_size // 2
    addresses = []
    for x, y in sites:
        center_x = int(np.rint(x))
        center_y = int(np.rint(y))
        for crop_y in range(crop_size):
            frame_y = center_y - half + crop_y
            for crop_x in range(crop_size):
                frame_x = center_x - half + crop_x
                if 0 <= frame_x < frame_width and 0 <= frame_y < frame_height:
                    addresses.append(frame_y * frame_width + frame_x)
                else:
                    addresses.append(-1)
    return addresses


def _round_shift_even(values, shift):
    values = np.asarray(values, dtype=np.int64)
    quotient = values >> shift
    remainder = values - (quotient << shift)
    halfway = 1 << (shift - 1)
    round_up = (remainder > halfway) | (
        (remainder == halfway) & ((quotient & 1) != 0)
    )
    return quotient + round_up


def _hex_signed(value, width):
    digits = (width + 3) // 4
    return f'{int(value) & ((1 << width) - 1):0{digits}X}'


def build(shape=None):
    shape = config.load() if shape is None else shape
    job = joblib.load(shape["model_path"])
    dataset_path = project_root / Path(job["h5_path"])

    with h5py.File(dataset_path, "r") as dataset:
        frame_height, frame_width = dataset["images"].shape[-2:]
        sites = dataset["sites"][:].astype(np.float64)

    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError(f"preprocess requires fixed sites shaped (sites, 2), got {sites.shape}")
    if shape["method"] != "lightweight_cnn_site_neighbor":
        raise ValueError(
            "preprocess currently implements the selected plain-neighbour model, "
            f"not {shape['method']}"
        )

    crop_size = int(shape["crop"])
    crop_pixels = crop_size * crop_size
    site_count = int(sites.shape[0])
    frame_pixels = int(frame_height * frame_width)
    crop_x_start = [int(np.rint(x)) - crop_size // 2 for x, _ in sites]
    crop_y_start = [int(np.rint(y)) - crop_size // 2 for _, y in sites]
    raw_width, _ = fixed_point.precision_parts(
        shape["precision"][("__raw__", "integer")]
    )
    input_width, _ = fixed_point.precision_parts(
        shape["precision"][("__input__", "result")]
    )
    neighbor_width, _ = fixed_point.precision_parts(
        shape["precision"][("__input___neighbor", "result")]
    )

    raw_values = np.arange(1 << raw_width, dtype=np.float32)
    normalized = (
        raw_values - np.asarray(shape["mean_image"], dtype=np.float32)
    ) / float(shape["scale"])
    raw_to_input = fixed_point.quantise(
        normalized, shape["precision"][("__input__", "result")]
    )

    neighbor_indices, _ = neighbor_table(sites)
    neighbor_indices = neighbor_indices.reshape(-1)

    feature_scale = float(job["neighbor_feature_scale"])
    noise_floor = float(job["neighbor_noise_floor"])
    neighbor_precision = shape["precision"][("__input___neighbor", "result")]
    missing_neighbor_code = int(
        fixed_point.quantise(noise_floor / feature_scale, neighbor_precision)
    )

    neighbor_fraction = fixed_point.fractional_bits(neighbor_precision)
    affine_scale = 1 << affine_fraction_bits
    multiplier = int(np.rint(
        (2.0 ** neighbor_fraction)
        / (crop_pixels * float(shape["scale"]) * feature_scale)
        * affine_scale
    ))
    offset = int(np.rint(
        (2.0 ** neighbor_fraction)
        * float(shape["mean_image"])
        / (float(shape["scale"]) * feature_scale)
        * affine_scale
    ))

    # Prove the integer affine form gives the same code as the mathematical
    # preprocessing for every possible sum of unsigned RAW_W crop pixels.
    sums = np.arange(crop_pixels * ((1 << raw_width) - 1) + 1, dtype=np.int64)
    exact = fixed_point.quantise(
        ((sums / crop_pixels - float(shape["mean_image"])) / float(shape["scale"]))
        / feature_scale,
        neighbor_precision,
    )
    approximate = _round_shift_even(sums * multiplier - offset, affine_fraction_bits)
    approximate = fixed_point.saturate(approximate, neighbor_width)
    if not np.array_equal(exact, approximate):
        mismatch = int(np.flatnonzero(exact != approximate)[0])
        raise ValueError(
            f"neighbor affine constants differ at raw crop sum {mismatch}: "
            f"expected {exact[mismatch]}, got {approximate[mismatch]}"
        )

    affine_width = 44
    if offset >= (1 << (affine_width - 1)):
        raise ValueError("neighbor affine offset does not fit NB_AFFINE_W")

    package = "\n".join(
        [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "",
            "use work.nn_pkg.all;",
            "",
            "package preprocess_pkg is",
            f"    constant FRAME_WIDTH    : integer := {frame_width};",
            f"    constant FRAME_HEIGHT   : integer := {frame_height};",
            f"    constant FRAME_PIXELS   : integer := {frame_pixels};",
            f"    constant FRAME_ADDR_W   : integer := {max(1, (frame_pixels - 1).bit_length())};",
            f"    constant SITE_COUNT     : integer := {site_count};",
            f"    constant SITE_INDEX_W   : integer := {max(1, (site_count - 1).bit_length())};",
            f"    constant CROP_SIDE       : integer := {crop_size};",
            f"    constant CROP_PIXELS    : integer := {crop_pixels};",
            f"    constant CROP_SUM_W     : integer := "
            f"{max(1, (crop_pixels * ((1 << raw_width) - 1)).bit_length())};",
            "",
            _format_integer_vector("CROP_X_START", crop_x_start),
            "",
            _format_integer_vector("CROP_Y_START", crop_y_start),
            "",
            _format_integer_vector(
                "CROP_FRAME_ADDR",
                _crop_frame_addresses((frame_height, frame_width), sites, crop_size),
            ),
            "",
            _format_integer_vector("NEIGHBOR_SITE", neighbor_indices),
            "",
            _format_integer_vector("RAW_TO_INPUT", raw_to_input),
            "",
            f"    constant MISSING_NEIGHBOR_CODE : integer := {missing_neighbor_code};",
            f"    constant NB_AFFINE_FRACTION    : integer := {affine_fraction_bits};",
            f"    constant NB_MULTIPLIER_W       : integer := {max(1, multiplier.bit_length() + 1)};",
            f"    constant NB_MULTIPLIER         : integer := {multiplier};",
            f"    constant NB_AFFINE_W           : integer := {affine_width};",
            "    constant NB_OFFSET : signed(NB_AFFINE_W - 1 downto 0) := "
            f"signed'(x\"{_hex_signed(offset, affine_width)}\");",
            "end package preprocess_pkg;",
            "",
        ]
    )

    return package, job, dataset_path


def write(shape=None):
    package, job, dataset_path = build(shape)
    src_dir.mkdir(parents=True, exist_ok=True)
    package_path = src_dir / "preprocess_pkg.vhd"
    package_path.write_text(package)

    test_indices = np.asarray(job["test_idx"][:frame_count], dtype=np.int64)
    with h5py.File(dataset_path, "r") as dataset:
        frames = np.stack([dataset["images"][int(index)] for index in test_indices])
    if not np.all(frames == np.rint(frames)):
        raise ValueError("raw-frame RTL vectors require integer camera samples")
    frames = np.rint(frames).astype(np.int64)

    raw_width, _ = fixed_point.precision_parts(
        (config.load() if shape is None else shape)["precision"][("__raw__", "integer")]
    )
    if frames.min() < 0 or frames.max() >= (1 << raw_width):
        raise ValueError(
            f"raw frames [{frames.min()}, {frames.max()}] do not fit unsigned {raw_width} bits"
        )
    digits = (raw_width + 3) // 4
    vectors_dir.mkdir(parents=True, exist_ok=True)
    raw_path = vectors_dir / "raw_frame.txt"
    raw_path.write_text(
        "".join(f"{int(value):0{digits}x}\n" for value in frames.reshape(-1))
    )
    return package_path, raw_path


if __name__ == "__main__":
    for path in write():
        print(f"wrote {path}")
