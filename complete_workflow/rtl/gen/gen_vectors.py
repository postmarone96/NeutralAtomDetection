"""Writes tb/vectors/*.txt: every tensor fixed_point.forward computes, as codes.

The codes come from the integer model itself, so they are exactly what the RTL
must produce, bit for bit. One code per line, two's complement hex, flattened
in (site, channel, row, column) order; tb/vectors/shapes.txt has each shape.
"""

from pathlib import Path

import numpy as np

import config
import fixed_point
import quantize
from fixed_point import precision_parts, twos_complement

vectors_dir = Path(__file__).resolve().parents[1] / "tb" / "vectors"
n_images = 8
file_names = {"__input__": "input_crop", "__input___neighbor": "input_neighbor"}


def write_codes(path, codes, width):
    digits = (width + 3) // 4
    packed = twos_complement(np.asarray(codes).reshape(-1), width)
    path.write_text("".join(f"{int(value):0{digits}x}\n" for value in packed))


def write(shape=None):
    shape = config.load() if shape is None else shape
    data = quantize.load(shape["model_path"])
    keep = n_images * data["site_count"]

    logits, stages = fixed_point.forward(
        data["model"], shape["precision"],
        data["crops"][:keep], data["neighbor_features"][:keep],
    )

    vectors_dir.mkdir(parents=True, exist_ok=True)
    for stale in vectors_dir.glob("*.txt"):
        stale.unlink()

    shapes = []
    written = {}
    for name, codes, precision in stages:
        file_name = file_names.get(name, name)
        width, integer = precision_parts(precision)
        write_codes(vectors_dir / f"{file_name}.txt", codes, width)
        shapes.append(f"{file_name} {'x'.join(str(n) for n in codes.shape)} "
                      f"fixed<{width},{integer}>")
        written[file_name] = codes.size

    decisions = (logits[:, 1] > logits[:, 0]).astype(np.int64)
    (vectors_dir / "decisions.txt").write_text(
        "".join(f"{int(value)}\n" for value in decisions)
    )
    written["decisions"] = decisions.size
    shapes.append(f"decisions {decisions.size} bit")

    (vectors_dir / "shapes.txt").write_text("\n".join(shapes) + "\n")
    (vectors_dir / "meta.txt").write_text(
        f"images {n_images}\nsites {data['site_count']}\ncrop {shape['crop']}\n"
        f"neighbors {shape['neighbors']}\n"
    )
    return written


if __name__ == "__main__":
    for name, count in write().items():
        print(f"{name:26} {count:7,} values")
