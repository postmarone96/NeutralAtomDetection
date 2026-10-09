"""Regenerates every file the RTL depends on: run after any retrain or width change."""

import config
import gen_activation_map
import gen_params
import gen_preprocess
import gen_vectors
import gen_weights


def main():
    shape = config.load()
    print(f"wrote {gen_params.write(shape)}")
    print(f"wrote {gen_activation_map.write(shape)}")

    table, weights, biases = gen_weights.write(shape)
    weight_words = sum(entry["weight_word_count"] for entry in table.values())
    bias_words = sum(entry["bias_word_count"] for entry in table.values())
    print(
        f"wrote weights.mem  {weight_words} x {gen_weights.weight_rom_bits}-bit words "
        f"({weights.size} weights)"
    )
    print(f"wrote weights_scalar.mem  {weights.size} scalar debug values")
    print(
        f"wrote biases.mem   {bias_words} x {gen_weights.bias_rom_bits}-bit words "
        f"({biases.size} biases)"
    )
    print(f"wrote biases_scalar.mem  {biases.size} scalar debug values")
    print("wrote weights_pkg.vhd and bias_pkg.vhd")

    written = gen_vectors.write(shape)
    print(f"wrote {len(written)} vector files to tb/vectors")
    preprocess_package, raw_vectors = gen_preprocess.write(shape)
    print(f"wrote {preprocess_package}")
    print(f"wrote {raw_vectors}")

    macs = config.macs_per_site(shape)
    total = sum(macs.values())
    cycles = sum(
        (count + shape["mac_units"] - 1) // shape["mac_units"]
        for count in macs.values()
    )
    print(f"\n{total:,} MACs/site, {9 * total:,} MACs/image")
    print(f"{shape['mac_units']} MACs -> at least {cycles:.0f} arithmetic cycles/image "
          f"= {cycles * 2.0 / 1000:.2f} us at 500 MHz")


if __name__ == "__main__":
    main()
