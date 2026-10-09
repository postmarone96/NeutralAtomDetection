from pathlib import Path

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def reads_every_bias_word_combinationally(dut):
    mem_path = Path(__file__).resolve().parents[2] / "src" / "biases.mem"
    words = [
        int(line, 16)
        for line in mem_path.read_text().splitlines()
        if line.strip()
    ]

    for address, expected in enumerate(words):
        dut.addr.value = address
        await Timer(1, unit="ns")
        assert dut.data_out.value.to_unsigned() == expected
