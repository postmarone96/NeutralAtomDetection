import cocotb
from cocotb.triggers import Timer
from cocotb.types import LogicArray

IN_W, IN_I = 16, 6
OUT_W, OUT_I = 8, 4

def ties_to_even_shift(raw, shift):
    if shift <= 0:
        return raw << (-shift)
    q = raw >> shift
    remainder = raw - (q << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (q & 1)):
        q += 1
    return q

def reference(raw):
    shift = (IN_W - IN_I) - (OUT_W - OUT_I)
    value = ties_to_even_shift(raw, shift)
    lo, hi = -(1 << (OUT_W - 1)), (1 << (OUT_W - 1)) - 1
    value = min(max(value, lo), hi)     # saturation
    return max(value, 0)                # ReLU

@cocotb.test
async def test_requant(dut):
    cases = [-32768, -25, -1, 0, 1, 23, 127, 1023, 32767]

    for raw in cases:
        dut.input.value = LogicArray.from_signed(raw, IN_W)
        await Timer(1, unit="ns")

        actual = dut.output.value.to_signed()
        expected = reference(raw)

        assert actual == expected, (
            f"raw={raw}: expected {expected}, got {actual}"
        )
