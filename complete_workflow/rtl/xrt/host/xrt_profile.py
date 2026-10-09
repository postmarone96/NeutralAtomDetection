"""Read native XRT transfer timestamps without taking Python timestamps.

Format: Xilinx/XRT 2024.2 native_writer.cpp, native_events.cpp, vtf_event.cpp.
"""

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path


@dataclass(frozen=True)
class TransferTiming:
    input_start_ms: float
    input_end_ms: float
    output_start_ms: float
    output_end_ms: float
    end_to_end_us: float


def summarize_native_trace(path: Path) -> TransferTiming:
    """Measure the single launch's input sync start through output sync end.

    The host performs exactly three syncs: result clear, frame upload, result
    download. The trace's READ/WRITE rows distinguish these from API rows.
    Reject unexpected transfer sequences rather than guessing their buffers.
    """
    strings, events = {}, []
    section = None
    resolution = None
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if not row:
                continue
            if len(row) == 1 and row[0] in (
                    "HEADER", "STRUCTURE", "MAPPING", "EVENTS", "DEPENDENCIES"):
                section = row[0]
                continue
            if section == "HEADER" and row[0] == "Resolution":
                resolution = row[1]
            elif section == "MAPPING":
                strings[int(row[0])] = ",".join(row[1:])
            elif section == "EVENTS":
                if len(row) != 6:
                    raise ValueError("unsupported native XRT event format")
                events.append((int(row[0]), int(row[1]), Decimal(row[2]),
                               int(row[3]), int(row[5])))
    if resolution != "ms":
        raise ValueError(f"expected XRT timestamps in ms, got {resolution!r}")

    starts, transfers = {}, []
    for event_id, start_id, timestamp, bucket, name_id in events:
        direction = strings.get(name_id)
        if direction not in ("WRITE", "READ"):
            continue
        if not timestamp.is_finite():
            raise ValueError("non-finite XRT transfer timestamp")
        if start_id == 0:
            if event_id in starts:
                raise ValueError("duplicate XRT transfer start")
            starts[event_id] = (timestamp, bucket, direction)
        else:
            try:
                start, start_bucket, start_direction = starts.pop(start_id)
            except KeyError as error:
                raise ValueError("XRT transfer end has no matching start") from error
            if (start_bucket, start_direction) != (bucket, direction) or timestamp < start:
                raise ValueError("inconsistent XRT transfer interval")
            transfers.append((start, timestamp, direction))
    if starts:
        raise ValueError("incomplete XRT transfers")
    transfers.sort()
    if [entry[2] for entry in transfers] != ["WRITE", "WRITE", "READ"]:
        raise ValueError("expected result clear, frame upload, then result download")
    clear, upload, download = transfers
    if clear[1] > upload[0] or upload[1] > download[0]:
        raise ValueError("overlapping XRT transfers in synchronous host")
    timing = TransferTiming(
        float(upload[0]), float(upload[1]), float(download[0]), float(download[1]),
        float((download[1] - upload[0]) * 1000))
    path.with_name("transfer_timing.json").write_text(
        json.dumps({"source": str(path.resolve()), "measurement":
                    "XRT native host API timestamps, including transfers and launch/wait",
                    **asdict(timing)}, indent=2) + "\n")
    return timing
