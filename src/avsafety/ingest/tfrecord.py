"""Minimal TFRecord reader/writer with no TensorFlow dependency.

Record layout (little-endian):
    uint64 length
    uint32 masked_crc32c(length bytes)
    bytes  data[length]
    uint32 masked_crc32c(data)

CRC verification uses the optional `google-crc32c` package. Without it, records
are read unverified and `crc_available()` returns False so callers can log it.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    import google_crc32c

    def _crc32c(data: bytes) -> int:
        return google_crc32c.value(data)

    _HAS_CRC = True
except ImportError:  # pragma: no cover - depends on environment
    _HAS_CRC = False

_MASK_DELTA = 0xA282EAD8


class TFRecordError(Exception):
    """Raised for a structurally corrupt or truncated record."""


@dataclass(frozen=True)
class Record:
    index: int
    offset: int
    data: bytes


def crc_available() -> bool:
    return _HAS_CRC


def _masked_crc(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def read_records(path: str | Path, verify_crc: bool = True) -> Iterator[Record]:
    """Yield records in file order. Raises TFRecordError on corruption.

    Corruption in the framing makes every later offset unreliable, so the
    reader stops rather than guessing; the caller records the failure.
    """
    check = verify_crc and _HAS_CRC
    with open(path, "rb") as f:
        index = 0
        while True:
            offset = f.tell()
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise TFRecordError(f"truncated header at offset {offset}")
            (length,) = struct.unpack("<Q", header[:8])
            (length_crc,) = struct.unpack("<I", header[8:])
            if check and _masked_crc(header[:8]) != length_crc:
                raise TFRecordError(f"length CRC mismatch at offset {offset}")
            data = f.read(length)
            footer = f.read(4)
            if len(data) < length or len(footer) < 4:
                raise TFRecordError(f"truncated record {index} at offset {offset}")
            if check and _masked_crc(data) != struct.unpack("<I", footer)[0]:
                raise TFRecordError(f"data CRC mismatch in record {index} at offset {offset}")
            yield Record(index=index, offset=offset, data=data)
            index += 1


def write_records(path: str | Path, payloads: list[bytes]) -> None:
    """Write records. Used for test fixtures; requires google-crc32c for valid CRCs."""
    with open(path, "wb") as f:
        for data in payloads:
            length = struct.pack("<Q", len(data))
            f.write(length)
            f.write(struct.pack("<I", _masked_crc(length) if _HAS_CRC else 0))
            f.write(data)
            f.write(struct.pack("<I", _masked_crc(data) if _HAS_CRC else 0))
