import pytest

from avsafety.ingest.tfrecord import TFRecordError, crc_available, read_records, write_records


def test_roundtrip(tmp_path):
    path = tmp_path / "x.tfrecord"
    payloads = [b"alpha", b"", b"\x00" * 1000]
    write_records(path, payloads)
    recs = list(read_records(path))
    assert [r.data for r in recs] == payloads
    assert [r.index for r in recs] == [0, 1, 2]
    assert recs[0].offset == 0


def test_truncated_file_raises_after_good_records(tmp_path):
    path = tmp_path / "x.tfrecord"
    write_records(path, [b"first", b"second-record"])
    path.write_bytes(path.read_bytes()[:-6])
    it = read_records(path)
    assert next(it).data == b"first"
    with pytest.raises(TFRecordError, match="truncated"):
        next(it)


@pytest.mark.skipif(not crc_available(), reason="google-crc32c not installed")
def test_corrupt_payload_fails_crc(tmp_path):
    path = tmp_path / "x.tfrecord"
    write_records(path, [b"payload"])
    raw = bytearray(path.read_bytes())
    raw[14] ^= 0xFF  # flip a byte inside the payload
    path.write_bytes(bytes(raw))
    with pytest.raises(TFRecordError, match="CRC"):
        list(read_records(path))
