"""Saved-prediction admission rejects corrupt/mismatched scientific payloads."""
import base64
import importlib.util
from pathlib import Path
import zlib

import numpy as np
import pytest


def reader():
    path = Path(__file__).resolve().parents[2] / 'tools/reconstruct_map_v2.py'
    spec = importlib.util.spec_from_file_location('saved_mask_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recorded_mask


def payload(raw, shape=(2, 2)):
    return {'shape':list(shape), 'dtype':'uint8',
            'zlib_base64':base64.b64encode(zlib.compress(raw)).decode()}


def test_saved_mask_preserves_classes_unknown_and_sensor_shape():
    expected = np.array([[0, 39], [30, 255]], np.uint8)
    actual = reader()(payload(expected.tobytes()), (2, 2))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize('raw',[b'\x00' * 3, b'\x00' * 5, b'\x00' * 100000])
def test_saved_mask_rejects_wrong_size_or_expansion(raw):
    with pytest.raises(ValueError, match='Invalid or oversized'):
        reader()(payload(raw), (2, 2))


def test_saved_mask_rejects_truncated_and_appended_streams():
    compressed = zlib.compress(bytes([0, 39, 30, 255]))
    for encoded in (compressed[:-1], compressed + b'extra'):
        item = payload(b'')
        item['zlib_base64'] = base64.b64encode(encoded).decode()
        with pytest.raises(ValueError, match='Invalid or oversized'):
            reader()(item, (2, 2))


def test_saved_mask_rejects_foreign_class_coordinates_and_metadata():
    admit = reader()
    with pytest.raises(ValueError, match='Unexpected NYU40'):
        admit(payload(bytes([0, 39, 40, 255])), (2, 2))
    with pytest.raises(ValueError, match='original sensor coordinates'):
        admit(payload(bytes(4), shape=(1, 4)), (2, 2))
    item = payload(bytes(4)); item['dtype'] = 'float32'
    with pytest.raises(ValueError, match='original sensor coordinates'):
        admit(item, (2, 2))
    item = payload(bytes(4)); item['future_target'] = []
    with pytest.raises(ValueError, match='Unexpected recorded semantic fields'):
        admit(item, (2, 2))
