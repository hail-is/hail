import struct
from typing import Union


class ByteReader:
    def __init__(self, byte_memview, offset=0):
        self._memview = byte_memview
        self._offset = offset

    def read_int32(self) -> int:
        res = struct.unpack('=i', self._memview[self._offset : self._offset + 4])[0]
        self._offset += 4
        return res

    def read_int64(self) -> int:
        res = struct.unpack('=q', self._memview[self._offset : self._offset + 8])[0]
        self._offset += 8
        return res

    def read_bool(self) -> bool:
        res = self._memview[self._offset] != 0
        self._offset += 1
        return res

    def read_float32(self) -> float:
        res = struct.unpack('=f', self._memview[self._offset : self._offset + 4])[0]
        self._offset += 4
        return res

    def read_float64(self) -> float:
        res = struct.unpack('=d', self._memview[self._offset : self._offset + 8])[0]
        self._offset += 8
        return res

    def read_bytes_view(self, num_bytes):
        res = self._memview[self._offset : self._offset + num_bytes]
        self._offset += num_bytes
        return res

    def read_bytes(self, num_bytes):
        return self.read_bytes_view(num_bytes).tobytes()

    def read_varint(self) -> int:
        result = 0
        shift = 0
        while shift < 7 * 10:
            b = self._memview[self._offset]
            self._offset += 1
            result |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                return result
            shift += 7
        raise ValueError('Invalid variable length integer, longer than 10 bytes')


class ByteWriter:
    def __init__(self, buf):
        self._buf: bytearray = buf

    def write_byte(self, v):
        self._buf += struct.pack('=B', v)

    def write_int32(self, v):
        self._buf += struct.pack('=i', v)

    def write_int64(self, v):
        self._buf += struct.pack('=q', v)

    def write_bool(self, v):
        self._buf += b'\x01' if v else b'\x00'

    def write_float32(self, v):
        self._buf += struct.pack('=f', v)

    def write_float64(self, v):
        self._buf += struct.pack('=d', v)

    def write_bytes(self, bs: Union[bytes, memoryview]):
        self._buf += bs

    def write_varint(self, v: int):
        if v < 0:
            raise ValueError(f'write_varint expected non-negative value, got {v}')
        while True:
            byte = v & 0x7F
            v >>= 7
            if v == 0:
                self._buf += struct.pack('=B', byte)
                return
            self._buf += struct.pack('=B', byte | 0x80)
