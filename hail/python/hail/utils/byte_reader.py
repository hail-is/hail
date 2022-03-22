import struct


class ByteReader:
    def __init__(self, byte_memview, offset=0):
        self._memview = byte_memview
        self._offset = offset

    def read_int32(self) -> int:
        res = struct.unpack('=i', self._memview[self._offset:self._offset + 4])[0]
        self._offset += 4
        return res

    def read_int64(self) -> int:
        res = struct.unpack('=q', self._memview[self._offset:self._offset + 8])[0]
        self._offset += 8
        return res

    def read_bool(self) -> bool:
        res = self._memview[self._offset] != 0
        self._offset += 1
        return res

    def read_float32(self) -> float:
        res = struct.unpack('=f', self._memview[self._offset:self._offset + 4])[0]
        self._offset += 4
        return res

    def read_float64(self) -> float:
        res = struct.unpack('=d', self._memview[self._offset:self._offset + 8])[0]
        self._offset += 8
        return res

    def read_bytes_view(self, num_bytes):
        res = self._memview[self._offset: self._offset + num_bytes]
        self._offset += num_bytes
        return res

    def read_bytes(self, num_bytes):
        return self.read_bytes_view(num_bytes).tobytes()
