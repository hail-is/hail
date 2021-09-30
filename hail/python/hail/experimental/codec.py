from hail.utils.java import Env


def encode(expression, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    v = Env.spark_backend('encode')._jbackend.encodeToBytes(Env.backend()._to_java_value_ir(expression._ir), codec)
    return (v._1(), v._2())


def decode_through_JSON(typ, ptype_string, bytes, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    return typ._from_json(
        Env.spark_backend('decode')._jbackend.decodeToJSON(ptype_string, bytes, codec))


def read_int(byte_array, offset):
    ans = (byte_array[offset] +
           (byte_array[offset + 1] << 8) +
           (byte_array[offset + 2] << 16) +
           (byte_array[offset + 3] << 24))

    return ans

class EType:
    def __init__(self, required):
        self.required = required

    def decode(self, byte_array, offset):
        pass

class EInt32(EType):
    def __init__(self, required=False):
        super().__init__(required)

    def decode(self, byte_array, offset):
        return read_int(byte_array, offset)


class EArray(EType):
    def __init__(self, element_type, required=False):
        super().__init__(required)
        self.element_type = element_type

    def decode(self, byte_array, offset):
        len = read_int(byte_array, offset)
        data_start = offset + 4 if self.element_type.required else None
        return [self.element_type.decode(byte_array, data_start + i * 4) for i in range(len)]

def decode(e_type, bytes):
    return e_type.decode(bytes, 4)