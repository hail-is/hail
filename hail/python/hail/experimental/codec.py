from hail.utils.java import Env

block_codec = '{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'
stream_codec = '{"name":"StreamBufferSpec"}'


def encode(expression, codec=stream_codec):
    return Env.spark_backend('encode')._jbackend.encodeToBytes(Env.backend()._to_java_value_ir(expression._ir), codec)


def decode_through_JSON(typ, ptype_string, bytes, codec=stream_codec):
    return typ._from_json(
        Env.spark_backend('decode')._jbackend.decodeToJSON(ptype_string, bytes, codec))


def read_int(byte_array, offset):
    ans = (byte_array[offset] +
           (byte_array[offset + 1] << 8) +
           (byte_array[offset + 2] << 16) +
           (byte_array[offset + 3] << 24))

    return ans


def decode(typ, bytes):
    return typ._from_encoding(bytes)


def lookup_bit(byte, which_bit):
    return (byte >> which_bit) & 1
