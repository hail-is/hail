from hail.utils.java import Env
import struct

block_codec = '{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'
stream_codec = '{"name":"StreamBufferSpec"}'


def encode(expression, codec=stream_codec):
    return Env.spark_backend('encode')._jbackend.encodeToBytes(Env.backend()._to_java_value_ir(expression._ir), codec)


def decode(typ, bytes):
    return typ._from_encoding(bytes)

