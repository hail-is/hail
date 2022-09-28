from hail.utils.java import Env

stream_codec = '{"name":"StreamBufferSpec"}'


def encode(expression, codec=stream_codec):
    jir = Env.backend()._to_java_value_ir(expression._ir)
    return Env.backend()._jhc.backend().executeEncode(jir, codec, False)._1()


def decode(typ, bytes):
    return typ._from_encoding(bytes)
