from hail.utils.java import Env


def encode(expression, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    v = Env.spark_backend('encode')._jbackend.encodeToBytes(Env.backend()._to_java_value_ir(expression._ir), codec)
    return (v._1(), v._2())


def decode(typ, ptype_string, bytes, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    return typ._from_json(
        Env.spark_backend('decode')._jbackend.decodeToJSON(ptype_string, bytes, codec))
