from hail.utils.java import Env


def encode(expression, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    v = Env.hc()._jhc.backend().encodeToBytes(Env.backend()._to_java_ir(expression._ir), codec)
    return (v._1(), v._2())


def decode(typ, ptype_string, bytes, codec='{"name":"BlockingBufferSpec","blockSize":65536,"child":{"name":"StreamBlockBufferSpec"}}'):
    return typ._from_json(
        Env.hc()._jhc.backend().decodeToJSON(ptype_string, bytes, codec))
