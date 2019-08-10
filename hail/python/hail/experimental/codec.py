from hail.utils.java import Env


def encode(expression, codec='default'):
    v = Env.hc()._jhc.backend().encode(Env.backend()._to_java_ir(expression._ir), codec)
    return (v._1(), v._2())


def decode(typ, ptype_string, bytes, codec='default'):
    return typ._from_json(
        Env.hc()._jhc.backend().decodeToJSON(ptype_string, bytes, codec))
