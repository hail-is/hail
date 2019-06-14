from hail.utils.java import Env


def encode(self, ir, codec='default'):
    v = Env.hc()._jhc.backend().encode(self._to_java_ir(ir), codec)
    return (v._1(), v._2())


def decode(self, typ, ptype_string, bytes, codec='default'):
    return typ._from_json(
        Env.hc()._jhc.backend().decodeToJSON(ptype_string, bytes, codec))
