
def jarray(gateway, jtype, lst):
    jarr = gateway.new_array(jtype, len(lst))
    for i, s in enumerate(lst):
        jarr[i] = s
    return jarr

def scala_object(jpackage, name):
    return getattr(getattr(jpackage, name + '$'), 'MODULE$')

def scala_package_object(jpackage):
    return scala_object(jpackage, 'package')

def jnone(jvm):
    return scala_object(jvm.scala, 'None')

def jsome(jvm, x):
    return jvm.scala.Some(x)

def joption(jvm, x):
    return jsome(jvm, x) if x else jnone(jvm)

def strip_option(x):
    return x.get() if x.isDefined() else None
