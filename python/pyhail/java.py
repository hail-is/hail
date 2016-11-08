
def jarray(gateway, jtype, lst):
    jarr = gateway.new_array(jtype, len(lst))
    for i, s in enumerate(lst):
        jarr[i] = s
    return jarr

def scala_object(jpackage, name):
    return getattr(getattr(jpackage, name + '$'), 'MODULE$')

def scala_package_object(jpackage):
    return scala_object(jpackage, 'package')
