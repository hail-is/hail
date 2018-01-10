from hail.utils.java import handle_py4j, Env

class FunctionDocumentation(object):
    @handle_py4j
    def types_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeTypesDocs(file_name)

    @handle_py4j
    def functions_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeFunctionsDocs(file_name)


def wrap_to_list(s):
    if isinstance(s, list):
        return s
    else:
        return [s]

def get_env_or_default(maybe, envvar, default):
    import os

    return maybe or os.environ.get(envvar) or default

@handle_py4j
def new_temp_file(n_char = 10, prefix=None, suffix=None):
    return Env.hc()._jhc.getTemporaryFile(n_char, joption(prefix), joption(suffix))
