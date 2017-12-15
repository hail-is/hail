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

def get_export_type(et):
    export_types = {None: Env.hail().utils.ExportType.CONCATENATED(),
                    'separate_header': Env.hail().utils.ExportType.PARALLEL_SEPARATE_HEADER(),
                    'header_per_shard': Env.hail().utils.ExportType.PARALLEL_HEADER_IN_SHARD()}

    assert(et in export_types)
    return export_types[et]
