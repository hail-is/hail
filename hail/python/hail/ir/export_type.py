from hail.typecheck import *

class ExportType:
    checker = enumeration('separate_header', 'header_per_shard')

    CONCATENATED = "concatenated"
    PARALLEL_SEPARATE_HEADER = "separate_header"
    PARALLEL_HEADER_IN_SHARD = "header_per_shard"

    @staticmethod
    def default(export_type):
        if export_type is None:
            return ExportType.CONCATENATED
        else:
            return export_type
