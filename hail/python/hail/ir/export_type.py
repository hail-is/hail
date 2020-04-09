from hail.typecheck import *

class ExportType:
    CONCATENATED = "concatenated"
    PARALLEL_SEPARATE_HEADER = "separate_header"
    PARALLEL_HEADER_IN_SHARD = "header_per_shard"

    checker = enumeration(
        ExportType.CONCATENATED,
        ExportType.PARALLEL_SEPARATE_HEADER,
        ExportType.PARALLEL_HEADER_IN_SHARD)

    @staticmethod
    def default(export_type):
        if export_type is None:
            return ExportType.CONCATENATED
        else:
            return export_type
