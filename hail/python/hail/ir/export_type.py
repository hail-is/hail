from hail.typecheck import enumeration


class ExportType:
    CONCATENATED = "concatenated"
    PARALLEL_SEPARATE_HEADER = "separate_header"
    PARALLEL_HEADER_IN_SHARD = "header_per_shard"
    PARALLEL_COMPOSABLE = "composable"

    checker = enumeration("concatenated", "separate_header", "header_per_shard",
                          "composable")

    @staticmethod
    def default(export_type):
        if export_type is None:
            return ExportType.CONCATENATED
        else:
            return export_type
