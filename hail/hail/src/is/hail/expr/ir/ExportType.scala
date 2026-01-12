package is.hail.expr.ir

object ExportType {
  val CONCATENATED = "concatenated"
  val PARALLEL_SEPARATE_HEADER = "separate_header"
  val PARALLEL_HEADER_IN_SHARD = "header_per_shard"
  val PARALLEL_COMPOSABLE = "composable"
}
