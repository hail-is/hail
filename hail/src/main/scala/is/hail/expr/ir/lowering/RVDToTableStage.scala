package is.hail.expr.ir.lowering

import is.hail.expr.ir.{IR, PartitionRVDReader, ReadPartition, StreamRange}
import is.hail.rvd.RVD

object RVDToTableStage {
  def apply(rvd: RVD, globals: IR): TableStage = {
    TableStage(
      globals = globals,
      partitioner = rvd.partitioner,
      contexts = StreamRange(0, rvd.getNumPartitions, 1),
      body = ReadPartition(_, rvd.rowType, PartitionRVDReader(rvd))
    )
  }
}