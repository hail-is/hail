package is.hail.methods

import is.hail.rvd.{RVD, RVDPartitioner, RVDType}
import is.hail.table.Table
import is.hail.utils.Interval
import is.hail.variant.MatrixTable

import scala.collection.JavaConverters._

object MatrixFilterIntervals {
  def apply(mt: MatrixTable, jintervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    val partitioner = RVDPartitioner.union(
      mt.rvd.typ.kType.virtualType,
      jintervals.asScala.toFastIndexedSeq,
      mt.rvd.typ.key.length - 1)
    mt.copy2(rvd = mt.rvd.filterIntervals(partitioner, keep))
  }
}

object TableFilterIntervals {
  def apply(ht: Table, jintervals: java.util.ArrayList[Interval], keep: Boolean): Table = {
    val partitioner = RVDPartitioner.union(
      ht.rvd.typ.kType.virtualType,
      jintervals.asScala.toFastIndexedSeq,
      ht.rvd.typ.key.length - 1)
    ht.copy2(rvd = ht.rvd.filterIntervals(partitioner, keep))
  }
}
