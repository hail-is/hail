package is.hail.methods

import is.hail.utils.{Interval, IntervalTree}
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    vsm.requirePartitionKeyLocus("filter_intervals")
    val locusField = vsm.rowType.fieldByName(vsm.rowPartitionKey(0))
    val iList = IntervalTree(locusField.typ.ordering,
      intervals.asScala.map { i =>
        Interval(Row(i.start), Row(i.end), i.includeStart, i.includeEnd)
      }.toArray)
    apply(vsm, iList, keep)
  }

  def apply[U](vsm: MatrixTable, intervals: IntervalTree[U], keep: Boolean): MatrixTable = {
    if (keep) {
      vsm.copy2(rvd = vsm.rvd.filterIntervals(intervals))
    } else {
      val intervalsBc = vsm.sparkContext.broadcast(intervals)
      val pord = vsm.rvd.typ.pkRowEOrd

      vsm.copy2(rvd = vsm.rvd.mapPartitionsPreservesPartitioning(vsm.rvd.typ) { it =>
        it.filter { rv =>
          !intervalsBc.value.contains(pord, rv)
        }
      })
    }
  }
}
