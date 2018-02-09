package is.hail.methods

import is.hail.annotations.UnsafeRow
import is.hail.expr.types.Type
import is.hail.utils.{Interval, IntervalTree}
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    vsm.requireRowKeyVariant("filter_intervals")
    val locusField = vsm.rowType.fieldByName(vsm.rowKey(0))
    val iList = IntervalTree(locusField.typ.ordering, intervals.asScala.toArray)
    apply(vsm, iList, keep)
  }

  def apply[U](vsm: MatrixTable, intervals: IntervalTree[U], keep: Boolean): MatrixTable = {
    if (keep) {
      val pkIntervals = IntervalTree(
        vsm.matrixType.orderedRVType.pkType.ordering,
        intervals.map { case (i, _) =>
          Interval(Row(i.start), Row(i.end))
        }.toArray)
      vsm.copy2(rvd = vsm.rvd.filterIntervals(pkIntervals))
    } else {
      val intervalsBc = vsm.sparkContext.broadcast(intervals)

      val locusField = vsm.rowType.fieldByName(vsm.rowKey(0))
      val locusIdx = locusField.index

      val fullRowType = vsm.rvRowType
      val localLocusOrdering = locusField.typ.ordering

      vsm.copy2(rvd = vsm.rvd.mapPartitionsPreservesPartitioning(vsm.rvd.typ) { it =>
        val ur = new UnsafeRow(fullRowType)
        it.filter { rv =>
          ur.set(rv)
          !intervalsBc.value.contains(localLocusOrdering, ur.get(locusIdx))
        }
      })
    }
  }
}
