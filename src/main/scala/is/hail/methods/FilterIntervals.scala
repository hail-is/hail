package is.hail.methods

import is.hail.expr.types.Type
import is.hail.utils.{Interval, IntervalTree}
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    vsm.requireRowKeyVariant("filter_intervals")
    val iList = IntervalTree(vsm.locusType.ordering, intervals.asScala.toArray)
    apply(vsm, iList, keep)
  }

  def apply[U](vsm: MatrixTable, intervals: IntervalTree[U], keep: Boolean): MatrixTable = {
    if (keep) {
      val pkIntervals = IntervalTree(
        vsm.matrixType.pkType.ordering,
        intervals.map { case (i, _) =>
          Interval(Row(i.start), Row(i.end))
        }.toArray)
      vsm.copy2(rdd2 = vsm.rdd2.filterIntervals(pkIntervals))
    } else {
      val intervalsBc = vsm.sparkContext.broadcast(intervals)
      val (t, p) = Type.partitionKeyProjection(vsm.vSignature)
      assert(t == vsm.locusType)
      val localLocusOrdering = vsm.locusType.ordering
      vsm.filterVariants { (v, va, gs) => !intervalsBc.value.contains(localLocusOrdering, p(v)) }
    }
  }
}
