package is.hail.methods

import is.hail.annotations.{UnsafeRow, WritableRegionValue}
import is.hail.utils.{Interval, IntervalTree}
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    val iList = IntervalTree(vsm.rvd.typ.pkType.ordering, intervals.asScala.toArray)
    apply(vsm, iList, keep)
  }

  def apply[U](vsm: MatrixTable, intervals: IntervalTree[U], keep: Boolean): MatrixTable = {
    if (keep) {
      vsm.copy2(rvd = vsm.rvd.filterIntervals(intervals))
    } else {
      val intervalsBc = vsm.sparkContext.broadcast(intervals)
      val pkType = vsm.rvd.typ.pkType
      val pkRowFieldIdx = vsm.rvd.typ.pkRowFieldIdx
      val rowType = vsm.rvd.typ.rowType

      vsm.copy2(rvd = vsm.rvd.mapPartitionsPreservesPartitioning(vsm.rvd.typ, { (ctx, it) =>
        val pkUR = new UnsafeRow(pkType)
        it.filter { rv =>
          ctx.rvb.start(pkType)
          ctx.rvb.selectRegionValue(rowType, pkRowFieldIdx, rv)
          pkUR.set(ctx.region, ctx.rvb.end())
          !intervalsBc.value.contains(pkType.ordering, pkUR)
        }
      }))
    }
  }
}
