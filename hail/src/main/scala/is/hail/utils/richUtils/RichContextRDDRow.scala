package is.hail.utils.richUtils

import is.hail.annotations.RegionValue
import is.hail.expr.types.physical.PStruct
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import org.apache.spark.sql.Row

class RichContextRDDRow(crdd: ContextRDD[RVDContext, Row]) {
  def toRegionValues(rowType: PStruct): ContextRDD[RVDContext, RegionValue] = {
    crdd.cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowType))
  }
}
