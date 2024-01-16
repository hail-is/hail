package is.hail.utils.richUtils
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.PStruct
import is.hail.utils._

import org.apache.spark.sql.Row

class RichContextRDDRow(crdd: ContextRDD[Row]) {
  def toRegionValues(rowType: PStruct): ContextRDD[Long] =
    crdd.cmapPartitions((ctx, it) => it.copyToRegion(ctx.region, rowType))
}
