package is.hail.sparkextras.implicits

import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.PStruct

import org.apache.spark.sql.Row

class RichContextRDDRow(crdd: ContextRDD[Row]) {
  def toRegionValues(rowType: PStruct): ContextRDD[Long] =
    crdd.cmapPartitions((ctx, it) => it.copyToRegion(ctx.region, rowType))
}
