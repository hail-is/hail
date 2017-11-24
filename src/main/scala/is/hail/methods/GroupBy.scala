package is.hail.methods

import is.hail.annotations.RegionValue
import is.hail.sparkextras.OrderedPartitioner2
import is.hail.variant.VariantSampleMatrix
import org.apache.spark.rdd.RDD

object GroupByRows {
  /// TODO: Tentatively, keyExpr takes an array or set of keys by which to count each row key.
  def apply(vsm: VariantSampleMatrix, keyName: String, keyExpr: String, aggExpr: String): VariantSampleMatrix = {
    //def apply(typ: OrderedRDD2Type, rdd: RDD[RegionValue], fastKeys: Option[RDD[RegionValue]], hintPartitioner: Option[OrderedPartitioner2]):
  }
}

class GroupByColumns(key: String, aggExpr: String) {

}
