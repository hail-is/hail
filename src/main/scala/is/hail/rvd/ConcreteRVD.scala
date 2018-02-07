package is.hail.rvd

import is.hail.annotations.RegionValue
import is.hail.expr.types.{TStruct, Type}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object ConcreteRVD {
  def empty(sc: SparkContext, rowType: Type): ConcreteRVD = new ConcreteRVD(rowType, sc.emptyRDD[RegionValue])
}

class ConcreteRVD(val typ: Type, val rdd: RDD[RegionValue]) extends RVD {
  override def rowType: TStruct = typ match {
    case t: TStruct => t
  }
}
