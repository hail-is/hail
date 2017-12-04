package is.hail.rvd

import is.hail.annotations.RegionValue
import is.hail.expr.Type
import org.apache.spark.rdd.RDD

class ConcreteRVD(val rowType: Type, val rdd: RDD[RegionValue]) extends RVD
