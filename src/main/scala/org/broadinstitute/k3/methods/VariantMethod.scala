package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

import scala.reflect.ClassTag

abstract class VariantMethod[T](implicit tt: ClassTag[T]) {
  def name: String
  def apply(vds: VariantDataset): RDD[(Variant, T)]
  def run(vds: VariantDataset): (String, RDD[(Variant, T)]) = (name, this(vds))
}
