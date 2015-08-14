package org.broadinstitute.k3.methods

import scala.collection.Map
import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

// FIXME need to account for all HomRef?
object nNoCallPerSample extends SampleMethod{
  def name = "nNoCall"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => if (g.isNotCalled) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}
