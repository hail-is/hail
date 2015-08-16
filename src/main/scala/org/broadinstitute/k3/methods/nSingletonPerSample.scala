package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nSingletonPerSample extends SampleMethod[Int] {
  def name = "nSingleton"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    val singletons = vds.sparkContext.broadcast(sSingletonVariants(vds))

    vds
      .mapValuesWithKeys((v,s,g) => if (g.isNonRef && singletons.value.contains(v)) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}
