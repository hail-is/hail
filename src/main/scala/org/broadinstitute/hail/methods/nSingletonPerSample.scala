package org.broadinstitute.hail.methods
import org.apache.spark.broadcast.Broadcast
import org.broadinstitute.hail.variant._

class nSingletonPerSample(singletons: Broadcast[Set[Variant]]) extends SumMethod {
  def name = "nSingleton"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) =
    if (g.isNonRef && singletons.value.contains(v)) 1 else 0
}
