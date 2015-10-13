package org.broadinstitute.k3.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._

object dpMeanPer extends DerivedMethod {

  def name = "dpMean"

  type T = Double

  override def map(vals: MethodValues) = vals.get(dpStatCounterPer).mean
}

object dpStDevPer extends DerivedMethod {

  def name = "dpMean"

  type T = Double

  override def map(vals: MethodValues) = vals.get(dpStatCounterPer).stdev
}


object dpStatCounterPer extends AggregateMethod {

  def name = "dpStatCounter"

  type T = StatCounter

  override def aggZeroValue = StatCounter()

  override def seqOpWithKeys(v: Variant, s: Int, g: Genotype, sc: StatCounter): StatCounter = {
    val mean = sc.mean
    if (g.isCalled)
      sc.merge(g.dp)
    sc
  }

  override def combOp(sc1: StatCounter, sc2: StatCounter) = sc1.merge(sc2)
}