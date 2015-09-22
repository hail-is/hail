package org.broadinstitute.k3.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._

object gqMeanPer extends DerivedMethod {

  def name = "gqMean"

  type T = Double

  override def map(vals: MethodValues) = vals.get(gqStatCounterPer).mean
}

object gqStDevPer extends DerivedMethod {

  def name = "gqMean"

  type T = Double

  override def map(vals: MethodValues) = vals.get(gqStatCounterPer).stdev
}


object gqStatCounterPer extends AggregateMethod {

  // FIXME - double underscore to indicate not to be called directly
  def name = "__gqStatCounter"

  type T = StatCounter

  override def aggZeroValue = StatCounter()

  override def seqOpWithKeys(v: Variant, s: Int, g: Genotype, sc: StatCounter): StatCounter = {
    if (g.isCalled)
      sc.merge(g.gq)
    sc
  }

  override def combOp(sc1: StatCounter, sc2: StatCounter) = sc1.merge(sc2)
}