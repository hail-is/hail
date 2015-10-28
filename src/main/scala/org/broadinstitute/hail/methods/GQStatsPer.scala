package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._

import scala.collection.mutable

object gqStatCounterPer extends AggregateMethod {

  def name = "gqMean\tgqStDev"

  type T = StatCounter

  override def aggZeroValue = StatCounter()

  override def seqOpWithKeys(v: Variant, s: Int, g: Genotype, sc: StatCounter): StatCounter = {
    if (g.isCalled)
      sc.merge(g.gq)
    sc
  }

  override def combOp(sc1: StatCounter, sc2: StatCounter) = sc1.merge(sc2)

  override def emit(sc: StatCounter, b: mutable.ArrayBuilder[Any]) {
    b += someIf(sc.count > 0, sc.mean)
    b += someIf(sc.count > 0, sc.stdev)
  }
}

object gqStatCounterPerGenotype extends AggregateMethod {
  def name = "gqMeanHomRef\tgqStDevHomRef\tgqMeanHet\tgqStDevHet\tgqMeanHomVar\tgqStDevHomVar"

  type T = (StatCounter, StatCounter, StatCounter)

  override def aggZeroValue: T = (StatCounter(), StatCounter(), StatCounter())

  override def seqOpWithKeys(v: Variant, s: Int, g: Genotype,
    scs: T): T = {
    if (g.isCalled) {
      if (g.isHomRef)
        scs._1.merge(g.gq)
      if (g.isHet)
        scs._2.merge(g.gq)
      if (g.isHomVar)
        scs._3.merge(g.gq)
    }
    scs
  }

  override def combOp(scs1: T, scs2: T): T = (scs1._1.merge(scs2._1), scs1._2.merge(scs2._2), scs1._3.merge(scs2._3))

  override def emit(scs: T, b: mutable.ArrayBuilder[Any]) {
    b += someIf(scs._1.count > 0, scs._1.mean)
    b += someIf(scs._1.count > 0, scs._1.stdev)

    b += someIf(scs._2.count > 0, scs._2.mean)
    b += someIf(scs._2.count > 0, scs._2.stdev)

    b += someIf(scs._3.count > 0, scs._3.mean)
    b += someIf(scs._3.count > 0, scs._3.stdev)
  }
}
