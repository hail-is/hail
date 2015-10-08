package org.broadinstitute.k3.methods

import org.apache.commons.math3.distribution.BinomialDistribution
import org.broadinstitute.k3.variant.Genotype

object AlleleDepthPerVariant extends MapRedMethod {
  def name = "AlleleDepth"
  type T = (Int, Int)
  override def map(g: Genotype): T = if (g.isHet) g.ad else (0, 0)
  def foldZeroValue = (0, 0)
  def fold(x: T, y: T): T = (x._1 + y._1, x._2 + y._2)
}

object AlleleBalancePerVariant extends DerivedMethod {
  type T = Double
  def name = "AlleleBalance"
  def map(values: MethodValues): T = {
    val (refd, altd) = values.get(AlleleDepthPerVariant)
    val d = new BinomialDistribution(refd + altd, 0.5)
    val mind = refd.min(altd)
    val minp = d.probability(mind)
    val mincp = d.cumulativeProbability(mind)
    val p = (2 * mincp - minp)
    p.min(1.0).max(0.0)
  }
}
