package org.broadinstitute.hail.methods

import org.apache.commons.math3.distribution.BinomialDistribution
import org.broadinstitute.hail.variant.Genotype

import scala.collection.mutable

object AlleleBalancePer extends MapRedMethod {
  def name = "AlleleBalance"
  type T = (Int, Int)
  override def map(g: Genotype): T = if (g.isHet) g.ad else (0, 0)
  def foldZeroValue = (0, 0)
  def fold(x: T, y: T): T = (x._1 + y._1, x._2 + y._2)

  override def emit(x: T, b: mutable.ArrayBuilder[Any]) {
    val (refd, altd) = x
    val d = new BinomialDistribution(refd + altd, 0.5)
    val mind = refd.min(altd)
    val minp = d.probability(mind)
    val mincp = d.cumulativeProbability(mind)
    val p = 2 * mincp - minp
    b += p.min(1.0).max(0.0)
  }
}
