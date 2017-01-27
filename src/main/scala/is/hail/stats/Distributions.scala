package is.hail.stats

import breeze.stats.distributions._
import org.apache.commons.math3.random.JDKRandomGenerator
import is.hail.utils._

sealed trait Distribution {
    def getBreezeDist(seed: Int): ContinuousDistr[Double]
}
case class UniformDist(a: Double, b: Double) extends Distribution{
  if (a >= b)
    fatal("a must less than b")

  override def getBreezeDist(seed: Int): ContinuousDistr[Double] = {
    val generator = new JDKRandomGenerator
    generator.setSeed(seed)
    val basis = new RandBasis(generator)

    new Uniform(a, b)(basis)
  }
}

case class BetaDist(a: Double, b: Double) extends Distribution {

  override def getBreezeDist(seed: Int): ContinuousDistr[Double] = {
    val generator = new JDKRandomGenerator
    generator.setSeed(seed)
    val basis = new RandBasis(generator)

    new Beta(a, b)(basis)
  }
}

trait DistributionFactory {

  def createDist(seed: Int): Distribution
}

case class UniformFactory(a: Double, b: Double) extends DistributionFactory {

  override def createDist(seed: Int): Distribution = ???
}