package is.hail.stats

import breeze.stats.distributions._
import org.apache.commons.math3.random.JDKRandomGenerator
import is.hail.utils._

sealed trait Distribution {
    def getBreezeDist(seed: Int): Rand[Double]
}
case class UniformDist(minVal: Double, maxVal: Double) extends Distribution{
  if (minVal >= maxVal)
    fatal("minVal must less than maxVal")



  override def getBreezeDist(seed: Int): Rand[Double] = {
    val generator = new JDKRandomGenerator
    generator.setSeed(seed)
    val basis = new RandBasis(generator)

    new Uniform(minVal, maxVal)(basis)
  }
}

case class BetaDist(a: Double, b: Double) extends Distribution {

  override def getBreezeDist(seed: Int): Rand[Double] = {
    val generator = new JDKRandomGenerator
    generator.setSeed(seed)
    val basis = new RandBasis(generator)

    new Beta(a, b)(basis)
  }
}

case class TruncatedBetaDist(a: Double, b: Double, minVal: Double, maxVal: Double) extends Distribution {
  if (minVal >= maxVal)
    fatal("minVal must be less than maxVal")

  override def getBreezeDist(seed: Int): Rand[Double] = {
    val generator = new JDKRandomGenerator
    generator.setSeed(seed)
    val basis = new RandBasis(generator)

    new TruncatedBeta(a, b, minVal, maxVal)(basis)
  }
}


class TruncatedBeta(a: Double, b: Double, minVal: Double, maxVal: Double)(randBasis: RandBasis) extends Rand[Double] {
  if (minVal >= maxVal)
    fatal("minVal must be less than maxVal")

  private val beta = new Beta(a, b)(randBasis)

  override def draw(): Double = {
    var drawn = beta.draw()

    while(drawn <= minVal || drawn >= maxVal) {
      drawn = beta.draw()
    }

    drawn
  }
}

