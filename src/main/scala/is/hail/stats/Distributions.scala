package is.hail.stats

import breeze.stats.distributions._
import is.hail.utils._

sealed trait Distribution {
    def getBreezeDist(randBasis: RandBasis): Rand[Double]
}
case class UniformDist(minVal: Double, maxVal: Double) extends Distribution {
  if (minVal >= maxVal)
    fatal("minVal must less than maxVal")

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new Uniform(minVal, maxVal)(randBasis)
  }
}

case class BetaDist(a: Double, b: Double) extends Distribution {

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new Beta(a, b)(randBasis)
  }
}

case class TruncatedBetaDist(a: Double, b: Double, minVal: Double, maxVal: Double) extends Distribution {
  if (minVal >= maxVal)
    fatal("minVal must be less than maxVal")

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new TruncatedBeta(a, b, minVal, maxVal)(randBasis)
  }
}


class TruncatedBeta(a: Double, b: Double, minVal: Double, maxVal: Double)(randBasis: RandBasis) extends Rand[Double] {
  if (minVal >= maxVal)
    fatal("minVal must be less than maxVal")

  if (minVal < 0)
    fatal("minVal cannot be less than 0")

  if(maxVal > 1)
    fatal("maxVal cannot be greater than 1")

  private val beta = new Beta(a, b)(randBasis)

  override def draw(): Double = {
    var drawn = beta.draw()

    while(drawn < minVal || drawn > maxVal) {
      drawn = beta.draw()
    }

    drawn
  }
}

