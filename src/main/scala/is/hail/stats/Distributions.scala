package is.hail.stats

import breeze.stats.distributions._
import is.hail.utils._

sealed trait Distribution {
    def getBreezeDist(randBasis: RandBasis): Rand[Double]
}
case class UniformDist(min: Double, max: Double) extends Distribution {
  if (min >= max)
    fatal(s"min $min must be less than max $max")

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new Uniform(min, max)(randBasis)
  }
}

case class BetaDist(a: Double, b: Double) extends Distribution {
  if (a <= 0 || b <= 0)
    fatal(s"a and b must be positive, got $a and $b")

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new Beta(a, b)(randBasis)
  }
}

case class TruncatedBetaDist(a: Double, b: Double, min: Double, max: Double) extends Distribution {
  if (min >= max)
    fatal(s"min $min must be less than max $max")

  override def getBreezeDist(randBasis: RandBasis): Rand[Double] = {
    new TruncatedBeta(a, b, min, max)(randBasis)
  }
}


class TruncatedBeta(a: Double, b: Double, min: Double, max: Double)(randBasis: RandBasis) extends Rand[Double] {
  if (min >= max)
    fatal(s"min $min must be less than max $max")

  if (min < 0)
    fatal(s"min $min must be at least 0")

  if(max > 1)
    fatal(s"max $min must be at most 1")

  private val beta = new Beta(a, b)(randBasis)

  // rejection sampling
  override def draw(): Double = {
    var drawn = beta.draw()

    while(drawn < min || drawn > max) {
      drawn = beta.draw()
    }

    drawn
  }
}

