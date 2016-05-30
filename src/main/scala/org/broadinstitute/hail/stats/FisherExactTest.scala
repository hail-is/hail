package org.broadinstitute.hail.stats

import org.apache.commons.math3.distribution.HypergeometricDistribution
import org.broadinstitute.hail.Utils.{fatal, divOption}

object FisherUtils {
  //based on C code in R source code called zeroin.c
  def uniroot(fn: Double => Double, min: Double, max: Double, tolerance: Double = 1.220703e-4): Option[Double] = {
    var a = min
    var b = max
    var c = min
    require(a < b, "interval start must be larger than end")

    var fa: Double = fn(a)
    var fb: Double = fn(b)
    var fc: Double = fa

    val epsilon = 2.2204460492503131e-16

    val maxIterations = 1001
    var numIterations = 0

    if (fa == 0d)
      Option(a)
    else if (fb == 0d)
      Option(b)
    else {
      while (numIterations != maxIterations) {
        numIterations += 1
        val previousStep = b - a
        var q = 0d
        var p = 0d
        val toleranceActual = 2 * epsilon * math.abs(b) + tolerance / 2
        var newStep = (c - b) / 2

        if (math.abs(fc) < math.abs(fb)) { // swap endpoints so b is best approx
          a = b; b = c; c = a
          fa = fb; fb = fc; fc = fa
        }

        if (math.abs(newStep) <= toleranceActual || fb == 0d) {
          return Option(b) // acceptable approximation is found
        }

        if (math.abs(previousStep) >= toleranceActual && math.abs(fa) > math.abs(fb)) { // try interpolation
          val cb = c - b

          if (a == c) { // if only two distinct points, linear interpolation can only be applied
          val t1 = fb / fa
            p = cb * t1
            q = 1.0 - t1
          } else { // quadric inverse interpolation
            q = fa / fc
            val t1 = fb / fc
            val t2 = fb / fa
            p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1d))
            q = (q - 1d) * (t1 - 1d) * (t2 - 1d)
          }

          if ( p > 0d ) //p was calculated with opposite sign
            q = -q
          else
            p = -p

          if (p < (0.75 * cb * q - math.abs(toleranceActual * q) / 2) &&
            p < math.abs(previousStep * q / 2))
            newStep = p / q
        }

        if (math.abs(newStep) < toleranceActual) {
          if (newStep > 0d)
            newStep = toleranceActual
          else
            newStep = -1 * toleranceActual
        }

        a = b; fa = fb
        b += newStep
        fb = fn(b)

        if  ((fb > 0d && fc > 0d) || (fb < 0d && fc < 0d)) {
          c = a
          fc = fa
        }
      }

      None //No value found within threshold with 1000 iterations
    }
  }
}

object FisherExactTest {
  def apply(a: Int, b: Int, c: Int, d: Int) = new FisherExactTest(a, b, c, d)
  def apply(t: (Int, Int, Int, Int)) = new FisherExactTest(t._1, t._2, t._3, t._4)
  def apply(a: Double, b: Double, c: Double, d: Double) = new FisherExactTest(a.toInt, b.toInt, c.toInt, d.toInt)
  def apply(a: Array[Double]) = new FisherExactTest(a(0).toInt, a(1).toInt, a(2).toInt, a(3).toInt)
}

class FisherExactTest(a: Int, b: Int, c: Int, d: Int) {
  private val popSize = a + b + c + d
  private val numSuccessPopulation = a + c
  private val sampleSize = a + b
  private val numSuccessSample = a

  require(popSize > 0 && sampleSize > 0 && sampleSize < popSize)

  private val low = math.max(0, (a + b) - (c + d))
  private val high = math.min(a + b, a + c)
  private val support = (low to high).toArray
  private val hgd = new HypergeometricDistribution(null, popSize, numSuccessPopulation, sampleSize)

  import FisherUtils._

  def result(oddsRatio: Double = 1d, confidence_level: Double = 0.95, alternative: String = "two.sided") = {

    if (confidence_level < 0d || confidence_level > 1d)
      fatal("Confidence level must be between 0 and 1")
    if (oddsRatio < 0d)
      fatal("Odds ratio must be between 0 and Inf")
    if (alternative != "greater" && alternative != "less" && alternative != "two.sided")
      fatal("Did not recognize test type string. Use one of greater, less, two.sided")

    def dhyper(k: Int, logProb: Boolean = false): Double = {
      if (logProb) hgd.logProbability(k) else hgd.probability(k)
    }

    def logdc = support.map(dhyper(_, logProb = true))

    def dnhyper(ncp: Double): Array[Double] = {
      var d = logdc.zipWithIndex.map { case (hr, i) => hr + math.log(ncp) * i }
      d = d.map { case dens => math.exp(dens - d.max) }
      d.map(_ / d.sum)
    }

    def phyper(k: Int, lower_tail: Boolean = true): Double = {
      if (lower_tail)
        hgd.cumulativeProbability(k)
      else
        hgd.upperCumulativeProbability(k)
    }

    def pnhyper(q: Int, ncp: Double = 1d, upper_tail: Boolean = false): Double = {
      if (ncp == 1d) {
        if (upper_tail)
          phyper(q, lower_tail = false)
        else
          phyper(q, lower_tail = true)
      } else if (ncp == 0d) {
        if (upper_tail)
          if (q <= low) 1d else 0d
        else if (q >= low) 1d else 0d
      } else if (ncp == Double.PositiveInfinity) {
        if (upper_tail)
          if (q <= high) 1d else 0d
        else if (q >= high) 1d else 0d
      } else {
        dnhyper(ncp)
          .zipWithIndex
          .filter { case (d, i) => if (upper_tail) support(i) >= q else support(i) <= q }
          .map { case (d, i) => d }
          .sum
      }
    }

    def mnhyper(ncp: Double): Double = {
      if (ncp == 0d)
        low
      else if (ncp == Double.PositiveInfinity)
        high
      else
        dnhyper(ncp).zipWithIndex.map { case (dnh, i) => dnh * support(i) }.sum
    }

    def unirootMnHyper(fn: Double => Double, x: Double)(t: Double) = mnhyper(fn(t)) - x

    def unirootPnHyper(fn: Double => Double, x: Int, upper_tail: Boolean, alpha: Double)(t: Double) = pnhyper(x, fn(t), upper_tail) - alpha

    def inverse(x: Double) = divOption(1d, x)

    def mle(x: Double): Option[Double] = {
      if (x == low)
        Option(0d)
      else if (x == high)
        Option(Double.PositiveInfinity) // Should be infinity
      else {
        val mu = mnhyper(1.0)
        if (mu > x)
          uniroot(unirootMnHyper(d => d, x) _, 0.0, 1.0)
        else if (mu < x)
          uniroot(unirootMnHyper(d => 1 / d, x) _, 2.220446e-16, 1d).flatMap(inverse)
        else
          Option(1d)
      }
    }

    def ncpLower(x: Int, alpha: Double): Option[Double] = {
      if (x == low)
        Option(0d)
      else {
        val p = pnhyper(x, 1, upper_tail = true)
        if (p > alpha)
          uniroot(unirootPnHyper(d => d, x, true, alpha)_, 0d, 1d)
        else if (p < alpha)
          uniroot(unirootPnHyper(d => 1 / d, x, true, alpha) _, 2.220446e-16, 1d).flatMap(inverse)
        else
          Option(1d)
      }
    }

    def ncpUpper(x: Int, alpha: Double): Option[Double] = {
      if (x == high) {
        Option(Double.PositiveInfinity)
      }
      else {
        val p = pnhyper(x, 1)
        if (p < alpha)
          uniroot(unirootPnHyper(d => d, x, false, alpha)_, 0d, 1d)
        else if (p > alpha)
          uniroot(unirootPnHyper(d => 1 / d, x, false, alpha) _, 2.220446e-16, 1d).flatMap(inverse)
        else
          Option(1d)
      }
    }

    val pvalue = alternative match {
      case "less" => pnhyper(numSuccessSample, oddsRatio)
      case "greater" => pnhyper(numSuccessSample, oddsRatio, upper_tail = true)
      case "two.sided" =>
        if (oddsRatio == 0)
          if (low == numSuccessSample) 1d else 0d

        else if (oddsRatio == Double.PositiveInfinity)
          if (high == numSuccessSample) 1d else 0d
        else {
          val relErr = 1d + 1e-7
          val d = dnhyper(oddsRatio)
          d.filter(_ <= d(numSuccessSample - low) * relErr).sum
        }

      case _ => throw new IllegalArgumentException("didn't recognize option for alternative. Use one of [less, greater, two.sided]")
    }

    if (pvalue < 0d || pvalue > 1.000000000002)
      fatal(s"pvalue not valid probability: $pvalue")

    val oddsRatioEstimate = mle(numSuccessSample)

    val confInterval = alternative match {
      case "less" => (Option(0d), ncpUpper(numSuccessSample, 1 - confidence_level))
      case "greater" => (ncpLower(numSuccessSample, 1 - confidence_level), Option(Double.PositiveInfinity))
      case "two.sided" =>
        val alpha = (1 - confidence_level) / 2d
        (ncpLower(numSuccessSample, alpha), ncpUpper(numSuccessSample, alpha))
    }

    Array(Option(pvalue), oddsRatioEstimate, confInterval._1, confInterval._2)
  }
}
