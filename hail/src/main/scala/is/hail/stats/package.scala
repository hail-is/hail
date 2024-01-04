package is.hail

import is.hail.types.physical.{PCanonicalStruct, PFloat64}
import is.hail.utils._

import net.sourceforge.jdistlib.{Beta, ChiSquare, NonCentralChiSquare, Normal, Poisson}
import net.sourceforge.jdistlib.disttest.{DistributionTest, TestKind}
import org.apache.commons.math3.distribution.HypergeometricDistribution

package object stats {

  def uniroot(fn: Double => Double, min: Double, max: Double, tolerance: Double = 1.220703e-4)
    : Option[Double] = {
    // based on C code in R source code called zeroin.c
    /* https://github.com/wch/r-source/blob/e5b21d0397c607883ff25cca379687b86933d730/src/library/stats/src/zeroin.c */

    require(min < max, "interval start must be larger than end")

    var a = min
    var b = max
    var c = min

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

        if (math.abs(fc) < math.abs(fb)) {
          // swap endpoints so b is best approx
          a = b
          b = c
          c = a
          fa = fb
          fb = fc
          fc = fa
        }

        if (math.abs(newStep) <= toleranceActual || fb == 0d) {
          return Option(b) // acceptable approximation is found
        }

        if (math.abs(previousStep) >= toleranceActual && math.abs(fa) > math.abs(fb)) {
          // try interpolation
          val cb = c - b

          if (a == c) {
            // if only two distinct points, linear interpolation can only be applied
            val t1 = fb / fa
            p = cb * t1
            q = 1.0 - t1
          } else {
            // quadric inverse interpolation
            q = fa / fc
            val t1 = fb / fc
            val t2 = fb / fa
            p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1d))
            q = (q - 1d) * (t1 - 1d) * (t2 - 1d)
          }

          if (p > 0d) // p was calculated with opposite sign
            q = -q
          else
            p = -p

          if (
            p < (0.75 * cb * q - math.abs(toleranceActual * q) / 2) &&
            p < math.abs(previousStep * q / 2)
          )
            newStep = p / q
        }

        if (math.abs(newStep) < toleranceActual) {
          if (newStep > 0d)
            newStep = toleranceActual
          else
            newStep = -1 * toleranceActual
        }

        a = b
        fa = fb
        b += newStep
        fb = fn(b)

        if ((fb > 0d && fc > 0d) || (fb < 0d && fc < 0d)) {
          c = a
          fc = fa
        }
      }

      None // No value found within threshold with 1000 iterations
    }
  }

  val hweStruct = PCanonicalStruct("het_freq_hwe" -> PFloat64(true), "p_value" -> PFloat64(true))

  def hardyWeinbergTest(nHomRef: Int, nHet: Int, nHomVar: Int, oneSided: Boolean): Array[Double] = {
    if (nHomRef < 0 || nHet < 0 || nHomVar < 0)
      fatal(
        s"hardy_weinberg_test: all arguments must be non-negative, got $nHomRef, $nHet, $nHomVar"
      )

    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    val pVal = if (oneSided) LH.rightMidP(nAB) else LH.exactMidP(nAB)
    Array(LH.getNumericalMean / n, pVal)
  }

  val chisqStruct = PCanonicalStruct("p_value" -> PFloat64(true), "odds_ratio" -> PFloat64(true))

  def chiSquaredTest(a0: Int, b0: Int, c0: Int, d0: Int): Array[Double] = {
    if (a0 < 0 || b0 < 0 || c0 < 0 || d0 < 0)
      fatal(s"chi_squared_test: all arguments must be non-negative, got $a0, $b0, $c0, $d0")

    val a = a0.toDouble
    val b = b0.toDouble
    val c = c0.toDouble
    val d = d0.toDouble

    val ad = a * d
    val bc = b * c
    val det = ad - bc
    val chiSquare = (a + b + c + d) * (det / ((a + b) * (c + d))) * (det / ((b + d) * (a + c)))

    Array(pchisqtail(chiSquare, 1), ad / bc)
  }

  def contingencyTableTest(a: Int, b: Int, c: Int, d: Int, minCellCount: Int): Array[Double] = {
    if (minCellCount < 0)
      fatal(s"contingency_table_test: 'min_cell_count' must be non-negative, found $minCellCount")

    if (a >= minCellCount && b >= minCellCount && c >= minCellCount && d >= minCellCount)
      chiSquaredTest(a, b, c, d)
    else
      fisherExactTest(a, b, c, d)
  }

  val fetStruct = PCanonicalStruct(
    "p_value" -> PFloat64(required = true),
    "odds_ratio" -> PFloat64(required = true),
    "ci_95_lower" -> PFloat64(required = true),
    "ci_95_upper" -> PFloat64(required = true),
  )

  def fisherExactTest(a: Int, b: Int, c: Int, d: Int): Array[Double] =
    fisherExactTest(a, b, c, d, 1.0, 0.95, "two.sided")

  def fisherExactTest(
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    oddsRatio: Double = 1d,
    confidenceLevel: Double = 0.95,
    alternative: String = "two.sided",
  ): Array[Double] = {

    if (!(a >= 0 && b >= 0 && c >= 0 && d >= 0))
      fatal(s"fisher_exact_test: all arguments must be non-negative, got $a, $b, $c, $d")

    if (confidenceLevel < 0d || confidenceLevel > 1d)
      fatal("Confidence level must be between 0 and 1")

    if (oddsRatio < 0d)
      fatal("Odds ratio must be non-negative")

    if (alternative != "greater" && alternative != "less" && alternative != "two.sided")
      fatal("Did not recognize test type string. Use one of greater, less, two.sided")

    val popSize = a + b + c + d
    val numSuccessPopulation = a + c
    val sampleSize = a + b
    val numSuccessSample = a

    if (
      !(popSize > 0 && sampleSize > 0 && sampleSize < popSize && numSuccessPopulation > 0 && numSuccessPopulation < popSize)
    )
      return Array(Double.NaN, Double.NaN, Double.NaN, Double.NaN)

    val low = math.max(0, (a + b) - (b + d))
    val high = math.min(a + b, a + c)
    val support = (low to high).toArray

    val hgd = new HypergeometricDistribution(null, popSize, numSuccessPopulation, sampleSize)
    val epsilon = 2.220446e-16

    def dhyper(k: Int, logProb: Boolean): Double =
      if (logProb) hgd.logProbability(k) else hgd.probability(k)

    val logdc = support.map(dhyper(_, logProb = true))

    def dnhyper(ncp: Double): Array[Double] = {
      var d = logdc.zipWithIndex.map { case (hr, i) => hr + math.log(ncp) * i }
      d = d.map(dens => math.exp(dens - d.max))
      d.map(_ / d.sum)
    }

    def phyper(k: Int, lower_tail: Boolean): Double =
      if (lower_tail)
        hgd.cumulativeProbability(k)
      else
        hgd.upperCumulativeProbability(k)

    def pnhyper(q: Int, ncp: Double = 1d, upper_tail: Boolean = false): Double = {
      if (ncp == 1d) {
        if (upper_tail)
          phyper(q, lower_tail = false)
        else
          phyper(q, lower_tail = true)
      } else if (ncp == 0d) {
        if (upper_tail)
          if (q <= low) 1d else 0d
        else if (q >= low) 1d
        else 0d
      } else if (ncp == Double.PositiveInfinity) {
        if (upper_tail)
          if (q <= high) 1d else 0d
        else if (q >= high) 1d
        else 0d
      } else {
        dnhyper(ncp).zipWithIndex
          .filter { case (_, i) => if (upper_tail) support(i) >= q else support(i) <= q }
          .map { case (dbl, _) => dbl }
        dnhyper(ncp)
          .zipWithIndex
          .filter { case (_, i) => if (upper_tail) support(i) >= q else support(i) <= q }
          .map { case (dbl, _) => dbl }
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

    def unirootPnHyper(
      fn: Double => Double,
      x: Int,
      upper_tail: Boolean,
      alpha: Double,
    )(
      t: Double
    ) =
      pnhyper(x, fn(t), upper_tail) - alpha

    def mle(x: Double): Double = {
      if (x == low)
        0.0
      else if (x == high)
        Double.PositiveInfinity
      else {
        val mu = mnhyper(1.0)
        if (mu > x)
          uniroot(unirootMnHyper(d => d, x), 0.0, 1.0).getOrElse(Double.NaN)
        else if (mu < x)
          1.0 / uniroot(unirootMnHyper(d => 1 / d, x), epsilon, 1d).getOrElse(Double.NaN)
        else
          1.0
      }
    }

    def ncpLower(x: Int, alpha: Double): Double = {
      if (x == low)
        0.0
      else {
        val p = pnhyper(x, upper_tail = true)
        if (p > alpha)
          uniroot(unirootPnHyper(d => d, x, upper_tail = true, alpha), 0d, 1d).getOrElse(Double.NaN)
        else if (p < alpha)
          1.0 / uniroot(
            unirootPnHyper(d => 1 / d, x, upper_tail = true, alpha),
            epsilon,
            1d,
          ).getOrElse(Double.NaN)
        else
          1.0
      }
    }

    def ncpUpper(x: Int, alpha: Double): Double = {
      if (x == high) {
        Double.PositiveInfinity
      } else {
        val p = pnhyper(x)
        if (p < alpha)
          uniroot(unirootPnHyper(d => d, x, upper_tail = false, alpha), 0d, 1d).getOrElse(
            Double.NaN
          )
        else if (p > alpha)
          1.0 / uniroot(
            unirootPnHyper(d => 1 / d, x, upper_tail = false, alpha),
            epsilon,
            1d,
          ).getOrElse(Double.NaN)
        else
          1.0
      }
    }

    val pvalue: Double = (alternative: @unchecked) match {
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
    }

    assert(pvalue >= 0d && pvalue <= 1.000000000002)

    val oddsRatioEstimate = mle(numSuccessSample)

    val confInterval = alternative match {
      case "less" => (0d, ncpUpper(numSuccessSample, 1 - confidenceLevel))
      case "greater" => (ncpLower(numSuccessSample, 1 - confidenceLevel), Double.PositiveInfinity)
      case "two.sided" =>
        val alpha = (1 - confidenceLevel) / 2d
        (ncpLower(numSuccessSample, alpha), ncpUpper(numSuccessSample, alpha))
    }

    Array(pvalue, oddsRatioEstimate, confInterval._1, confInterval._2)
  }

  def dnorm(x: Double, mu: Double, sigma: Double, logP: Boolean): Double =
    Normal.density(x, mu, sigma, logP)

  def dnorm(x: Double): Double = dnorm(x, mu = 0, sigma = 1, logP = false)

  // Returns the p for which p = Prob(Z < x) with Z a standard normal RV
  def pnorm(x: Double, mu: Double, sigma: Double, lowerTail: Boolean, logP: Boolean): Double =
    Normal.cumulative(x, mu, sigma, lowerTail, logP)

  def pnorm(x: Double): Double = pnorm(x, mu = 0, sigma = 1, lowerTail = true, logP = false)

  // Returns the x for which p = Prob(Z < x) with Z a standard normal RV
  def qnorm(p: Double, mu: Double, sigma: Double, lowerTail: Boolean, logP: Boolean): Double =
    Normal.quantile(p, mu, sigma, lowerTail, logP)

  def qnorm(p: Double): Double = qnorm(p, mu = 0, sigma = 1, lowerTail = true, logP = false)

  /* Returns the p for which p = Prob(Z < x) with Z a RV having the T distribution with n degrees of
   * freedom */
  def pT(x: Double, n: Double, lower_tail: Boolean, log_p: Boolean): Double =
    net.sourceforge.jdistlib.T.cumulative(x, n, lower_tail, log_p)

  def pF(x: Double, df1: Double, df2: Double, lower_tail: Boolean, log_p: Boolean): Double =
    net.sourceforge.jdistlib.F.cumulative(x, df1, df2, lower_tail, log_p)

  def dchisq(x: Double, df: Double, logP: Boolean): Double = ChiSquare.density(x, df, logP)

  def dchisq(x: Double, df: Double): Double = dchisq(x, df, logP = false)

  def dnchisq(x: Double, df: Double, ncp: Double, logP: Boolean): Double =
    NonCentralChiSquare.density(x, df, ncp, logP)

  def dnchisq(x: Double, df: Double, ncp: Double): Double = dnchisq(x, df, ncp, logP = false)

  // Returns the p for which p = Prob(Z^2 > x) with Z^2 a chi-squared RV with df degrees of freedom
  def pchisqtail(x: Double, df: Double, lowerTail: Boolean, logP: Boolean): Double =
    ChiSquare.cumulative(x, df, lowerTail, logP)

  def pchisqtail(x: Double, df: Double): Double = pchisqtail(x, df, lowerTail = false, logP = false)

  def pnchisqtail(x: Double, df: Double, ncp: Double, lowerTail: Boolean, logP: Boolean): Double =
    NonCentralChiSquare.cumulative(x, df, ncp, lowerTail, logP)

  def pnchisqtail(x: Double, df: Double, ncp: Double): Double =
    pnchisqtail(x, df, ncp, lowerTail = false, logP = false)

  // Returns the x for which p = Prob(Z^2 > x) with Z^2 a chi-squared RV with df degrees of freedom
  def qchisqtail(p: Double, df: Double, lowerTail: Boolean, logP: Boolean): Double =
    ChiSquare.quantile(p, df, lowerTail, logP)

  def qchisqtail(p: Double, df: Double): Double = qchisqtail(p, df, lowerTail = false, logP = false)

  def qnchisqtail(p: Double, df: Double, ncp: Double, lowerTail: Boolean, logP: Boolean): Double =
    NonCentralChiSquare.quantile(p, df, ncp, lowerTail, logP)

  def qnchisqtail(p: Double, df: Double, ncp: Double): Double =
    qnchisqtail(p, df, ncp, lowerTail = false, logP = false)

  def pgenchisq(
    x: Double,
    w: IndexedSeq[Double],
    k: IndexedSeq[Int],
    lam: IndexedSeq[Double],
    sigma: Double,
    lim: Int,
    acc: Double,
  ): DaviesResultForPython =
    GeneralizedChiSquaredDistribution.cdfReturnExceptions(
      x,
      k.toArray,
      w.toArray,
      lam.toArray,
      sigma,
      lim,
      acc,
    )

  def dbeta(x: Double, a: Double, b: Double): Double = Beta.density(x, a, b, false)

  def dpois(x: Double, lambda: Double, logP: Boolean): Double = new Poisson(lambda).density(x, logP)

  def dpois(x: Double, lambda: Double): Double = dpois(x, lambda, logP = false)

  def ppois(x: Double, lambda: Double, lowerTail: Boolean, logP: Boolean): Double =
    new Poisson(lambda).cumulative(x, lowerTail, logP)

  def ppois(x: Double, lambda: Double): Double = ppois(x, lambda, lowerTail = true, logP = false)

  def qpois(x: Double, lambda: Double, lowerTail: Boolean, logP: Boolean): Int = {
    val result = new Poisson(lambda).quantile(x, lowerTail, logP)
    assert(result.isValidInt, s"qpois result is not a valid int. Found $result.")
    result.toInt
  }

  def qpois(x: Double, lambda: Double): Int = qpois(x, lambda, lowerTail = true, logP = false)

  def binomTest(nSuccess: Int, n: Int, p: Double, alternative: Int): Double = {
    val kind = alternative match {
      case 0 => TestKind.TWO_SIDED
      case 1 => TestKind.LOWER
      case 2 => TestKind.GREATER
      case _ =>
        fatal(s"""Invalid alternative "$alternative". Must be "two-sided", "less" or "greater".""")
    }

    DistributionTest.binomial_test(nSuccess, n, p, kind)(1)
  }

  def entropy[T](xs: Iterable[T]): Double = {
    val counts = xs.counter()

    var length = 0
    var acc = 0.0
    counts.valuesIterator.foreach { count =>
      length += count
      acc += count * math.log(count)
    }

    if (length != 0)
      (math.log(length) - (acc / length)) / math.log(2)
    else
      0.0
  }

  def uninitialized[T]: T = null.asInstanceOf[T]
}
