package is.hail.stats

import is.hail.utils._

import org.apache.commons.math3.distribution.AbstractIntegerDistribution
import org.apache.commons.math3.random.RandomGenerator

// Efficient implementation of the Levene-Haldane distribution, used in exact tests of Hardy-Weinberg equilibrium
// See docs/LeveneHaldane.pdf

class LeveneHaldane(
  val n: Int,
  val nA: Int,
  val mode: Int,
  pRU: Stream[Double],
  pLU: Stream[Double],
  pN: Double,
  rng: RandomGenerator,
) extends AbstractIntegerDistribution(rng) {

  // The probability mass function P(nAB), computing no more than necessary
  def probability(nAB: Int): Double =
    if (nAB < 0 || nAB > nA || nAB % 2 != nA % 2)
      0.0
    else if (nAB >= mode)
      pRU((nAB - mode) / 2) / pN
    else
      pLU((mode - nAB) / 2) / pN

  /* P(n0 < nAB <= n1), implemented to minimize round-off error but take advantage of the
   * sub-geometric tails */
  override def cumulativeProbability(n0: Int, n1: Int): Double =
    if (n0 >= n1 || n0 >= nA || n1 < nA % 2)
      0.0
    else if (n0 >= mode) {
      val cutoff = pRU((n0 - mode) / 2 + 1) * 1.0e-16
      pRU.slice((n0 - mode) / 2 + 1, (n1 - mode) / 2 + 1).takeWhile(_ > cutoff).sum / pN
    } else if (n1 < mode) {
      val cutoff = pLU((mode - n1 + 1) / 2) * 1.0e-16
      pLU.slice((mode - n1 + 1) / 2, (mode - n0 + 1) / 2).takeWhile(_ > cutoff).sum / pN
    } else {
      val cutoff = 1.0e-16
      (pLU.slice(1, (mode - n0 + 1) / 2).takeWhile(_ > cutoff).sum
        + pRU.slice(0, (n1 - mode) / 2 + 1).takeWhile(_ > cutoff).sum) / pN
    }

  // P(nAB <= n1)
  def cumulativeProbability(n1: Int): Double =
    cumulativeProbability(-1, n1)

  // P(nAB > n0)
  def survivalFunction(n0: Int): Double =
    cumulativeProbability(n0, nA)

  // Exact tests with the mid-p-value correction:
  /* half the probability of the observed outcome nAB plus the probabilities of those "more
   * extreme", i.e. */
  // greater, lesser, or of smaller probability, respectively
  // (in the latter case weighting outcomes of equal probability by 1/2)
  def rightMidP(nAB: Int) =
    survivalFunction(nAB) + 0.5 * probability(nAB)

  def leftMidP(nAB: Int) =
    cumulativeProbability(nAB) - 0.5 * probability(nAB)

  def exactMidP(nAB: Int) = {
    val p0U = probability(nAB) * pN
    if (D_==(p0U, 0.0))
      0.0
    else {
      val cutoff = p0U * 0.5e-16
      def mpU(s: Stream[Double]): Double = {
        val (sEq, sLess) =
          s.dropWhile(D_>(_, p0U, tolerance = 1.0e-12)).span(D_==(_, p0U, tolerance = 1.0e-12))
        0.5 * sEq.sum + sLess.takeWhile(_ > cutoff).sum
      }
      (mpU(pLU.tail) + mpU(pRU)) / pN
    }
  }

  def nB: Int = 2 * n - nA
  def getNumericalMean: Double = 1.0 * nA * nB / (2 * n - 1)

  def getNumericalVariance: Double =
    1.0 * nA * nB / (2 * n - 1) * (1 + (nA - 1.0) * (nB - 1) / (2 * n - 3) - 1.0 * nA * nB / (2 * n - 1))

  def isSupportConnected: Boolean = true // interpreted as restricted to the even or odd integers,
  def getSupportUpperBound: Int = nA
  def getSupportLowerBound: Int = nA % 2
}

object LeveneHaldane {

  def apply(n: Int, nA: Int, rng: RandomGenerator): LeveneHaldane = {
    require(nA >= 0 && nA <= n)

    val nB = 2 * n - nA
    val parity = nA % 2
    val mode = ((x: Double) => 2 * math.round((x - parity) / 2) + parity)(
      (nA + 1.0) * (nB + 1) / (2 * n + 3)
    ).toInt

    def pRUfrom(nAB: Int, p: Double): Stream[Double] =
      p #:: pRUfrom(nAB + 2, p * (nA - nAB) * (nB - nAB) / ((nAB + 2.0) * (nAB + 1)))
    def pLUfrom(nAB: Int, p: Double): Stream[Double] =
      p #:: pLUfrom(nAB - 2, p * nAB * (nAB - 1) / ((nA - nAB + 2.0) * (nB - nAB + 2)))

    // Unnormalized probability mass function going right, respectively left, from the mode by 2s
    val pRU = pRUfrom(mode, 1.0)
    val pLU = pLUfrom(mode, 1.0)

    // Normalization constant
    val pN = pRU.takeWhile(_ > 1.0e-16).sum + pLU.takeWhile(_ > 1.0e-16).sum - 1.0

    new LeveneHaldane(n, nA, mode, pRU, pLU, pN, rng)
  }

  // If we ever want to sample, it seems standard to replace `null` with a
  // default Random Generator `new Well19937c()`
  def apply(n: Int, nA: Int): LeveneHaldane = LeveneHaldane(n, nA, null)
}
