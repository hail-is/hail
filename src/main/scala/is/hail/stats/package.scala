package is.hail

import breeze.linalg.Matrix
import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.disttest.{DistributionTest, TestKind}
import net.sourceforge.jdistlib.{Beta, ChiSquare, Normal, Poisson}
import org.apache.commons.math3.distribution.HypergeometricDistribution
import org.apache.spark.sql.Row

package object stats {

  def uniroot(fn: Double => Double, min: Double, max: Double, tolerance: Double = 1.220703e-4): Option[Double] = {
    // based on C code in R source code called zeroin.c
    // https://github.com/wch/r-source/blob/e5b21d0397c607883ff25cca379687b86933d730/src/library/stats/src/zeroin.c

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

          if (p > 0d) //p was calculated with opposite sign
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

  val hweStruct = TStruct("r_expected_het_freq" -> TFloat64(), "p_hwe" -> TFloat64())
  
  def hweTest(nHomRef: Int, nHet: Int, nHomVar: Int): Array[Double] = {
    if (nHomRef < 0 || nHet < 0 || nHomVar < 0)
      fatal(s"hwe: all arguments must be non-negative, got $nHomRef, $nHet, $nHomVar")
  
    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    Array(LH.getNumericalMean / n, LH.exactMidP(nAB))
  }
  
  val chisqStruct = TStruct("p_value" -> TFloat64(), "odds_ratio" -> TFloat64())
  
  def chisqTest(a: Int, b: Int, c: Int, d: Int): Array[Double] = {
    if (a < 0 || b < 0 || c < 0 || d < 0)
      fatal(s"chisq: all arguments must be non-negative, got $a, $b, $c, $d")

    val ad = a * d
    val bc = (b * c).toDouble
    val det = ad - bc
    val chiSquare = (det * det * (a + b + c + d)) / ((a + b) * (c + d) * (b + d) * (a + c))
    
    Array(chiSquaredTail(chiSquare, 1), ad / bc)
  }
  
  def contingencyTableTest(a: Int, b: Int, c: Int, d: Int, minCellCount: Int): Array[Double] = {
    if (minCellCount < 0)
      fatal(s"ctt: 'min_cell_count' must be non-negative, found $minCellCount")
    
    if (a >= minCellCount && b >= minCellCount && c >= minCellCount && d >= minCellCount)
      chisqTest(a, b, c, d)
    else
      fisherExactTest(a, b, c, d)
  }
  
  val fetStruct = TStruct(
    "p_value" -> TFloat64(),
    "odds_ratio" -> TFloat64(),
    "ci_95_lower" -> TFloat64(),
    "ci_95_upper" -> TFloat64())

  def fisherExactTest(a: Int, b: Int, c: Int, d: Int): Array[Double] =
    fisherExactTest(a, b, c, d, 1.0, 0.95, "two.sided")
  
  def fisherExactTest(a: Int, b: Int, c: Int, d: Int,
    oddsRatio: Double = 1d,
    confidenceLevel: Double = 0.95,
    alternative: String = "two.sided"): Array[Double] = {
    
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

    if (!(popSize > 0 && sampleSize > 0 && sampleSize < popSize && numSuccessPopulation > 0 && numSuccessPopulation < popSize))
      return Array(Double.NaN, Double.NaN, Double.NaN, Double.NaN)

    val low = math.max(0, (a + b) - (b + d))
    val high = math.min(a + b, a + c)
    val support = (low to high).toArray

    val hgd = new HypergeometricDistribution(null, popSize, numSuccessPopulation, sampleSize)
    val epsilon = 2.220446e-16

    def dhyper(k: Int, logProb: Boolean = false): Double = {
      if (logProb) hgd.logProbability(k) else hgd.probability(k)
    }

    val logdc = support.map(dhyper(_, logProb = true))

    def dnhyper(ncp: Double): Array[Double] = {
      var d = logdc.zipWithIndex.map { case (hr, i) => hr + math.log(ncp) * i }
      d = d.map(dens => math.exp(dens - d.max))
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
          .filter { case (dbl, i) => if (upper_tail) support(i) >= q else support(i) <= q }
          .map { case (dbl, i) => dbl }
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

    def unirootPnHyper(fn: Double => Double, x: Int, upper_tail: Boolean, alpha: Double)(t: Double) =
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
          1.0 / uniroot(unirootPnHyper(d => 1 / d, x, upper_tail = true, alpha), epsilon, 1d).getOrElse(Double.NaN)
        else
          1.0
      }
    }

    def ncpUpper(x: Int, alpha: Double): Double = {
      if (x == high) {
        Double.PositiveInfinity
      }
      else {
        val p = pnhyper(x)
        if (p < alpha)
          uniroot(unirootPnHyper(d => d, x, upper_tail = false, alpha), 0d, 1d).getOrElse(Double.NaN)
        else if (p > alpha)
          1.0 / uniroot(unirootPnHyper(d => 1 / d, x, upper_tail = false, alpha), epsilon, 1d).getOrElse(Double.NaN)
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

  // Returns the p for which p = Prob(Z < x) with Z a standard normal RV
  def pnorm(x: Double): Double = Normal.cumulative(x, 0, 1)

  // Returns the x for which p = Prob(Z < x) with Z a standard normal RV
  def qnorm(p: Double): Double = Normal.quantile(p, 0, 1, true, false)

  // Returns the p for which p = Prob(Z^2 > x) with Z^2 a chi-squared RV with df degrees of freedom
  def chiSquaredTail(x: Double, df: Double): Double = ChiSquare.cumulative(x, df, false, false)

  // Returns the x for which p = Prob(Z^2 > x) with Z^2 a chi-squared RV with df degrees of freedom
  def inverseChiSquaredTail(p: Double, df: Double): Double = ChiSquare.quantile(p, df, false, false)

  def dbeta(x: Double, a: Double, b: Double): Double = Beta.density(x, a, b, false)

  def rpois(lambda: Double): Double = new Poisson(lambda).random()

  def rpois(n: Int, lambda: Double): Array[Double] = new Poisson(lambda).random(n)

  def dpois(x: Double, lambda: Double, logP: Boolean): Double = new Poisson(lambda).density(x, logP)

  def dpois(x: Double, lambda: Double): Double = dpois(x, lambda, logP = false)

  def ppois(x: Double, lambda: Double, lowerTail: Boolean, logP: Boolean): Double = new Poisson(lambda).cumulative(x, lowerTail, logP)

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
      case _ => fatal(s"""Invalid alternative "$alternative". Must be "two_sided", "less" or "greater".""")
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

  def vdsFromCallMatrix(hc: HailContext)(
    callMat: Matrix[BoxedCall],
    samplesIdsOpt: Option[Array[String]] = None,
    nPartitions: Int = hc.sc.defaultMinPartitions): MatrixTable = {

    require(samplesIdsOpt.forall(_.length == callMat.rows))
    require(samplesIdsOpt.forall(_.areDistinct()))

    val sampleIds = samplesIdsOpt.getOrElse((0 until callMat.rows).map(_.toString).toArray)

    val rdd = hc.sc.parallelize(
      (0 until callMat.cols).map { j =>
        (Annotation(Locus("1", j + 1), FastIndexedSeq("A", "C")),
          (0 until callMat.rows).map { i =>
            Genotype(callMat(i, j))
          }: Iterable[Annotation])
      },
      nPartitions)

    MatrixTable.fromLegacy(hc, MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowPartitionKey = Array("locus"), rowKey = Array("locus", "alleles"),
      rowType = TStruct("locus" -> TLocus(ReferenceGenome.defaultReference),
        "alleles" -> TArray(TString())),
      entryType = Genotype.htsGenotypeType),
      Annotation.empty, sampleIds.map(Annotation(_)), rdd)
  }

  // genotypes(i,j) is genotype of variant j in sample i; i and j are 0-based indices
  // sample i is "i" by default
  // variant j is ("1", j + 1, "A", C") since 0 is not a valid position.
  def vdsFromGpMatrix(hc: HailContext)(
    nAlleles: Int,
    gpMat: Matrix[Array[Double]],
    samplesIdsOpt: Option[Array[String]] = None,
    nPartitions: Int = hc.sc.defaultMinPartitions): MatrixTable = {

    require(samplesIdsOpt.forall(_.length == gpMat.rows))
    require(samplesIdsOpt.forall(_.areDistinct()))

    val sampleIds = samplesIdsOpt.getOrElse((0 until gpMat.rows).map(_.toString).toArray)

    val rdd = hc.sc.parallelize(
      (0 until gpMat.cols).map { j =>
        (Annotation(Locus("1", j + 1), FastIndexedSeq("A", "C")),
          (0 until gpMat.rows).map { i =>
            Row(gpMat(i, j): IndexedSeq[Annotation])
          }: Iterable[Annotation])
      },
      nPartitions)

    MatrixTable.fromLegacy(hc, MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowPartitionKey = Array("locus"), rowKey = Array("locus", "alleles"),
      rowType = TStruct("locus" -> TLocus(ReferenceGenome.defaultReference),
        "alleles" -> TArray(TString())),
      entryType = TStruct(
        "GP" -> TArray(TFloat64()))),
      Annotation.empty, sampleIds.map(Annotation(_)), rdd)
  }
}
