package is.hail.utils

import is.hail.HailContext
import is.hail.annotations._
import is.hail.variant._

import scala.util.Random

object TestRDDBuilder {

  // set default distributions
  val gqMin = Some(10)
  val gqMax = Some(99)
  val gqMean = 75
  val gqStDev = 20

  val dpMin = Some(10)
  val dpMax = None
  val dpMean = 30
  val dpStDev = 10

  val plDefault = 120
  val plMax = 1800

  val defaultAF = 0.25

  val defaultRef = "A"
  val defaultAlt = "T"

  def normInt[T](mean: T, stDev: T, floor: Option[Int] = None, ceil: Option[Int] = None)
                (implicit ev: T => Double): Int = {
    // pulls a random normal variable defined by a mean, stDev, floor, and ceiling, and returns an Int
    val pull = (Random.nextGaussian() * stDev + mean).toInt
    var result = floor match {
      case Some(x) => math.max(pull, x)
      case None => pull
    }
    result = ceil match {
      case Some(x) => math.min(result, x)
      case None => result
    }
    result
  }

  def pullGT(): Int = {
    // pulls a genotype call according to HWE and defaultMAF
    val pull = Random.nextFloat()
    val homVarThreshold = defaultAF * defaultAF
    val hetThreshold = homVarThreshold + 2 * defaultAF * (1 - defaultAF)
    pull match {
      case x if x < homVarThreshold => 2
      case x if x < hetThreshold => 1
      case _ => 0
    }
  }

  def plFromGQ(gq: Int, gt: Option[Int]): Option[Array[Int]] = {
    // supplies example PL values from a GQ and genotype
    gt match {
      case Some(0) => Some(Array(0, gq, plMax))
      case Some(1) => Some(Array(gq, 0, plDefault))
      case Some(2) => Some(Array(plMax, gq, 0))
      case _ => None
    }
  }

  def adFromDP(dp: Int, gt: Option[Int]): Option[Array[Int]] = {
    // if a homozygous genotype is supplied, assigns all reads to that allele.  If het, 50/50.  If uncalled, (0, 0)
    gt match {
      case Some(0) => Some(Array(dp, 0))
      case Some(2) => Some(Array(0, dp))
      case Some(1) =>
        val refReads = dp / 2
        val altReads = dp - refReads
        Some(Array(refReads, altReads))
      case _ => None
    }
  }

  def buildRDD(nSamples: Int, nVariants: Int, hc: HailContext,
                      gqArray: Option[Array[Array[Int]]] = None,
                      dpArray: Option[Array[Array[Int]]] = None,
                      gtArray: Option[Array[Array[Int]]] = None): VariantSampleMatrix = {
    /* Takes the arguments:
    nSamples(Int) -- number of samples (columns) to produce in VCF
    nVariants(Int) -- number of variants(rows) to produce in VCF
    sc(SparkContext) -- spark context in which to operate
    gqArray(Array[Array[Int]]] -- Int array of dimension (nVariants x nSamples)
    dpArray(Array[Array[Int]]] -- Int array of dimension (nVariants x nSamples)
    Returns a test VDS of the given parameters */

    // create list of dummy sample IDs

    val sampleList = (0 until nSamples).map(i => "Sample" + i).toArray

    // create array of (Variant, gq[Int], dp[Int])
    val variantArray = (0 until nVariants).map(i =>
      (Variant("1", i + 1, defaultRef, defaultAlt),
        (gtArray.map(_(i)),
          gqArray.map(_(i)),
          dpArray.map(_(i)))))
      .map {
        case (variant, (gtArr, gqArr, dpArr)) =>
          val b = new ArrayBuilder[Genotype]()
          for (sample <- 0 until nSamples) {
            val gt = gtArr match {
              case Some(arr) => Some(arr(sample))
              case None => Some(pullGT())
            }
            val gq = gqArr match {
              case Some(arr) => arr(sample)
              case None => normInt(gqMean, gqStDev, floor = gqMin, ceil = gqMax)
            }
            val dp = dpArr match {
              case Some(arr) => arr(sample)
              case None => normInt(dpMean, dpStDev, floor = dpMin, ceil = dpMax)
            }
            val ad = adFromDP(dp, gt)
            val pl = plFromGQ(gq, gt)

            // FIXME gq
            b += Genotype(gt, ad, Some(dp), Some(gq), pl)
          }
          (variant, (Annotation.empty, b.result(): Iterable[Genotype]))
      }.toArray
    
    val variantRDD = hc.sc.parallelize(variantArray)
    VariantSampleMatrix.fromLegacy(hc, VSMFileMetadata(sampleList), variantRDD)
  }
}