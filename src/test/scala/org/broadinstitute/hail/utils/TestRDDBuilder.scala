package org.broadinstitute.hail.utils

import scala.util.Random
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.apache.spark.SparkContext
import scala.math

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

  val defaultMAF = 0.25

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
    val homVarThreshold = defaultMAF * defaultMAF
    val hetThreshold = homVarThreshold + 2 * defaultMAF * (1 - defaultMAF)
    pull match {
      case x if x < homVarThreshold => 2
      case x if x < hetThreshold => 1
      case _ => 0
    }
  }

  def plFromGQ(gq: Int, gt: Int): (Int, Int, Int) = {
    // supplies example PL values from a GQ and genotype
    gt match {
      case 0 => (0, gq, plMax)
      case 1 => (gq, 0, plDefault)
      case 2 => (plMax, gq, 0)
      case -1 => (0, 0, plDefault)
    }
  }

  def adFromDP(dp: Int, gt: Int): (Int, Int) = {
    // if a homozygous genotype is supplied, assigns all reads to that allele.  If het, 50/50.  If uncalled, (0, 0)
    gt match {
      case 0 => (dp, 0)
      case 2 => (0, dp)
      case 1 =>
        val refReads = dp / 2
        val altReads = dp - refReads
        (refReads, altReads)
      case -1 => (0, 0)
    }
  }

  def buildRDD(nSamples: Int, nVariants: Int, sc: SparkContext,
               gtArray: Option[Array[Array[Int]]] = None,
               gqArray: Option[Array[Array[Int]]] = None,
               dpArray: Option[Array[Array[Int]]] = None,
               sampleIds: Option[Array[String]] = None): VariantDataset = {
    /* Takes the arguments:
    nSamples(Int) -- number of samples (columns) to produce in VCF
    nVariants(Int) -- number of variants(rows) to produce in VCF
    sc(SparkContext) -- spark context in which to operate
    vsmtype(String) -- sparky, tuple, or managed
    gqArray(Array[Array[Int]]] -- Int array of dimension (nVariants x nSamples)
    dpArray(Array[Array[Int]]] -- Int array of dimension (nVariants x nSamples)
    Returns a test VDS of the given parameters */

    // create list of dummy sample IDs
    val sampleList = sampleIds match {
      case Some(arr) => arr
      case None => (0 until nSamples).map(i => "Sample" + i).toArray
    }

    // create array of (Variant, gq[Int], dp[Int])
    val variantArray = (0 until nVariants).map(i =>
      (Variant("1", i, defaultRef, defaultAlt),
        (gqArray.map(_ (i)),
          dpArray.map(_ (i)),
          gtArray.map(_ (i)))))
      .toArray

    val variantRDD = sc.parallelize(variantArray)

    val streamRDD = variantRDD.map {
      case (variant, (gqArr, dpArr, gtArr)) =>
        val b = new GenotypeStreamBuilder(variant)
        for (sample <- 0 until nSamples) {
          val gt = gtArr match {
            case Some(arr) => arr(sample)
            case None => pullGT()
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

          b += Genotype(gt, ad, dp, pl)
        }
        (variant, new AnnotationData(Map(), Map()), b.result(): Iterable[Genotype])
    }
    VariantSampleMatrix(VariantMetadata(sampleList), streamRDD)
  }
}