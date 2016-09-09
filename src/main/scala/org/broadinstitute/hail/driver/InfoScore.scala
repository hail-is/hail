package org.broadinstitute.hail.driver

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.variant._

class InfoScoreCombiner extends Serializable {
  var result = 0d
  var expectedAlleleCount = 0d
  var totalDosage = 0d
  var nIncluded = 0

  def expectedVariance(dosage: Array[Double]): Double = {
    val mean = dosage(1) + 2 * dosage(2)
    dosage(1) + 4 * dosage(2) - mean * mean
  }

  def merge(g: Genotype): InfoScoreCombiner = {
      g.dosage.foreach { dx =>
        result += expectedVariance(dx)
        expectedAlleleCount += dx(1) + 2 * dx(2)
        totalDosage += dx.sum
        nIncluded += 1
      }

    this
  }

  def merge(that: InfoScoreCombiner): InfoScoreCombiner = {
    result += that.result
    expectedAlleleCount += that.expectedAlleleCount
    totalDosage += that.totalDosage
    nIncluded += that.nIncluded

    this
  }

  def thetaMLE = divOption(expectedAlleleCount, totalDosage)

  def imputeInfoScore(theta: Double) =
    if (theta == 1.0 || theta == 0.0)
      Some(1d)
    else if (nIncluded == 0)
      None
    else
      Some(1d - ((result / nIncluded) / (2 * theta * (1 - theta))))

  def asAnnotation: Annotation = {
    val imputeScore = thetaMLE.flatMap{theta => imputeInfoScore(theta)}

    Annotation(imputeScore.orNull, nIncluded)
  }
}

object InfoScore extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "infoscore"

  def description = "Compute info scores per variant from dosage data."

  def results(vds: VariantDataset): RDD[(Variant, InfoScoreCombiner)] =
    vds.aggregateByVariant(new InfoScoreCombiner)((comb, g) => comb.merge(g), (comb1, comb2) => comb1.merge(comb2))

  def supportsMultiallelic = false

  def requiresVDS = true

  val signature = TStruct(
    "impute" -> TDouble,
    "nIncluded" -> TInt
  )

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!vds.isDosage)
      fatal("Genotypes must be dosages in order to compute an info score.")

    // don't recompute info scores in case there are multiple downstream actions
    val r = results(vds).persist(StorageLevel.MEMORY_AND_DISK)

    val (newVAS, insertInfoScore) = vds.vaSignature.insert(signature, "infoscore")

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.zipPartitions(r, preservesPartitioning = true) { case (it, jt) =>
          it.zip(jt).map { case ((v, (va, gs)), (v2, comb)) =>
            assert(v == v2)
            (v, (insertInfoScore(va, Some(comb.asAnnotation)), gs))
          }
        }.asOrderedRDD[Locus], vaSignature = newVAS)
    )
  }
}

