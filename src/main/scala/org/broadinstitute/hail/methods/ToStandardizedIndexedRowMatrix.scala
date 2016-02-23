package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.stat.Statistics
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.broadinstitute.hail.Utils._

object ToStandardizedIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.length
    val nSamples = vds.nLocalSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)

    val standardized = vds
      .rdd
      .map { case (v, va, gs) =>
        val (count, sum) = gs.foldLeft((0, 0)) { case ((c, s), g) =>
          g.nNonRefAlleles match {
            case Some(n) => (c + 1, s + n)
            case None => (c, s)
          }
        }
        // Will the denominator be computed every time the function is applied?
        def standardize(n: Int): Double =
          if (sum == 0 || sum == 2 * count)
            0.0
          else {
            val p = sum.toDouble / (2 * count)
            (n - 2 * p) / math.sqrt(2 * p * (1 - p) * nVariants)
          }
        IndexedRow(variantIdxBroadcast.value(v),
          Vectors.dense(gs.map(_.nNonRefAlleles.map(standardize).getOrElse(0.0)).toArray))
      }

    (variants, new IndexedRowMatrix(standardized.cache(), nVariants, nSamples))
  }

  //      vds
  //      .mapWithKeys((v, s, g) => (s, (variantIdxBroadcast.value(v), g.nNonRef)))
  //      .groupByKey()
  //      .map { case (s, indexGT) =>
  //      val a = Array.fill[Double](nVariants)(0.0)
  //      indexGT.foreach { case (j, gt) => a(j) = gt.toDouble }
  //      IndexedRow(s.toLong, Vectors.dense(a))
  //    }.cache()  // FIXME

  // Per variant normalization by standard deviation, estimated as a function of
  // the mean assuming a Binomial(n = 2, p) model, i.e. Hardy-Weinberg equilibrium;
  // see equation (3) in Patterson, Price, Reich (2006).
  // Ignore variants with MAF below a cutoff
  //
  //val cutoff = 0.0
  //
  //def sdNorm (m: Double): Double = {
  //val p = m / 2
  //if (math.min (p, 1 - p) <= cutoff)
  //0.0
  //else
  //1.0 / math.sqrt (2 * p * (1 - p) )
  //}
  //
  //val mean = Statistics.colStats (unnormalized.map (_.vector) ).mean
  //val normalized = unnormalized.map (ir => (ir - mean) :* mean.map (sdNorm) )
  //
  //(variants, new IndexedRowMatrix (normalized.cache (), nSamples, nVariants) )
  //}

}
