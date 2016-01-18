package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.stat.Statistics
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.broadinstitute.hail.Utils._

object ToNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.size
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)

    val unnormalized = vds
      .mapWithKeys((v, s, g) => (s, (variantIdxBroadcast.value(v), g.nNonRef)))
      .groupByKey()
      .map { case (s, indexGT) =>
      val a = Array.fill[Double](nVariants)(0.0)
      indexGT.foreach { case (j, gt) => a(j) = gt.toDouble }
      IndexedRow(s.toLong, Vectors.dense(a))
    }.cache()  // FIXME

    def std(m: Double): Double = math.sqrt(2 * (m / 2) * (1 - m / 2))
    val summary = Statistics.colStats(unnormalized.map(_.vector))
    val normalized = unnormalized.map(ir => (ir - summary.mean) :/ summary.mean.map(std))
    
    (variants, new IndexedRowMatrix(normalized.cache(), nSamples, nVariants))
  }
}
