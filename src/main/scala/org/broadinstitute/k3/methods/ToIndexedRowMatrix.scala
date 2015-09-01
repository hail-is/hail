package org.broadinstitute.k3.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.stat.Statistics
import org.broadinstitute.k3.variant.{Variant, VariantDataset}
import org.broadinstitute.k3.Utils._

object ToIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.size
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)

    val unnormalized = vds
              .mapWithKeys((v, s, g) => (s, (variantIdxBroadcast.value(v), g.nNonRef)))
              .groupByKey()
              .map{ case (s, i) =>
      val a = Array.fill[Double](nVariants)(0.0)
      i.foreach { case (i, gt) => a(i) = gt.toDouble }
      IndexedRow(s.toLong, Vectors.dense(a))
              }
    
    val summary = Statistics.colStats(unnormalized.map(_.vector))
    val normalized = unnormalized.map(ir =>
      ir - summary.mean)

    (variants, new IndexedRowMatrix(normalized, nSamples, nVariants))
  }
}
