package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.{Vector => SVector}
import org.broadinstitute.hail.variant.VariantDataset

class SamplePCA(k: Int) {
  def name = "SamplePCA"
  def apply(vds: VariantDataset): Map[Int, SVector] = {
    val (variants, mat) = ToNormalizedIndexedRowMatrix(vds)
    val svd = mat.computeSVD(k)
    mat
      .multiply(svd.V)
      .rows
      .map(ir => (ir.index.toInt, ir.vector))
      .collectAsMap()
      .toMap
  }
}
