package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.{Vector => SVector, Matrix}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.variant.VariantDataset

class SamplePCA(k: Int, l: Boolean, e: Boolean) {
  def name = "SamplePCA"

  def apply(vds: VariantDataset): (Matrix, RDD[(Variant, SVector)], Array[Double])  = {

    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)

    val svd = mat.computeSVD(k, computeU = l)

    val scores =
      svd.V.multiply(DenseMatrix.diag(svd.s))

    val loadings =
      if (l)
        svd.U.rows.map(ir =>
          (variants(ir.index.toInt), ir.vector))
      else
        null

    val eigenvalues =
      if (e)
        svd.s.toArray.map(x => x * x)
      else
        null

    (scores, loadings, eigenvalues)
  }
}
