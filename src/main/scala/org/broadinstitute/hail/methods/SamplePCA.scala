package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.{Matrix, DenseMatrix}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.variant.VariantDataset

class SamplePCA(k: Int, computeLoadings: Boolean, computeEigenvalues: Boolean) {
  def name = "SamplePCA"

  def apply(vds: VariantDataset): (Matrix, Option[RDD[(Variant, Array[Double])]], Option[Array[Double]])  = {

    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)
    val sc = vds.sparkContext
    val variantsB = sc.broadcast(variants)

    val svd = mat.computeSVD(k, computeU = computeLoadings)

    val scores =
      svd.V.multiply(DenseMatrix.diag(svd.s))

    val loadings =
      if (computeLoadings)
        Some(svd.U.rows.map(ir =>
          (variantsB.value(ir.index.toInt), ir.vector.toArray)))
      else
        None

    val eigenvalues =
      if (computeEigenvalues)
        Some(svd.s.toArray.map(x => x * x))
      else
        None

    (scores, loadings, eigenvalues)
  }
}
