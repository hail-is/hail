package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.{Vector => SVector}
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.variant.VariantDataset

class SamplePCA(k: Int, l: Boolean, e: Boolean) {
  def name = "SamplePCA"

  def apply(vds: VariantDataset): (DenseMatrix, Option[Map[Int, (Variant, SVector)]], Option[Array[Double]])  = {

    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)

    val svd = mat.computeSVD(k, computeU = l)

    val scores =
      svd.V.multiply(DenseMatrix.diag(svd.s))

    val loadings =
      if (l)
        Some(svd.U.rows.map(ir => (ir.index.toInt, (variants(ir.index.toInt), ir.vector))).collectAsMap().toMap)
      else
        None

    val eigenvalues =
      if (e)
        Some(svd.s.toArray.map(x => x * x))
      else
        None

    (scores, loadings, eigenvalues)
  }
}
