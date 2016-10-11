package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._

object SamplePCA {

  def pcSchema(asArray: Boolean, k: Int) =
    if (asArray)
      TArray(TDouble)
    else
      TStruct((1 until k).map(i => (s"PC$i", TDouble)): _*)

  def makeAnnotation(is: IndexedSeq[Double], asArray: Boolean): Annotation =
    if (asArray)
      is
    else
      Annotation.fromSeq(is)


  def apply(vds: VariantDataset, k: Int, computeLoadings: Boolean, computeEigenvalues: Boolean,
    asArray: Boolean): (Map[String, Annotation], Option[RDD[(Variant, Annotation)]], Option[Annotation]) = {

    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)
    val sc = vds.sparkContext

    val svd = mat.computeSVD(k, computeU = computeLoadings)

    val scores = svd.V.multiply(DenseMatrix.diag(svd.s))
    val sampleScores = vds.sampleIds.zipWithIndex.map { case (id, i) =>
      (id, makeAnnotation((0 until k).map(j => scores(i, j)), asArray))
    }

    val loadings = someIf(computeLoadings, {
      val variantsBc = sc.broadcast(variants)
      svd.U.rows.map(ir =>
        (variantsBc.value(ir.index.toInt), makeAnnotation(ir.vector.toArray, asArray)))
    })

    val eigenvalues = someIf(computeEigenvalues, makeAnnotation(svd.s.toArray.map(math.pow(_, 2)), asArray))

    (sampleScores.toMap, loadings, eigenvalues)
  }
}
