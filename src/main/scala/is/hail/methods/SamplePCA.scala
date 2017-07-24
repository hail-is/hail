package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.rdd.RDD

object SamplePCA {

  def pcSchema(asArray: Boolean, k: Int) =
    if (asArray)
      TArray(TDouble)
    else
      TStruct((1 to k).map(i => (s"PC$i", TDouble)): _*)

  def makeAnnotation(is: IndexedSeq[Double], asArray: Boolean): Annotation =
    if (asArray)
      is
    else
      Annotation.fromSeq(is)


  def apply(vds: VariantDataset, k: Int, computeLoadings: Boolean, computeEigenvalues: Boolean,
    asArray: Boolean): (Map[Annotation, Annotation], Option[RDD[(Variant, Annotation)]], Option[Annotation]) = {

    val (variants, mat) = ToHWENormalizedIndexedRowMatrix(vds)
    val sc = vds.sparkContext

    val svd = mat.computeSVD(k, computeU = computeLoadings)

    if (svd.s.size < k)
      fatal(
        s"""Found only ${svd.s.size} non-zero (or nearly zero) singular values, but user requested ${k}
           |principal components per sample.
         """.stripMargin)

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
