package is.hail.methods

import is.hail.annotations.{Annotation, MemoryBuffer, RegionValue, RegionValueBuilder}
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.{Variant, VariantSampleMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector}
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix

trait PCA {
  def pcSchema(k: Int, asArray: Boolean = false): Type =
    if (asArray)
      TArray(TFloat64())
    else
      TStruct((1 to k).map(i => (s"PC$i", TFloat64())): _*)

  //returns (sample scores, variant loadings, eigenvalues)
  def apply(vsm: VariantSampleMatrix[_, _, _], k: Int, computeLoadings: Boolean, computeEigenvalues: Boolean, asArray: Boolean = false): (DenseMatrix, Option[KeyTable], Option[IndexedSeq[Double]]) = {
    val sc = vsm.sparkContext
    val (maybeVariants, mat) = doubleMatrixFromVSM(vsm, computeLoadings)
    val svd = mat.computeSVD(k, computeLoadings)
    if (svd.s.size < k)
      fatal(
        s"""Found only ${ svd.s.size } non-zero (or nearly zero) eigenvalues, but user requested ${ k }
           |principal components.""".stripMargin)

    val optionLoadings = someIf(computeLoadings, {
      val rowType = TStruct("v" -> vsm.vSignature, "pcaLoadings" -> pcSchema(k, asArray))
      val rowTypeBc = vsm.sparkContext.broadcast(rowType)
      val variantsBc = vsm.sparkContext.broadcast(maybeVariants.get)
      val rdd = svd.U.rows.mapPartitions[RegionValue] { it =>
        val region = MemoryBuffer()
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)
        it.map { ir =>
          rvb.start(rowTypeBc.value)
          rvb.startStruct()
          rvb.addAnnotation(rowTypeBc.value.fieldType(0), variantsBc.value(ir.index.toInt))
          if (asArray) rvb.startArray(k) else rvb.startStruct()
          var i = 0
          while (i < k) {
            rvb.addDouble(ir.vector(i))
            i += 1
          }
          if (asArray) rvb.endArray() else rvb.endStruct()
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }
      new KeyTable(vsm.hc, rdd, rowType, Array("v"))
    })

    (svd.V.multiply(DenseMatrix.diag(svd.s)), optionLoadings, someIf(computeEigenvalues, svd.s.toArray.map(math.pow(_, 2))))
  }

  def doubleMatrixFromVSM(vsm: VariantSampleMatrix[_, _, _], getVariants: Boolean): (Option[Array[Variant]], IndexedRowMatrix)
}

object SamplePCA extends PCA {
  override def doubleMatrixFromVSM(vsm: VariantSampleMatrix[_, _, _], getVariants: Boolean): (Option[Array[Variant]], IndexedRowMatrix) = {
    val (variants, mat) = ToHWENormalizedIndexedRowMatrix(vsm.makeVariantConcrete())
    (if (getVariants) Option(variants) else None, mat)
  }
}
