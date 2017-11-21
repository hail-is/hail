package is.hail.methods

import is.hail.annotations._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats.{RegressionUtils, ToHWENormalizedIndexedRowMatrix}
import is.hail.utils._
import is.hail.variant.{HTSGenotypeView, Variant, VariantSampleMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.Row

trait PCA {
  def pcSchema(k: Int, asArray: Boolean = false): Type =
    if (asArray)
      TArray(TFloat64())
    else
      TStruct((1 to k).map(i => (s"PC$i", TFloat64())): _*)

  //returns (sample scores, variant loadings, eigenvalues)
  def apply(vsm: VariantSampleMatrix, k: Int, computeLoadings: Boolean, computeEigenvalues: Boolean, asArray: Boolean = false): (DenseMatrix, Option[KeyTable], Option[IndexedSeq[Double]]) = {
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

  def doubleMatrixFromVSM(vsm: VariantSampleMatrix, getVariants: Boolean): (Option[Array[Any]], IndexedRowMatrix)
}

class ExprPCA(val expr: String) extends PCA {
  def doubleMatrixFromVSM(vsm: VariantSampleMatrix, getVariants: Boolean): (Option[Array[Any]], IndexedRowMatrix) = {

    val partitionSizes = vsm.rdd2.mapPartitions(it => Iterator.single(it.size)).collect().scanLeft(0)(_ + _)
    assert(partitionSizes.length == vsm.rdd2.getNumPartitions + 1)
    val pSizeBc = vsm.sparkContext.broadcast(partitionSizes)

    val rowType = vsm.rowType
    val samplesBc = vsm.sampleIdsBc
    val sampleAnnotationsBc = vsm.sampleAnnotationsBc

    val ec = EvalContext(Map(
      "global" -> (0, vsm.globalSignature),
      "v" -> (1, vsm.vSignature),
      "va" -> (2, vsm.vaSignature),
      "s" -> (3, vsm.sSignature),
      "sa" -> (4, vsm.saSignature),
      "g" -> (5, vsm.genotypeSignature)))
    val f = RegressionUtils.parseExprAsDouble(expr, ec)
    ec.set(0, vsm.globalAnnotation)

    val mat = vsm.rdd2.mapPartitionsWithIndex {case (i, it) =>
      val pStartIdx = pSizeBc.value(i)
      var j = 0
      val ur = new UnsafeRow(rowType)
      it.map {rv =>
        ur.set(rv)
        ec.set(1, ur.get(1))
        ec.set(2, ur.get(2))
        val gs = ur.getAs[IndexedSeq[Any]](3)
        val row = IndexedRow(pStartIdx + j, Vectors.dense(gs.indices.map {k =>
          ec.set(3, samplesBc.value(k))
          ec.set(4, sampleAnnotationsBc.value(k))
          ec.set(5, gs(k))
          f().toDouble
        }.toArray))
        j += 1
        row
      }
    }

    (someIf(getVariants,
      vsm.rdd2.mapPartitions{it =>
        val ur = new UnsafeRow(rowType)
        it.map{ rv =>
          ur.set(rv)
          ur.get(1)
        }
      }.collect()
    ),
    new IndexedRowMatrix(mat, partitionSizes(partitionSizes.length - 1), vsm.sampleIds.length))
  }
}

object SamplePCA extends PCA {
  def doubleMatrixFromVSM(vsm: VariantSampleMatrix, getVariants: Boolean): (Option[Array[Any]], IndexedRowMatrix) = {
    val (variants, mat) = ToHWENormalizedIndexedRowMatrix(vsm)
    (if (getVariants) Option(variants.asInstanceOf[Array[Any]]) else None, mat)
  }
}
