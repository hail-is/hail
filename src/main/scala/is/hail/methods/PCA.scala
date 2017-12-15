package is.hail.methods

import breeze.linalg.{*, DenseMatrix, DenseVector}
import is.hail.annotations._
import is.hail.expr._
import is.hail.keytable.Table
import is.hail.utils._
import is.hail.variant.MatrixTable

object PCA {
  def pcSchema(k: Int, asArray: Boolean = false): Type =
    if (asArray)
      TArray(TFloat64())
    else
      TStruct((1 to k).map(i => (s"PC$i", TFloat64())): _*)

  //returns (sample scores, variant loadings, eigenvalues)
  def apply(vsm: MatrixTable, expr: String, k: Int, computeLoadings: Boolean, asArray: Boolean = false): (IndexedSeq[Double], DenseMatrix[Double], Option[Table]) = {
    val sc = vsm.sparkContext
    val (irm, optionVariants) = vsm.toIndexedRowMatrix(expr, computeLoadings)
    val svd = irm.computeSVD(k, computeLoadings)
    if (svd.s.size < k)
      fatal(
        s"""Found only ${ svd.s.size } non-zero (or nearly zero) eigenvalues, but user requested ${ k }
           |principal components.""".stripMargin)

    val optionLoadings = someIf(computeLoadings, {
      val rowType = TStruct("v" -> vsm.vSignature, "pcaLoadings" -> pcSchema(k, asArray))
      val rowTypeBc = vsm.sparkContext.broadcast(rowType)
      val variantsBc = vsm.sparkContext.broadcast(optionVariants.get)
      val rdd = svd.U.rows.mapPartitions[RegionValue] { it =>
        val region = Region()
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
      new Table(vsm.hc, rdd, rowType, Array("v"))
    })

    val data =
      if (!svd.V.isTransposed)
        svd.V.asInstanceOf[org.apache.spark.mllib.linalg.DenseMatrix].values
      else
        svd.V.toArray
    
    val V = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, data)
    val S = DenseVector(svd.s.toArray)

    val eigenvalues = svd.s.toArray.map(math.pow(_, 2))
    val scaledEigenvectors = V(*, ::) :* S
    
    (eigenvalues, scaledEigenvectors, optionLoadings)
  }
}
