package is.hail.stats

import breeze.linalg._
import is.hail.annotations.RegionValue
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.MatrixTable

object RegressionUtils {  
  def setMeanImputedDoubles(data: Array[Double],
    offset: Int,
    completeColIdx: Array[Int],
    missingCompleteCols: ArrayBuilder[Int],
    rv: RegionValue,
    rvRowType: TStruct,
    entryArrayType: TArray,
    entryType: TStruct,
    entryArrayIdx: Int,
    fieldIdx: Int) : Unit = {

    missingCompleteCols.clear()
    val n = completeColIdx.length
    var sum = 0.0
    val region = rv.region
    val entryArrayOffset = rvRowType.loadField(rv, entryArrayIdx)

    var j = 0
    while (j < n) {
      val k = completeColIdx(j)
      if (entryArrayType.isElementDefined(region, entryArrayOffset, k)) {
        val entryOffset = entryArrayType.loadElement(region, entryArrayOffset, k)
        if (entryType.isFieldDefined(region, entryOffset, fieldIdx)) {
          val fieldOffset = entryType.loadField(region, entryOffset, fieldIdx)
          val e = region.loadDouble(fieldOffset)
          sum += e
          data(offset + j) = e
        } else
          missingCompleteCols += j
      } else
        missingCompleteCols += j
      j += 1
    }

    val nMissing = missingCompleteCols.size
    val mean = sum / (n - nMissing)
    var i = 0
    while (i < nMissing) {
      data(offset + missingCompleteCols(i)) = mean
      i += 1
    }
  }

  def parseFloat64Expr(expr: String, ec: EvalContext): () => java.lang.Double = {
    val (xt, xf0) = Parser.parseExpr(expr, ec)
    assert(xt.isOfType(TFloat64()))
    () => xf0().asInstanceOf[java.lang.Double]
  }

  // IndexedSeq indexed by samples, Array by annotations
  def getSampleAnnotations(vds: MatrixTable, annots: Array[String], ec: EvalContext): IndexedSeq[Array[Option[Double]]] = {
    val aQs = annots.map(parseFloat64Expr(_, ec))

    vds.colValues.value.map { sa =>
      ec.set(0, sa)
      aQs.map { aQ =>
        val a = aQ()
        if (a != null)
          Some(a: Double)
        else
          None
      }
    }
  }

  def getPhenoCovCompleteSamples(
    vsm: MatrixTable,
    yExpr: String,
    covExpr: Array[String]): (DenseVector[Double], DenseMatrix[Double], Array[Int]) = {
    
    val (y, covs, completeSamples) = getPhenosCovCompleteSamples(vsm, Array(yExpr), covExpr)
    
    (DenseVector(y.data), covs, completeSamples)
  }
  
  def getPhenosCovCompleteSamples(
    vsm: MatrixTable,
    yExpr: Array[String],
    covExpr: Array[String]): (DenseMatrix[Double], DenseMatrix[Double], Array[Int]) = {

    val nPhenos = yExpr.length
    val nCovs = covExpr.length + 1 // intercept

    if (nPhenos == 0)
      fatal("No phenotypes present.")

    val symTab = Map(
      "sa" -> (0, vsm.colType))

    val ec = EvalContext(symTab)

    val yIS = getSampleAnnotations(vsm, yExpr, ec)
    val covIS = getSampleAnnotations(vsm, covExpr, ec)

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until vsm.numCols)
        .zipped
        .filter((y, c, s) => y.forall(_.isDefined) && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.flatMap(_.map(_.get)).toArray
    val y = new DenseMatrix(rows = n, cols = nPhenos, data = yArray, offset = 0, majorStride = nPhenos, isTranspose = true)

    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(rows = n, cols = nCovs, data = covArray, offset = 0, majorStride = nCovs, isTranspose = true)

    if (n < vsm.numCols)
      warn(s"${ vsm.numCols - n } of ${ vsm.numCols } samples have a missing phenotype or covariate.")

    (y, cov, completeSamples.toArray)
  }
}
