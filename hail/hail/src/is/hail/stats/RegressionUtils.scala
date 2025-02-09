package is.hail.stats

import is.hail.annotations.Region
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue}
import is.hail.types.physical.{PArray, PStruct}
import is.hail.types.virtual.TFloat64
import is.hail.utils._

import breeze.linalg._
import org.apache.spark.sql.Row

object RegressionUtils {
  def setMeanImputedDoubles(
    data: Array[Double],
    offset: Int,
    completeColIdx: Array[Int],
    missingCompleteCols: IntArrayBuilder,
    rv: Long,
    rvRowType: PStruct,
    entryArrayType: PArray,
    entryType: PStruct,
    entryArrayIdx: Int,
    fieldIdx: Int,
  ): Unit = {

    missingCompleteCols.clear()
    val n = completeColIdx.length
    var sum = 0.0
    val entryArrayOffset = rvRowType.loadField(rv, entryArrayIdx)

    var j = 0
    while (j < n) {
      val k = completeColIdx(j)
      if (entryArrayType.isElementDefined(entryArrayOffset, k)) {
        val entryOffset = entryArrayType.loadElement(entryArrayOffset, k)
        if (entryType.isFieldDefined(entryOffset, fieldIdx)) {
          val fieldOffset = entryType.loadField(entryOffset, fieldIdx)
          val e = Region.loadDouble(fieldOffset)
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

  // IndexedSeq indexed by column, Array by field
  def getColumnVariables(mv: MatrixValue, names: Array[String])
    : IndexedSeq[Array[Option[Double]]] = {
    assert(names.forall(name => mv.typ.colType.field(name).typ == TFloat64))
    val fieldIndices = names.map { name =>
      val field = mv.typ.colType.field(name)
      assert(field.typ == TFloat64)
      field.index
    }
    mv.colValues
      .javaValue
      .map { a =>
        val struct = a.asInstanceOf[Row]
        fieldIndices.map(i => Option(struct.get(i)).map(_.asInstanceOf[Double]))
      }
  }

  def getPhenoCovCompleteSamples(
    mv: MatrixValue,
    yField: String,
    covFields: Array[String],
  ): (DenseVector[Double], DenseMatrix[Double], Array[Int]) = {

    val (y, covs, completeSamples) = getPhenosCovCompleteSamples(mv, Array(yField), covFields)

    (DenseVector(y.data), covs, completeSamples)
  }

  def getPhenosCovCompleteSamples(
    mv: MatrixValue,
    yFields: Array[String],
    covFields: Array[String],
  ): (DenseMatrix[Double], DenseMatrix[Double], Array[Int]) = {

    val nPhenos = yFields.length
    val nCovs = covFields.length

    if (nPhenos == 0)
      fatal("No phenotypes present.")

    val yIS = getColumnVariables(mv, yFields)
    val covIS = getColumnVariables(mv, covFields)

    val nCols = mv.nCols
    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, 0 until nCols)
        .zipped
        .filter((y, c, s) => y.forall(_.isDefined) && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.flatMap(_.map(_.get)).toArray
    val y = new DenseMatrix(rows = n, cols = nPhenos, data = yArray, offset = 0,
      majorStride = nPhenos, isTranspose = true)

    val covArray = covForCompleteSamples.flatMap(_.map(_.get)).toArray
    val cov = new DenseMatrix(rows = n, cols = nCovs, data = covArray, offset = 0,
      majorStride = nCovs, isTranspose = true)

    if (n < nCols)
      warn(s"${nCols - n} of $nCols samples have a missing phenotype or covariate.")

    (y, cov, completeSamples.toArray)
  }
}
