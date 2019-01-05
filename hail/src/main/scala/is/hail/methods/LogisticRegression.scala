package is.hail.methods

import breeze.linalg._
import is.hail.annotations._
import is.hail.expr.ir.{TableLiteral, TableValue}
import is.hail.expr.types.virtual.{TArray, TFloat64, TStruct}
import is.hail.expr.types.TableType
import is.hail.stats._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._

object LogisticRegression {

  def apply(vsm: MatrixTable,
    test: String,
    yField: String,
    xField: String,
    _covFields: java.util.ArrayList[String],
    _passThrough: java.util.ArrayList[String]): Table = {
    val covFields = _covFields.asScala.toArray
    val passThrough = _passThrough.asScala.toArray
    val logRegTest = LogisticRegressionTest.tests(test)

    val (y, cov, completeColIdx) = RegressionUtils.getPhenoCovCompleteSamples(vsm, yField, covFields)


    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, y must be bool or numeric with all present values equal to 0 or 1")
    val sumY = sum(y)
    if (sumY == 0d || sumY == y.length)
      fatal(s"For logistic regression, y must be non-constant")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x) implies $d degrees of freedom.")

    info(s"logistic_regression_rows: running $test on $n samples for response variable y,\n"
      + s"    with input variable x, and ${ k } additional ${ plural(k, "covariate") }...")

    val nullModel = new LogisticRegressionModel(cov, y)
    var nullFit = nullModel.fit()

    if (!nullFit.converged)
      if (logRegTest == LogisticFirthTest)
        nullFit = GLMFit(nullModel.bInterceptOnly(),
          None, None, 0, nullFit.nIter, exploded = nullFit.exploded, converged = false)
      else
        fatal("Failed to fit logistic regression null model (standard MLE with covariates only): " + (
          if (nullFit.exploded)
            s"exploded at Newton iteration ${ nullFit.nIter }"
          else
            "Newton iteration failed to converge"))

    val sc = vsm.sparkContext
    val completeColIdxBc = sc.broadcast(completeColIdx)

    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val fullRowType = vsm.rvRowType.physicalType
    val entryArrayType = vsm.matrixType.entryArrayType.physicalType
    val entryType = vsm.entryType.physicalType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType.isOfType(TFloat64()))

    val entryArrayIdx = vsm.entriesIndex
    val fieldIdx = entryType.fieldIdx(xField)

    val passThroughType = TStruct(passThrough.map(f => f -> vsm.rowType.field(f).typ): _*)
    val tableType = TableType(vsm.rowKeyStruct ++ passThroughType ++ logRegTest.schema, vsm.rowKey, TStruct())
    val newRVDType = tableType.canonicalRVDType
    val copiedFieldIndices = (vsm.rowKey ++ passThrough).map(vsm.rvRowType.fieldIdx(_)).toArray

    val newRVD = vsm.rvd.mapPartitions(newRVDType) { it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      val missingCompleteCols = new ArrayBuilder[Int]()

      val X = XBc.value.copy
      it.map { rv =>
        RegressionUtils.setMeanImputedDoubles(X.data, n * k, completeColIdxBc.value, missingCompleteCols, 
          rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)

        val logregAnnot = logRegTestBc.value.test(X, yBc.value, nullFitBc.value, "logistic")

        rvb.set(rv.region)
        rvb.start(newRVDType.rowType)
        rvb.startStruct()
        rvb.addFields(fullRowType, rv, copiedFieldIndices)
        logregAnnot.addToRVB(rvb)
        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    new Table(vsm.hc, TableLiteral(TableValue(tableType, BroadcastRow.empty(sc), newRVD)))
  }

  def apply(vsm: MatrixTable,
            test: String,
            _yFields: java.util.ArrayList[String],
            xField: String,
            _covFields: java.util.ArrayList[String],
            _passThrough: java.util.ArrayList[String]): Table = {

    val yFields = _yFields.asScala.toArray
    val covFields = _covFields.asScala.toArray
    val passThrough = _passThrough.asScala.toArray
    val logRegTest = LogisticRegressionTest.tests(test)
    val multiPhenoSchema = TStruct(("logistic_regression", TArray(logRegTest.schema)))


    val (yVecs, cov, completeColIdx) = RegressionUtils.getPhenosCovCompleteSamples(vsm, yFields, covFields)

    (0 until yVecs.cols).foreach(col => {
      if (!yVecs(::, col).forall(yi => yi == 0d || yi == 1d))
        fatal(s"For logistic regression, y at index ${col} must be bool or numeric with all present values equal to 0 or 1")
      val sumY = sum(yVecs(::,col))
      if (sumY == 0d || sumY == yVecs(::,col).length)
        fatal(s"For logistic regression, y at index ${col} must be non-constant")
    })

    val n = yVecs.rows
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x) implies $d degrees of freedom.")

    info(s"logistic_regression_rows: running $test on $n samples for response variable y,\n"
      + s"    with input variable x, and ${ k } additional ${ plural(k, "covariate") }...")

    val nullFits = (0 until yVecs.cols).map(col => {
      val nullModel = new LogisticRegressionModel(cov, yVecs(::, col))
      var nullFit = nullModel.fit()

      if (!nullFit.converged)
        if (logRegTest == LogisticFirthTest)
          nullFit = GLMFit(nullModel.bInterceptOnly(),
            None, None, 0, nullFit.nIter, exploded = nullFit.exploded, converged = false)
        else
          fatal("Failed to fit logistic regression null model (standard MLE with covariates only): " + (
            if (nullFit.exploded)
              s"exploded at Newton iteration ${nullFit.nIter}"
            else
              "Newton iteration failed to converge"))
      nullFit
    })
    val sc = vsm.sparkContext
    val completeColIdxBc = sc.broadcast(completeColIdx)

    val yVecsBc = sc.broadcast(yVecs)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFits)
    val logRegTestBc = sc.broadcast(logRegTest)
    val resultSchemaBc = sc.broadcast(logRegTest.schema)

    val fullRowType = vsm.rvRowType.physicalType
    val entryArrayType = vsm.matrixType.entryArrayType.physicalType
    val entryType = vsm.entryType.physicalType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType.isOfType(TFloat64()))

    val entryArrayIdx = vsm.entriesIndex
    val fieldIdx = entryType.fieldIdx(xField)

    val passThroughType = TStruct(passThrough.map(f => f -> vsm.rowType.field(f).typ): _*)
    val tableType = TableType(vsm.rowKeyStruct ++ passThroughType ++ multiPhenoSchema, vsm.rowKey, TStruct())
    val newRVDType = tableType.canonicalRVDType
    val copiedFieldIndices = (vsm.rowKey ++ passThrough).map(vsm.rvRowType.fieldIdx(_)).toArray

    val newRVD = vsm.rvd.mapPartitions(newRVDType) { it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      val missingCompleteCols = new ArrayBuilder[Int]()
      val _nullFits = nullFitBc.value
      val _yVecs = yVecsBc.value
      val _resultSchema = resultSchemaBc.value
      val X = XBc.value.copy
      it.map { rv =>
        RegressionUtils.setMeanImputedDoubles(X.data, n * k, completeColIdxBc.value, missingCompleteCols,
          rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)
        val logregAnnotations = (0 until _yVecs.cols).map(col => {
          logRegTestBc.value.test(X, _yVecs(::,col), _nullFits(col), "logistic")
        })

        rvb.set(rv.region)
        rvb.start(newRVDType.rowType)
        rvb.startStruct()
        rvb.addFields(fullRowType, rv, copiedFieldIndices)
        rvb.startArray(_yVecs.cols)
        logregAnnotations.foreach(stats => {
          rvb.startStruct()
          stats.addToRVB(rvb)
          rvb.endStruct()

        })
        rvb.endArray()
        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    new Table(vsm.hc, TableLiteral(TableValue(tableType, BroadcastRow.empty(sc), newRVD)))
  }
}
