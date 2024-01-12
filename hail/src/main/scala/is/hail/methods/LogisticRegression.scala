package is.hail.methods

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.rvd.RVDType
import is.hail.stats._
import is.hail.types.{MatrixType, TableType}
import is.hail.types.virtual.{TArray, TFloat64, TStruct}
import is.hail.utils._

import breeze.linalg._

case class LogisticRegression(
  test: String,
  yFields: Seq[String],
  xField: String,
  covFields: Seq[String],
  passThrough: Seq[String],
  maxIterations: Int,
  tolerance: Double,
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val logRegTest = LogisticRegressionTest.tests(test)
    val multiPhenoSchema = TStruct(("logistic_regression", TArray(logRegTest.schema)))
    val passThroughType = TStruct(passThrough.map(f => f -> childType.rowType.field(f).typ): _*)
    TableType(
      childType.rowKeyStruct ++ passThroughType ++ multiPhenoSchema,
      childType.rowKey,
      TStruct.empty,
    )
  }

  def preservesPartitionCounts: Boolean = true

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val logRegTest = LogisticRegressionTest.tests(test)
    val tableType = typ(mv.typ)
    val newRVDType = tableType.canonicalRVDType

    val multiPhenoSchema = TStruct(("logistic_regression", TArray(logRegTest.schema)))

    val (yVecs, cov, completeColIdx) =
      RegressionUtils.getPhenosCovCompleteSamples(mv, yFields.toArray, covFields.toArray)

    (0 until yVecs.cols).foreach { col =>
      if (!yVecs(::, col).forall(yi => yi == 0d || yi == 1d))
        fatal(
          s"For logistic regression, y at index $col must be bool or numeric with all present values equal to 0 or 1"
        )
      val sumY = sum(yVecs(::, col))
      if (sumY == 0d || sumY == yVecs(::, col).length)
        fatal(s"For logistic regression, y at index $col must be non-constant")
    }

    val n = yVecs.rows
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(
        s"$n samples and ${k + 1} ${plural(k, "covariate")} (including x) implies $d degrees of freedom."
      )

    info(s"logistic_regression_rows: running $test on $n samples for response variable y,\n"
      + s"    with input variable x, and $k additional ${plural(k, "covariate")}...")

    val nullFits = (0 until yVecs.cols).map { col =>
      val nullModel = new LogisticRegressionModel(cov, yVecs(::, col))
      var nullFit = nullModel.fit(maxIter = maxIterations, tol = tolerance)

      if (!nullFit.converged)
        if (logRegTest == LogisticFirthTest)
          nullFit = GLMFit(
            nullModel.bInterceptOnly(),
            None,
            None,
            0,
            nullFit.nIter,
            exploded = nullFit.exploded,
            converged = false,
          )
        else
          fatal(
            "Failed to fit logistic regression null model (standard MLE with covariates only): " + (
              if (nullFit.exploded)
                s"exploded at Newton iteration ${nullFit.nIter}"
              else
                "Newton iteration failed to converge"
            )
          )
      nullFit
    }

    val backend = HailContext.backend
    val completeColIdxBc = backend.broadcast(completeColIdx)

    val yVecsBc = backend.broadcast(yVecs)
    val XBc =
      backend.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = backend.broadcast(nullFits)
    val logRegTestBc = backend.broadcast(logRegTest)

    val fullRowType = mv.rvRowPType
    val entryArrayType = mv.entryArrayPType
    val entryType = mv.entryPType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType == TFloat64)

    val entryArrayIdx = mv.entriesIdx
    val fieldIdx = entryType.fieldIdx(xField)

    val copiedFieldIndices = (mv.typ.rowKey ++ passThrough).map(mv.rvRowType.fieldIdx(_)).toArray

    val newRVD = mv.rvd.mapPartitions(newRVDType) { (ctx, it) =>
      val rvb = ctx.rvb

      val missingCompleteCols = new IntArrayBuilder()
      val _nullFits = nullFitBc.value
      val _yVecs = yVecsBc.value
      val X = XBc.value.copy
      it.map { ptr =>
        RegressionUtils.setMeanImputedDoubles(
          X.data,
          n * k,
          completeColIdxBc.value,
          missingCompleteCols,
          ptr,
          fullRowType,
          entryArrayType,
          entryType,
          entryArrayIdx,
          fieldIdx,
        )
        val logregAnnotations = (0 until _yVecs.cols).map { col =>
          logRegTestBc.value.test(
            X,
            _yVecs(::, col),
            _nullFits(col),
            "logistic",
            maxIter = maxIterations,
            tol = tolerance,
          )
        }

        rvb.start(newRVDType.rowType)
        rvb.startStruct()
        rvb.addFields(fullRowType, ctx.r, ptr, copiedFieldIndices)
        rvb.startArray(_yVecs.cols)
        logregAnnotations.foreach { stats =>
          rvb.startStruct()
          stats.addToRVB(rvb)
          rvb.endStruct()

        }
        rvb.endArray()
        rvb.endStruct()
        rvb.end()
      }
    }

    TableValue(ctx, tableType, BroadcastRow.empty(ctx), newRVD)
  }
}
