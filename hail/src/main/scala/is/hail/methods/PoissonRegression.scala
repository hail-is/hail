package is.hail.methods

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IntArrayBuilder, MatrixValue, TableValue}
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.stats._
import is.hail.types.{MatrixType, TableType}
import is.hail.types.virtual.{TFloat64, TStruct}
import is.hail.utils._

import breeze.linalg._

case class PoissonRegression(
  test: String,
  yField: String,
  xField: String,
  covFields: Seq[String],
  passThrough: Seq[String],
  maxIterations: Int,
  tolerance: Double,
) extends MatrixToTableFunction {

  override def typ(childType: MatrixType): TableType = {
    val poisRegTest = PoissonRegressionTest.tests(test)
    val passThroughType = TStruct(passThrough.map(f => f -> childType.rowType.field(f).typ): _*)
    TableType(
      childType.rowKeyStruct ++ passThroughType ++ poisRegTest.schema,
      childType.rowKey,
      TStruct.empty,
    )
  }

  def preservesPartitionCounts: Boolean = true

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val poisRegTest = PoissonRegressionTest.tests(test)
    val tableType = typ(mv.typ)
    val newRVDType = tableType.canonicalRVDType

    val (y, cov, completeColIdx) =
      RegressionUtils.getPhenoCovCompleteSamples(mv, yField, covFields.toArray)

    if (!y.forall(yi => math.floor(yi) == yi && yi >= 0))
      fatal(s"For poisson regression, y must be numeric with all values non-negative integers")
    if (sum(y) == 0)
      fatal(s"For poisson regression, y must have at least one non-zero value")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(
        s"$n samples and ${k + 1} ${plural(k, "covariate")} (including x) implies $d degrees of freedom."
      )

    info(s"poisson_regression_rows: running $test on $n samples for response variable y,\n"
      + s"    with input variable x, and $k additional ${plural(k, "covariate")}...")

    val nullModel = new PoissonRegressionModel(cov, y)
    var nullFit = nullModel.fit(None, maxIter = maxIterations, tol = tolerance)

    if (!nullFit.converged)
      fatal("Failed to fit poisson regression null model (standard MLE with covariates only): " + (
        if (nullFit.exploded)
          s"exploded at Newton iteration ${nullFit.nIter}"
        else
          "Newton iteration failed to converge"
      ))

    val backend = HailContext.backend
    val completeColIdxBc = backend.broadcast(completeColIdx)

    val yBc = backend.broadcast(y)
    val XBc =
      backend.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = backend.broadcast(nullFit)
    val poisRegTestBc = backend.broadcast(poisRegTest)

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

        rvb.start(newRVDType.rowType)
        rvb.startStruct()
        rvb.addFields(fullRowType, ctx.r, ptr, copiedFieldIndices)
        poisRegTestBc.value
          .test(X, yBc.value, nullFitBc.value, "poisson", maxIter = maxIterations, tol = tolerance)
          .addToRVB(rvb)
        rvb.endStruct()
        rvb.end()
      }
    }

    TableValue(ctx, tableType, BroadcastRow.empty(ctx), newRVD)
  }
}
