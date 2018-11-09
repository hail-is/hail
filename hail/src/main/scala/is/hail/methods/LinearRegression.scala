package is.hail.methods

import breeze.linalg._
import breeze.numerics.sqrt
import is.hail.annotations._
import is.hail.expr.ir.{TableLiteral, TableValue}
import is.hail.expr.types.{virtual, _}
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32, TStruct}
import is.hail.stats._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._

case class ChainedLinregInput(
  n: Int,
  y: DenseMatrix[Double],
  completeColIndex: Array[Int],
  Qt: DenseMatrix[Double],
  Qty: DenseMatrix[Double],
  yyp: DenseVector[Double],
  d: Int)

case class ChainedLinregResult(
  n: Int,
  AC: DenseVector[Double],
  ytx: DenseMatrix[Double],
  b: DenseMatrix[Double],
  se: DenseMatrix[Double],
  t: DenseMatrix[Double],
  p: DenseMatrix[Double]
)

object LinearRegression {
  val schema = TStruct(
    ("n", TInt32()),
    ("sum_x", TFloat64()),
    ("y_transpose_x", TArray(TFloat64())),
    ("beta", TArray(TFloat64())),
    ("standard_error", TArray(TFloat64())),
    ("t_stat", TArray(TFloat64())),
    ("p_value", TArray(TFloat64())))

  val chainedSchema = TStruct(
    ("n", TArray(TInt32())),
    ("sum_x", TArray(TFloat64())),
    ("y_transpose_x", TArray(TArray(TFloat64()))),
    ("beta", TArray(TArray(TFloat64()))),
    ("standard_error", TArray(TArray(TFloat64()))),
    ("t_stat", TArray(TArray(TFloat64()))),
    ("p_value", TArray(TArray(TFloat64()))))

  def single_group(vsm: MatrixTable,
    _yFields: java.util.ArrayList[String],
    xField: String,
    _covFields: java.util.ArrayList[String],
    rowBlockSize: Int,
    _passThrough: java.util.ArrayList[String]
  ): Table = {

    val yFields = _yFields.asScala.toArray
    val covFields = _covFields.asScala.toArray
    val passThrough = _passThrough.asScala.toArray

    val (y, cov, completeColIdx) = RegressionUtils.getPhenosCovCompleteSamples(vsm, yFields, covFields)

    val n = y.rows // n_complete_samples
    val k = cov.cols // nCovariates
    val d = n - k - 1
    val dRec = 1d / d

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x) implies $d degrees of freedom.")

    info(s"linear_regression_rows: running on $n samples for ${ y.cols } response ${ plural(y.cols, "variable") } y,\n"
       + s"    with input variable x, and ${ k } additional ${ plural(k, "covariate") }...")

    val Qt =
      if (k > 0)
        qr.reduced.justQ(cov).t
      else
        DenseMatrix.zeros[Double](0, n)

    val Qty = Qt * y

    val sc = vsm.sparkContext
    val completeColIdxBc = sc.broadcast(completeColIdx)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(y.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r))

    val fullRowType = vsm.rvRowType.physicalType
    val entryArrayType = vsm.matrixType.entryArrayType.physicalType
    val entryType = vsm.entryType.physicalType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType.isOfType(TFloat64()))

    val entryArrayIdx = vsm.entriesIndex
    val fieldIdx = entryType.fieldIdx(xField)

    val passThroughType = TStruct(passThrough.map(f => f -> vsm.rowType.field(f).typ): _*)
    val tableType = TableType(vsm.rowKeyStruct ++ passThroughType ++ LinearRegression.schema, vsm.rowKey, TStruct())
    val newRVDType = tableType.rvdType
    val copiedFieldIndices = (vsm.rowKey ++ passThrough).map(vsm.rvRowType.fieldIdx(_)).toArray
    val nDependentVariables = yFields.length

    val newRVD = vsm.rvd.boundary.mapPartitions(
      newRVDType, { (ctx, it) =>
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        val missingCompleteCols = new ArrayBuilder[Int]
        val data = new Array[Double](n * rowBlockSize)

        val blockWRVs = new Array[WritableRegionValue](rowBlockSize)
        var i = 0
        while (i < rowBlockSize) {
          blockWRVs(i) = WritableRegionValue(fullRowType, ctx.freshRegion)
          i += 1
        }

        it.trueGroupedIterator(rowBlockSize)
          .flatMap { git =>
            var i = 0
            while (git.hasNext) {
              val rv = git.next()
              RegressionUtils.setMeanImputedDoubles(data, i * n, completeColIdxBc.value, missingCompleteCols,
                rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)
              blockWRVs(i).set(rv)
              i += 1
            }
            val blockLength = i

            val X = new DenseMatrix[Double](n, blockLength, data)

            val AC: DenseVector[Double] = X.t(*, ::).map(r => sum(r))
            assert(AC.length == blockLength)

            val qtx: DenseMatrix[Double] = QtBc.value * X
            val qty: DenseMatrix[Double] = QtyBc.value
            val xxpRec: DenseVector[Double] = 1.0 / (X.t(*, ::).map(r => r dot r) - qtx.t(*, ::).map(r => r dot r))
            val ytx: DenseMatrix[Double] = yBc.value.t * X
            assert(ytx.rows == yBc.value.cols && ytx.cols == blockLength)

            val xyp: DenseMatrix[Double] = ytx - (qty.t * qtx)
            val yyp: DenseVector[Double] = yypBc.value

            // resuse xyp
            val b = xyp
            i = 0
            while (i < blockLength) {
              xyp(::, i) :*= xxpRec(i)
              i += 1
            }

            val se = sqrt(dRec * (yyp * xxpRec.t - (b *:* b)))

            val t = b /:/ se
            val p = t.map(s => 2 * T.cumulative(-math.abs(s), d, true, false))

            (0 until blockLength).iterator.map { i =>
              val wrv = blockWRVs(i)
              rvb.set(wrv.region)
              rvb.start(newRVDType.rowType)
              rvb.startStruct()
              rvb.addFields(fullRowType, wrv.region, wrv.offset, copiedFieldIndices)
              rvb.addInt(n)
              rvb.addDouble(AC(i))

              def addSlice(dm: DenseMatrix[Double]) {
                rvb.startArray(nDependentVariables)
                var j = 0
                while (j < nDependentVariables) {
                  rvb.addDouble(dm(j, i))
                  j += 1
                }
                rvb.endArray()
              }

              addSlice(ytx)
              addSlice(b)
              addSlice(se)
              addSlice(t)
              addSlice(p)

              rvb.endStruct()
              rv2.set(wrv.region, rvb.end())
              rv2
            }
          }
      }).persist(StorageLevel.MEMORY_AND_DISK)
    new Table(vsm.hc, TableLiteral(TableValue(tableType, BroadcastRow.empty(sc), newRVD)))
  }

  def chain(vsm: MatrixTable,
    _yFields: java.util.ArrayList[java.util.ArrayList[String]],
    xField: String,
    _covFields: java.util.ArrayList[String],
    rowBlockSize: Int,
    _passThrough: java.util.ArrayList[String]
  ): Table = {

    val yFields = _yFields.asScala.map(_.asScala.toArray).toArray
    val covFields = _covFields.asScala.toArray
    val passThrough = _passThrough.asScala.toArray

    val localData = yFields.map(RegressionUtils.getPhenosCovCompleteSamples(vsm, _, covFields))

    val k = covFields.length // nCovariates
    val bcData = localData.zipWithIndex.map { case ((y, cov, completeColIdx), i) =>
      val n = y.rows
      val d = n - k - 1
      if (d < 1)
        fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x) implies $d degrees of freedom.")

      info(s"linear_regression_rows[$i]: running on $n samples for ${ y.cols } response ${ plural(y.cols, "variable") } y,\n"
        + s"    with input variable x, and ${ k } additional ${ plural(k, "covariate") }...")

      val Qt =
        if (k > 0)
          qr.reduced.justQ(cov).t
        else
          DenseMatrix.zeros[Double](0, n)
      val Qty = Qt * y
      val yyp = y.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r)

      ChainedLinregInput(n, y, completeColIdx, Qt, Qty, yyp, d)
    }

    val bc = vsm.sparkContext.broadcast(bcData)
    val nGroups = bcData.length

    val fullRowType = vsm.rvRowType.physicalType
    val entryArrayType = vsm.matrixType.entryArrayType.physicalType
    val entryType = vsm.entryType.physicalType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.virtualType.isOfType(TFloat64()))

    val entryArrayIdx = vsm.entriesIndex
    val fieldIdx = entryType.fieldIdx(xField)

    val passThroughType = TStruct(passThrough.map(f => f -> vsm.rowType.field(f).typ): _*)
    val tableType = TableType(vsm.rowKeyStruct ++ passThroughType ++ LinearRegression.chainedSchema, vsm.rowKey, TStruct())
    val newRVDType = tableType.rvdType
    val copiedFieldIndices = (vsm.rowKey ++ passThrough).map(vsm.rvRowType.fieldIdx(_)).toArray

    val newRVD = vsm.rvd.boundary.mapPartitions(
      newRVDType, { (ctx, it) =>
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        val inputData = bc.value
        val builder = new ArrayBuilder[Int]
        val data = inputData.map(cri => new Array[Double](cri.n * rowBlockSize))

        val blockWRVs = new Array[WritableRegionValue](rowBlockSize)
        var i = 0
        while (i < rowBlockSize) {
          blockWRVs(i) = WritableRegionValue(fullRowType, ctx.freshRegion)
          i += 1
        }

        it.trueGroupedIterator(rowBlockSize)
          .flatMap { git =>
            var i = 0
            while (git.hasNext) {
              val rv = git.next()
              var j = 0
              while (j < nGroups) {
                RegressionUtils.setMeanImputedDoubles(data(j), i * inputData(j).n, inputData(j).completeColIndex, builder,
                  rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)
                j += 1
              }
              blockWRVs(i).set(rv)
              i += 1
            }
            val blockLength = i

            val results = Array.tabulate(nGroups) { j =>
              val cri = inputData(j)
              val X = new DenseMatrix[Double](cri.n, blockLength, data(j))

              val AC: DenseVector[Double] = X.t(*, ::).map(r => sum(r))
              assert(AC.length == blockLength)

              val qtx: DenseMatrix[Double] = cri.Qt * X
              val qty: DenseMatrix[Double] = cri.Qty
              val xxpRec: DenseVector[Double] = 1.0 / (X.t(*, ::).map(r => r dot r) - qtx.t(*, ::).map(r => r dot r))
              val ytx: DenseMatrix[Double] = cri.y.t * X
              assert(ytx.rows == cri.y.cols && ytx.cols == blockLength)

              val xyp: DenseMatrix[Double] = ytx - (qty.t * qtx)
              val yyp: DenseVector[Double] = cri.yyp
              // resuse xyp
              val b = xyp
              i = 0
              while (i < blockLength) {
                xyp(::, i) :*= xxpRec(i)
                i += 1
              }
              val se = sqrt((1d / cri.d) * (yyp * xxpRec.t - (b *:* b)))

              val t = b /:/ se
              val p = t.map(s => 2 * T.cumulative(-math.abs(s), cri.d, true, false))

              ChainedLinregResult(cri.n, AC, ytx, b, se, t, p)
            }

            (0 until blockLength).iterator.map { i =>
              val wrv = blockWRVs(i)
              rvb.set(wrv.region)
              rvb.start(newRVDType.rowType)
              rvb.startStruct()
              rvb.addFields(fullRowType, wrv.region, wrv.offset, copiedFieldIndices)

              // FIXME: the below has horrible cache behavior, but hard to get around
              // FIXME: it when doing a two-way in-memory transpose like this

              rvb.startArray(nGroups)
              results.foreach(r => rvb.addInt(r.n))
              rvb.endArray()

              rvb.startArray(nGroups)
              results.foreach(r => rvb.addDouble(r.AC(i)))
              rvb.endArray()

              def addSlice(dm: DenseMatrix[Double]) {
                val size = dm.rows
                rvb.startArray(size)
                var j = 0
                while (j < size) {
                  rvb.addDouble(dm(j, i))
                  j += 1
                }
                rvb.endArray()
              }

              rvb.startArray(nGroups)
              results.foreach(r => addSlice(r.ytx))
              rvb.endArray()

              rvb.startArray(nGroups)
              results.foreach(r => addSlice(r.b))
              rvb.endArray()

              rvb.startArray(nGroups)
              results.foreach(r => addSlice(r.se))
              rvb.endArray()

              rvb.startArray(nGroups)
              results.foreach(r => addSlice(r.t))
              rvb.endArray()

              rvb.startArray(nGroups)
              results.foreach(r => addSlice(r.p))
              rvb.endArray()

              rvb.endStruct()
              rv2.set(wrv.region, rvb.end())
              rv2
            }
          }
      }).persist(StorageLevel.MEMORY_AND_DISK)
    new Table(vsm.hc, TableLiteral(TableValue(tableType, BroadcastRow.empty(vsm.sparkContext), newRVD)))
  }
}
