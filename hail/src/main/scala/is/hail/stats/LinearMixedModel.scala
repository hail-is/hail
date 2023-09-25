package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.annotations.{BroadcastRow, Region, RegionValue, RegionValueBuilder}
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkTaskContext
import is.hail.expr.ir.{TableIR, TableLiteral, TableValue}
import is.hail.linalg.RowMatrix
import is.hail.rvd.{RVD, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalStruct, PFloat64, PInt64}
import is.hail.types.virtual.TStruct
import is.hail.utils._
import org.apache.spark.storage.StorageLevel

case class LMMData(gamma: Double, residualSq: Double, py: BDV[Double], px: BDM[Double], d: BDV[Double],
  ydy: Double, xdy: BDV[Double], xdx: BDM[Double], yOpt: Option[BDV[Double]], xOpt: Option[BDM[Double]])

object LinearMixedModel {
  def pyApply(gamma: Double, residualSq: Double, py: Array[Double], px: BDM[Double], d: Array[Double],
    ydy: Double, xdy: Array[Double], xdx: BDM[Double],
    // yOpt, xOpt can be null
    yOpt: Array[Double], xOpt: BDM[Double]): LinearMixedModel = {

    new LinearMixedModel(
      LMMData(gamma, residualSq, BDV(py), px, BDV(d), ydy, BDV(xdy), xdx, Option(yOpt).map(BDV(_)), Option(xOpt)))
  }

  private val rowType = PCanonicalStruct(true,
      "idx" -> PInt64(),
      "beta" -> PFloat64(),
      "sigma_sq" -> PFloat64(),
      "chi_sq" -> PFloat64(),
      "p_value" -> PFloat64())

  private val tableType = TableType(rowType.virtualType, FastSeq("idx"), TStruct.empty)

  def toTableIR(ctx: ExecuteContext, rvd: RVD): TableIR = {
    TableLiteral(TableValue(ctx, tableType, BroadcastRow.empty(ctx), rvd), ctx.theHailClassLoader)
  }
}

class LinearMixedModel(lmmData: LMMData) {
  def fit(ctx: ExecuteContext, pa_t: RowMatrix, a_t: Option[RowMatrix]): TableIR =
    if (a_t.isDefined) {
      assert(lmmData.yOpt.isDefined && lmmData.xOpt.isDefined)
      fitLowRank(ctx, pa_t, a_t.get)
    } else {
      assert(lmmData.yOpt.isEmpty && lmmData.xOpt.isEmpty)
      fitFullRank(ctx, pa_t)
    }

  def fitLowRank(ctx: ExecuteContext, pa_t: RowMatrix, a_t: RowMatrix): TableIR = {
    if (pa_t.nRows != a_t.nRows)
      fatal(s"pa_t and a_t must have the same number of rows, but found ${pa_t.nRows} and ${a_t.nRows}")
    else if (!(pa_t.partitionCounts() sameElements a_t.partitionCounts()))
      fatal(s"pa_t and a_t both have ${pa_t.nRows} rows, but row partitions are not aligned")

    val lmmDataBc = ctx.backend.broadcast(lmmData)
    val rowType = LinearMixedModel.rowType

    val rdd = pa_t.rows.zipPartitions(a_t.rows) { case (itPAt, itAt) =>
      val LMMData(gamma, nullResidualSq, py, px, d, ydy, xdy0, xdx0, Some(y), Some(x)) = lmmDataBc.value
      val xdy = xdy0.copy
      val xdx = xdx0.copy
      val n = x.rows
      val f = x.cols + 1
      val dof = n - f
      val r0 = 0 to 0
      val r1 = 1 until f

      val region = Region(pool = SparkTaskContext.get().getRegionPool())
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(ctx.stateManager, region)

      itPAt.zip(itAt).map { case ((i, pa0), (i2, a0)) =>
        assert(i == i2)

        val a = BDV(a0)
        val pa = BDV(pa0)
        val dpa = d *:* pa

        xdy(0) = (py dot dpa) + gamma * (y dot a)
        xdx(0, 0) = (pa dot dpa) + gamma * (a dot a)
        xdx(r1, r0) := (dpa.t * px).t + gamma * (a.t * x).t // if px and x are not copied, the forms px.t * dpa and x.t * a result in a subtle bug
        xdx(r0, r1) := xdx(r1, r0).t

        region.clear()
        rvb.start(rowType)
        try {
          val beta = xdx \ xdy
          val residualSq = ydy - (xdy dot beta)
          val sigmaSq = residualSq / dof
          val chiSq = n * math.log(nullResidualSq / residualSq)
          val pValue = pchisqtail(chiSq, 1)

          rvb.startStruct()
          rvb.addLong(i)
          rvb.addDouble(beta(0))  // could expand to return all coefficients
          rvb.addDouble(sigmaSq)
          rvb.addDouble(chiSq)
          rvb.addDouble(pValue)
          rvb.endStruct()
        } catch {
          case _: breeze.linalg.MatrixSingularException =>
            rvb.startStruct()
            rvb.addLong(i)
            rvb.setMissing()
            rvb.setMissing()
            rvb.setMissing()
            rvb.setMissing()
            rvb.endStruct()
        }
        rv.setOffset(rvb.end())
        rv
      }
    }
    val rvd = RVD(
      RVDType(rowType, LinearMixedModel.tableType.key),
      pa_t.partitioner(),
      ContextRDD.weaken(rdd).toCRDDPtr).persist(ctx, StorageLevel.MEMORY_AND_DISK)

    LinearMixedModel.toTableIR(ctx, rvd)
  }

  def fitFullRank(ctx: ExecuteContext, pa_t: RowMatrix): TableIR = {
    val lmmDataBc = ctx.backend.broadcast(lmmData)
    val rowType = LinearMixedModel.rowType

    val rdd = pa_t.rows.mapPartitions { itPAt =>
      val LMMData(_, nullResidualSq, py, px, d, ydy, xdy0, xdx0, _, _) = lmmDataBc.value
      val xdy = xdy0.copy
      val xdx = xdx0.copy
      val n = px.rows
      val f = px.cols + 1
      val dof = n - f
      val r0 = 0 to 0
      val r1 = 1 until f

      val region = Region(pool = SparkTaskContext.get().getRegionPool())
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(ctx.stateManager, region)

      itPAt.map { case (i, pa0) =>
        val pa = BDV(pa0)
        val dpa = d *:* pa

        xdy(0) = py dot dpa
        xdx(0, 0) = pa dot dpa
        xdx(r1, r0) := (dpa.t * px).t // if px is not copied, the form px.t * dpa results in a subtle bug
        xdx(r0, r1) := xdx(r1, r0).t

        region.clear()
        rvb.start(rowType)
        try {
          val beta = xdx \ xdy
          val residualSq = ydy - (xdy dot beta)
          val sigmaSq = residualSq / dof
          val chiSq = n * math.log(nullResidualSq / residualSq)
          val pValue = pchisqtail(chiSq, 1)

          rvb.startStruct()
          rvb.addLong(i)
          rvb.addDouble(beta(0))  // could expand to return all coefficients, or switch to block matrix projection trick
          rvb.addDouble(sigmaSq)
          rvb.addDouble(chiSq)
          rvb.addDouble(pValue)
          rvb.endStruct()
        } catch {
          case _: breeze.linalg.MatrixSingularException =>
            rvb.startStruct()
            rvb.addLong(i)
            rvb.setMissing()
            rvb.setMissing()
            rvb.setMissing()
            rvb.setMissing()
            rvb.endStruct()
        }
        rv.setOffset(rvb.end())
        rv
      }
    }

    val rvd = RVD(
      RVDType(rowType, LinearMixedModel.tableType.key),
      pa_t.partitioner(),
      ContextRDD.weaken(rdd).toCRDDPtr).persist(ctx, StorageLevel.MEMORY_AND_DISK)

    LinearMixedModel.toTableIR(ctx, rvd)
  }
}
