package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.HailContext
import is.hail.annotations.{BroadcastRow, Region, RegionValue, RegionValueBuilder}
import is.hail.expr.ir.{TableLiteral, TableValue}
import is.hail.expr.types.{TFloat64, TInt64, TStruct, TableType}
import is.hail.linalg.RowMatrix
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner, OrderedRVDType, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

case class LMMData(gamma: Double, residualSq: Double, py: BDV[Double], px: BDM[Double], d: BDV[Double],
  ydy: Double, xdy: BDV[Double], xdx: BDM[Double], yOpt: Option[BDV[Double]], xOpt: Option[BDM[Double]])

object LinearMixedModel {
  def apply(hc: HailContext, gamma: Double, residualSq: Double, py: Array[Double], px: BDM[Double], d: Array[Double],
    ydy: Double, xdy: Array[Double], xdx: BDM[Double],
    yOpt: Option[Array[Double]], xOpt: Option[BDM[Double]]): LinearMixedModel = {

    new LinearMixedModel(hc,
      LMMData(gamma, residualSq, BDV(py), px, BDV(d), ydy, BDV(xdy), xdx, yOpt.map(BDV(_)), xOpt))
  }
  
  private val rowType = TStruct(
      "idx" -> TInt64(),
      "beta" -> TFloat64(),
      "sigma_sq" -> TFloat64(),
      "chi_sq" -> TFloat64(),
      "p_value" -> TFloat64())
  
  def toTable(hc: HailContext,
    orderedRVDPartitioner: OrderedRVDPartitioner,
    rdd: RDD[RegionValue]): Table = {
    
    val typ = TableType(rowType, Some(FastIndexedSeq("idx")), globalType = TStruct())
    
    val orderedRVD = OrderedRVD(new OrderedRVDType(Array("idx"), typ.rowType),
      orderedRVDPartitioner, ContextRDD.weaken[RVDContext](rdd))
    
    new Table(hc, TableLiteral(TableValue(typ, BroadcastRow(Row(), typ.globalType, hc.sc), orderedRVD)))
  }
}

class LinearMixedModel(hc: HailContext, lmmData: LMMData) {
  def fit(pa_t: RowMatrix, a_t: Option[RowMatrix]): Table =
    if (a_t.isDefined) {
      assert(lmmData.yOpt.isDefined && lmmData.xOpt.isDefined)
      fitLowRank(pa_t, a_t.get)
    } else {
      assert(lmmData.yOpt.isEmpty && lmmData.xOpt.isEmpty)
      fitFullRank(pa_t)
    }
 
  def fitLowRank(pa_t: RowMatrix, a_t: RowMatrix): Table = {
    if (pa_t.nRows != a_t.nRows)
      fatal(s"pa_t and a_t must have the same number of rows, but found ${pa_t.nRows} and ${a_t.nRows}")
    else if (!(pa_t.partitionCounts() sameElements a_t.partitionCounts()))
      fatal(s"pa_t and a_t both have ${pa_t.nRows} rows, but row partitions are not aligned")
        
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(LinearMixedModel.rowType)
    
    val rdd = pa_t.rows.zipPartitions(a_t.rows) { case (itPAt, itAt) =>
      val LMMData(gamma, nullResidualSq, py, px, d, ydy, xdy0, xdx0, Some(y), Some(x)) = lmmDataBc.value
      val xdy = xdy0.copy
      val xdx = xdx0.copy
      val n = x.rows
      val f = x.cols + 1
      val dof = n - f
      val r0 = 0 to 0
      val r1 = 1 until f

      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value

      itPAt.zip(itAt).map { case ((i, pa0), (i2, a0)) =>
        assert(i == i2)

        val a = BDV(a0)
        val pa = BDV(pa0)
        val dpa = d *:* pa

        xdy(0) = (py dot dpa) + gamma * (y dot a)
        xdx(0, 0) = (pa dot dpa) + gamma * (a dot a)
        xdx(r0, r1) := (px.t * dpa) + gamma * (x.t * a)  // could group rows or compute xtc with block matrix multiply
        xdx(r1, r0) := xdx(r0, r1).t
        
        region.clear()
        rvb.start(rowType)
        try {
          val beta = xdx \ xdy
          val residualSq = ydy - (xdy dot beta)
          val sigmaSq = residualSq / dof
          val chiSq = n * math.log(nullResidualSq / residualSq)
          val pValue = chiSquaredTail(chiSq, 1)

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
    
    LinearMixedModel.toTable(hc, pa_t.orderedRVDPartitioner(), rdd)
  }
  
  def fitFullRank(pa_t: RowMatrix): Table = {
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(LinearMixedModel.rowType)

    val rdd = pa_t.rows.zipPartitions(pa_t.rows) { case (itPAt, itPAt2) =>
      info("part")
      
      val LMMData(_, nullResidualSq, py, px, d, ydy, xdy0, xdx0, _, _) = lmmDataBc.value
      val xdy = xdy0.copy
      val xdx = xdx0.copy
 
      val n = px.rows
      val f = px.cols + 1
      val dof = n - f
      val r0 = 0 to 0
      val r1 = 1 until f

      val py0a = py.toArray
      val px0a = px.toArray
      val d0a = d.toArray
      val xdy0a = xdy0(1 until f).toArray
      val xdx0a = xdx0(1 until f, 1 until f).toArray
      
      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value

      itPAt.zip(itPAt2).map { case ((i, pa0), (i2, pa02)) =>
        val pa = BDV(pa0)       
        val dpa = d *:* pa

        xdy(0) = py dot dpa
        xdx(0, 0) = pa dot dpa
        xdx(r0, r1) := px.t * dpa
        xdx(r1, r0) := xdx(r0, r1).t
        
        region.clear()
        rvb.start(rowType)
        try {
          val beta = xdx \ xdy
          val residualSq = ydy - (xdy dot beta)
          val sigmaSq = residualSq / dof
          val chiSq = n * math.log(nullResidualSq / residualSq)
          val pValue = chiSquaredTail(chiSq, 1)

          def state(): String = s"state\n:" +
            s"i=$i, ${i / 4096}, ${i % 4096}\n\n" +
            s"i2=$i2, ${i2 / 4096}, ${i2 % 4096}\n\n" +   
            s"pa=${pa(0 until 10)}\n\n" +
            s"pa2=${pa(0 until 10)}\n\n" +
            s"py=${py(0 until 10)}\n\n" +
            s"py0=${py0a.slice(0, 10)}\n\n" +
            s"px=${px(0 until 10, 0 until 10)}\n\n" +
            s"px0=${px0a.slice(0, 10)}\n\n" +
            s"d=${d(0 until 10)}\n\n" +
            s"d0=${d0a.slice(0, 10)}\n\n" +
            s"xdy=$xdy\n\n" +
            s"xdy0=$xdy0\n\n" +
            s"xdx=$xdx\n\n" +
            s"xdx0=$xdx0\n\n" +
            s"beta=$beta\n\n" +
            s"pval=$pValue\n\n\n"
          
          if (!(py.toArray sameElements py0a))
            info(s"py changed\n\n${state()}\n\n")
          if (!(px.toArray sameElements px0a))
            info(s"px changed\n\n${state()}\n\n")
          if (!(d.toArray sameElements d0a))
            info(s"d changed\n\n${state()}\n\n")
          if (!(xdy(1 until f).toArray sameElements xdy0a))
            info(s"xdy changed\n\n${state()}\n\n")
          if (!(xdx(1 until f, 1 until f).toArray sameElements xdx0a))
            info(s"xdx changed\n\n${state()}\n\n")
          if (!(xdx(1 until f, 0).toArray sameElements xdx(0, 1 until f).t.toArray))
            info(s"xdx assymetric\n\n${state()}\n\n")
          
          if (pa != BDV(pa02))
            info(s"pa not equal pa2\n\n${state()}\n\n")
          else if (pValue == 1)
            info(s"pval is 1\n\n${state()}\n\n")
          
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
    
    LinearMixedModel.toTable(hc, pa_t.orderedRVDPartitioner(), rdd)
  }
}
