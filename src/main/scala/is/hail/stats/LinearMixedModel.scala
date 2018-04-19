package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.expr.types.{TFloat64, TInt64, TStruct}
import is.hail.linalg.RowMatrix
import is.hail.table.Table
import is.hail.utils._

case class LMMData(gamma: Double, residualSq: Double, py: BDV[Double], pc: BDM[Double], d: BDV[Double],
  ydy: Double, cdy: BDV[Double], cdc: BDM[Double], yOpt: Option[BDV[Double]], cOpt: Option[BDM[Double]])

object LinearMixedModel {
  def apply(hc: HailContext, gamma: Double, residualSq: Double, py: Array[Double], pc: BDM[Double], d: Array[Double],
    ydy: Double, cdy: Array[Double], cdc: BDM[Double],
    yOpt: Option[Array[Double]], cOpt: Option[BDM[Double]]): LinearMixedModel = {

    new LinearMixedModel(hc,
      LMMData(gamma, residualSq, BDV(py), pc, BDV(d), ydy, BDV(cdy), cdc, yOpt.map(BDV(_)), cOpt))
  }
}

class LinearMixedModel(hc: HailContext, lmmData: LMMData) {
  private val rowType = TStruct(
      "idx" -> TInt64(),
      "beta" -> TFloat64(),
      "sigma_sq" -> TFloat64(),
      "chi_sq" -> TFloat64(),
      "p_value" -> TFloat64())

  def fit(pathPXt: String, pathXt: Option[String], partitionSize: Int): Table =
    if (pathXt.isDefined) {
      assert(lmmData.yOpt.isDefined && lmmData.cOpt.isDefined)
      fitLowRank(pathPXt, pathXt.get, partitionSize)
    } else {
      assert(lmmData.yOpt.isEmpty && lmmData.cOpt.isEmpty)
      fitFullRank(pathPXt, partitionSize)
    }
 
  def fitLowRank(pathPXt: String, pathXt: String, partitionSize: Int): Table = {
    val PXt = RowMatrix.readBlockMatrix(hc, pathPXt, partitionSize)
    val Xt = RowMatrix.readBlockMatrix(hc, pathXt, partitionSize)
    
    assert(PXt.nRows == Xt.nRows)
    
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)
    
    val rdd = PXt.rows.zipPartitions(Xt.rows) { case (itPx, itx) =>
      val LMMData(gamma, nullResidualSq, py, pc, d, ydy, cdy0, cdc0, Some(y), Some(c)) = lmmDataBc.value
      val cdy = cdy0.copy
      val cdc = cdc0.copy
      val n = c.rows
      val k = c.cols + 1
      val dof = n - k
      val r0 = 0 to 0
      val r1 = 1 until k

      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value

      itPx.zip(itx).map { case ((i, px0), (i2, x0)) =>
        assert(i == i2)

        val x = BDV(x0)
        val px = BDV(px0)
        val dpx = d *:* px

        cdy(0) = (py dot dpx) + gamma * (y dot x)
        cdc(0, 0) = (px dot dpx) + gamma * (x dot x)
        cdc(r0, r1) := (pc.t * dpx) + gamma * (c.t * x)  // FIXME group rows or compute xtc with block matrix multiply
        cdc(r1, r0) := cdc(r0, r1).t
        
        region.clear()
        rvb.start(rowType)
        try {
          val beta = cdc \ cdy
          val residualSq = ydy - (cdy dot beta)
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
    new Table(hc, rdd, rowType, Array("idx")) // FIXME hand partitioner to OrderedRVD once Table is updated
  }
  
  def fitFullRank(pathPXt: String, partitionSize: Int): Table = {
    val PXt = RowMatrix.readBlockMatrix(hc, pathPXt, partitionSize)    

    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)
    
    val rdd = PXt.rows.mapPartitions { itPx =>
      val LMMData(gamma, nullResidualSq, py, pc, d, ydy, cdy0, cdc0, _, _) = lmmDataBc.value
      val cdy = cdy0.copy
      val cdc = cdc0.copy
      val n = lmmData.pc.rows
      val k = lmmData.pc.cols + 1
      val dof = n - k
      val r0 = 0 to 0
      val r1 = 1 until k
      
      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value

      itPx.map { case (i, px0) =>
        val px = BDV(px0)
        val dpx = d *:* px

        cdy(0) = py dot dpx
        cdc(0, 0) = px dot dpx
        cdc(r0, r1) := pc.t * dpx
        cdc(r1, r0) := cdc(r0, r1).t
        
        region.clear()
        rvb.start(rowType)
        try {          
          val beta = cdc \ cdy
          val residualSq = ydy - (cdy dot beta)
          val sigmaSq = residualSq / dof
          val chiSq = n * math.log(nullResidualSq / residualSq)
          val pValue = chiSquaredTail(chiSq, 1)

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
    new Table(hc, rdd, rowType, Array("idx")) // FIXME hand partitioner to OrderedRVD once Table is updated
  }
}
