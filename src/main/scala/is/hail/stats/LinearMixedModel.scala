package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.expr.types.{TFloat64, TInt64, TStruct}
import is.hail.linalg.RowMatrix
import is.hail.table.Table

case class LMMData(gamma: Double, residualSq: Double, py: BDV[Double], px: BDM[Double], d: BDV[Double],
  ydy: Double, xdy: BDV[Double], xdx: BDM[Double], yOpt: Option[BDV[Double]], xOpt: Option[BDM[Double]])

object LinearMixedModel {
  def apply(hc: HailContext, gamma: Double, residualSq: Double, py: Array[Double], px: BDM[Double], d: Array[Double],
    ydy: Double, xdy: Array[Double], xdx: BDM[Double],
    yOpt: Option[Array[Double]], xOpt: Option[BDM[Double]]): LinearMixedModel = {

    new LinearMixedModel(hc,
      LMMData(gamma, residualSq, BDV(py), px, BDV(d), ydy, BDV(xdy), xdx, yOpt.map(BDV(_)), xOpt))
  }
}

class LinearMixedModel(hc: HailContext, lmmData: LMMData) {
  private val rowType = TStruct(
      "idx" -> TInt64(),
      "beta" -> TFloat64(),
      "sigma_sq" -> TFloat64(),
      "chi_sq" -> TFloat64(),
      "p_value" -> TFloat64())

  def fit(pathPAt: String, pathAt: Option[String], partitionSize: Int): Table =
    if (pathAt.isDefined) {
      assert(lmmData.yOpt.isDefined && lmmData.xOpt.isDefined)
      fitLowRank(pathPAt, pathAt.get, partitionSize)
    } else {
      assert(lmmData.yOpt.isEmpty && lmmData.xOpt.isEmpty)
      fitFullRank(pathPAt, partitionSize)
    }
 
  def fitLowRank(pathPAt: String, pathAt: String, partitionSize: Int): Table = {
    val PAt = RowMatrix.readBlockMatrix(hc, pathPAt, partitionSize)
    val At = RowMatrix.readBlockMatrix(hc, pathAt, partitionSize)
    
    assert(PAt.nRows == At.nRows)
    
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)
    
    val rdd = PAt.rows.zipPartitions(At.rows) { case (itPAt, itAt) =>
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
        xdx(r0, r1) := (px.t * dpa) + gamma * (x.t * a)  // FIXME group rows or compute xtc with block matrix multiply
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
    new Table(hc, rdd, rowType, Array("idx")) // FIXME hand partitioner to OrderedRVD once Table is updated
  }
  
  def fitFullRank(pathPAt: String, partitionSize: Int): Table = {
    val PAt = RowMatrix.readBlockMatrix(hc, pathPAt, partitionSize)    

    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)
    
    val rdd = PAt.rows.mapPartitions { itPAt =>
      val LMMData(_, nullResidualSq, py, px, d, ydy, xdy0, xdx0, _, _) = lmmDataBc.value
      val xdy = xdy0.copy
      val xdx = xdx0.copy
      val n = lmmData.px.rows
      val f = lmmData.px.cols + 1
      val dof = n - f
      val r0 = 0 to 0
      val r1 = 1 until f
      
      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value

      itPAt.map { case (i, pa0) =>
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
