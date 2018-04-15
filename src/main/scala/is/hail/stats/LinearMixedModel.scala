package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.expr.types.{TFloat64, TInt64, TStruct}
import is.hail.linalg.RowMatrix
import is.hail.table.Table
import is.hail.utils._

case class LMMData(gamma: Double, residualSq: Double, y: BDV[Double], C: BDM[Double],
  Py: BDV[Double], PC: BDM[Double], Z: BDV[Double], ydy: Double, Cdy: BDV[Double], CdC: BDM[Double])

object LinearMixedModel {
  def apply(hc: HailContext, gamma: Double, residualSq: Double, y: Array[Double], C: BDM[Double],
    Py: Array[Double], PC: BDM[Double], Z: Array[Double], ydy: Double, Cdy: Array[Double], CdC: BDM[Double]) =

    new LinearMixedModel(hc, LMMData(gamma, residualSq, BDV(y), C, BDV(Py), PC, BDV(Z), ydy, BDV(Cdy), CdC))
}

class LinearMixedModel(hc: HailContext, lmmData: LMMData) {
  def fit(pathXt: String, pathPXt: String, partitionSize: Int): Table = {
    val rowType = TStruct(
      "idx" -> TInt64(),
      "beta" -> TFloat64(),
      "sigma_sq" -> TFloat64(),
      "chi_sq" -> TFloat64(),
      "p_value" -> TFloat64())
    
    val Xt = RowMatrix.readBlockMatrix(hc, pathXt, partitionSize)
    val PXt = RowMatrix.readBlockMatrix(hc, pathPXt, partitionSize)
    
    if (Xt.nRows != PXt.nRows)
      fatal("Numbers disagree")
    
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)

    val rdd = Xt.rows.zipPartitions(PXt.rows) { case (itx, itPx) =>
      val LMMData(gamma, nullResidualSq, y, c, py, pc, z, ydy, cdy0, cdc0) = lmmDataBc.value
      val cdy = cdy0.copy
      val cdc = cdc0.copy

      val n = c.rows
      val k = c.cols
      val dof = n - k - 1
      val r0 = 0 to 0
      val r1 = 1 to k

      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val rowType = rowTypeBc.value
      
      itx.zip(itPx).map { case ((i, x0), (i2, px0)) =>
        assert(i == i2)

        val x = BDV(x0)
        val px = BDV(px0)
        val zpx = z *:* px

        cdy(0) = (y dot x) * gamma + (py dot zpx)

        cdc(0, 0) = (x dot x) * gamma + (px dot zpx)
        cdc(r0, r1) := (c.t * x) * gamma + pc.t * zpx
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
          rvb.addDouble(beta(0))
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
