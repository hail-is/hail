package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{HailSuite, TestUtils}
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.variant.MatrixTable
import org.testng.annotations.Test
import scala.language.postfixOps

import scala.sys.process._

class PCRelateSuite extends HailSuite {
  private val blockSize: Int = BlockMatrix.defaultBlockSize

  private def toD(a: java.lang.Double): Double =
    a.asInstanceOf[Double]

  private def toBoxedD(a: Any): java.lang.Double =
    a.asInstanceOf[java.lang.Double]

  private def quadMap[T,U](f: T => U): (T, T, T, T) => (U, U, U, U) =
    { case (x, y, z, w) => (f(x), f(y), f(z), f(w)) }

  def runPcRelateHail(
    vds: MatrixTable,
    pcs: BDM[Double],
    maf: Double,
    minKinship: Double = PCRelate.defaultMinKinship,
    statistics: PCRelate.StatisticSubset = PCRelate.defaultStatisticSubset): Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)] =
    PCRelate(hc, vds, pcs, maf, blockSize, minKinship, statistics)
      .collect()
      .map(r => ((r(0), r(1)), (r(2), r(3), r(4), r(5))))
      .toMap
      .mapValues(quadMap(toBoxedD).tupled)
      .asInstanceOf[Map[(String, String), (java.lang.Double, java.lang.Double, java.lang.Double, java.lang.Double)]]

  def compareDoubleQuadruplet(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)

  @Test
  def trivialReference() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromCallMatrix(hc)(TestUtils.unphasedDiploidGtIndicesToBoxedCall(genotypeMatrix))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new BDM(4, 2, pcsArray)
    val us = runPcRelateHail(vds, pcs, maf=0.0).mapValues(quadMap(toD).tupled)
    val truth = PCRelateReferenceImplementation(vds, pcs)._1
    assert(mapSameElements(us, truth, compareDoubleQuadruplet((x, y) => math.abs(x - y) < 1e-14)))
  }
}
