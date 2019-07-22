package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{HailSuite, TestUtils}
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.expr.ir.{BlockMatrixLiteral, BlockMatrixToTableApply, Literal}
import is.hail.annotations.Annotation
import is.hail.expr.types.virtual.{TArray, TFloat64}
import is.hail.table.Table
import is.hail.variant.{Call, HardCallView, MatrixTable}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.testng.annotations.Test
import scala.language.postfixOps

class PCRelateSuite extends HailSuite {
  private val blockSize: Int = BlockMatrix.defaultBlockSize

  private def toD(a: java.lang.Double): Double =
    a.asInstanceOf[Double]

  private def toBoxedD(a: Any): java.lang.Double =
    a.asInstanceOf[java.lang.Double]

  type Quad[T] = (T, T, T, T)
  type Double4 = Quad[Double]
  type JavaDouble4 = Quad[java.lang.Double]

  private def quadMap[T,U](f: T => U): (T, T, T, T) => (U, U, U, U) =
    { case (x, y, z, w) => (f(x), f(y), f(z), f(w)) }

  private def vdsToMeanImputedMatrix(vds: MatrixTable): BlockMatrix = {
    val nSamples = vds.numCols
    val localRowPType = vds.rvRowPType
    val partStarts = vds.partitionStarts()
    val partStartsBc = vds.hc.backend.broadcast(partStarts)
    val rdd = vds.rvd.mapPartitionsWithIndex { (partIdx, it) =>
      val view = HardCallView(localRowPType)
      val missingIndices = new ArrayBuilder[Int]()

      var rowIdx = partStartsBc.value(partIdx)
      it.map { rv =>
        view.setRegion(rv)

        missingIndices.clear()
        var sum = 0
        var nNonMissing = 0
        val a = new Array[Double](nSamples)
        var i = 0
        while (i < nSamples) {
          view.setGenotype(i)
          if (view.hasGT) {
            val gt = Call.unphasedDiploidGtIndex(view.getGT)
            sum += gt
            a(i) = gt
            nNonMissing += 1
          } else
            missingIndices += i
          i += 1
        }

        val mean = sum.toDouble / nNonMissing

        i = 0
        while (i < missingIndices.length) {
          a(missingIndices(i)) = mean
          i += 1
        }

        rowIdx += 1
        IndexedRow(rowIdx - 1, Vectors.dense(a))
      }
    }

    val irr = new IndexedRowMatrix(rdd.cache(), partStarts.last, nSamples)
    BlockMatrix.fromIRM(irr, blockSize).cache()
  }

  def runPcRelateHail(vds: MatrixTable, pcs: BDM[Double], maf: Double): Map[(String, String), JavaDouble4] = {
    val gIR = new BlockMatrixLiteral(vdsToMeanImputedMatrix(vds))
    val pcsIR = Literal(TArray(TArray(TFloat64())),
      IndexedSeq.tabulate(pcs.rows) { row =>
        IndexedSeq.tabulate(pcs.cols)(pcs(row, _)) })

    val sampleIds = vds.stringSampleIds.toArray[Annotation]
    val tir = BlockMatrixToTableApply(gIR, pcsIR, new PCRelate(maf, blockSize))

    new Table(hc, tir)
      .collect()
      .map { r =>
        val i = r(0).asInstanceOf[Int]
        val j = r(1).asInstanceOf[Int]
        ((sampleIds(i), sampleIds(j)), (r(2), r(3), r(4), r(5)))
      }
      .toMap
      .mapValues(quadMap(toBoxedD).tupled)
      .asInstanceOf[Map[(String, String), JavaDouble4]]
  }

  private def compareDoubleQuadruplet(cmp: (Double, Double) => Boolean)(x: Double4, y: Double4): Boolean =
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
