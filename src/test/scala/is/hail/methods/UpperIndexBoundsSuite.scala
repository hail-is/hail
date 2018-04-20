package is.hail.methods

import is.hail.expr.types.{TInt32, TString, TStruct}
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class UpperIndexBoundsSuite extends SparkSuite {

  def makeSquareBlockMatrix(nRows: Int, blockSize: Int): BlockMatrix = {
    val arbitraryEntries = new BDM[Double](nRows, nRows, Array.fill[Double](nRows * nRows)(0))
    BlockMatrix.fromBreezeMatrix(sc, arbitraryEntries, blockSize)
  }

  val tbl = {
    val rows = IndexedSeq[(String, Int)](("X", 5), ("X", 7), ("X", 13), ("X", 14), ("X", 17),
      ("X", 65), ("X", 70), ("X", 73), ("Y", 74), ("Y", 75), ("Y", 200), ("Y", 300))
      .map { case (contig, pos) => Row(contig, pos) }
    Table.parallelize(hc, rows, TStruct("contig" -> TString(), "pos" -> TInt32()), IndexedSeq[String](), None)
  }

  @Test def testGroupPositionsByContig() {
    val groupedPositions = UpperIndexBounds.groupPositionsByKey(tbl.keyBy("contig"))
    val expected = Array(Array(5, 7, 13, 14, 17, 65, 70, 73), Array(74, 75, 200, 300))
    assert((groupedPositions.collect(), expected).zipped
      .forall { case (positions, expectedPositions) => positions.toSet == expectedPositions.toSet })
  }

  @Test def testComputeUpperIndexBoundsOnSingleArray() {
    val positions = Array(1, 3, 4, 5, 8, 10, 13, 14, 16, 17, 18, 20)
    val bounds = UpperIndexBounds.computeUpperIndexBounds(positions, radius = 10)
    val expected = Array(5, 6, 7, 7, 10, 11, 11, 11, 11, 11, 11, 11)
    assert(bounds sameElements expected)
  }

  @Test def testShiftUpperIndexBounds() {
    val groupedPositions = UpperIndexBounds.groupPositionsByKey(tbl.keyBy("contig"))
    val bounds = UpperIndexBounds.shiftUpperIndexBounds(groupedPositions.map { positions =>
      scala.util.Sorting.quickSort(positions)
      UpperIndexBounds.computeUpperIndexBounds(positions, radius = 10)
    })
    val expected = Array(3, 4, 4, 4, 4, 7, 7, 7, 9, 9, 10, 11)
    assert(bounds sameElements expected)
  }

  @Test def testComputeBlocksWithinRadius() {
    val blockSizes = Array(1, 2)
    val expecteds = Array(Array(12, 24, 25, 36, 37, 38, 49, 50, 51, 77, 89, 90, 116),
      Array(0, 6, 7, 12, 13, 20, 21, 28))
    val diagonalBlocksToExclude = Array(Array(0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143), Array(14, 35))
    
    for (i <- 0 to 1) {
      val bm = makeSquareBlockMatrix(tbl.count().toInt, blockSize = blockSizes(i))
      val blocks = UpperIndexBounds.computeBlocksWithinRadiusAndAboveDiagonal(tbl.keyBy("contig"), 
        bm.gp, radius = 10, includeDiagonal = false)
      val blocksWithDiagonal = UpperIndexBounds.computeBlocksWithinRadiusAndAboveDiagonal(tbl.keyBy("contig"), 
        bm.gp, radius=10, includeDiagonal = true)
      assert(blocks sameElements expecteds(i))
      assert(blocksWithDiagonal.sorted sameElements (expecteds(i)++diagonalBlocksToExclude(i)).sorted)
    }
  }
}
