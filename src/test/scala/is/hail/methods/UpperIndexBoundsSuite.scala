package is.hail.methods

import is.hail.expr.types.{TInt32, TString, TStruct}
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.SparkSuite
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class UpperIndexBoundsSuite extends SparkSuite {
  lazy val t: Table = {
    val rows = IndexedSeq[(String, Int)](("X", 5), ("X", 7), ("X", 13), ("X", 14), ("X", 17),
      ("X", 65), ("X", 70), ("X", 73), ("Y", 74), ("Y", 75), ("Y", 200), ("Y", 300))
      .map { case (contig, pos) => Row(contig, pos) }
    Table.parallelize(hc, rows, TStruct("contig" -> TString(), "pos" -> TInt32()), None, None)
  }

  @Test def testGroupPositionsByContig() {
    val groupedPositions = UpperIndexBounds.groupPositionsByKey(t.keyBy("contig"))
    val expected = Array(Array(5, 7, 13, 14, 17, 65, 70, 73), Array(74, 75, 200, 300))
    assert((groupedPositions.collect(), expected).zipped
      .forall { case (positions, expectedPositions) => positions.toSet == expectedPositions.toSet })
  }

  @Test def testComputeUpperIndexBoundsOnSingleArray() {
    val positions = Array(1, 3, 4, 5, 8, 10, 13, 14, 16, 17, 18, 20)
    val bounds = UpperIndexBounds.computeUpperIndexBounds(positions, radius = 10)
    val expected = Array(6, 7, 8, 8, 11, 12, 12, 12, 12, 12, 12, 12)
    assert(bounds sameElements expected)
  }

  @Test def testShiftUpperIndexBounds() {
    val groupedPositions = UpperIndexBounds.groupPositionsByKey(t.keyBy("contig"))
    val bounds = UpperIndexBounds.shiftUpperIndexBounds(groupedPositions.map { positions =>
      scala.util.Sorting.quickSort(positions)
      UpperIndexBounds.computeUpperIndexBounds(positions, radius = 10)
    })
    val expected = Array(4, 5, 5, 5, 5, 8, 8, 8, 10, 10, 11, 12)
    assert(bounds sameElements expected)
  }

  @Test def testComputeBlocksWithinRadius() {
    val blockSizes = Array(1, 2)
    val expecteds = Array(Array(12, 24, 25, 36, 37, 38, 49, 50, 51, 77, 89, 90, 116),
      Array(0, 6, 7, 12, 13, 20, 21, 28))
    val blocksWithOnlyDiagonalEntriesWithinRadius = Array(Array(0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143), 
      Array(14, 35))
    
    for (i <- Seq(0, 1)) {
      val nRows = t.count().toInt
      val bm = BlockMatrix.fromBreezeMatrix(sc, BDM.zeros(nRows, nRows), blockSize = blockSizes(i))
      val blocks = UpperIndexBounds.computeCoverByUpperTriangularBlocks(t.keyBy("contig"), 
        bm.gp, radius = 10, includeDiagonal = false)
      val blocksWithDiagonal = UpperIndexBounds.computeCoverByUpperTriangularBlocks(t.keyBy("contig"), 
        bm.gp, radius=10, includeDiagonal = true)
      assert(blocks sameElements expecteds(i))
      assert(blocksWithDiagonal.sorted sameElements (expecteds(i) ++ blocksWithOnlyDiagonalEntriesWithinRadius(i)).sorted)
    }
  }
}
