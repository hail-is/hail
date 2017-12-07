package is.hail.distributedmatrix

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GridPartitionerSuite extends TestNGSuite {

  private def assertLayout(hg: GridPartitioner, layout: ((Int, Int), Int)*) {
    layout.foreach { case ((i, j), p) =>
      assert(hg.partitionIdFromBlockIndices(i, j) === p, s"at coordiantes ${(i,j)}")
    }
    layout.foreach { case ((i, j), p) =>
      assert(hg.blockCoordinates(p) === (i, j), s"at pid $p")
    }
  }

  @Test
  def squareIsColumnMajor() {
    assertLayout(GridPartitioner(2, 4, 4),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (0, 1) -> 2,
      (1, 1) -> 3
    )
  }

  @Test
  def rectangleMoreRowsIsColumnMajor() {
    assertLayout(GridPartitioner(2, 6, 4),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (2, 0) -> 2,
      (0, 1) -> 3,
      (1, 1) -> 4,
      (2, 1) -> 5
    )
  }

  @Test
  def rectangleMoreColsIsColumnMajor() {
    assertLayout(GridPartitioner(2, 4, 6),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (0, 1) -> 2,
      (1, 1) -> 3,
      (0, 2) -> 4,
      (1, 2) -> 5
    )
  }
}
