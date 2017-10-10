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
  def untransposedSquareIsColumnMajor() {
    assertLayout(GridPartitioner(4, 4, 2),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (0, 1) -> 2,
      (1, 1) -> 3
    )
  }

  @Test
  def untransposedRectangleMoreRowsIsColumnMajor() {
    assertLayout(GridPartitioner(6, 4, 2),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (2, 0) -> 2,
      (0, 1) -> 3,
      (1, 1) -> 4,
      (2, 1) -> 5
    )
  }

  @Test
  def untransposedRectangleMoreColsIsColumnMajor() {
    assertLayout(GridPartitioner(4, 6, 2),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (0, 1) -> 2,
      (1, 1) -> 3,
      (0, 2) -> 4,
      (1, 2) -> 5
    )
  }

  @Test
  def transposedSquare() {
    assertLayout(GridPartitioner(4, 4, 2, transposed=true),
      (0, 0) -> 0,
      (1, 0) -> 2,
      (0, 1) -> 1,
      (1, 1) -> 3
    )
  }

  @Test
  def transposedRectangleMoreRows() {
    // nb: physical layout is transposed and column major, so:
    //   0 2 4
    //   1 3 5
    assertLayout(GridPartitioner(6, 4, 2, transposed=true),
      (0, 0) -> 0,
      (1, 0) -> 2,
      (2, 0) -> 4,
      (0, 1) -> 1,
      (1, 1) -> 3,
      (2, 1) -> 5
    )
  }

  @Test
  def transposedRectangleMoreCols() {
    // nb: physical layout is transposed and column major, so:
    //   0 3
    //   1 4
    //   2 5
    assertLayout(GridPartitioner(4, 6, 2, transposed=true),
      (0, 0) -> 0,
      (1, 0) -> 3,
      (0, 1) -> 1,
      (1, 1) -> 4,
      (0, 2) -> 2,
      (1, 2) -> 5
    )
  }

}
