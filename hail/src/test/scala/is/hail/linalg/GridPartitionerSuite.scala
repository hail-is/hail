package is.hail.linalg

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GridPartitionerSuite extends TestNGSuite {

  private def assertLayout(hg: GridPartitioner, layout: ((Int, Int), Int)*) {
    layout.foreach { case ((i, j), p) =>
      assert(hg.coordinatesBlock(i, j) === p, s"at coordinates ${(i, j)}")
    }
    layout.foreach { case ((i, j), p) =>
      assert(hg.blockCoordinates(p) === ((i, j)), s"at pid $p")
    }
  }

  @Test
  def squareIsColumnMajor() {
    assertLayout(GridPartitioner(2, 4, 4), (0, 0) -> 0, (1, 0) -> 1, (0, 1) -> 2, (1, 1) -> 3)
  }

  @Test
  def rectangleMoreRowsIsColumnMajor() {
    assertLayout(
      GridPartitioner(2, 6, 4),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (2, 0) -> 2,
      (0, 1) -> 3,
      (1, 1) -> 4,
      (2, 1) -> 5,
    )
  }

  @Test
  def rectangleMoreColsIsColumnMajor() {
    assertLayout(
      GridPartitioner(2, 4, 6),
      (0, 0) -> 0,
      (1, 0) -> 1,
      (0, 1) -> 2,
      (1, 1) -> 3,
      (0, 2) -> 4,
      (1, 2) -> 5,
    )
  }

  @Test
  def bandedBlocksTest() {
    // 0  3  6  9
    // 1  4  7 10
    // 2  5  8 11
    val gp1 = GridPartitioner(10, 30, 40)
    val gp2 = GridPartitioner(10, 21, 31)

    for (gp <- Seq(gp1, gp2)) {
      assert(gp.bandBlocks(0, 0) sameElements Array(0, 4, 8))

      assert(gp.bandBlocks(-1, 0) sameElements Array(0, 1, 4, 5, 8))
      assert(gp.bandBlocks(-1, 0) sameElements gp.bandBlocks(-10, 0))

      assert(gp.bandBlocks(0, 1) sameElements Array(0, 3, 4, 7, 8, 11))
      assert(gp.bandBlocks(0, 1) sameElements gp.bandBlocks(0, 10))

      assert(gp.bandBlocks(-1, 1) sameElements Array(0, 1, 3, 4, 5, 7, 8, 11))
      assert(gp.bandBlocks(-1, 1) sameElements gp.bandBlocks(-10, 10))

      assert(gp.bandBlocks(-11, 0) sameElements Array(0, 1, 2, 4, 5, 8))

      assert(gp.bandBlocks(0, 11) sameElements Array(0, 3, 4, 6, 7, 8, 10, 11))
      assert(gp.bandBlocks(0, 20) sameElements gp.bandBlocks(0, 11))
      assert(gp.bandBlocks(0, 21) sameElements Array(0, 3, 4, 6, 7, 8, 9, 10, 11))

      assert(gp.bandBlocks(-1000, 1000) sameElements (0 until 12))
    }
  }

  @Test
  def rectangularBlocksTest() {
    // 0  3  6  9
    // 1  4  7 10
    // 2  5  8 11
    val gp1 = GridPartitioner(10, 30, 40)
    val gp2 = GridPartitioner(10, 21, 31)

    for (gp <- Seq(gp1, gp2)) {
      assert(gp.rectangleBlocks(Array(0, 1, 0, 1)) sameElements Array(0))
      assert(gp.rectanglesBlocks(Array(Array(0, 1, 0, 1))) sameElements Array(0))

      assert(gp.rectangleBlocks(Array(0, 10, 0, 10)) sameElements Array(0))

      assert(gp.rectangleBlocks(Array(9, 11, 9, 11)) sameElements Array(0, 1, 3, 4))
      assert(gp.rectanglesBlocks(Array(Array(9, 11, 9, 11))) sameElements Array(0, 1, 3, 4))

      assert(gp.rectangleBlocks(Array(10, 20, 10, 30)) sameElements Array(4, 7))

      assert(gp.rectanglesBlocks(Array(
        Array(9, 11, 9, 11),
        Array(10, 20, 10, 30),
        Array(0, 1, 20, 21),
        Array(20, 21, 20, 31),
      ))
        sameElements Array(0, 1, 3, 4, 6, 7, 8, 11))

      assert(gp.rectangleBlocks(Array(0, 21, 0, 31)) sameElements (0 until 12))
    }
  }
}
