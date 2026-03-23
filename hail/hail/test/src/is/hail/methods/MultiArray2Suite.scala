package is.hail.methods

import is.hail.utils.MultiArray2

class MultiArray2Suite extends munit.FunSuite {
  test("fill with size 0") {
    MultiArray2.fill[Int](0, 0)(0): Unit
  }

  test("apply on size 0 throws") {
    intercept[IllegalArgumentException] {
      val ma0 = MultiArray2.fill[Int](0, 0)(0)
      ma0(0, 0)
    }: Unit
  }

  test("row slice out of bounds") {
    intercept[ArrayIndexOutOfBoundsException] {
      val foo = MultiArray2.fill[Int](5, 5)(0)
      foo.row(0)(5)
    }: Unit
  }

  test("negative row count") {
    intercept[IllegalArgumentException] {
      MultiArray2.fill[Int](-5, 5)(0)
    }: Unit
  }

  test("negative column count") {
    intercept[IllegalArgumentException] {
      MultiArray2.fill[Int](5, -5)(0)
    }: Unit
  }

  test("update and apply") {
    val ma1 = MultiArray2.fill[Int](10, 3)(0)
    for ((i, j) <- ma1.indices)
      ma1.update(i, j, i * j)
    assertEquals(ma1(2, 2), 4)
    assertEquals(ma1(6, 1), 6)

    intercept[IllegalArgumentException] {
      ma1(100, 100)
    }: Unit

    val ma2 = MultiArray2.fill[Int](10, 3)(0)
    for ((i, j) <- ma2.indices)
      ma2.update(i, j, i + j)

    assertEquals(ma2(2, 2), 4)
    assertEquals(ma2(6, 1), 7)

    // zip two int arrays
    val ma3 = ma1.zip(ma2)
    assertEquals(ma3(2, 2), (4, 4))
    assertEquals(ma3(6, 1), (6, 7))

    // zip arrays of different types
    val ma4 = MultiArray2.fill[String](10, 3)("foo")
    val ma5 = ma1.zip(ma4)
    assertEquals(ma5(2, 2), (4, "foo"))
    assertEquals(ma5(0, 0), (0, "foo"))

    // row slice
    for {
      row <- ma5.rows
      idx <- 0 until row.length
    }
      assertEquals(row(idx), (row.i * idx, "foo"))

    intercept[IllegalArgumentException] {
      ma5.row(100)
    }: Unit

    intercept[ArrayIndexOutOfBoundsException] {
      val x = ma5.row(0)
      x(100)
    }: Unit

    intercept[IllegalArgumentException] {
      ma5.row(-5)
    }: Unit

    intercept[IllegalArgumentException] {
      ma5.column(100)
    }: Unit

    intercept[IllegalArgumentException] {
      ma5.column(-5)
    }: Unit

    intercept[ArrayIndexOutOfBoundsException] {
      val x = ma5.column(0)
      x(100)
    }: Unit

    // column slice
    for {
      column <- ma5.columns
      idx <- 0 until column.length
    }
      assertEquals(column(idx), (column.j * idx, "foo"))
  }
}
