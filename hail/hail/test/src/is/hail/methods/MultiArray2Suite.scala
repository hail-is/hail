package is.hail.methods

import is.hail.TestUtils._
import is.hail.utils.MultiArray2

import org.junit.jupiter.api.Test

class MultiArray2Suite {
  @Test def test(): Unit = {

    // test multiarray of size 0 will be created
    MultiArray2.fill[Int](0, 0)(0): Unit

    // test multiarray of size 0 that apply nothing out
    intercept[IllegalArgumentException] {
      val ma0 = MultiArray2.fill[Int](0, 0)(0)
      ma0(0, 0)
    }: Unit

    // test array index out of bounds on row slice
    intercept[ArrayIndexOutOfBoundsException] {
      val foo = MultiArray2.fill[Int](5, 5)(0)
      foo.row(0)(5)
    }: Unit

    // bad multiarray initiation -- negative number
    intercept[IllegalArgumentException] {
      MultiArray2.fill[Int](-5, 5)(0)
    }: Unit

    // bad multiarray initiation -- negative number
    intercept[IllegalArgumentException] {
      MultiArray2.fill[Int](5, -5)(0)
    }: Unit

    val ma1 = MultiArray2.fill[Int](10, 3)(0)
    for ((i, j) <- ma1.indices)
      ma1.update(i, j, i * j)
    assertEq(ma1(2, 2), 4)
    assertEq(ma1(6, 1), 6)

    // Catch exception if try to apply value that is not in indices of multiarray
    intercept[IllegalArgumentException] {
      ma1(100, 100)
    }: Unit

    val ma2 = MultiArray2.fill[Int](10, 3)(0)
    for ((i, j) <- ma2.indices)
      ma2.update(i, j, i + j)

    assertEq(ma2(2, 2), 4)
    assertEq(ma2(6, 1), 7)

    // Test zip with two ints
    val ma3 = ma1.zip(ma2)
    assertEq(ma3(2, 2), (4, 4))
    assertEq(ma3(6, 1), (6, 7))

    // Test zip with multi-arrays of different types
    val ma4 = MultiArray2.fill[String](10, 3)("foo")
    val ma5 = ma1.zip(ma4)
    assertEq(ma5(2, 2), (4, "foo"))
    assertEq(ma5(0, 0), (0, "foo"))

    // Test row slice
    for {
      row <- ma5.rows
      idx <- 0 until row.length
    }
      assertEq(row(idx), (row.i * idx, "foo"))

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

    // Test column slice
    for {
      column <- ma5.columns
      idx <- 0 until column.length
    }
      assertEq(column(idx), (column.j * idx, "foo"))

  }
}
