package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test

class MultiArray2Suite extends SparkSuite{
  @Test def test() = {

    // test multiarray of size 0 will be created
    val ma0 = MultiArray2.fill[Int](0, 0)(0)

    // test multiarray of size 0 that get nothing out
    intercept[IllegalArgumentException] {
      val ma0 = MultiArray2.fill[Int](0, 0)(0)
      ma0(0,0)
    }

    // test array index out of bounds on row slice
    intercept[ArrayIndexOutOfBoundsException] {
      val foo = MultiArray2.fill[Int](5, 5)(0)
      foo.rowSlice(0)(5)
    }

    // bad multiarray initiation -- negative number
    intercept[IllegalArgumentException] {
      val a = MultiArray2.fill[Int](-5,5)(0)
    }

    // bad multiarray initiation -- negative number
    intercept[IllegalArgumentException] {
      val a = MultiArray2.fill[Int](5,-5)(0)
    }

    val ma1 = MultiArray2.fill[Int](10,3)(0)
    for ((i,j) <- ma1.indices) {
      ma1.update(i,j,i*j)
    }
    assert(ma1(2,2) == 4)
    assert(ma1(6,1) == 6)

    // Catch exception if try to get value that is not in indices of multiarray
    intercept[IllegalArgumentException] {
      val foo = ma1(100,100)
    }

    val ma2 = MultiArray2.fill[Int](10,3)(0)
    for ((i,j) <- ma2.indices) {
      ma2.update(i,j,i+j)
    }


    assert(ma2(2,2) == 4)
    assert(ma2(6,1) == 7)

    // Test zip with two ints
    val ma3 = ma1.zip(ma2)
    assert(ma3(2,2) == (4,4))
    assert(ma3(6,1) == (6,7))

    // Test zip with multi-arrays of different types
    val ma4 = MultiArray2.fill[String](10,3)("foo")
    val ma5 = ma1.zip(ma4)
    assert(ma5(2,2) == (4,"foo"))
    assert(ma5(0,0) == (0,"foo"))

    // Test row slice
    for (row <- ma5.rows; idx <- 0 until row.length) {
      assert(row(idx) == (row.i*idx,"foo"))
    }

    intercept[IllegalArgumentException] {
      val x = ma5.rowSlice(100)
    }

    intercept[ArrayIndexOutOfBoundsException] {
      val x = ma5.rowSlice(0)
      val y = x(100)
    }

    intercept[IllegalArgumentException] {
      val x = ma5.rowSlice(-5)
    }

    intercept[IllegalArgumentException] {
      val x = ma5.columnSlice(100)
    }

    intercept[IllegalArgumentException] {
      val x = ma5.columnSlice(-5)
    }

    intercept[ArrayIndexOutOfBoundsException] {
      val x = ma5.columnSlice(0)
      val y = x(100)
    }

    // Test column slice
    for (column <- ma5.columns; idx <- 0 until column.length) {
      assert(column(idx) == (column.j*idx,"foo"))
    }

  }
}
