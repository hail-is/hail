package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test

class MultiArray2Suite extends SparkSuite{
  @Test def test() = {
    val ma1 = MultiArray2.fill[Int](10,3)(0)
    for ((i,j) <- ma1.indices) {
      ma1.update(i,j,i*j)
    }
    assert(ma1(2,2) == 4)
    assert(ma1(6,1) == 6)

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

    for (row <- ma5.rows) {
      assert(row.data == Iterable((row.i*0,"foo"),(row.i*1,"foo"),(row.i*2,"foo")))
    }

    for (column <- ma5.columns) {
      assert(column.data == Iterable((column.j*0,"foo"),(column.j*1,"foo"),(column.j*2,"foo"),(column.j*3,"foo"),
        (column.j*4,"foo"),(column.j*5,"foo"),(column.j*6,"foo"),(column.j*7,"foo"),(column.j*8,"foo"),
        (column.j*9,"foo")))
    }
  }
}
