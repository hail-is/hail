package is.hail.utils

import is.hail.expr.ir.IntArrayBuilder

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class ArrayBuilderSuite extends TestNGSuite {
  @Test def addOneElement(): Unit = {
    val ab = new IntArrayBuilder(0)
    ab += 3
    val a = ab.result()
    assert(a.length == 1)
    assert(a(0) == 3)
  }

  @Test def addArray(): Unit = {
    val ab = new IntArrayBuilder(0)
    ab ++= Array.fill[Int](5)(2)
    val a = ab.result()
    assert(a.length == 5)
    assert(a.forall(_ == 2))

    val ab2 = new IntArrayBuilder(0)
    ab2 ++= (Array.fill[Int](4)(3), 2)
    val a2 = ab2.result()
    assert(a2.length == 2)
    assert(a2.forall(_ == 3))

    ab2(0) = 5
    val ab2Update = ab2.result()
    assert(ab2Update sameElements Array(5, 3))

    val ab3 = new IntArrayBuilder
    ab3 += 1
    ab3 += 5
    ab3 ++= Array.fill[Int](2)(3)
    ab3 += 4
    ab3 += 6

    assert(ab3.result() sameElements Array(1, 5, 3, 3, 4, 6))

    ab3(2) = 20
    assert(ab3.result() sameElements Array(1, 5, 20, 3, 4, 6))
  }
}
