package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class ArrayBuilderSuite extends SparkSuite {
  @Test def addOneElement() {
    val ab = new ArrayBuilder[Int](0)
    ab += 3
    val a = ab.result()
    assert(a.length == 1)
    assert(a(0) == 3)
  }

  @Test def addArray() {
    val ab = new ArrayBuilder[Int](0)
    ab ++= Array.fill[Int](5)(2)
    val a = ab.result()
    assert(a.length == 5)
    assert(a.indices.forall(a(_) == 2))

    val ab2 = new ArrayBuilder[Int](0)
    ab2 ++= (Array.fill[Int](4)(3), 2)
    val a2 = ab2.result()
    assert(a2.length == 2)
    assert(a2.indices.forall(a2(_) == 3))

    ab2(0) = 5
    val ab2Update = ab2.result()
    ab2Update.zipWithIndex.forall { case (v, i) => if (i != 0) v == 3 else v == 5 }
  }
}
