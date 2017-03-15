package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class ArrayBuilderSuite extends SparkSuite {
  @Test def addOneElement() {
    val ab = new ArrayBuilder[Int](0)
    ab += 3
    val a = ab.result()
    assert(a.size == 1)
    assert(a(0) == 3)
  }
}
