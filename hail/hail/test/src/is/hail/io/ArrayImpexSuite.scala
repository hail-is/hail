package is.hail.io

import is.hail.HailSuite

import org.testng.annotations.Test

class ArrayImpexSuite extends HailSuite {
  @Test def testArrayImpex(): Unit = {
    val file = ctx.createTmpPath("test")
    val a = Array.fill[Double](100)(util.Random.nextDouble())
    val a2 = new Array[Double](100)

    ArrayImpex.exportToDoubles(fs, file, a, bufSize = 32)
    ArrayImpex.importFromDoubles(fs, file, a2, bufSize = 16)
    assert(a === a2)

    interceptFatal("Premature") {
      ArrayImpex.importFromDoubles(fs, file, new Array[Double](101), bufSize = 64)
    }
  }
}
