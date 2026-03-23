package is.hail.io

import is.hail.HailSuite

class ArrayImpexSuite extends HailSuite {
  test("ArrayImpex") {
    val file = ctx.createTmpPath("test")
    val a = Array.fill[Double](100)(util.Random.nextDouble())
    val a2 = new Array[Double](100)

    ArrayImpex.exportToDoubles(fs, file, a, bufSize = 32)
    ArrayImpex.importFromDoubles(fs, file, a2, bufSize = 16)
    assertEquals(a.toSeq, a2.toSeq)

    interceptFatal("Premature") {
      ArrayImpex.importFromDoubles(fs, file, new Array[Double](101), bufSize = 64)
    }
  }
}
