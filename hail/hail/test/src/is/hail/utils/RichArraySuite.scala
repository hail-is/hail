package is.hail.utils

import is.hail.HailSuite
import is.hail.utils.richUtils.RichArray

import org.scalatest
import org.testng.annotations.Test

class RichArraySuite extends HailSuite {
  @Test def testArrayImpex(): scalatest.Assertion = {
    val file = ctx.createTmpPath("test")
    val a = Array.fill[Double](100)(util.Random.nextDouble())
    val a2 = new Array[Double](100)

    RichArray.exportToDoubles(fs, file, a, bufSize = 32)
    RichArray.importFromDoubles(fs, file, a2, bufSize = 16)
    assert(a === a2)

    interceptFatal("Premature") {
      RichArray.importFromDoubles(fs, file, new Array[Double](101), bufSize = 64)
    }
  }
}
