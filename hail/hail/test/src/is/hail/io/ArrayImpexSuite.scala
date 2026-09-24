package is.hail.io

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext

import org.junit.jupiter.api.Test

class ArrayImpexSuite {
  @Test def testArrayImpex(implicit ctx: ExecuteContext): Unit = {
    val file = ctx.createTmpPath("test")
    val a = Array.fill[Double](100)(util.Random.nextDouble())
    val a2 = new Array[Double](100)

    ArrayImpex.exportToDoubles(ctx.fs, file, a, bufSize = 32)
    ArrayImpex.importFromDoubles(ctx.fs, file, a2, bufSize = 16)
    assert(java.util.Arrays.equals(a, a2))

    interceptFatal("Premature") {
      ArrayImpex.importFromDoubles(ctx.fs, file, new Array[Double](101), bufSize = 64)
    }
  }
}
