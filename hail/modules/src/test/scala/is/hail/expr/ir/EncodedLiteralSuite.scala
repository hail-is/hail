package is.hail.expr.ir

import is.hail.HailSuite

import org.testng.annotations.Test

class EncodedLiteralSuite extends HailSuite {

  @Test
  def testWrappedByteArrayEquality(): Unit = {
    val byteArray1 = Array[Byte](1, 2, 1, 1)
    val byteArray2 = Array[Byte](1, 2, 1, 1)
    val byteArray3 = Array[Byte](0, 0, 1, 0)
    val wba1 = new WrappedByteArrays(Array(byteArray1))
    val wba2 = new WrappedByteArrays(Array(byteArray2))
    val wba3 = new WrappedByteArrays(Array(byteArray3))

    assert(wba1 == wba1)
    assert(wba1 == wba2)
    assert(wba1 != wba3)
  }
}
