package is.hail.io

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class ByteArrayReaderSuite extends TestNGSuite {
  @Test
  def readLongReadsALong(): Unit = {
    val a = Array(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff)
      .map(_.toByte)
    assert(new ByteArrayReader(a).readLong() == -1L)
  }

  @Test
  def readLongReadsALong2(): Unit = {
    val a = Array(0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8)
      .map(_.toByte)
    assert(new ByteArrayReader(a).readLong() == 0xf8f7f6f5f4f3f2f1L)
  }

  @Test
  def readLongReadsALong3(): Unit = {
    val a = Array(0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1)
      .map(_.toByte)
    assert(new ByteArrayReader(a).readLong() == 0xf1f2f3f4f5f6f7f8L)
  }
}
