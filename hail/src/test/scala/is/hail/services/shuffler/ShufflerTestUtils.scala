package is.hail.services.shuffler

import is.hail.annotations._
import is.hail.types.virtual._
import is.hail.types.physical._

object ShufflerTestUtils {
  private[this] lazy val region = Region()
  private[this] lazy val rvb = new RegionValueBuilder(region)

  def arrayOfUnsafeRow(elementPType: PStruct, array: Array[Long]): Array[UnsafeRow] =
    array.map(new UnsafeRow(elementPType, null, _)).toArray

  val structIntPType = PCanonicalStruct("x" -> PInt32())

  def struct(x: Int): Long = {
    rvb.start(structIntPType)
    rvb.startStruct()
    rvb.addInt(x)
    rvb.endStruct()
    rvb.end()
  }

  val structIntStringPType = PCanonicalStruct("x" -> PInt32(), "y" -> PCanonicalString())

  def struct(x: Int, y: String): Long = {
    rvb.start(structIntStringPType)
    rvb.startStruct()
    rvb.addInt(x)
    rvb.addString(y)
    rvb.endStruct()
    rvb.end()
  }

  def assertStrictlyIncreasingPrefix(
    ord: UnsafeOrdering,
    values: Array[UnsafeRow],
    prefixLength: Int
  ): Unit = {
    if (!(prefixLength <= values.length)) {
      throw new AssertionError(s"$prefixLength <= ${values.length}")
    }

    if (values.length <= 1) {
      return
    }

    var prev = values(0)
    var i = 1
    while (i < prefixLength) {
      assert(ord.lt(prev.offset, values(i).offset),
        s"""values are not strictly increasing on [0, $prefixLength). We saw
           |${prev} and ${values(i)} at $i. Context: ${values.slice(i-3, i+3).toIndexedSeq}
           |""".stripMargin)
      prev = values(i)
      i += 1
    }
  }
}
