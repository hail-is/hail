package is.hail.shuffler

import is.hail.annotations._
import is.hail.types.virtual._
import is.hail.types.physical._

object ShuffleTestUtils {
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
}

