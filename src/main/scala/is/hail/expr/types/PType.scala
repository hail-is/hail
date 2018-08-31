package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s._

abstract class PType {
  def virtualType: Type

  def byteSize: Long

  def alignment: Long
}

case class PDefault(virtualType: Type) extends PType {
  def byteSize: Long = virtualType.byteSize

  def alignment: Long = virtualType.alignment
}

abstract class PStruct extends PType {

  override def virtualType: TBaseStruct

  def allocate(region: Region): Long

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean =
    isFieldDefined(rv.region, rv.offset, fieldIdx)

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(region, offset, fieldIdx)

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int): Unit

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit]

  def fieldOffset(offset: Long, fieldIdx: Int): Long

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long]

  def loadField(rv: RegionValue, fieldIdx: Int): Long = loadField(rv.region, rv.offset, fieldIdx)

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long]

  def fieldType(field: Int): PType
}

abstract class PConstructableStruct extends PStruct {
  def clearMissingBits(region: Region, off: Long): Unit

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit]

}

abstract class PArray extends PType {
  override def virtualType: TIterable

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int]

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] = !isElementDefined(region, aoff, i)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def elementOffset(aoff: Long, length: Int, i: Int): Long

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def elementType: PType
}

abstract class PConstructableArray extends PArray {
  def clearMissingBits(region: Region, aoff: Long, length: Int)

  def initialize(region: Region, aoff: Long, length: Int): Unit

  def initialize(region: Code[Region], aoff: Code[Long], length: Code[Int], a: Settable[Int]): Unit

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementMissing(region: Region, aoff: Long, i: Int): Unit
}

// TODO: the primitive system can be relaxed to include non-canonical primitives
abstract class PPrimitive(val byteSize: Long) extends PType {
  def alignment: Long = byteSize
}

case object PInt32 extends PPrimitive(4) {
  def virtualType: Type = TInt32()
}

case object PInt64 extends PPrimitive(8) {
  def virtualType: Type = TInt64()
}

case object PFloat32 extends PPrimitive(4) {
  def virtualType: Type = TFloat32()
}

case object PFloat64 extends PPrimitive(8) {
  def virtualType: Type = TFloat64()
}

case object PBool extends PPrimitive(1) {
  def virtualType: Type = TBoolean()
}

case object PBinary extends PType {
  def virtualType: Type = TBinary()

  val byteSize: Long = 8
  val alignment: Long = 8
}