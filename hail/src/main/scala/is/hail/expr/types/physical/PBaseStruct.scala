package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.utils._

trait PBaseStruct extends PType {
  def types: Array[PType]

  def fields: IndexedSeq[PField]

  def fieldRequired: Array[Boolean]

  def fieldIdx: Map[String, Int]

  def fieldNames: Array[String]

  def fieldByName(name: String): PField

  def index(str: String): Option[Int]

  def selfField(name: String): Option[PField]

  def hasField(name: String): Boolean

  def field(name: String): PField

  def fieldType(name: String): PType

  def size: Int

  def isIsomorphicTo(other: PBaseStruct): Boolean

  def isPrefixOf(other: PBaseStruct): Boolean

  def isCompatibleWith(other: PBaseStruct): Boolean

  def truncate(newSize: Int): PBaseStruct

  def nMissingBytes: Int

  protected def missingIdx: Array[Int]

  def byteOffsets: Array[Long]

  def allocate(region: Region): Long

  def allocate(region: Code[Region]): Code[Long]

  def setAllMissing(off: Code[Long]): Code[Unit]

  def clearMissingBits(region: Region, off: Long): Unit

  def clearMissingBits(off: Code[Long]): Code[Unit]

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit]

  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean

  def isFieldDefined(offset: Long, fieldIdx: Int): Boolean

  def isFieldMissing(off: Long, fieldIdx: Int): Boolean

  def isFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def isFieldDefined(offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int): Unit

  def setFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Unit]

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit]

  def setFieldPresent(region: Region, offset: Long, fieldIdx: Int): Unit

  def setFieldPresent(offset: Code[Long], fieldIdx: Int): Code[Unit]

  def setFieldPresent(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit]

  def fieldOffset(offset: Long, fieldIdx: Int): Long

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long]

  def loadField(rv: RegionValue, fieldIdx: Int): Long

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long

  def loadField(offset: Long, fieldIdx: Int): Long

  def loadField(offset: Code[Long], fieldIdx: Int): Code[Long]

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long]
}