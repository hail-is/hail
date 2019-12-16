package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.utils._

trait PBaseStruct extends PType {
  def types: Array[PType]

  def fields: IndexedSeq[PField]

  def fieldRequired: Array[Boolean]

  lazy val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  def fieldNames: Array[String]

  def fieldByName(name: String): PField = fields(fieldIdx(name))

  def index(str: String): Option[Int]

  def selfField(name: String): Option[PField]

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def field(name: String): PField = fields(fieldIdx(name))

  def fieldType(name: String): PType = types(fieldIdx(name))

  lazy val size: Int = fields.length

  def isIsomorphicTo(other: PBaseStruct): Boolean =
    size == other.size && isCompatibleWith(other)

  def isPrefixOf(other: PBaseStruct): Boolean =
    size <= other.size && isCompatibleWith(other)

  def isCompatibleWith(other: PBaseStruct): Boolean =
    fields.zip(other.fields).forall{ case (l, r) => l.typ isOfType r.typ }

  def truncate(newSize: Int): PBaseStruct

  def nMissingBytes: Int

  protected def missingIdx: Array[Int]

  def byteOffsets: Array[Long]

  def allocate(region: Region): Long

  def allocate(region: Code[Region]): Code[Long]

  def setAllMissing(off: Code[Long]): Code[Unit] = {
    var c: Code[Unit] = Code._empty
    var i = 0
    while (i < nMissingBytes) {
      c = Code(c, Region.storeByte(off + i.toLong, const(0xFF.toByte)))
      i += 1
    }
    c
  }

  def clearMissingBits(region: Region, off: Long) {
    var i = 0
    while (i < nMissingBytes) {
      Region.storeByte(off + i, 0.toByte)
      i += 1
    }
  }

  def clearMissingBits(off: Code[Long]): Code[Unit] = {
    var c: Code[Unit] = Code._empty
    var i = 0
    while (i < nMissingBytes) {
      c = Code(c, Region.storeByte(off + i.toLong, const(0)))
      i += 1
    }
    c
  }

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit] =
    clearMissingBits(off)

  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean =
    isFieldDefined(rv.region, rv.offset, fieldIdx)

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !Region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldDefined(offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !Region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldMissing(off: Long, fieldIdx: Int): Boolean = !isFieldDefined(off, fieldIdx)

  def isFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      Region.loadBit(offset, missingIdx(fieldIdx).toLong)

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    isFieldMissing(offset, fieldIdx)

  def isFieldDefined(offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(offset, fieldIdx)

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    isFieldDefined(offset, fieldIdx)

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx).toLong)
  }

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit] =
    setFieldMissing(offset, fieldIdx)

  def setFieldPresent(region: Region, offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.clearBit(offset, missingIdx(fieldIdx))
  }

  def setFieldPresent(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    Region.clearBit(offset, missingIdx(fieldIdx))
  }

  def setFieldPresent(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit] =
    setFieldPresent(offset, fieldIdx)

  def fieldOffset(offset: Long, fieldIdx: Int): Long =
    offset + byteOffsets(fieldIdx)

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long] =
    offset + byteOffsets(fieldIdx)

  def loadField(rv: RegionValue, fieldIdx: Int): Long = loadField(rv.region, rv.offset, fieldIdx)

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long = loadField(offset, fieldIdx)

  def loadField(offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    types(fieldIdx).fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadField(offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(fieldOffset(offset, fieldIdx), types(fieldIdx))

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(fieldOffset(offset, fieldIdx), types(fieldIdx))

  private def loadField(fieldOffset: Code[Long], fieldType: PType): Code[Long] = {
    fieldType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(fieldOffset)
      case _ => fieldOffset
    }
  }
}