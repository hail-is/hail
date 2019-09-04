package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.check.Gen
import is.hail.cxx
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, classTag}

object PBaseStruct {
  def getMissingness(types: Array[PType], missingIdx: Array[Int]): Int = {
    assert(missingIdx.length == types.length)
    var i = 0
    types.zipWithIndex.foreach { case (t, idx) =>
      missingIdx(idx) = i
      if (!t.required)
        i += 1
    }
    i
  }

  def getByteSizeAndOffsets(types: Array[PType], nMissingBytes: Long, byteOffsets: Array[Long]): Long = {
    assert(byteOffsets.length == types.length)
    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    types.zipWithIndex.foreach { case (t, i) =>
      val fSize = t.byteSize
      val fAlignment = t.alignment

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          byteOffsets(i) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          byteOffsets(i) = offset
          offset += fSize
      }
    }
    offset
  }

  def alignment(types: Array[PType]): Long = {
    if (types.isEmpty)
      1
    else
      types.map(_.alignment).max
  }
}

abstract class PBaseStruct extends PType {
  def types: Array[PType]

  def fields: IndexedSeq[PField]

  def fieldRequired: Array[Boolean]

  def size: Int

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  def isIsomorphicTo(other: PBaseStruct): Boolean =
    size == other.size && isCompatibleWith(other)

  def isPrefixOf(other: PBaseStruct): Boolean =
    size <= other.size && isCompatibleWith(other)

  def isCompatibleWith(other: PBaseStruct): Boolean =
    fields.zip(other.fields).forall{ case (l, r) => l.typ isOfType r.typ }

  def truncate(newSize: Int): PBaseStruct

  override def unsafeOrdering(): UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    require(this.isOfType(rightType))

    val right = rightType.asInstanceOf[PBaseStruct]
    val fieldOrderings: Array[UnsafeOrdering] =
      types.zip(right.types).map { case (l, r) => l.unsafeOrdering(r)}

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < types.length) {
          val leftDefined = isFieldDefined(r1, o1, i)
          val rightDefined = right.isFieldDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(r1, o1, i), r2, right.loadField(r2, o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }
          i += 1
        }
        0
      }
    }
  }

  def nMissingBytes: Int

  def missingIdx: Array[Int]

  def byteOffsets: Array[Long]

  def allocate(region: Region): Long = {
    region.allocate(alignment, byteSize)
  }

  def allocate(region: Code[Region]): Code[Long] = region.allocate(alignment, byteSize)

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
      region.storeByte(off + i, 0)
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
    fieldRequired(fieldIdx) || !region.loadBit(offset, missingIdx(fieldIdx))

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
    region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit] =
    setFieldMissing(offset, fieldIdx)

  def setFieldPresent(region: Region, offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    region.clearBit(offset, missingIdx(fieldIdx))
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

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    types(fieldIdx).fundamentalType match {
      case _: PArray | _: PBinary => region.loadAddress(off)
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

  override def containsPointers: Boolean = types.exists(_.containsPointers)

  def cxxIsFieldMissing(o: cxx.Code, fieldIdx: Int): cxx.Code = {
    s"load_bit($o, ${ missingIdx(fieldIdx) })"
  }

  def cxxLoadField(o: cxx.Code, fieldIdx: Int): cxx.Code = {
    val a = s"(((char *)$o) + (${ byteOffsets(fieldIdx) }))"
    cxx.loadIRIntermediate(fields(fieldIdx).typ, a)
  }
}
