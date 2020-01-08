package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.utils._

object PBaseStruct {
  def getContentsByteSizeAndSetOffsets(types: Array[PType], nMissingBytes: Long, byteOffsets: Array[Long]): Long = {
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
  val types: Array[PType]

  val fields: IndexedSeq[PField]

  lazy val allFieldsRequired: Boolean = types.forall(_.required)

  lazy val fieldRequired: Array[Boolean] = types.map(_.required)

  lazy val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  lazy val fieldNames: Array[String] = fields.map(_.name).toArray

  def fieldByName(name: String): PField = fields(fieldIdx(name))

  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[PField] = fieldIdx.get(name).map(i => fields(i))

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def field(name: String): PField = fields(fieldIdx(name))

  def fieldType(name: String): PType = types(fieldIdx(name))

  def size: Int = fields.length

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  def identBase: String
  
  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append(identBase)
    sb.append("_of_")
    types.foreachBetween { ty =>
      sb.append(ty.asIdent)
    } {
      sb.append("AND")
    }
    sb.append("END")
    sb.result()
  }

  def codeOrdering(mb: EmitMethodBuilder, so: Array[SortOrder]): CodeOrdering =
    codeOrdering(mb, this, so)

  def codeOrdering(mb: EmitMethodBuilder, other: PType, so: Array[SortOrder]): CodeOrdering

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

  def nMissing: Int

  def nMissingBytes: Int

  def missingIdx: Array[Int]

  def byteOffsets: Array[Long]

  def allocate(region: Region): Long = {
    region.allocate(alignment, this.byteSize)
  }

  def allocate(region: Code[Region]): Code[Long] = region.allocate(alignment, this.byteSize)

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

  def copyFrom(region: Region, srcOff: Long): Long = {
    val destOff = allocate(region)
    Region.copyFrom(srcOff,  destOff, this.byteSize)
    destOff
  }

  def copyFrom(mb: MethodBuilder, region: Code[Region], srcOff: Code[Long]): Code[Long] = {
    val destOff = mb.newField[Long]
    Code(
      destOff := allocate(region),
      this.storeShallowAtOffset(srcOff,  destOff),
      destOff
    )
  }

  override def storeShallowAtOffset(destAddress: Code[Long], srcAddress: Code[Long]): Code[Unit] = {
    Region.copyFrom(srcAddress, destAddress, this.byteSize)
  }

  override def storeShallowAtOffset(destAddress: Long, srcAddress: Long) {
    Region.copyFrom(srcAddress, destAddress, this.byteSize)
  }

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long],
    allowDowncast: Boolean = false, forceDeep: Boolean = false): Code[Long] = {
    assert(srcPType.isInstanceOf[PBaseStruct])

    val sourceType = srcPType.asInstanceOf[PBaseStruct]

    assert(sourceType.size == this.size)

    val destTypes = this.types
    val srcTypes = sourceType.types

    if(this.types == sourceType.types) {
      if(!forceDeep || this.types.forall(_.isPrimitive)) {
        return this.copyFrom(mb, region, srcAddress)
      }
    }

    val dstAddress = mb.newField[Long]
    val numberOfElements = mb.newLocal[Int]
    val currentElementAddress = mb.newLocal[Long]
    val currentIdx = mb.newLocal[Int]

    var c: Code[_] = Code(
      dstAddress := this.allocate(region),
      this.stagedInitialize(dstAddress, numberOfElements),
      currentElementAddress := this.firstElementOffset(dstAddress, numberOfElements),
      currentIdx := const(0)
    )

    var loop: Code[Unit] = if (!sourceType.elementType.isPrimitive) {
      // recurse
      Region.storeAddress(
        currentElementAddress,
        this.elementType.copyFromType(
          mb,
          region,
          sourceType.elementType,
          sourceType.loadElementAddress(srcAddress, numberOfElements, currentIdx),
          allowDowncast,
          forceDeep
        )
      )
    } else {
      sourceType.elementType.storeShallowAtOffset(
        currentElementAddress,
        sourceType.loadElement(region, srcAddress, numberOfElements, currentIdx)
      )
    }

    if(!sourceType.elementType.required && this.elementType.required) {
      if (!allowDowncast) {
        return Code._fatal("Downcast isn't allowed and source elementType isn't required")
      }

      c = Code(sourceType.hasMissingValues(srcAddress).orEmpty(
        Code._fatal("Found missing values. Cannot copy to type whose elements are required.")
      ), c)
    } else if(!this.elementType.required) {
      loop = sourceType.isElementMissing(srcAddress, currentIdx).mux(
        this.setElementMissing(dstAddress, currentIdx),
        loop
      )
    }

    Code(
      c,
      Code.whileLoop(currentIdx < numberOfElements,
        loop,
        currentElementAddress := this.nextElementAddress(currentElementAddress),
        currentIdx := currentIdx + const(1)
      ),
      dstAddress
    )
  }


  override def containsPointers: Boolean = types.exists(_.containsPointers)
}
