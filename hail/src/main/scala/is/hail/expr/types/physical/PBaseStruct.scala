package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.utils._

object PBaseStruct {
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
    require(this == rightType)

    val right = rightType.asInstanceOf[PBaseStruct]
    val fieldOrderings: Array[UnsafeOrdering] =
      types.zip(right.types).map { case (l, r) => l.unsafeOrdering(r)}

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < types.length) {
          val leftDefined = isFieldDefined(o1, i)
          val rightDefined = right.isFieldDefined(o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(o1, i), r2, right.loadField(o2, i))
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
    region.allocate(alignment, byteSize)
  }

  def allocate(region: Code[Region]): Code[Long] =
    region.allocate(alignment, byteSize)

  def initialize(structAddress: Long, setMissing: Boolean = false): Unit = {
    if (allFieldsRequired) {
      return
    }

    Region.setMemory(structAddress, nMissingBytes.toLong, if (setMissing) 0xFF.toByte else 0.toByte)
  }

  def stagedInitialize(structAddress: Code[Long], setMissing: Boolean = false): Code[Unit] = {
    if (allFieldsRequired) {
      return Code._empty
    }

    Region.setMemory(structAddress, const(nMissingBytes.toLong), const(if (setMissing) 0xFF.toByte else 0.toByte))
  }

  def isFieldDefined(offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !Region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldMissing(off: Long, fieldIdx: Int): Boolean = !isFieldDefined(off, fieldIdx)

  def isFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      Region.loadBit(offset, missingIdx(fieldIdx).toLong)

  def isFieldDefined(offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(offset, fieldIdx)

  def setFieldMissing(offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx).toLong)
  }

  def setFieldPresent(offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.clearBit(offset, missingIdx(fieldIdx))
  }

  def setFieldPresent(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    Region.clearBit(offset, missingIdx(fieldIdx).toLong)
  }

  def fieldOffset(structAddress: Long, fieldIdx: Int): Long =
    structAddress + byteOffsets(fieldIdx)

  def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    structAddress + byteOffsets(fieldIdx)

  def loadField(offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    types(fieldIdx).fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadField(offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(fieldOffset(offset, fieldIdx), types(fieldIdx))

  private def loadField(fieldOffset: Code[Long], fieldType: PType): Code[Long] = {
    fieldType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(fieldOffset)
      case _ => fieldOffset
    }
  }

  def deepPointerCopy(mb: MethodBuilder, region: Code[Region], dstStructAddress: Code[Long]): Code[Unit] = {
    var c: Code[Unit] = Code._empty

    var i = 0
    while(i < this.size) {
      val dstFieldType = this.fields(i).typ.fundamentalType
      if(dstFieldType.containsPointers) {
        val dstFieldAddress = mb.newField[Long]
        c = Code(
          c,
          this.isFieldDefined(dstStructAddress, i).orEmpty(
            Code(
              dstFieldAddress := this.fieldOffset(dstStructAddress, i),
              dstFieldType match {
                case t@(_: PBinary | _: PArray) =>
                  Region.storeAddress(dstFieldAddress, t.copyFromType(mb, region, dstFieldType, Region.loadAddress(dstFieldAddress), forceDeep = true))
                case t: PBaseStruct =>
                  t.deepPointerCopy(mb, region, dstFieldAddress)
                case t: PType =>
                  fatal(s"Field type isn't supported ${t}")
              }
            )
          )
        )
      }
      i += 1
    }

    c
  }

  def deepPointerCopy(region: Region, dstStructAddress: Long) {
    var i = 0
    while(i < this.size) {
      val dstFieldType = this.fields(i).typ.fundamentalType
      if(dstFieldType.containsPointers && this.isFieldDefined(dstStructAddress, i)) {
        val dstFieldAddress = this.fieldOffset(dstStructAddress, i)
        dstFieldType match {
          case t@(_: PBinary | _: PArray) =>
            Region.storeAddress(dstFieldAddress, t.copyFromType(region, dstFieldType, Region.loadAddress(dstFieldAddress), forceDeep = true))
          case t: PBaseStruct =>
            t.deepPointerCopy(region, dstFieldAddress)
          case t: PType =>
            fatal(s"Field type isn't supported ${t}")
        }
      }
      i += 1
    }
  }

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcStructAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    val sourceType = srcPType.asInstanceOf[PBaseStruct]
    assert(sourceType.size == this.size)

    if (this == sourceType && !forceDeep)
      srcStructAddress
    else {
      val addr = mb.newLocal[Long]
          Code(
            addr := allocate(region),
            constructAtAddress(mb, addr, region, sourceType, srcStructAddress, forceDeep),
            addr
          )
    }
  }

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcStructAddress: Long, forceDeep: Boolean): Long = {
    val sourceType = srcPType.asInstanceOf[PBaseStruct]
    if (this == sourceType && !forceDeep)
      srcStructAddress
    else {
      val newAddr = allocate(region)
      constructAtAddress(newAddr, region, sourceType, srcStructAddress, forceDeep)
      newAddr
    }
  }

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] = {
    val srcStruct = srcPType.asInstanceOf[PBaseStruct]
    val addrVar = mb.newLocal[Long]
    if (srcStruct == this) {
      var c: Code[Unit] = Code(
        addrVar := addr,
        Region.copyFrom(srcAddress, addrVar, byteSize))
      if (forceDeep) {
        c = Code(c, deepPointerCopy(mb, region, addrVar))
      }
      c
    } else {
      val srcAddrVar = mb.newLocal[Long]

      Code(
        srcAddrVar := srcAddress,
        addrVar := addr,
        stagedInitialize(addrVar, setMissing = true),
        Code(fields.zip(srcStruct.fields).map { case (dest, src) =>
          assert(dest.typ.required <= src.typ.required)
          val idx = dest.index
          assert(idx == src.index)
          srcStruct.isFieldDefined(srcAddrVar, idx).orEmpty(
            dest.typ.constructAtAddress(mb, fieldOffset(addrVar, idx), region, src.typ, srcStruct.loadField(srcAddrVar, idx), forceDeep))
        }: _*
      ).asInstanceOf[Code[Unit]])
    }
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit = {
    val srcStruct = srcPType.asInstanceOf[PBaseStruct]
    if (srcStruct == this) {
      Region.copyFrom(srcAddress, addr, byteSize)
      if (forceDeep)
        deepPointerCopy(region, addr)
    } else {
      initialize(addr, setMissing = true)
      var idx = 0
      while (idx < types.length) {
        idx += 1
        val dest = types(idx)
        val src = srcStruct.types(idx)
        assert(dest.required <= src.required)

        if (srcStruct.isFieldDefined(srcAddress, idx))
          dest.constructAtAddress(fieldOffset(addr, idx), region, src, srcStruct.loadField(srcAddress, idx), forceDeep)
      }
    }
  }

  override def containsPointers: Boolean = types.exists(_.containsPointers)
}
