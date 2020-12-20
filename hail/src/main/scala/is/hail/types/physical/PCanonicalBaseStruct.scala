package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.BaseStruct
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerCode, SBaseStructPointerSettable}
import is.hail.types.physical.stypes.interfaces.SStruct
import is.hail.utils._

abstract class PCanonicalBaseStruct(val types: Array[PType]) extends PBaseStruct {
  if (!types.forall(_.isRealizable)) {
    throw new AssertionError(
      s"found non realizable type(s) ${ types.filter(!_.isRealizable).mkString(", ") } in ${ types.mkString(", ") }")
  }

  val (missingIdx: Array[Int], nMissing: Int) = BaseStruct.getMissingIndexAndCount(types.map(_.required))
  val nMissingBytes: Int = UnsafeUtils.packBitsToBytes(nMissing)
  val byteOffsets: Array[Long] = new Array[Long](size)
  override val byteSize: Long = PBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override val alignment: Long = PBaseStruct.alignment(types)


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

  def isFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      Region.loadBit(offset, missingIdx(fieldIdx).toLong)

  def setFieldMissing(offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    if (!fieldRequired(fieldIdx))
      Region.setBit(offset, missingIdx(fieldIdx).toLong)
    else
      Code._fatal[Unit](s"Required field cannot be missing: $fieldIdx (${ fieldNames(fieldIdx) })")
  }

  def setFieldPresent(offset: Long, fieldIdx: Int) {
    if (!fieldRequired(fieldIdx))
      Region.clearBit(offset, missingIdx(fieldIdx))
  }

  def setFieldPresent(offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    if (!fieldRequired(fieldIdx))
      Region.clearBit(offset, missingIdx(fieldIdx).toLong)
    else
      Code._empty
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

  def deepPointerCopy(cb: EmitCodeBuilder, region: Value[Region], dstStructAddress: Code[Long]): Unit = {
    val dstAddr = cb.newLocal[Long]("pcbs_dpcopy_dst", dstStructAddress)
    fields.foreach { f =>
      val dstFieldType = f.typ.fundamentalType
      if (dstFieldType.containsPointers) {
        cb.ifx(isFieldDefined(dstAddr, f.index),
          {
            dstFieldType match {
              case t@(_: PBinary | _: PArray) =>
                val fieldAddr = cb.newLocal[Long]("pcbs_dpcopy_field", fieldOffset(dstAddr, f.index))
                t.storeAtAddress(cb, fieldAddr, region, t.loadCheapPCode(cb, Region.loadAddress(fieldAddr)), deepCopy = true)
              case t: PCanonicalBaseStruct =>
                t.deepPointerCopy(cb, region, fieldOffset(dstAddr, f.index))
            }
          })
      }
    }
  }

  def deepPointerCopy(region: Region, dstStructAddress: Long) {
    var i = 0
    while (i < this.size) {
      val dstFieldType = this.fields(i).typ.fundamentalType
      if (dstFieldType.containsPointers && this.isFieldDefined(dstStructAddress, i)) {
        val dstFieldAddress = this.fieldOffset(dstStructAddress, i)
        dstFieldType match {
          case t@(_: PBinary | _: PArray) =>
            Region.storeAddress(dstFieldAddress, t.copyFromAddress(region, dstFieldType, Region.loadAddress(dstFieldAddress), deepCopy = true))
          case t: PCanonicalBaseStruct =>
            t.deepPointerCopy(region, dstFieldAddress)
        }
      }
      i += 1
    }
  }

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    if (equalModuloRequired(srcPType) && !deepCopy)
      return srcAddress

    val newAddr = allocate(region)
    unstagedStoreAtAddress(newAddr, region, srcPType.asInstanceOf[PBaseStruct], srcAddress, deepCopy)
    newAddr
  }

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcStruct = srcPType.asInstanceOf[PBaseStruct]
    if (equalModuloRequired(srcStruct)) {
      Region.copyFrom(srcAddress, addr, byteSize)
      if (deepCopy)
        deepPointerCopy(region, addr)
    } else {
      initialize(addr, setMissing = true)
      var idx = 0
      while (idx < types.length) {
        if (srcStruct.isFieldDefined(srcAddress, idx)) {
          setFieldPresent(addr, idx)
          types(idx).unstagedStoreAtAddress(
            fieldOffset(addr, idx), region, srcStruct.types(idx), srcStruct.loadField(srcAddress, idx), deepCopy)
        } else
          assert(!fieldRequired(idx))
        idx += 1
      }
    }
  }

  def sType: SStruct = SBaseStructPointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SBaseStructPointerCode(SBaseStructPointer(this), addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SBaseStructPointer(t) if t.equalModuloRequired(this) && !deepCopy =>
        value.asInstanceOf[SBaseStructPointerCode].a
      case _ =>
        val newAddr = cb.newLocal[Long]("pcbasestruct_store_newaddr")
        cb.assign(newAddr, allocate(region))
        storeAtAddress(cb, newAddr, region, value, deepCopy)
        newAddr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    value.st match {
      case SBaseStructPointer(t) if t.equalModuloRequired(this) =>
        val pcs = value.asBaseStruct.memoize(cb, "pcbasestruct_store_src").asInstanceOf[SBaseStructPointerSettable]
        val addrVar = cb.newLocal[Long]("pcbasestruct_store_dest_addr1", addr)
        cb += Region.copyFrom(pcs.a, addrVar, byteSize)
        if (deepCopy)
          deepPointerCopy(cb, region, addrVar)
      case _ =>
        val addrVar = cb.newLocal[Long]("pcbasestruct_store_dest_addr2", addr)
        val pcs = value.asBaseStruct.memoize(cb, "pcbasestruct_store_src")
        cb += stagedInitialize(addrVar, setMissing = false)

        fields.foreach { f =>
          pcs.loadField(cb, f.index)
            .consume(cb,
              {
                cb += setFieldMissing(addrVar, f.index)
              },
              {
                f.typ.storeAtAddress(cb, fieldOffset(addrVar, f.index), region, _, deepCopy)
              })
        }
    }
  }
}
