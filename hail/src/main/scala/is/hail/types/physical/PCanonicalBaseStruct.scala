package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeRow, UnsafeUtils}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.BaseStruct
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerValue}
import is.hail.utils._

import org.apache.spark.sql.Row

abstract class PCanonicalBaseStruct(val types: Array[PType]) extends PBaseStruct {
  if (!types.forall(_.isRealizable)) {
    throw new AssertionError(
      s"found non realizable type(s) ${types.filter(!_.isRealizable).mkString(", ")} in ${types.mkString(", ")}"
    )
  }

  override val (missingIdx: Array[Int], nMissing: Int) =
    BaseStruct.getMissingIndexAndCount(types.map(_.required))

  val nMissingBytes: Int = UnsafeUtils.packBitsToBytes(nMissing)
  val byteOffsets: Array[Long] = new Array[Long](size)

  override val byteSize: Long =
    getByteSizeAndOffsets(types.map(_.byteSize), types.map(_.alignment), nMissingBytes, byteOffsets)

  override val alignment: Long = PBaseStruct.alignment(types)

  override def allocate(region: Region): Long =
    region.allocate(alignment, byteSize)

  override def allocate(region: Code[Region]): Code[Long] =
    region.allocate(alignment, byteSize)

  override def initialize(structAddress: Long, setMissing: Boolean = false): Unit = {
    if (allFieldsRequired) {
      return
    }

    Region.setMemory(structAddress, nMissingBytes.toLong, if (setMissing) 0xff.toByte else 0.toByte)
  }

  override def stagedInitialize(
    cb: EmitCodeBuilder,
    structAddress: Code[Long],
    setMissing: Boolean = false,
  ): Unit =
    if (!allFieldsRequired) {
      cb += Region.setMemory(
        structAddress,
        const(nMissingBytes.toLong),
        const(if (setMissing) 0xff.toByte else 0.toByte),
      )
    }

  override def isFieldDefined(offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !Region.loadBit(offset, missingIdx(fieldIdx))

  override def isFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int)
    : Value[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      cb.memoize(Region.loadBit(offset, missingIdx(fieldIdx).toLong))

  override def setFieldMissing(offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    Region.setBit(offset, missingIdx(fieldIdx))
  }

  override def setFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Unit =
    if (!fieldRequired(fieldIdx))
      cb += Region.setBit(offset, missingIdx(fieldIdx).toLong)
    else {
      cb._fatal(s"Required field cannot be missing.")
    }

  override def setFieldPresent(offset: Long, fieldIdx: Int) {
    if (!fieldRequired(fieldIdx))
      Region.clearBit(offset, missingIdx(fieldIdx))
  }

  override def setFieldPresent(cb: EmitCodeBuilder, offset: Code[Long], fieldIdx: Int): Unit =
    if (!fieldRequired(fieldIdx))
      cb += Region.clearBit(offset, missingIdx(fieldIdx).toLong)

  override def fieldOffset(structAddress: Long, fieldIdx: Int): Long =
    structAddress + byteOffsets(fieldIdx)

  override def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    structAddress + byteOffsets(fieldIdx)

  override def loadField(offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    types(fieldIdx).unstagedLoadFromNested(off)
  }

  override def loadField(offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(fieldOffset(offset, fieldIdx), types(fieldIdx))

  private def loadField(fieldOffset: Code[Long], fieldType: PType): Code[Long] =
    fieldType.loadFromNested(fieldOffset)

  def deepPointerCopy(cb: EmitCodeBuilder, region: Value[Region], dstStructAddress: Code[Long])
    : Unit = {
    val dstAddr = cb.newLocal[Long]("pcbs_dpcopy_dst", dstStructAddress)
    fields.foreach { f =>
      val dstFieldType = f.typ
      if (dstFieldType.containsPointers) {
        cb.if_(
          isFieldDefined(cb, dstAddr, f.index), {
            val fieldAddr = cb.newLocal[Long]("pcbs_dpcopy_field", fieldOffset(dstAddr, f.index))
            dstFieldType.storeAtAddress(
              cb,
              fieldAddr,
              region,
              dstFieldType.loadCheapSCode(cb, dstFieldType.loadFromNested(fieldAddr)),
              deepCopy = true,
            )
          },
        )
      }
    }
  }

  def deepPointerCopy(sm: HailStateManager, region: Region, dstStructAddress: Long) {
    var i = 0
    while (i < this.size) {
      val dstFieldType = this.fields(i).typ
      if (dstFieldType.containsPointers && this.isFieldDefined(dstStructAddress, i)) {
        val dstFieldAddress = this.fieldOffset(dstStructAddress, i)
        val dstFieldAddressFromNested = dstFieldType.unstagedLoadFromNested(dstFieldAddress)
        dstFieldType.unstagedStoreAtAddress(sm, dstFieldAddress, region, dstFieldType,
          dstFieldAddressFromNested, true)
      }
      i += 1
    }
  }

  override def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long = {
    if (equalModuloRequired(srcPType) && !deepCopy)
      return srcAddress

    val newAddr = allocate(region)
    unstagedStoreAtAddress(
      sm,
      newAddr,
      region,
      srcPType.asInstanceOf[PBaseStruct],
      srcAddress,
      deepCopy,
    )
    newAddr
  }

  override def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit = {
    val srcStruct = srcPType.asInstanceOf[PBaseStruct]
    if (equalModuloRequired(srcStruct)) {
      Region.copyFrom(srcAddress, addr, byteSize)
      if (deepCopy)
        deepPointerCopy(sm, region, addr)
    } else {
      initialize(addr, setMissing = true)
      var idx = 0
      while (idx < types.length) {
        if (srcStruct.isFieldDefined(srcAddress, idx)) {
          setFieldPresent(addr, idx)
          types(idx).unstagedStoreAtAddress(
            sm,
            fieldOffset(addr, idx),
            region,
            srcStruct.types(idx),
            srcStruct.loadField(srcAddress, idx),
            deepCopy,
          )
        } else
          assert(!fieldRequired(idx))
        idx += 1
      }
    }
  }

  override def sType: SBaseStructPointer =
    SBaseStructPointer(setRequired(false).asInstanceOf[PCanonicalBaseStruct])

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBaseStructPointerValue =
    new SBaseStructPointerValue(sType, cb.memoize(addr))

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] = {
    value.st match {
      case SBaseStructPointer(t) if t.equalModuloRequired(this) && !deepCopy =>
        value.asInstanceOf[SBaseStructPointerValue].a
      case _ =>
        val newAddr = cb.memoize(allocate(region))
        storeAtAddress(cb, newAddr, region, value, deepCopy)
        newAddr
    }
  }

  override def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit = {
    value.st match {
      case SBaseStructPointer(t) if t.equalModuloRequired(this) =>
        val pcs = value.asInstanceOf[SBaseStructPointerValue]
        val addrVar = cb.newLocal[Long]("pcbasestruct_store_dest_addr1", addr)
        cb += Region.copyFrom(pcs.a, addrVar, byteSize)
        if (deepCopy)
          deepPointerCopy(cb, region, addrVar)
      case _ =>
        val addrVar = cb.newLocal[Long]("pcbasestruct_store_dest_addr2", addr)
        val pcs = value.asBaseStruct
        stagedInitialize(cb, addrVar, setMissing = false)

        fields.foreach { f =>
          pcs.loadField(cb, f.index)
            .consume(
              cb,
              setFieldMissing(cb, addrVar, f.index),
              sv => f.typ.storeAtAddress(cb, fieldOffset(addrVar, f.index), region, sv, deepCopy),
            )
        }
    }
  }

  def constructFromFields(
    cb: EmitCodeBuilder,
    region: Value[Region],
    emitFields: IndexedSeq[EmitCode],
    deepCopy: Boolean,
  ): SBaseStructPointerValue = {
    require(emitFields.length == size)
    val addr = cb.newLocal[Long]("pcbs_construct_fields", allocate(region))
    stagedInitialize(cb, addr, setMissing = false)
    emitFields.zipWithIndex.foreach { case (ev, i) =>
      ev.toI(cb)
        .consume(
          cb,
          setFieldMissing(cb, addr, i),
          sc => types(i).storeAtAddress(cb, fieldOffset(addr, i), region, sc, deepCopy = deepCopy),
        )
    }

    new SBaseStructPointerValue(sType, addr)
  }

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region)
    : Long = {
    val addr = allocate(region)
    unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
    addr
  }

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit = {
    initialize(addr)
    val row = annotation.asInstanceOf[Row]
    row match {
      case ur: UnsafeRow =>
        this.unstagedStoreAtAddress(sm, addr, region, ur.t, ur.offset, region.ne(ur.region))
      case _: Row =>
        this.types.zipWithIndex.foreach { case (fieldPt, fieldIdx) =>
          if (row(fieldIdx) == null) {
            setFieldMissing(addr, fieldIdx)
          } else {
            val fieldAddress = fieldOffset(addr, fieldIdx)
            fieldPt.unstagedStoreJavaObjectAtAddress(sm, fieldAddress, row(fieldIdx), region)
          }
        }
    }

  }

  override def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr
}
