package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.BaseStruct
import is.hail.utils._

abstract class PCanonicalBaseStruct(val types: Array[PType]) extends PBaseStruct {
  if (!types.forall(_.isRealizable)) {
    throw new AssertionError(
      s"found non realizable type(s) ${types.filter(!_.isRealizable).mkString(", ")} in ${types.mkString(", ")}")
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
      Code._fatal[Unit](s"Required field cannot be missing")
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

  def deepPointerCopy(mb: EmitMethodBuilder[_], region: Value[Region], dstStructAddress: Code[Long]): Code[Unit] = {
    Code.memoize(dstStructAddress, "pcbs_dpcopy_dst") { dstStructAddress =>
      var c: Code[Unit] = Code._empty
      var i = 0
      while (i < size) {
        val dstFieldType = fields(i).typ.fundamentalType
        if (dstFieldType.containsPointers) {
          val dstFieldAddress = mb.genFieldThisRef[Long]()
          c = Code(
            c,
            isFieldDefined(dstStructAddress, i).orEmpty(
              Code(
                dstFieldAddress := fieldOffset(dstStructAddress, i),
                dstFieldType match {
                  case t@(_: PBinary | _: PArray) =>
                    Region.storeAddress(dstFieldAddress, t.copyFromType(mb, region, dstFieldType, Region.loadAddress(dstFieldAddress), deepCopy = true))
                  case t: PCanonicalBaseStruct =>
                    t.deepPointerCopy(mb, region, dstFieldAddress)
                }
              )
            )
          )
        }
        i += 1
      }

      c
    }
  }

  def deepPointerCopy(region: Region, dstStructAddress: Long) {
    var i = 0
    while(i < this.size) {
      val dstFieldType = this.fields(i).typ.fundamentalType
      if(dstFieldType.containsPointers && this.isFieldDefined(dstStructAddress, i)) {
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

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcStructAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    val sourceType = srcPType.asInstanceOf[PBaseStruct]
    assert(sourceType.size == this.size)

    if (this == sourceType && !deepCopy)
      srcStructAddress
    else {
      val addr = mb.newLocal[Long]()
      Code(
        addr := allocate(region),
        constructAtAddress(mb, addr, region, sourceType, srcStructAddress, deepCopy),
        addr
      )
    }
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    if (equalModuloRequired(srcPType) && !deepCopy)
      return srcAddress

    val newAddr = allocate(region)
    constructAtAddress(newAddr, region, srcPType.asInstanceOf[PBaseStruct], srcAddress, deepCopy)
    newAddr
  }

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val srcStruct = srcPType.asInstanceOf[PBaseStruct]
    val addrVar = mb.newLocal[Long]()

    if (srcStruct == this) {
      var c: Code[Unit] = Code(
        addrVar := addr,
        Region.copyFrom(srcAddress, addrVar, byteSize))
      if (deepCopy)
        c = Code(c, deepPointerCopy(mb, region, addrVar))
      c
    } else {
      val srcAddrVar = mb.newLocal[Long]()
      Code(
        srcAddrVar := srcAddress,
        addrVar := addr,
        stagedInitialize(addrVar, setMissing = true),
        Code(fields.zip(srcStruct.fields).map { case (dest, src) =>
          assert(dest.typ.required <= src.typ.required, s"${dest.typ} <- ${src.typ}\n  src: $srcPType\n  dst: $this")
          val idx = dest.index
          assert(idx == src.index)
          srcStruct.isFieldDefined(srcAddrVar, idx).orEmpty(Code(
            setFieldPresent(addrVar, idx),
            dest.typ.constructAtAddress(mb, fieldOffset(addrVar, idx), region, src.typ, srcStruct.loadField(srcAddrVar, idx), deepCopy))
          )
        }))
    }
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
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
          types(idx).constructAtAddress(
            fieldOffset(addr, idx), region, srcStruct.types(idx), srcStruct.loadField(srcAddress, idx), deepCopy)
        } else
          assert(!fieldRequired(idx))
        idx += 1
      }
    }
  }

  override def load(src: Code[Long]): PCanonicalBaseStructCode =
    new PCanonicalBaseStructCode(this, src)
}

object PCanonicalBaseStructSettable {
  def apply(sb: SettableBuilder, pt: PCanonicalBaseStruct, name: String): PCanonicalBaseStructSettable = {
    new PCanonicalBaseStructSettable(pt, sb.newSettable(name))
  }
}

class PCanonicalBaseStructSettable(
  val pt: PCanonicalBaseStruct,
  val a: Settable[Long]
) extends PBaseStructValue with PSettable {
  def get: PBaseStructCode = new PCanonicalBaseStructCode(pt, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a)

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    IEmitCode(cb,
      pt.isFieldMissing(a, fieldIdx),
      pt.fields(fieldIdx).typ.load(pt.fieldOffset(a, fieldIdx)))
  }

  def store(pv: PCode): Code[Unit] = {
    a := pv.asInstanceOf[PCanonicalBaseStructCode].a
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] = {
    this.pt.isFieldMissing(a, fieldIdx)
  }
}

class PCanonicalBaseStructCode(val pt: PCanonicalBaseStruct, val a: Code[Long]) extends PBaseStructCode {
  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PBaseStructValue = {
    val s = PCanonicalBaseStructSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, deepCopy = false)
}
