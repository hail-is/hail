package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.utils._

trait PPrimitive extends PType {
  def byteSize: Long

  def _construct(mb: EmitMethodBuilder[_], region: Value[Region], pc: PCode): PCode = pc

  override def containsPointers: Boolean = false

  override def encodableType: PType = this

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    if (!deepCopy)
      return srcAddress

    // FIXME push down
    val addr = region.allocate(byteSize, byteSize)
    constructAtAddress(addr, region, srcPType, srcAddress, deepCopy)
    addr
  }

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    assert(this.isOfType(srcPType))
    if (deepCopy) {
      val addr = mb.newLocal[Long]()
      Code(
        addr := region.allocate(byteSize, byteSize),
        constructAtAddress(mb, addr, region, srcPType, srcAddress, deepCopy),
        addr
      )
    } else srcAddress
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] = {
    assert(this.isOfType(srcPType))
    stackValue
  }

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    assert(srcPType.isOfType(this))
    Region.copyFrom(srcAddress, addr, byteSize)
  }

  override def constructAtAddressFromValue(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, src: Code[_], deepCopy: Boolean): Code[Unit] = {
    assert(this.isOfType(srcPType))
    storePrimitiveAtAddress(addr, srcPType, src)
  }

  def storePrimitiveAtAddress(addr: Code[Long], srcPType: PType, value: Code[_]): Code[Unit]

  def setRequired(required: Boolean): PPrimitive = {
    if (required == this.required)
      this
    else
      this match {
        case _: PBoolean => PBoolean(required)
        case _: PInt32 => PInt32(required)
        case _: PInt64 => PInt64(required)
        case _: PFloat32 => PFloat32(required)
        case _: PFloat64 => PFloat64(required)
      }
  }
}

class PPrimitiveCode(val pt: PType, val code: Code[_]) extends PCode {
  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(code)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], a: Code[Long]): Code[Unit] =
    Region.storeIRIntermediate(pt)(a, code)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)
}
