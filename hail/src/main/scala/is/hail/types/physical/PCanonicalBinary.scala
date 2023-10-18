package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerValue}
import is.hail.utils._

case object PCanonicalBinaryOptional extends PCanonicalBinary(false)

case object PCanonicalBinaryRequired extends PCanonicalBinary(true)

class PCanonicalBinary(val required: Boolean) extends PBinary {
  def _asIdent = "binary"

  override def copiedType: PType = this

  override def byteSize: Long = 8

  override def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcBinary = srcPType.asInstanceOf[PCanonicalBinary]
    if (srcBinary == this) {
      if (deepCopy) {
        val len = srcBinary.loadLength(srcAddress)
        val newAddr = allocate(region, len)
        Region.copyFrom(srcAddress, newAddr, contentByteSize(len))
        newAddr
      } else
        srcAddress
    } else {
      val len = srcBinary.loadLength(srcAddress)
      val newAddr = allocate(region, len)
      storeLength(newAddr, len)
      Region.copyFrom(srcAddress + srcBinary.lengthHeaderBytes, newAddr + lengthHeaderBytes, len)
      newAddr
    }
  }


  override def containsPointers: Boolean = true

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCBinary")

  def contentAlignment: Long = 4

  def lengthHeaderBytes: Long = 4

  def contentByteSize(length: Int): Long = 4 + length

  def contentByteSize(length: Code[Int]): Code[Long] = (const(4) + length).toL

  def allocate(region: Region, length: Int): Long =
    region.allocate(contentAlignment, contentByteSize(length))

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    region.allocate(const(contentAlignment), contentByteSize(length))

  def loadLength(boff: Long): Int = Region.loadInt(boff)

  def loadLength(boff: Code[Long]): Code[Int] = Region.loadInt(boff)

  def loadBytes(bAddress: Code[Long], length: Code[Int]): Code[Array[Byte]] =
    Region.loadBytes(this.bytesAddress(bAddress), length)

  def loadBytes(bAddress: Code[Long]): Code[Array[Byte]] =
    Code.memoize(bAddress, "pcbin_load_bytes_addr") { bAddress =>
      loadBytes(bAddress, this.loadLength(bAddress))
    }

  def loadBytes(bAddress: Long, length: Int): Array[Byte] = {
    assert(length >= 0, s"Length was: $length")
    Region.loadBytes(this.bytesAddress(bAddress), length)
  }

  def loadBytes(bAddress: Long): Array[Byte] =
    this.loadBytes(bAddress, this.loadLength(bAddress))

  def storeLength(boff: Long, len: Int): Unit = Region.storeInt(boff, len)

  def storeLength(cb: EmitCodeBuilder, boff: Code[Long], len: Code[Int]): Unit = cb += Region.storeInt(boff, len)

  def bytesAddress(boff: Long): Long = boff + lengthHeaderBytes

  def bytesAddress(boff: Code[Long]): Code[Long] = boff + lengthHeaderBytes

  def store(addr: Long, bytes: Array[Byte]) {
    Region.storeInt(addr, bytes.length)
    Region.storeBytes(bytesAddress(addr), bytes)
  }

  override def store(cb: EmitCodeBuilder, _addr: Code[Long], _bytes: Code[Array[Byte]]): Unit = {
    val addr = cb.memoize(_addr, "pcanonical_binary_store_addr")
    val bytes = cb.memoize(_bytes, "pcanonical_binary_store_bytes")
    cb += Region.storeInt(addr, bytes.length())
    cb += Region.storeBytes(bytesAddress(addr), bytes)
  }

  def constructFromByteArray(cb: EmitCodeBuilder, region: Value[Region], bytes: Code[Array[Byte]]): SBinaryPointerValue = {
    val ba = cb.newLocal[Array[Byte]]("pcbin_ba", bytes)
    val len = cb.newLocal[Int]("pcbin_len", ba.length())
    val addr = cb.newLocal[Long]("pcbin_addr", allocate(region, len))
    store(cb, addr, ba)
    loadCheapSCode(cb, addr)
  }

  def constructAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Unit = {
    val srcBinary = srcPType.asInstanceOf[PBinary]
    cb += Region.storeAddress(addr, constructOrCopy(cb, region, srcBinary, srcAddress, deepCopy))
  }

  private def constructOrCopy(cb: EmitCodeBuilder, region: Value[Region], srcBinary: PBinary, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    if (srcBinary == this) {
      if (deepCopy) {
        val srcAddrVar = cb.newLocal[Long]("pcanonical_binary_construct_or_copy_src_addr")
        val len = cb.newLocal[Int]("pcanonical_binary_construct_or_copy_len")
        val newAddr = cb.newLocal[Long]("pcanonical_binary_construct_or_copy_new_addr")
        cb.assign(srcAddrVar, srcAddress)
        cb.assign(len, srcBinary.loadLength(srcAddrVar))
        cb.assign(newAddr, allocate(region, len))
        cb += Region.copyFrom(srcAddrVar, newAddr, contentByteSize(len))
        newAddr
      } else
        srcAddress
    } else {
      val srcAddrVar = cb.newLocal[Long]("pcanonical_binary_construct_or_copy_src_addr")
      val len = cb.newLocal[Int]("pcanonical_binary_construct_or_copy_len")
      val newAddr = cb.newLocal[Long]("pcanonical_binary_construct_or_copy_new_addr")
      cb.assign(srcAddrVar, srcAddress)
      cb.assign(len, srcBinary.loadLength(srcAddrVar))
      cb.assign(newAddr, allocate(region, len))
      storeLength(cb, newAddr, len)
      cb += Region.copyFrom(srcAddrVar + srcBinary.lengthHeaderBytes, newAddr + lengthHeaderBytes, len.toL)
      newAddr
    }
  }

  def sType: SBinaryPointer = SBinaryPointer(setRequired(false))

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBinaryPointerValue =
    new SBinaryPointerValue(sType, cb.memoize(addr))

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    value.st match {
      case SBinaryPointer(PCanonicalBinary(_)) =>
        if (deepCopy) {
          val bv = value.asInstanceOf[SBinaryPointerValue]
          val len = bv.loadLength(cb)
          val newAddr = cb.memoize(allocate(region, len))
          storeLength(cb, newAddr, len)
          cb += Region.copyFrom(bytesAddress(bv.a), bytesAddress(newAddr), len.toL)
          newAddr
        } else
          value.asInstanceOf[SBinaryPointerValue].a
      case _ =>
        val bv = value.asBinary
        val len = bv.loadLength(cb)
        val newAddr = cb.memoize(allocate(region, len))
        storeLength(cb, newAddr, len)
        cb += Region.storeBytes(bytesAddress(newAddr), bv.loadBytes(cb))
        newAddr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, copyFromAddress(sm, region, srcArray, srcAddress, deepCopy))
  }

  def setRequired(required: Boolean): PCanonicalBinary =
    if (required == this.required) this else PCanonicalBinary(required)

  def loadFromNested(addr: Code[Long]): Code[Long] = Region.loadAddress(addr)

  override def unstagedLoadFromNested(addr: Long): Long = Region.loadAddress(addr)

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeAddress(addr, unstagedStoreJavaObject(sm, annotation, region))
  }

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val bytes = annotation.asInstanceOf[Array[Byte]]
    val valueAddress = allocate(region, bytes.length)
    store(valueAddress, bytes)
    valueAddress
  }
}

object PCanonicalBinary {
  def apply(required: Boolean = false): PCanonicalBinary = if (required) PCanonicalBinaryRequired else PCanonicalBinaryOptional

  def unapply(t: PBinary): Option[Boolean] = Option(t.required)
}
