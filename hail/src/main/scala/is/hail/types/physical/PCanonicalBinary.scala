package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerCode, SBinaryPointerSettable}
import is.hail.types.physical.stypes.interfaces.SBinary
import is.hail.utils._

case object PCanonicalBinaryOptional extends PCanonicalBinary(false)

case object PCanonicalBinaryRequired extends PCanonicalBinary(true)

class PCanonicalBinary(val required: Boolean) extends PBinary {
  def _asIdent = "binary"

  override def byteSize: Long = 8

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
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

  def loadBytes(bAddress: Long, length: Int): Array[Byte] =
    Region.loadBytes(this.bytesAddress(bAddress), length)

  def loadBytes(bAddress: Long): Array[Byte] =
    this.loadBytes(bAddress, this.loadLength(bAddress))

  def storeLength(boff: Long, len: Int): Unit = Region.storeInt(boff, len)

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit] = Region.storeInt(boff, len)

  def bytesAddress(boff: Long): Long = boff + lengthHeaderBytes

  def bytesAddress(boff: Code[Long]): Code[Long] = boff + lengthHeaderBytes

  def store(addr: Long, bytes: Array[Byte]) {
    Region.storeInt(addr, bytes.length)
    Region.storeBytes(bytesAddress(addr), bytes)
  }

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] =
    Code.memoize(addr, "pcbin_store_addr") { addr =>
      Code.memoize(bytes, "pcbin_store_bytes") { bytes =>
        Code(
          Region.storeInt(addr, bytes.length),
          Region.storeBytes(bytesAddress(addr), bytes))
      }
    }

  def constructFromByteArray(cb: EmitCodeBuilder, region: Value[Region], bytes: Code[Array[Byte]]): SBinaryPointerCode = {
    val ba = cb.newLocal[Array[Byte]]("pcbin_ba", bytes)
    val len = cb.newLocal[Int]("pcbin_len", ba.length())
    val addr = cb.newLocal[Long]("pcbin_addr", allocate(region, len))
    cb += store(addr, ba)
    loadCheapSCode(cb, addr)
  }

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = {
    val srcBinary = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, constructOrCopy(mb, region, srcBinary, srcAddress, deepCopy))
  }

  private def constructOrCopy(mb: EmitMethodBuilder[_], region: Value[Region], srcBinary: PBinary, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    if (srcBinary == this) {
      if (deepCopy) {
        val srcAddrVar = mb.newLocal[Long]()
        val len = mb.newLocal[Int]()
        val newAddr = mb.newLocal[Long]()
        Code(
          srcAddrVar := srcAddress,
          len := srcBinary.loadLength(srcAddrVar),
          newAddr := allocate(region, len),
          Region.copyFrom(srcAddrVar, newAddr, contentByteSize(len)),
          newAddr)
      } else
        srcAddress
    } else {
      val srcAddrVar = mb.newLocal[Long]()
      val len = mb.newLocal[Int]()
      val newAddr = mb.newLocal[Long]()
      Code(
        srcAddrVar := srcAddress,
        len := srcBinary.loadLength(srcAddrVar),
        newAddr := allocate(region, len),
        storeLength(newAddr, len),
        Region.copyFrom(srcAddrVar + srcBinary.lengthHeaderBytes, newAddr + lengthHeaderBytes, len.toL),
        newAddr
      )
    }
  }

  def sType: SBinaryPointer = SBinaryPointer(setRequired(false).asInstanceOf[PCanonicalBinary])

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBinaryPointerCode = new SBinaryPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SBinaryPointer(PCanonicalBinary(_)) =>
        if (deepCopy) {
          val bv = value.asInstanceOf[SBinaryPointerCode].memoize(cb, "pcbin_store")
          val len = cb.newLocal[Int]("pcbinary_store_len", bv.loadLength())
          val newAddr = cb.newLocal[Long]("pcbinary_store_newaddr", allocate(region, len))
          cb += storeLength(newAddr, len)
          cb += Region.copyFrom(bytesAddress(bv.a), bytesAddress(newAddr), len.toL)
          newAddr
        } else
          value.asInstanceOf[SBinaryPointerCode].a
      case _ =>
        val bv = value.asBinary.memoize(cb, "pcbin_store")
        val len = cb.newLocal[Int]("pcbinary_store_len", bv.loadLength())
        val newAddr = cb.newLocal[Long]("pcbinary_store_newaddr", allocate(region, len))
        cb += storeLength(newAddr, len)
        cb += Region.storeBytes(bytesAddress(newAddr), bv.loadBytes())
        newAddr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, copyFromAddress(region, srcArray, srcAddress, deepCopy))
  }

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalBinary(required)

  def loadFromNested(addr: Code[Long]): Code[Long] = Region.loadAddress(addr)

  override def unstagedLoadFromNested(addr: Long): Long = Region.loadAddress(addr)

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeAddress(addr, unstagedStoreJavaObject(annotation, region))
  }

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
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

