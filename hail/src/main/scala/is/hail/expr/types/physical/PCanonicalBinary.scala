package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.utils._

case object PCanonicalBinaryOptional extends PCanonicalBinary(false)
case object PCanonicalBinaryRequired extends PCanonicalBinary(true)

class PCanonicalBinary(val required: Boolean) extends PBinary {
  def _asIdent = "binary"

  override def byteSize: Long = 8

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
      constructOrCopy(mb, region, srcPType.asInstanceOf[PBinary], srcAddress, forceDeep)
  }

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long = {
    val srcBinary = srcPType.asInstanceOf[PBinary]
    constructOrCopy(region, srcBinary, srcAddress, forceDeep)
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
    Region.loadBytes(this.bytesOffset(bAddress), length)

  def loadBytes(bAddress: Code[Long]): Code[Array[Byte]] =
    Code.memoize(bAddress, "pcbin_load_bytes_addr") { bAddress =>
      loadBytes(bAddress, this.loadLength(bAddress))
    }

  def loadBytes(bAddress: Long, length: Int): Array[Byte] =
    Region.loadBytes(this.bytesOffset(bAddress), length)

  def loadBytes(bAddress: Long): Array[Byte] =
    this.loadBytes(bAddress, this.loadLength(bAddress))

  def storeLength(boff: Long, len: Int): Unit = Region.storeInt(boff, len)

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit] = Region.storeInt(boff, len)

  def bytesOffset(boff: Long): Long = boff + lengthHeaderBytes

  def bytesOffset(boff: Code[Long]): Code[Long] = boff + lengthHeaderBytes

  def store(addr: Long, bytes: Array[Byte]) {
    Region.storeInt(addr, bytes.length)
    Region.storeBytes(bytesOffset(addr), bytes)
  }

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] =
    Code.memoize(addr, "pcbin_store_addr") { addr =>
      Code.memoize(bytes, "pcbin_store_bytes") { bytes =>
        Code(
          Region.storeInt(addr, bytes.length),
          Region.storeBytes(bytesOffset(addr), bytes))
      }
    }

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] = {
    val srcBinary = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, constructOrCopy(mb, region, srcBinary, srcAddress, forceDeep))
  }

  private def constructOrCopy(mb: MethodBuilder, region: Code[Region], srcBinary: PBinary, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    if (srcBinary == this) {
      if (forceDeep) {
        val srcAddrVar = mb.newLocal[Long]
        val len = mb.newLocal[Int]
        val newAddr = mb.newLocal[Long]
        Code(
          srcAddrVar := srcAddress,
          len := srcBinary.loadLength(srcAddrVar),
          newAddr := allocate(region, len),
          Region.copyFrom(srcAddrVar, newAddr, contentByteSize(len)),
          newAddr)
      } else
        srcAddress
    } else {
      val srcAddrVar = mb.newLocal[Long]
      val len = mb.newLocal[Int]
      val newAddr = mb.newLocal[Long]
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

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, constructOrCopy(region, srcArray, srcAddress, forceDeep))
  }

  private def constructOrCopy(region: Region, srcBinary: PBinary, srcAddress: Long, forceDeep: Boolean): Long = {
    if (srcBinary == this) {
      if (forceDeep) {
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

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalBinary(required)
}

object PCanonicalBinary {
  def apply(required: Boolean = false): PBinary = if (required) PCanonicalBinaryRequired else PCanonicalBinaryOptional

  def unapply(t: PBinary): Option[Boolean] = Option(t.required)
}
