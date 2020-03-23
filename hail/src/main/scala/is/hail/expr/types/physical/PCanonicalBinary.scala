package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

case object PCanonicalBinaryOptional extends PCanonicalBinary(false)
case object PCanonicalBinaryRequired extends PCanonicalBinary(true)

class PCanonicalBinary(val required: Boolean) extends PBinary {
  def _asIdent = "binary"

  override def byteSize: Long = 8

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
      constructOrCopy(mb, region, srcPType.asInstanceOf[PBinary], srcAddress, deepCopy)
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    val srcBinary = srcPType.asInstanceOf[PBinary]
    constructOrCopy(region, srcBinary, srcAddress, deepCopy)
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

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    val srcArray = srcPType.asInstanceOf[PBinary]
    Region.storeAddress(addr, constructOrCopy(region, srcArray, srcAddress, deepCopy))
  }

  private def constructOrCopy(region: Region, srcBinary: PBinary, srcAddress: Long, deepCopy: Boolean): Long = {
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

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalBinary(required)
}

object PCanonicalBinary {
  def apply(required: Boolean = false): PBinary = if (required) PCanonicalBinaryRequired else PCanonicalBinaryOptional

  def unapply(t: PBinary): Option[Boolean] = Option(t.required)
}
