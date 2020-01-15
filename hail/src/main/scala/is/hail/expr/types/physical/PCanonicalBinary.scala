package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.utils._

case object PCanonicalBinaryOptional extends PCanonicalBinary(false)
case object PCanonicalBinaryRequired extends PCanonicalBinary(true)

class PCanonicalBinary(val required: Boolean) extends PBinary {
  def _asIdent = "binary"
  def _toPretty = "Binary"

  override def byteSize: Long = 8

  override def copyFromType(mb: MethodBuilder, region: Code[Region], sourcePType: PType, sourceAddress: Code[Long],
  allowDowncast: Boolean, forceDeep: Boolean): Code[Long] = {
    if(this == sourcePType && !forceDeep) {
      return sourceAddress
    }

    assert(this isOfType sourcePType)

    val dstAddress = mb.newField[Long]
    val length = mb.newLocal[Int]

    // srcAddress must point to data by copyFromType semantics, so no runtime null-check needed
    if(this.required > sourcePType.required) {
      assert(allowDowncast)
    }

    Code(
      length := PCanonicalBinary.loadLength(region, sourceAddress),
      dstAddress := PCanonicalBinary.allocate(region, length),
      Region.copyFrom(sourceAddress, dstAddress, PCanonicalBinary.contentByteSize(length)),
      dstAddress
    )
  }

  override def copyFromType(region: Region, sourcePType: PType, sourceAddress: Long,
    allowDowncast: Boolean, forceDeep: Boolean): Long = {
    if(this == sourcePType && !forceDeep) {
      return sourceAddress
    }

    assert(this isOfType sourcePType)

    if(this.required > sourcePType.required) {
      assert(allowDowncast)
    }

    val length = PCanonicalBinary.loadLength(region, sourceAddress)
    val dstAddress = PCanonicalBinary.allocate(region, length)
    Region.copyFrom(sourceAddress, dstAddress, PCanonicalBinary.contentByteSize(length))
    dstAddress
  }

  override def containsPointers: Boolean = true

  override def storeShallowAtOffset(dstAddress: Code[Long], valueAddress: Code[Long]): Code[Unit] = {
    Region.storeAddress(dstAddress, valueAddress)
  }

  override def storeShallowAtOffset(dstAddress: Long, valueAddress: Long) {
    Region.storeAddress(dstAddress, valueAddress)
  }
}

object PCanonicalBinary {
  def apply(required: Boolean = false): PBinary = if (required) PCanonicalBinaryRequired else PCanonicalBinaryOptional

  def unapply(t: PBinary): Option[Boolean] = Option(t.required)

  def contentAlignment: Long = 4

  def lengthHeaderBytes: Long = 4

  def contentByteSize(length: Int): Long = 4 + length

  def contentByteSize(length: Code[Int]): Code[Long] = (const(4) + length).toL

  def loadLength(boff: Long): Int = Region.loadInt(boff)

  def loadLength(region: Region, boff: Long): Int =
    Region.loadInt(boff)

  def loadLength(boff: Code[Long]): Code[Int] =
    Region.loadInt(boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] = loadLength(boff)

  def storeLength(boff: Long, len: Int): Unit = Region.storeInt(boff, len)

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit] = Region.storeInt(boff, len)

  def bytesOffset(boff: Long): Long = boff + lengthHeaderBytes

  def bytesOffset(boff: Code[Long]): Code[Long] = boff + lengthHeaderBytes

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentAlignment, contentByteSize(length))
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] = {
    region.allocate(const(contentAlignment), contentByteSize(length))
  }

  def store(addr: Long, bytes: Array[Byte]): Unit = {
    Region.storeInt(addr, bytes.length)
    Region.storeBytes(bytesOffset(addr), bytes)
  }

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] =
    Code.invokeScalaObject[Long, Array[Byte], Unit](PBinary.getClass, "store", addr, bytes)
}
