package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder, const}

case object PCanonicalStringOptional extends PCanonicalString(false)
case object PCanonicalStringRequired extends PCanonicalString(true)

abstract class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCString")

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PBinary = PBinary(required)

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] = {
    assert(srcPType isOfType this)
    this.fundamentalType.copyFromType(
      mb, region, srcPType.asInstanceOf[PString].fundamentalType, srcAddress, forceDeep
    )
  }

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long  = {
    assert(srcPType isOfType this)
    this.fundamentalType.copyFromType(
      region, srcPType.asInstanceOf[PString].fundamentalType, srcAddress, forceDeep
    )
  }

  override def containsPointers: Boolean = true

  override def storeShallowAtOffset(dstAddress: Code[Long], valueAddress: Code[Long]): Code[Unit] =
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)

  override def storeShallowAtOffset(dstAddress: Long, valueAddress: Long) {
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)
  }

  def bytesOffset(boff: Long): Long =
    this.fundamentalType.bytesOffset(boff)

  def bytesOffset(boff: Code[Long]): Code[Long] =
    this.fundamentalType.bytesOffset(boff)

  def loadLength(boff: Long): Int =
    this.fundamentalType.loadLength(boff)

  def loadLength(boff: Code[Long]): Code[Int] =
    this.fundamentalType.loadLength(boff)

  def loadString(bAddress: Long): String =
    new String(this.fundamentalType.loadBytes(bAddress))

  def loadString(bAddress: Code[Long]): Code[String] =
    Code.newInstance[String, Array[Byte]](this.fundamentalType.loadBytes(bAddress))

  def allocate(region: Region, length: Int): Long =
    this.fundamentalType.allocate(region, length)

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    this.fundamentalType.allocate(region, length)

  def store(addr: Long, bytes: Array[Byte]) {
    this.fundamentalType.store(addr, bytes)
  }

  def store(addr: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] =
    this.fundamentalType.store(addr, bytes)
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)
}
