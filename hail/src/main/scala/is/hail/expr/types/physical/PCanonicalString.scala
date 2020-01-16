package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{???, Code, MethodBuilder}

case object PCanonicalStringOptional extends PCanonicalString(false)
case object PCanonicalStringRequired extends PCanonicalString(true)

abstract class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCString")

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PBinary = PBinary(required)

  override def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcOffset: Code[Long], forceDeep: Boolean): Code[Long] = {
    assert(srcPType isOfType this)
    this.fundamentalType.copyFromType(
      mb, region, srcPType.asInstanceOf[PString].fundamentalType, srcOffset, forceDeep
    )
  }

  override def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long = ???

  override def containsPointers: Boolean = true

  override def storeShallowAtOffset(dstAddress: Code[Long], valueAddress: Code[Long]): Code[Unit] =
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)

  override def storeShallowAtOffset(dstAddress: Long, valueAddress: Long) {
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)
  }
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)

  def loadString(bAddress: Long): String =
    new String(PBinary.loadBytes(bAddress))

  def loadString(region: Region, boff: Long): String =
    loadString(boff)

  def loadString(bAddress: Code[Long]): Code[String] =
    Code.newInstance[String, Array[Byte]](PBinary.loadBytes(bAddress))

  def loadString(region: Code[Region], bAddress: Code[Long]): Code[String] =
    loadString(bAddress)

  def loadLength(region: Region, bAddress: Long): Int =
    PBinary.loadLength(region, bAddress)

  def loadLength(region: Code[Region], bAddress: Code[Long]): Code[Int] =
    PBinary.loadLength(region, bAddress)
}
