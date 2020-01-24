package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder}

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

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], forceDeep)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long  = {
    assert(srcPType isOfType this)
    this.fundamentalType.copyFromType(
      region, srcPType.asInstanceOf[PString].fundamentalType, srcAddress, forceDeep
    )
  }

  override def containsPointers: Boolean = true

  def storeShallowAtOffset(dstAddress: Code[Long], valueAddress: Code[Long]): Code[Unit] =
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)

  def storeShallowAtOffset(dstAddress: Long, valueAddress: Long) {
    this.fundamentalType.storeShallowAtOffset(dstAddress, valueAddress)
  }
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)

  def loadString(bAddress: Long): String =
    new String(PBinary.loadBytes(bAddress))

  def loadString(bAddress: Code[Long]): Code[String] =
    Code.newInstance[String, Array[Byte]](PBinary.loadBytes(bAddress))

  def loadLength(bAddress: Long): Int =
    PBinary.loadLength(bAddress)

  def loadLength(bAddress: Code[Long]): Code[Int] =
    PBinary.loadLength(bAddress)
}
