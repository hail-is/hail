package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code

case object PCanonicalStringOptional extends PCanonicalString(false)
case object PCanonicalStringRequired extends PCanonicalString(true)

abstract class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCString")

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PBinary = PBinary(required)

  override def containsPointers: Boolean = true
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)

  def loadString(boff: Long): String = {
    val length = PBinary.loadLength(boff)
    new String(Region.loadBytes(PBinary.bytesOffset(boff), length))
  }

  def loadString(region: Region, boff: Long): String =
    loadString(boff)

  def loadString(boff: Code[Long]): Code[String] = {
    val length = PBinary.loadLength(boff)
    Code.newInstance[String, Array[Byte]](
      Region.loadBytes(PBinary.bytesOffset(boff), length))
  }

  def loadString(region: Code[Region], boff: Code[Long]): Code[String] =
    loadString(boff)

  def loadLength(region: Region, boff: Long): Int =
    PBinary.loadLength(region, boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] =
    PBinary.loadLength(region, boff)
}
