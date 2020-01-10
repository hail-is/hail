package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{???, Code, MethodBuilder}

class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"
  def _toPretty = "String"

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PBinary = PBinary(required)

  override def copyFromType(mb: MethodBuilder, region: Code[Region], sourcePType: PType, sourceOffset: Code[Long],
  allowDowncast: Boolean = false, forceDeep: Boolean = false): Code[Long] = {
    assert(sourcePType isOfType this)
    this.fundamentalType.copyFromType(
      mb, region, sourcePType.asInstanceOf[PString].fundamentalType, sourceOffset, allowDowncast, forceDeep
    )
  }

  override def containsPointers: Boolean = true

  override def storeShallowAtOffset(destOffset: Code[Long], valueAddress: Code[Long]): Code[Unit] = {
    Region.storeAddress(destOffset, valueAddress)
  }
}

object PCanonicalString {
  def apply(required: Boolean = false) = new PCanonicalString(required)

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
