package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder}

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

  override def storeShallowAtOffset(destOffset: Code[Long], valueAddress: Code[Long]): Code[Unit] =
    this.fundamentalType.storeShallowAtOffset(destOffset, valueAddress)

  override def storeShallowAtOffset(destOffset: Long, valueAddress: Long) {
    this.fundamentalType.storeShallowAtOffset(destOffset, valueAddress)
  }
}

object PCanonicalString {
  def apply(required: Boolean = false) = new PCanonicalString(required)

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
