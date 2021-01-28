package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SStringPointer, SStringPointerCode}

case object PCanonicalStringOptional extends PCanonicalString(false)

case object PCanonicalStringRequired extends PCanonicalString(true)

class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCString")

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PCanonicalBinary = PCanonicalBinary(required)
  override lazy val fundamentalType: PCanonicalBinary = binaryFundamentalType
  override lazy val binaryEncodableType: PCanonicalBinary = PCanonicalBinary(required)

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    fundamentalType.copyFromAddress(region, srcPType.asInstanceOf[PString].fundamentalType, srcAddress, deepCopy)

  override def containsPointers: Boolean = true

  def loadLength(boff: Long): Int =
    this.fundamentalType.loadLength(boff)

  def loadLength(boff: Code[Long]): Code[Int] =
    this.fundamentalType.loadLength(boff)

  def loadString(bAddress: Long): String =
    new String(this.fundamentalType.loadBytes(bAddress))

  def loadString(bAddress: Code[Long]): Code[String] =
    Code.newInstance[String, Array[Byte]](this.fundamentalType.loadBytes(bAddress))

  def allocateAndStoreString(region: Region, str: String): Long = {
    val byteRep = str.getBytes()
    val dstAddrss = this.fundamentalType.allocate(region, byteRep.length)
    this.fundamentalType.store(dstAddrss, byteRep)
    dstAddrss
  }

  def allocateAndStoreString(mb: EmitMethodBuilder[_], region: Value[Region], str: Code[String]): Code[Long] = {
    val dstAddress = mb.genFieldThisRef[Long]()
    val byteRep = mb.genFieldThisRef[Array[Byte]]()
    Code(
      byteRep := str.invoke[Array[Byte]]("getBytes"),
      dstAddress := fundamentalType.allocate(region, byteRep.length),
      fundamentalType.store(dstAddress, byteRep),
      dstAddress)
  }

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    fundamentalType.unstagedStoreAtAddress(addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalString(required)

  def sType: SStringPointer = SStringPointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SStringPointerCode(SStringPointer(this), addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SStringPointer(t) if t.equalModuloRequired(this) && !deepCopy =>
        value.asInstanceOf[SStringPointerCode].a
      case _ =>
        binaryFundamentalType.store(cb, region, value.asString.asBytes(), deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  def loadFromNested(cb: EmitCodeBuilder, addr: Code[Long]): Code[Long] = binaryFundamentalType.loadFromNested(cb, addr)
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)
}
