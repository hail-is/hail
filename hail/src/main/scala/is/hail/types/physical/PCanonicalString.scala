package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.utils.FastIndexedSeq

case object PCanonicalStringOptional extends PCanonicalString(false)
case object PCanonicalStringRequired extends PCanonicalString(true)

class PCanonicalString(val required: Boolean) extends PString {
  def _asIdent = "string"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCString")

  override def byteSize: Long = 8

  lazy val binaryFundamentalType: PCanonicalBinary = PCanonicalBinary(required)
  override lazy val binaryEncodableType: PCanonicalBinary =  PCanonicalBinary(required)

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = {
    this.fundamentalType.copyFromType(
      mb, region, srcPType.asInstanceOf[PString].fundamentalType, srcAddress, deepCopy
    )
  }

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    this.copyFromType(mb, region, srcPType, stackValue.asInstanceOf[Code[Long]], deepCopy)

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

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    fundamentalType.constructAtAddress(mb, addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    fundamentalType.constructAtAddress(addr, region, srcPType.fundamentalType, srcAddress, deepCopy)

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalString(required)
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)
}

class PCanonicalStringCode(val pt: PCanonicalString, a: Code[Long]) extends PStringCode {
  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def loadLength(): Code[Int] = pt.loadLength(a)

  def loadString(): Code[String] = pt.loadString(a)

  def asBytes(): PBinaryCode = new PCanonicalBinaryCode(pt.binaryFundamentalType, a)

  def memoize(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeImpl(cb, name)

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = defaultMemoizeFieldImpl(cb, name)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}
