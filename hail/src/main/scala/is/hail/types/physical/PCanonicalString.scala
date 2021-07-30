package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
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

  lazy val binaryRepresentation: PCanonicalBinary = PCanonicalBinary(required)

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    binaryRepresentation.copyFromAddress(region, srcPType.asInstanceOf[PString].binaryRepresentation, srcAddress, deepCopy)

  override def containsPointers: Boolean = true

  def loadLength(boff: Long): Int =
    this.binaryRepresentation.loadLength(boff)

  def loadLength(boff: Code[Long]): Code[Int] =
    this.binaryRepresentation.loadLength(boff)

  def loadString(bAddress: Long): String =
    new String(this.binaryRepresentation.loadBytes(bAddress))

  def loadString(bAddress: Code[Long]): Code[String] =
    Code.newInstance[String, Array[Byte]](this.binaryRepresentation.loadBytes(bAddress))

  def allocateAndStoreString(region: Region, str: String): Long = {
    val byteRep = str.getBytes()
    val dstAddrss = this.binaryRepresentation.allocate(region, byteRep.length)
    this.binaryRepresentation.store(dstAddrss, byteRep)
    dstAddrss
  }

  def allocateAndStoreString(mb: EmitMethodBuilder[_], region: Value[Region], str: Code[String]): Code[Long] = {
    val dstAddress = mb.genFieldThisRef[Long]()
    val byteRep = mb.genFieldThisRef[Array[Byte]]()
    Code(
      byteRep := str.invoke[Array[Byte]]("getBytes"),
      dstAddress := binaryRepresentation.allocate(region, byteRep.length),
      binaryRepresentation.store(dstAddress, byteRep),
      dstAddress)
  }

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    binaryRepresentation.unstagedStoreAtAddress(addr, region, srcPType.asInstanceOf[PString].binaryRepresentation, srcAddress, deepCopy)

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalString(required)

  def sType: SStringPointer = SStringPointer(setRequired(false).asInstanceOf[PCanonicalString])

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SCode = new SStringPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SStringPointer(t) if t.equalModuloRequired(this) && !deepCopy =>
        value.asInstanceOf[SStringPointerCode].a
      case _ =>
        binaryRepresentation.store(cb, region, value.asString.toBytes(), deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    cb += Region.storeAddress(addr, store(cb, region, value, deepCopy))
  }

  def loadFromNested(addr: Code[Long]): Code[Long] = binaryRepresentation.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long = binaryRepresentation.unstagedLoadFromNested(addr)

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    binaryRepresentation.unstagedStoreJavaObject(annotation.asInstanceOf[String].getBytes(), region)
  }

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    binaryRepresentation.unstagedStoreJavaObjectAtAddress(addr, annotation.asInstanceOf[String].getBytes(), region)
  }
}

object PCanonicalString {
  def apply(required: Boolean = false): PCanonicalString = if (required) PCanonicalStringRequired else PCanonicalStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)
}
