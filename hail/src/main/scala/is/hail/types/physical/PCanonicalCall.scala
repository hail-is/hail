package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SCanonicalCallValue}
import is.hail.types.physical.stypes.interfaces.SCall
import is.hail.utils._

final case class PCanonicalCall(required: Boolean = false) extends PCall {
  def _asIdent = "call"

  override def copiedType: PType = this

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCCall")

  val representation: PInt32 = PInt32(required)

  def byteSize: Long = representation.byteSize
  override def alignment: Long = representation.alignment

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering =
    representation.unsafeOrdering(sm) // this was a terrible idea

  def setRequired(required: Boolean) =
    if (required == this.required) this else PCanonicalCall(required)

  override def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    srcPType match {
      case pt: PCanonicalCall =>
        representation.unstagedStoreAtAddress(
          sm,
          addr,
          region,
          pt.representation,
          srcAddress,
          deepCopy,
        )
    }

  override def containsPointers: Boolean = representation.containsPointers

  override def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    srcPType match {
      case pt: PCanonicalCall =>
        representation._copyFromAddress(sm, region, pt.representation, srcAddress, deepCopy)
    }

  def sType: SCall = SCanonicalCall

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SCanonicalCallValue =
    new SCanonicalCallValue(cb.memoize(Region.loadInt(addr)))

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] = {
    value.st match {
      case SCanonicalCall =>
        val newAddr = cb.memoize(region.allocate(representation.alignment, representation.byteSize))
        storeAtAddress(cb, newAddr, region, value, deepCopy)
        newAddr
    }
  }

  def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit =
    cb += Region.storeInt(addr, value.asCall.canonicalCall(cb))

  def loadFromNested(addr: Code[Long]): Code[Long] = representation.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long =
    representation.unstagedLoadFromNested(addr)

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region)
    : Long =
    representation.unstagedStoreJavaObject(sm, annotation, region)

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit =
    representation.unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
}
