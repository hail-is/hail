package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBinaryPointerCode, SCanonicalShufflePointer, SCanonicalShufflePointerCode}
import is.hail.types.virtual._

final case class PCanonicalShuffle(
  val tShuffle: TShuffle,
  required: Boolean = false
) extends PShuffle {
  def _asIdent = "shuffle"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PCShuffle")

  val representation: PCanonicalBinary = PCanonicalBinary(required)

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalShuffle(tShuffle, required)

  def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def containsPointers: Boolean = representation.containsPointers

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    srcPType match {
      case t: PCanonicalShuffle =>
        representation.copyFromAddress(region, t.representation, srcAddress, deepCopy)
    }
  }

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    this.representation.unstagedStoreAtAddress(addr, region, srcPType.asInstanceOf[PCanonicalShuffle].representation, srcAddress, deepCopy)

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit =
    this.representation.unstagedStoreJavaObjectAtAddress(addr, annotation, region)

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long =
    this.representation.unstagedStoreJavaObject(annotation, region)

  def loadBinary(cb: EmitCodeBuilder, addr: Code[Long]): SBinaryPointerCode = representation.loadCheapPCode(cb, addr).asInstanceOf[SBinaryPointerCode]

  def sType: SCanonicalShufflePointer = SCanonicalShufflePointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SCanonicalShufflePointerCode(sType, representation.loadCheapPCode(cb, addr))

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SCanonicalShufflePointer(t) =>
        representation.store(cb, region, value.asInstanceOf[SCanonicalShufflePointerCode].shuffle, deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    value.st match {
      case SCanonicalShufflePointer(t) =>
        representation.storeAtAddress(cb, addr, region, value.asInstanceOf[SCanonicalShufflePointerCode].shuffle, deepCopy)
    }
  }

  def loadLength(bAddress: Long): Int = representation.loadLength(bAddress)

  def loadLength(bAddress: Code[Long]): Code[Int] = representation.loadLength(bAddress)

  def bytesAddress(boff: Long): Long = representation.bytesAddress(boff)

  def bytesAddress(boff: Code[Long]): Code[Long] = representation.bytesAddress(boff)

  def storeLength(boff: Long, len: Int): Unit = representation.storeLength(boff, len)

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit] = representation.storeLength(boff, len)

  def allocate(region: Region, length: Int): Long = representation.allocate(region, length)

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] = representation.allocate(region, length)

  def loadFromNested(addr: Code[Long]): Code[Long] = representation.loadFromNested(addr)

  def unstagedLoadFromNested(addr: Long): Long = representation.unstagedLoadFromNested(addr)
}
