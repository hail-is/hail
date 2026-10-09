package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SPackedLocus, SPackedLocusValue}
import is.hail.variant.Locus

/** A locus stored as a single packed integer, `(contig_index << 32) | position`, where the contig
  * index is measured against the session's registered reference genome named `rgName`.
  */
final case class PPackedLocus(rgName: String, required: Boolean = false)
    extends PLocus with PPrimitive {

  override def rg: String = rgName

  override def byteSize: Long = 8

  override def _asIdent = s"packed_locus_$rgName"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    sb ++= "PPackedLocus(" ++= rgName += ')': Unit

  override def setRequired(required: Boolean): PPackedLocus =
    if (required == this.required) this else PPackedLocus(rgName, required)

  override def sType: SPackedLocus = SPackedLocus(rgName)

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = new UnsafeOrdering {
    override def compare(o1: Long, o2: Long): Int =
      java.lang.Long.compare(Region.loadLong(o1), Region.loadLong(o2))
  }

  def contigIdx(address: Long): Int = (Region.loadLong(address) >>> 32).toInt

  override def contig(sm: HailStateManager, address: Long): String =
    sm.referenceGenomes(rgName).getContig(contigIdx(address))

  override def position(address: Long): Int = Region.loadLong(address).toInt

  override def position(address: Code[Long]): Code[Int] = Region.loadLong(address).toI

  override def packed(sm: HailStateManager, address: Long): Long = Region.loadLong(address)

  override def positionType: PInt32 = PInt32(true)

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SPackedLocusValue =
    new SPackedLocusValue(sType, cb.memoize(Region.loadLong(addr)))

  override def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit =
    cb += Region.storeLong(addr, value.asLocus.packed(cb))

  override def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    srcPType match {
      case pt: PLocus =>
        assert(pt.isOfType(this))
        Region.storeLong(addr, pt.packed(sm, srcAddress))
    }

  override def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    srcPType match {
      case _: PPackedLocus =>
        super._copyFromAddress(sm, region, srcPType, srcAddress, deepCopy)
      case _ =>
        val addr = region.allocate(alignment, byteSize)
        unstagedStoreAtAddress(sm, addr, region, srcPType, srcAddress, deepCopy)
        addr
    }

  override def unstagedStoreLocus(
    sm: HailStateManager,
    addr: Long,
    contig: String,
    position: Int,
    region: Region,
  ): Unit =
    Region.storeLong(
      addr,
      PLocus.pack(sm.referenceGenomes(rgName).getContigIndex(contig), position),
    )

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit = {
    val locus = annotation.asInstanceOf[Locus]
    unstagedStoreLocus(sm, addr, locus.contig, locus.position, region)
  }
}
