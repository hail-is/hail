package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SPackedLocus, SPackedLocusValue}
import is.hail.variant.{Locus, ReferenceGenome}

/** A locus stored as a single packed integer, `(contig_index << 32) | position`, where the contig
  * index is measured against `rg`.
  */
final case class PPackedLocus(rg: ReferenceGenome, required: Boolean = false)
    extends PLocus with PPrimitive {

  override def byteSize: Long = 8

  override def _asIdent = s"packed_locus_${rg.name}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit =
    sb ++= "PPackedLocus(" ++= rg.name += ')': Unit

  override def setRequired(required: Boolean): PPackedLocus =
    if (required == this.required) this else PPackedLocus(rg, required)

  override def sType: SPackedLocus = SPackedLocus(rg)

  override def unsafeOrdering: UnsafeOrdering = new UnsafeOrdering {
    override def compare(o1: Long, o2: Long): Int =
      java.lang.Long.compare(Region.loadLong(o1), Region.loadLong(o2))
  }

  def contigIdx(address: Long): Int = (Region.loadLong(address) >>> 32).toInt

  override def contig(address: Long): String =
    rg.getContig(contigIdx(address))

  override def position(address: Long): Int = Region.loadLong(address).toInt

  override def position(address: Code[Long]): Code[Int] = Region.loadLong(address).toI

  override def packed(address: Long): Long = Region.loadLong(address)

  override def positionType: PInt32 = PInt32(true)

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SPackedLocusValue =
    new SPackedLocusValue(sType, cb.memoize(Region.loadLong(addr)))

  override def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit =
    cb += Region.storeLong(addr, value.asLocus.packed(cb))

  override def unstagedStoreAtAddress(
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    srcPType match {
      case pt: PLocus =>
        assert(pt.isOfType(this))
        Region.storeLong(addr, pt.packed(srcAddress))
    }

  override def _copyFromAddress(
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    srcPType match {
      case _: PPackedLocus =>
        super._copyFromAddress(region, srcPType, srcAddress, deepCopy)
      case _ =>
        val addr = region.allocate(alignment, byteSize)
        unstagedStoreAtAddress(addr, region, srcPType, srcAddress, deepCopy)
        addr
    }

  override def unstagedStoreLocus(
    addr: Long,
    contig: String,
    position: Int,
    region: Region,
  ): Unit =
    Region.storeLong(
      addr,
      PLocus.pack(rg.getContigIndex(contig), position),
    )

  override def unstagedStoreJavaObjectAtAddress(
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit = {
    val locus = annotation.asInstanceOf[Locus]
    unstagedStoreLocus(addr, locus.contig, locus.position, region)
  }
}
