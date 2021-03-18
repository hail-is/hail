package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SCanonicalLocusPointer, SCanonicalLocusPointerCode, SStringPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils.FastIndexedSeq
import is.hail.variant._

object PCanonicalLocus {
  def apply(rg: ReferenceGenome): PCanonicalLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PCanonicalLocus = PCanonicalLocus(rg.broadcastRG, required)

  private def representation(required: Boolean = false): PCanonicalStruct = PCanonicalStruct(
    required,
    "contig" -> PCanonicalString(required = true),
    "position" -> PInt32(required = true))

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): PType = rg match {
    case Some(ref) => PCanonicalLocus(ref, required)
    case None => representation(required)
  }
}

final case class PCanonicalLocus(rgBc: BroadcastRG, required: Boolean = false) extends PLocus {

  def byteSize: Long = representation.byteSize
  override def alignment: Long = representation.alignment

  def rg: ReferenceGenome = rgBc.value

  def _asIdent = "locus"

  override def _pretty(sb: StringBuilder, indent: Call, compact: Boolean): Unit = sb.append(s"PCLocus($rg)")

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalLocus(this.rgBc, required)

  val representation: PCanonicalStruct = PCanonicalLocus.representation(required)

  private[physical] def contigAddr(address: Code[Long]): Code[Long] = representation.loadField(address, 0)

  private[physical] def contigAddr(address: Long): Long = representation.loadField(address, 0)

  def contig(address: Long): String = contigType.loadString(contigAddr(address))
  def position(address: Long): Int = Region.loadInt(representation.fieldOffset(address, 1))

  lazy val contigType: PCanonicalString = representation.field("contig").typ.asInstanceOf[PCanonicalString]

  def position(off: Code[Long]): Code[Int] = Region.loadInt(representation.loadField(off, 1))

  lazy val positionType: PInt32 = representation.field("position").typ.asInstanceOf[PInt32]

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(): UnsafeOrdering = {
    val localRGBc = rgBc
    val binaryOrd = representation.fieldType("contig").unsafeOrdering()

    new UnsafeOrdering {
      def compare(o1: Long, o2: Long): Int = {
        val cOff1 = representation.loadField(o1, 0)
        val cOff2 = representation.loadField(o2, 0)

        if (binaryOrd.compare(cOff1, cOff2) == 0) {
          val posOff1 = representation.loadField(o1, 1)
          val posOff2 = representation.loadField(o2, 1)
          java.lang.Integer.compare(Region.loadInt(posOff1), Region.loadInt(posOff2))
        } else {
          val contig1 = contigType.loadString(cOff1)
          val contig2 = contigType.loadString(cOff2)
          localRGBc.value.compare(contig1, contig2)
        }
      }
    }
  }

  override def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    srcPType match {
      case pt: PCanonicalLocus => representation.unstagedStoreAtAddress(addr, region, pt.representation, srcAddress, deepCopy)
    }
  }

  override def containsPointers: Boolean = representation.containsPointers

  def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    srcPType match {
      case pt: PCanonicalLocus => representation._copyFromAddress(region, pt.representation, srcAddress, deepCopy)
    }
  }

  def sType: SCanonicalLocusPointer = SCanonicalLocusPointer(this)

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SCanonicalLocusPointerCode(sType, addr)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    value.st match {
      case SCanonicalLocusPointer(pt) =>
        representation.store(cb, region, pt.representation.loadCheapPCode(cb, value.asInstanceOf[SCanonicalLocusPointerCode].a), deepCopy)
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    value.st match {
      case SCanonicalLocusPointer(pt) =>
        representation.storeAtAddress(cb, addr, region, pt.representation.loadCheapPCode(cb, value.asInstanceOf[SCanonicalLocusPointerCode].a), deepCopy)
    }
  }

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  def unstagedStoreLocus(addr: Long, contig: String, position: Int, region: Region): Unit = {
    contigType.unstagedStoreJavaObjectAtAddress(representation.fieldOffset(addr, 0), contig, region)
    positionType.unstagedStoreJavaObjectAtAddress(representation.fieldOffset(addr, 1), position, region)
  }

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    val myLocus = annotation.asInstanceOf[Locus]
    unstagedStoreLocus(addr, myLocus.contig, myLocus.position, region)
  }

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val addr = representation.allocate(region)
    unstagedStoreJavaObjectAtAddress(addr, annotation, region)
    addr
  }

  def constructFromPositionAndString(cb: EmitCodeBuilder, r: Value[Region], contig: Code[String], pos: Code[Int]): SCanonicalLocusPointerCode = {
    val contigType = representation.fieldType("contig").asInstanceOf[PCanonicalString]
    val contigCode = SStringPointer(contigType).constructFromString(cb, r, contig)
    val repr = representation.constructFromFields(cb, r, FastIndexedSeq(EmitCode.present(cb.emb, contigCode), EmitCode.present(cb.emb, primitive(pos))), deepCopy = false)
    new SCanonicalLocusPointerCode(SCanonicalLocusPointer(this), repr.a)
  }
}
