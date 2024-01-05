package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SCanonicalLocusPointer, SCanonicalLocusPointerValue, SStackStruct}
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils.FastSeq
import is.hail.variant._

object PCanonicalLocus {
  private def representation(required: Boolean = false): PCanonicalStruct = PCanonicalStruct(
    required,
    "contig" -> PCanonicalString(required = true),
    "position" -> PInt32(required = true))

  def schemaFromRG(rg: Option[String], required: Boolean = false): PType = rg match {
    case Some(name) => PCanonicalLocus(name, required)
    case None => representation(required)
  }
}

final case class PCanonicalLocus(rgName: String, required: Boolean = false) extends PLocus {

  def byteSize: Long = representation.byteSize
  override def alignment: Long = representation.alignment

  override def copiedType: PType = this

  def rg: String = rgName

  def _asIdent = s"locus_$rgName"

  override def _pretty(sb: StringBuilder, indent: Call, compact: Boolean): Unit = sb.append(s"PCLocus($rgName)")

  def setRequired(required: Boolean): PCanonicalLocus =
    if (required == this.required) this else PCanonicalLocus(this.rgName, required)

  val representation: PCanonicalStruct = PCanonicalLocus.representation(required)

  private[physical] def contigAddr(address: Code[Long]): Code[Long] = representation.loadField(address, 0)

  private[physical] def contigAddr(address: Long): Long = representation.loadField(address, 0)

  def contig(address: Long): String = contigType.loadString(contigAddr(address))
  def position(address: Long): Int = Region.loadInt(representation.fieldOffset(address, 1))

  lazy val contigType: PCanonicalString = representation.field("contig").typ.asInstanceOf[PCanonicalString]

  def position(off: Code[Long]): Code[Int] = Region.loadInt(representation.loadField(off, 1))

  lazy val positionType: PInt32 = representation.field("position").typ.asInstanceOf[PInt32]

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = {
    val localRg = sm.referenceGenomes(rgName)
    val binaryOrd = representation.fieldType("contig").unsafeOrdering(sm)

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
          localRg.compare(contig1, contig2)
        }
      }
    }
  }

  override def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    srcPType match {
      case pt: PCanonicalLocus => representation.unstagedStoreAtAddress(sm, addr, region, pt.representation, srcAddress, deepCopy)
    }
  }

  override def containsPointers: Boolean = representation.containsPointers

  override def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    srcPType match {
      case pt: PCanonicalLocus => representation._copyFromAddress(sm, region, pt.representation, srcAddress, deepCopy)
    }
  }

  def sType: SCanonicalLocusPointer = SCanonicalLocusPointer(setRequired(false))

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SCanonicalLocusPointerValue = {
    val a = cb.memoize(addr)
    new SCanonicalLocusPointerValue(sType, a, cb.memoize(contigAddr(a)), cb.memoize(position(a)))
  }


  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    value.st match {
      case SCanonicalLocusPointer(pt) =>
        representation.store(cb, region, pt.representation.loadCheapSCode(cb, value.asInstanceOf[SCanonicalLocusPointerValue].a), deepCopy)
      case _ =>
        val addr = cb.memoize(representation.allocate(region))
        storeAtAddress(cb, addr, region, value, deepCopy)
        addr
    }
  }

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    value.st match {
      case SCanonicalLocusPointer(pt) =>
        representation.storeAtAddress(cb, addr, region, pt.representation.loadCheapSCode(cb, value.asInstanceOf[SCanonicalLocusPointerValue].a), deepCopy)
      case _ =>
        val loc = value.asLocus
        representation.storeAtAddress(cb, addr, region,
          SStackStruct.constructFromArgs(cb, region, representation.virtualType,
            EmitCode.present(cb.emb, loc.contig(cb)), EmitCode.present(cb.emb, primitive(loc.position(cb)))),
          deepCopy)
    }
  }

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  override def unstagedStoreLocus(sm: HailStateManager, addr: Long, contig: String, position: Int, region: Region): Unit = {
    contigType.unstagedStoreJavaObjectAtAddress(sm, representation.fieldOffset(addr, 0), contig, region)
    positionType.unstagedStoreJavaObjectAtAddress(sm, representation.fieldOffset(addr, 1), position, region)
  }

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
    val myLocus = annotation.asInstanceOf[Locus]
    unstagedStoreLocus(sm, addr, myLocus.contig, myLocus.position, region)
  }

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val addr = representation.allocate(region)
    unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
    addr
  }

  def constructFromContigAndPosition(cb: EmitCodeBuilder, r: Value[Region], contig: Code[String], pos: Code[Int]): SCanonicalLocusPointerValue = {
    val position = cb.memoize(pos)
    val contigType = representation.fieldType("contig").asInstanceOf[PCanonicalString]
    val contigCode = contigType.sType.constructFromString(cb, r, contig)
    val repr = representation.constructFromFields(cb, r, FastSeq(EmitCode.present(cb.emb, contigCode), EmitCode.present(cb.emb, primitive(position))), deepCopy = false)
    new SCanonicalLocusPointerValue(SCanonicalLocusPointer(setRequired(false)), repr.a, contigCode.a, position)
  }
}
