package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SCanonicalLocusPointer, SCanonicalLocusPointerCode}
import is.hail.variant._
import org.apache.spark.sql.Row

object PCanonicalLocus {
  def apply(rg: ReferenceGenome): PLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PCanonicalLocus(rg.broadcastRG, required)

  private def representation(required: Boolean = false): PStruct = PCanonicalStruct(
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
  override lazy val fundamentalType: PStruct = representation.fundamentalType

  def rg: ReferenceGenome = rgBc.value

  def _asIdent = "locus"

  override def _pretty(sb: StringBuilder, indent: Call, compact: Boolean): Unit = sb.append(s"PCLocus($rg)")

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalLocus(this.rgBc, required)

  val representation: PStruct = PCanonicalLocus.representation(required)

  private[physical] def contigAddr(address: Code[Long]): Code[Long] = representation.loadField(address, 0)

  private[physical] def contigAddr(address: Long): Long = representation.loadField(address, 0)

  def contig(address: Long): String = contigType.loadString(contigAddr(address))
  def position(address: Long): Int = Region.loadInt(representation.fieldOffset(address, 1))

  lazy val contigType: PCanonicalString = representation.field("contig").typ.asInstanceOf[PCanonicalString]

  def position(off: Code[Long]): Code[Int] = Region.loadInt(representation.loadField(off, 1))

  lazy val positionType: PInt32 = representation.field("position").typ.asInstanceOf[PInt32]

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(): UnsafeOrdering = {
    val repr = representation.fundamentalType

    val localRGBc = rgBc
    val binaryOrd = repr.fieldType("contig").asInstanceOf[PBinary].unsafeOrdering()

    new UnsafeOrdering {
      def compare(o1: Long, o2: Long): Int = {
        val cOff1 = repr.loadField(o1, 0)
        val cOff2 = repr.loadField(o2, 0)

        if (binaryOrd.compare(cOff1, cOff2) == 0) {
          val posOff1 = repr.loadField(o1, 1)
          val posOff2 = repr.loadField(o2, 1)
          java.lang.Integer.compare(Region.loadInt(posOff1), Region.loadInt(posOff2))
        } else {
          val contig1 = contigType.loadString(cOff1)
          val contig2 = contigType.loadString(cOff2)
          localRGBc.value.compare(contig1, contig2)
        }
      }
    }
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Long
      val bincmp = representation.fundamentalType.fieldType("contig").asInstanceOf[PBinary].codeOrdering(mb)

      override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
        val c1 = mb.newLocal[Long]("c1")
        val c2 = mb.newLocal[Long]("c2")

        val s1 = contigType.loadString(c1)
        val s2 = contigType.loadString(c2)

        val cmp = bincmp.compareNonnull(coerce[bincmp.T](c1), coerce[bincmp.T](c2))
        val codeRG = mb.getReferenceGenome(rg)

        Code.memoize(x, "plocus_code_ord_x", y, "plocus_code_ord_y") { (x, y) =>
          val p1 = Region.loadInt(representation.fieldOffset(x, 1))
          val p2 = Region.loadInt(representation.fieldOffset(y, 1))

          Code(
            c1 := representation.loadField(x, 0),
            c2 := representation.loadField(y, 0),
            cmp.ceq(0).mux(
              Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", p1, p2),
              codeRG.invoke[String, String, Int]("compare", s1, s2)))
        }
      }
    }
  }

  override def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    srcPType match {
      case pt: PCanonicalLocus => representation.unstagedStoreAtAddress(addr, region, pt.representation, srcAddress, deepCopy)
    }
  }

  override def encodableType: PType = representation.encodableType

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

  def loadFromNested(cb: EmitCodeBuilder, addr: Code[Long]): Code[Long] = addr

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation): Unit = {
    representation.unstagedStoreJavaObjectAtAddress(addr, annotation)

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    val myLocus = annotation.asInstanceOf[Locus]
    representation.unstagedStoreJavaObjectAtAddress(addr, Row(myLocus.contig, myLocus.position), region)
  }

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val addr = representation.allocate(region)
    unstagedStoreJavaObjectAtAddress(addr, annotation, region)
    addr
  }
}
