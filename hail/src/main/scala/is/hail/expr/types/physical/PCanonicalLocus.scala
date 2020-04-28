package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder}
import is.hail.utils.FastIndexedSeq
import is.hail.variant._

object PCanonicalLocus {
  def apply(rg: ReferenceGenome): PCanonicalLocus = PCanonicalLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PCanonicalLocus = PCanonicalLocus(rg.broadcastRG, required)

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
  def rg: ReferenceGenome = rgBc.value

  def _asIdent = "locus"

  override def _pretty(sb: StringBuilder, indent: Call, compact: Boolean): Unit = sb.append(s"PCLocus($rg)")

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalLocus(this.rgBc, required)

  val representation: PStruct = PCanonicalLocus.representation(required)

  private[physical] def contigAddr(address: Code[Long]): Code[Long] = representation.loadField(address, 0)

  private[physical] def contigAddr(address: Long): Long = representation.loadField(address, 0)

  def contig(address: Long): String = contigType.loadString(contigAddr(address))

  lazy val contigType: PCanonicalString = representation.field("contig").typ.asInstanceOf[PCanonicalString]

  private[physical] def positionAddr(address: Code[Long]): Code[Long] = representation.loadField(address, 1)

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
      val bettercmp = PBetterLocus(rgBc, required).codeOrdering(mb, other)

      override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
        bettercmp.compareNonnull(coerce[bettercmp.T](toBetterLocus(mb, x)), coerce[bettercmp.T](y))
      }
    }
  }

  def toBetterLocus(mb: EmitMethodBuilder[_], address: Code[Long]): Code[Long] = {
    Code.memoize(address, "to_better_locus_addr") { address =>
      val s = contigType.loadString(contigAddr(address))
      val rgc = mb.getReferenceGenome(rg)
      val contigIdx = rgc.invoke[String, Int]("contigIndex", s)

      (contigIdx.toL << 32) | (position(address).toL)
    }
  }
}

object PCanonicalLocusSettable {
  def apply(sb: SettableBuilder, pt: PCanonicalLocus, name: String): PCanonicalLocusSettable = {
    new PCanonicalLocusSettable(pt,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Long](s"${ name }_contig"),
      sb.newSettable[Int](s"${ name }_position"))
  }
}

class PCanonicalLocusSettable(
  val pt: PCanonicalLocus,
  val a: Settable[Long],
  _contig: Settable[Long],
  _position: Settable[Int]
) extends PLocusValue with PSettable {
  def get = new PCanonicalLocusCode(pt, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, _contig, _position)

  def store(pc: PCode): Code[Unit] = {
    Code(
      a := pc.asInstanceOf[PCanonicalLocusCode].a,
      _contig := pt.contigAddr(a),
      _position := pt.position(a))
  }

  def contig(mb: EmitMethodBuilder[_]): PStringCode = new PCanonicalStringCode(pt.contigType, _contig)

  def position(): Code[Int] = _position.get
}

class PCanonicalLocusCode(val pt: PCanonicalLocus, val a: Code[Long]) extends PLocusCode {
  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def contig(mb: EmitMethodBuilder[_]): PStringCode = new PCanonicalStringCode(pt.contigType, pt.contigAddr(a))

  def position(): Code[Int] = pt.position(a)

  def getLocusObj(mb: EmitMethodBuilder[_]): Code[Locus] = {
    Code.memoize(a, "get_locus_code_memo") { a =>
      Code.invokeStatic2[Locus, String, Int, Locus]("apply",
        pt.contigType.loadString(pt.contigAddr(a)),
        pt.position(a))
    }
  }

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): PLocusValue = {
    val s = PCanonicalLocusSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): PLocusValue = memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] = Region.storeAddress(dst, a)
}
