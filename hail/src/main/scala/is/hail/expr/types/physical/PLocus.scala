package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, coerce}
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TLocus
import is.hail.utils._
import is.hail.variant._

object PLocus {
  def apply(rg: ReferenceGenome): PLocus = PLocus(rg.broadcastRG)

  def apply(rg: ReferenceGenome, required: Boolean): PLocus = PLocus(rg.broadcastRG, required)

  def representation(required: Boolean = false): PStruct = PStruct(
      required,
      "contig" -> PString(required = true),
      "position" -> PInt32(required = true))

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): PType = rg match {
    case Some(ref) => PLocus(ref)
    case None => PLocus.representation(required)
  }
}

case class PLocus(rgBc: BroadcastRG, override val required: Boolean = false) extends ComplexPType {
  def rg: ReferenceGenome = rgBc.value

  lazy val virtualType: TLocus = TLocus(rgBc, required)

  def _asIdent = "locus"
  def _toPretty = s"Locus($rg)"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("locus<")
    sb.append(prettyIdentifier(rg.name))
    sb.append('>')
  }

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(): UnsafeOrdering = {
    val repr = representation.fundamentalType

    val localRGBc = rgBc
    val binaryOrd = repr.fieldType("contig").asInstanceOf[PBinary].unsafeOrdering()

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        if (binaryOrd.compare(r1, cOff1, r2, cOff2) == 0) {
          val posOff1 = repr.loadField(r1, o1, 1)
          val posOff2 = repr.loadField(r2, o2, 1)
          java.lang.Integer.compare(Region.loadInt(posOff1), Region.loadInt(posOff2))
        } else {
          val contig1 = PString.loadString(r1, cOff1)
          val contig2 = PString.loadString(r2, cOff2)
          localRGBc.value.compare(contig1, contig2)
        }
      }
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Long
      val contigBin = representation.fundamentalType.fieldType("contig").asInstanceOf[PBinary]
      val bincmp = contigBin.codeOrdering(mb)

      override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
        val c1 = mb.newLocal[Long]("c1")
        val c2 = mb.newLocal[Long]("c2")

        val s1 = PString.loadString(c1)
        val s2 = PString.loadString(c2)

        val p1 = Region.loadInt(representation.fieldOffset(x, 1))
        val p2 = Region.loadInt(representation.fieldOffset(y, 1))

        val cmp = bincmp.compareNonnull(coerce[bincmp.T](c1), coerce[bincmp.T](c2))
        val codeRG = mb.getReferenceGenome(rg.asInstanceOf[ReferenceGenome])

        Code(
          c1 := representation.loadField(x, 0),
          c2 := representation.loadField(y, 0),
          cmp.ceq(0).mux(
            Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", p1, p2),
            codeRG.invoke[String, String, Int]("compare", s1, s2)))
      }
    }
  }

  val representation: PStruct = PLocus.representation(required)

  def contig(region: Code[Region], off: Code[Long]): Code[Long] = representation.loadField(region, off, 0)

  def position(region: Code[Region], off: Code[Long]): Code[Int] = Region.loadInt(representation.loadField(region, off, 1))
}
