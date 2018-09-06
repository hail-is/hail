package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.check._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.TLocus
import is.hail.utils._
import is.hail.variant._

import scala.reflect.{ClassTag, classTag}

object PLocus {
  def representation(required: Boolean = false): PStruct = {
    val rep = PStruct(
      "contig" -> +PString(),
      "position" -> +PInt32())
    rep.setRequired(required).asInstanceOf[PStruct]
  }

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): PType = rg match {
    case Some(ref) => PLocus(ref)
    case None => PLocus.representation(required)
  }
}

case class PLocus(rg: RGBase, override val required: Boolean = false) extends ComplexPType {
  def virtualType: TLocus = TLocus(rg, required)

  def _toPretty = s"Locus($rg)"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("locus<")
    sb.append(prettyIdentifier(rg.name))
    sb.append('>')
  }

  override def scalaClassTag: ClassTag[Locus] = classTag[Locus]

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val repr = representation.fundamentalType

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        val contig1 = PString.loadString(r1, cOff1)
        val contig2 = PString.loadString(r2, cOff2)

        val c = rg.compare(contig1, contig2)
        if (c != 0)
          return c

        val posOff1 = repr.loadField(r1, o1, 1)
        val posOff2 = repr.loadField(r2, o2, 1)
        java.lang.Integer.compare(r1.loadInt(posOff1), r2.loadInt(posOff2))
      }
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Long

      override def compareNonnull(rx: Code[Region], x: Code[Long], ry: Code[Region], y: Code[Long], missingGreatest: Boolean): Code[Int] = {
        val cmp = mb.newLocal[Int]

        val c1 = representation.loadField(rx, x, 0)
        val c2 = representation.loadField(ry, y, 0)

        val s1 = Code.invokeScalaObject[Region, Long, String](PString.getClass, "loadString", rx, c1)
        val s2 = Code.invokeScalaObject[Region, Long, String](PString.getClass, "loadString", ry, c2)

        val p1 = rx.loadInt(representation.fieldOffset(x, 1))
        val p2 = ry.loadInt(representation.fieldOffset(y, 1))

        val codeRG = mb.getReferenceGenome(rg.asInstanceOf[ReferenceGenome])

        Code(
          cmp := codeRG.invoke[String, String, Int]("compare", s1, s2),
          cmp.ceq(0).mux(
            Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", p1, p2),
            cmp))
      }
    }
  }

  val representation: PStruct = PLocus.representation(required)

  override def unify(concrete: PType): Boolean = concrete match {
    case PLocus(crg, _) => rg.unify(crg)
    case _ => false
  }

  override def clear(): Unit = rg.clear()

  override def subst() = copy(rg = rg.subst())
  
  def contig(region: Code[Region], off: Code[Long]): Code[Long] = representation.loadField(region, off, 0)
  
  def position(region: Code[Region], off: Code[Long]): Code[Int] = region.loadInt(representation.loadField(region, off, 1))
}
