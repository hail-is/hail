package is.hail.expr.types

import is.hail.annotations._
import is.hail.check._
import is.hail.utils._
import is.hail.variant._

import scala.reflect.ClassTag
import scala.reflect.classTag

object TLocus {
  def representation(required: Boolean = false): TStruct = {
    val rep = TStruct(
      "contig" -> +TString(),
      "position" -> +TInt32())
    rep.setRequired(required).asInstanceOf[TStruct]
  }

  def schemaFromGR(rg: Option[ReferenceGenome], required: Boolean = false): Type = rg match {
    case Some(ref) => TLocus(ref)
    case None => TLocus.representation(required)
  }
}

case class TLocus(rg: RGBase, override val required: Boolean = false) extends ComplexType {
  def _toString = s"Locus($rg)"

  override def _toPyString(sb: StringBuilder): Unit = {
    sb.append("locus<")
    sb.append(prettyIdentifier(rg.name))
    sb.append('>')
  }
  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Locus.gen(rg.asInstanceOf[ReferenceGenome])

  override def desc: String = "A ``Locus(GR)`` is a Hail data type representing a specific genomic location in the Variant Dataset. It is parameterized by a reference genome (RG) such as GRCh37 or GRCh38."

  override def scalaClassTag: ClassTag[Locus] = classTag[Locus]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(rg.locusOrdering)

  // FIXME: Remove when representation of contig/position is a naturally-ordered Long
  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val repr = representation.fundamentalType

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val cOff1 = repr.loadField(r1, o1, 0)
        val cOff2 = repr.loadField(r2, o2, 0)

        val contig1 = TString.loadString(r1, cOff1)
        val contig2 = TString.loadString(r2, cOff2)

        val c = rg.compare(contig1, contig2)
        if (c != 0)
          return c

        val posOff1 = repr.loadField(r1, o1, 1)
        val posOff2 = repr.loadField(r2, o2, 1)
        java.lang.Integer.compare(r1.loadInt(posOff1), r2.loadInt(posOff2))
      }
    }
  }

  val representation: TStruct = TLocus.representation(required)

  def locusOrdering: Ordering[Locus] = rg.locusOrdering

  override def unify(concrete: Type): Boolean = concrete match {
    case TLocus(crg, _) => rg.unify(crg)
    case _ => false
  }

  override def clear(): Unit = rg.clear()

  override def subst() = rg.subst().locusType
}
