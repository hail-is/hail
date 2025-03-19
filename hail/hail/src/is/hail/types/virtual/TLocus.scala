package is.hail.types.virtual

import is.hail.annotations._
import is.hail.utils._
import is.hail.variant._

object TLocus {
  val representation: TStruct =
    TStruct(
      "contig" -> TString,
      "position" -> TInt32,
    )

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): Type = rg match {
    // must match tlocus.schema_from_rg
    case Some(name) => TLocus(name)
    case None => TLocus.representation
  }
}

case class TLocus(rg: ReferenceGenome) extends Type {

  def _toPretty = s"Locus($rg)"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("locus<")
    sb.append(prettyIdentifier(rg.name))
    sb.append('>')
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Locus]

  override def mkOrdering(missingEqual: Boolean = true): ExtendedOrdering =
    rg.extendedLocusOrdering

  lazy val representation: TStruct = TLocus.representation

  override def unify(concrete: Type): Boolean =
    isIsomorphicTo(concrete)

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case l: TLocus => rg == l.rg
      case _ => false
    }
}
