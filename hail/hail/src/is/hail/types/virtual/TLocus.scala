package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
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
    case Some(rg) => TLocus(rg)
    case None => TLocus.representation
  }
}

/** A locus on the reference genome `rg`. Two locus types are equal when their reference genomes
  * are, so loci on distinct copies of one layout have the same type.
  */
case class TLocus(rg: ReferenceGenome) extends Type {

  override def _toPretty = s"Locus(${rg.name})"

  override def pyString(sb: StringBuilder): Unit =
    sb ++= "locus<" ++= prettyIdentifier(rg.name) += '>': Unit

  override def _typeCheck(a: Any): Boolean = a.isInstanceOf[Locus]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean = true): ExtendedOrdering =
    rg.extendedLocusOrdering

  lazy val representation: TStruct = TLocus.representation

  def locusOrdering(sm: HailStateManager): Ordering[Locus] =
    rg.locusOrdering

  override def unify(concrete: Type): Boolean = concrete match {
    case TLocus(crg) => rg == crg
    case _ => false
  }

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case l: TLocus => rg == l.rg
      case _ => false
    }
}
