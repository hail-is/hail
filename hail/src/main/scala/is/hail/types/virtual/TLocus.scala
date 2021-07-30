package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.BroadcastValue
import is.hail.check._
import is.hail.types.physical.PLocus
import is.hail.types.virtual.TCall.representation
import is.hail.utils._
import is.hail.variant._

import scala.reflect.{ClassTag, classTag}

object TLocus {
  def apply(rg: ReferenceGenome): TLocus = TLocus(rg.broadcastRG)

  val representation: TStruct = {
    TStruct(
      "contig" -> TString,
      "position" -> TInt32)
  }

  def schemaFromRG(rg: Option[ReferenceGenome], required: Boolean = false): Type = rg match {
    case Some(ref) => TLocus(ref)
    case None => TLocus.representation
  }
}

case class TLocus(rgBc: BroadcastRG) extends Type {

  def rg: ReferenceGenome = rgBc.value

  def _toPretty = s"Locus($rg)"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("locus<")
    sb.append(prettyIdentifier(rg.name))
    sb.append('>')
  }
  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Locus]

  override def genNonmissingValue: Gen[Annotation] = Locus.gen(rg)

  override def scalaClassTag: ClassTag[Locus] = classTag[Locus]

  override lazy val ordering: ExtendedOrdering = mkOrdering()

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(rg.locusOrdering, missingEqual)

  lazy val representation: TStruct = TLocus.representation

  def locusOrdering: Ordering[Locus] = rg.locusOrdering

  override def unify(concrete: Type): Boolean = concrete match {
    case TLocus(crgBc) => rg == crgBc.value
    case _ => false
  }
}
