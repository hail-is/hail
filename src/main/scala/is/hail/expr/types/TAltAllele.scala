package is.hail.expr.types

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.utils._
import is.hail.variant.AltAllele

import scala.reflect.{ClassTag, _}

case object TAltAlleleOptional extends TAltAllele(false)
case object TAltAlleleRequired extends TAltAllele(true)

class TAltAllele(override val required: Boolean) extends ComplexType {
  def _toString = "AltAllele"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[AltAllele]

  override def genNonmissingValue: Gen[Annotation] = AltAllele.gen

  override def desc: String = "An ``AltAllele`` is a Hail data type representing an alternate allele in the Variant Dataset."

  override def scalaClassTag: ClassTag[AltAllele] = classTag[AltAllele]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[AltAllele]])

  val representation: TStruct = TAltAllele.representation(required)
}

object TAltAllele {
  def apply(required: Boolean = false): TAltAllele = if (required) TAltAlleleRequired else TAltAlleleOptional

  def unapply(t: TAltAllele): Option[Boolean] = Option(t.required)

  def representation(required: Boolean = false): TStruct = {
    val t = TStruct(
      "ref" -> +TString(),
      "alt" -> +TString())
    t.setRequired(required).asInstanceOf[TStruct]
  }

}
