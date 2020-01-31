package is.hail.expr.types.virtual

import is.hail.annotations._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.types.physical.PString
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TStringOptional extends TString(false)
case object TStringRequired extends TString(true)

class TString(override val required: Boolean) extends Type {
  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("str")
  }

  override def _showStr(a: Annotation): String = "\"" + a.asInstanceOf[String] + "\""

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]])

  override def fundamentalType: Type = TBinary(required)
}

object TString {
  def apply(required: Boolean = false): TString = if (required) TStringRequired else TStringOptional

  def unapply(t: TString): Option[Boolean] = Option(t.required)
}
