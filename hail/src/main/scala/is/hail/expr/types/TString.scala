package is.hail.expr.types

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PString
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TStringOptional extends TString(false)
case object TStringRequired extends TString(true)

class TString(override val required: Boolean) extends Type {
  lazy val physicalType: PString = PString(required)

  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("str")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]])

  override def fundamentalType: Type = TBinary(required)

  override def _showStr(a: Annotation, cfg: ShowStrConfig, sb: StringBuilder): Unit = {
    sb.append('"')
    sb.append(StringEscapeUtils.escapeString(a.asInstanceOf[String]))
    sb.append('"')
  }
}

object TString {
  def apply(required: Boolean = false): TString = if (required) TStringRequired else TStringOptional

  def unapply(t: TString): Option[Boolean] = Option(t.required)
}
