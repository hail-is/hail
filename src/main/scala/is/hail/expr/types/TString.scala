package is.hail.expr.types

import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TStringOptional extends TString(false)
case object TStringRequired extends TString(true)

class TString(override val required: Boolean) extends Type {
  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("str")
  }
  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = TBinary(required).unsafeOrdering(missingGreatest)

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]])

  override def byteSize: Long = 8

  override def fundamentalType: Type = TBinary(required)

  def codeOrdering(mb: EmitMethodBuilder): CodeOrdering = TBinary(required).codeOrdering(mb)
}

object TString {
  def apply(required: Boolean = false): TString = if (required) TStringRequired else TStringOptional

  def unapply(t: TString): Option[Boolean] = Option(t.required)

  def loadString(region: Region, boff: Long): String = {
    val length = TBinary.loadLength(region, boff)
    new String(region.loadBytes(TBinary.bytesOffset(boff), length))
  }
}
