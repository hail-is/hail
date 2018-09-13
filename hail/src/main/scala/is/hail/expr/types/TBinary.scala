package is.hail.expr.types

import is.hail.annotations.CodeOrdering
import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PBinary
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TBinaryOptional extends TBinary(false)

case object TBinaryRequired extends TBinary(true)

class TBinary(override val required: Boolean) extends Type {
  lazy val physicalType: PBinary = PBinary(required)

  def _toPretty = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(Ordering.Iterable[Byte])

  override def byteSize: Long = 8
}

object TBinary {
  def apply(required: Boolean = false): TBinary = if (required) TBinaryRequired else TBinaryOptional

  def unapply(t: TBinary): Option[Boolean] = Option(t.required)

  def contentAlignment: Long = 4

  def contentByteSize(length: Int): Long = 4 + length

  def contentByteSize(length: Code[Int]): Code[Long] = (const(4) + length).toL

  def loadLength(region: Region, boff: Long): Int =
    region.loadInt(boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] =
    region.loadInt(boff)

  def bytesOffset(boff: Long): Long = boff + 4

  def bytesOffset(boff: Code[Long]): Code[Long] = boff + 4L

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentAlignment, contentByteSize(length))
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] = {
    region.allocate(const(contentAlignment), contentByteSize(length))
  }

}
