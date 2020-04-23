package is.hail.types.virtual

import is.hail.annotations._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.SortField
import is.hail.types.physical._
import is.hail.types.encoded._
import is.hail.shuffler._
import is.hail.io._

import scala.reflect.{ClassTag, _}

case class TShuffle (
  keyFields: Array[SortField],
  rowType: TStruct,
  rowEType: EBaseStruct,
  keyEType: EBaseStruct
) extends Type {
  val bufferSpec = shuffleBufferSpec

  val keyType = rowType.typeAfterSelectNames(keyFields.map(_.field))

  val rowCodecSpec = new TypedCodecSpec(rowEType, rowType, bufferSpec)

  val rowDecodedPType = rowCodecSpec.decodedPType()

  val keyCodecSpec = new TypedCodecSpec(keyEType, keyType, bufferSpec)

  val keyDecodedPType = keyCodecSpec.decodedPType()

  def _toPretty = "Shuffle"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = ???

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  val ordering: ExtendedOrdering = ExtendedOrdering.iterableOrdering(new ExtendedOrdering {
    override def compareNonnull(x: Any, y: Any): Int =
      java.lang.Integer.compare(
        java.lang.Byte.toUnsignedInt(x.asInstanceOf[Byte]),
        java.lang.Byte.toUnsignedInt(y.asInstanceOf[Byte]))
  })
}
