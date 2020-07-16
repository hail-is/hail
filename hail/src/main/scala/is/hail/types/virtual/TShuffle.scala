package is.hail.types.virtual

import is.hail.annotations._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.types.encoded._
import is.hail.services.shuffler._
import is.hail.io._
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case class TShuffle (
  keyFields: IndexedSeq[SortField],
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

  def _toPretty: String = {
    val sb = new StringBuilder()
    sb.append("Shuffle{")
    sb.append(Pretty.prettySortFields(keyFields))
    sb.append(",")
    sb.append(rowType.parsableString())
    sb.append(",")
    sb.append(rowEType.parsableString())
    sb.append(",")
    sb.append(keyEType.parsableString())
    sb.append("}")
    sb.result()
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = ???

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  override def mkOrdering(missingEqual: Boolean = true): ExtendedOrdering = TBinary.mkOrdering(missingEqual)
}
