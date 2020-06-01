package is.hail.types.encoded
import is.hail.annotations.Region
import is.hail.asm4s.{Code, LocalRef, Value, coerce}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils._

case class ENDArray(elementType: EType, required: Boolean = false) extends EContainer {

  override def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    assert(pnd.elementType.required)
    val ndarray = coerce[Long](v)

    val shapeVariables: IndexedSeq[LocalRef[Long]] = (0 until pnd.nDims).map(i => mb.newLocal[Long](s"shape_$i"))
    val readShapes = shapeVariables.zipWithIndex.map{ case (shapeVar, i) => shapeVar := pnd.loadShape(ndarray, i)}
    val writeShapes = shapeVariables.map(shapeVar => out.get.writeLong(shapeVar))

    val loopVariables: IndexedSeq[LocalRef[Long]] = (0 until pnd.nDims).map(i => mb.newLocal[Long](s"index_$i"))


    val writeElemF = elementType.buildEncoder(pnd.elementType, mb.ecb)
    val code = Code(
      Code(readShapes),
      Code(writeShapes),
      Code(loopVariables.map(loopVar => loopVar := 0L))
    )
    ???
  }

  override def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = ???

  override def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = ???

  def _decodedPType(requestedType: Type): PType = {
    val requestedTNDArray = requestedType.asInstanceOf[TNDArray]
    val elementPType = elementType.decodedPType(requestedTNDArray.elementType)
    PCanonicalNDArray(elementPType, requestedTNDArray.nDims, required)
  }
  override def setRequired(required: Boolean): EType = ENDArray(elementType, required)

  override def _asIdent = s"ndarray_of_${elementType.asIdent}"
  override def _toPretty = s"ENDArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("ENDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }
}
