package is.hail.types.encoded
import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.PType
import is.hail.types.virtual.Type

class ENDArray(val elementType: EType) extends EContainer {

  override def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = ???

  override def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = ???

  override def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = ???

  override def _asIdent: String = ???

  override def _toPretty: String = ???

  override def _decodedPType(requestedType: Type): PType = ???

  override def setRequired(required: Boolean): EType = ???

  override def required: Boolean = ???
}
