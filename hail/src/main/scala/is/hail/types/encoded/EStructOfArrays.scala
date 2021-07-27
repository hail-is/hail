package is.hail.types.encoded
import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.Type

case class EStructOfArrays(override val required: Boolean = false) extends EType {
  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = ???

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = ???

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = ???

  def _asIdent: String = ???

  def _toPretty: String = ???

  def _decodedSType(requestedType: Type): SType = ???

  def setRequired(required: Boolean): EType = if (required == this.required) this else EStructOfArrays(required)
}
