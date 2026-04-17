package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.asm4s.implicits.{
  valueToRichCodeInputBuffer, valueToRichCodeOutputBuffer, valueToRichCodeRegion,
}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual._



final case class EArray(override val elementType: EType, override val required: Boolean = false)
    extends EArrayCommon(elementType, required) {
  override def writeLength(cb: EmitCodeBuilder, out: Value[OutputBuffer], len: Code[Int]): Unit = cb += out.writeInt(len)
  override def readLength(cb: EmitCodeBuilder, in: Value[InputBuffer]): Value[Int] = cb.memoize(in.readInt(), "len")

  override def _asIdent = s"array_of_${elementType.asIdent}"
  override def _toPretty = s"EArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb ++= "EArray["
    elementType.pretty(sb, indent, compact)
    sb += ']'
  }

  override def setRequired(newRequired: Boolean): EArray = EArray(elementType, newRequired)
}
