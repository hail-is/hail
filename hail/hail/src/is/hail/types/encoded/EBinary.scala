package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeInputBuffer, valueToRichCodeOutputBuffer}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue, SString}
import is.hail.types.virtual._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EBinaryCommon(required) {
  override def writeLength(cb: EmitCodeBuilder, out: Value[OutputBuffer], len: Code[Int]): Unit =
    cb += out.writeInt(len)

  override def readLength(cb: EmitCodeBuilder, in: Value[InputBuffer]): Value[Int] =
    cb.memoize(in.readInt(), "len")

  override def _asIdent = "binary"
  override def _toPretty = "EBinary"

  override def setRequired(newRequired: Boolean): EBinary = EBinary(newRequired)
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}
