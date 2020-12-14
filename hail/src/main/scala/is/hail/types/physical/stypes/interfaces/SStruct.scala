package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitSCode}
import is.hail.types.physical.{PBaseStruct}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}

trait SStruct extends SType

trait SStructSettable extends SBaseStructValue with SSettable

trait SBaseStructValue extends SValue {
  def pt: PBaseStruct

  def isFieldMissing(fieldIdx: Int)(implicit line: LineNumber): Code[Boolean]

  def isFieldMissing(fieldName: String)(implicit line: LineNumber): Code[Boolean] =
    isFieldMissing(pt.fieldIdx(fieldName))

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitSCode

  def loadField(cb: EmitCodeBuilder, fieldName: String): IEmitSCode =
    loadField(cb, pt.fieldIdx(fieldName))
}

trait SBaseStructCode extends SCode {
  def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue
}

