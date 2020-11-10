package is.hail.types.physical

import is.hail.asm4s.{Code, LineNumber, TypeInfo, coerce}

abstract class PNumeric extends PType {
  type NType <: PType

  def zero(implicit line: LineNumber): Code[NType]

  def add(a: Code[_], b: Code[_])(implicit line: LineNumber): Code[NType]

  def multiply(a: Code[_], b: Code[_])(implicit line: LineNumber): Code[NType]
}
