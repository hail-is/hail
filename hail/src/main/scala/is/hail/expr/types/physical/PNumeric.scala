package is.hail.expr.types.physical

import is.hail.asm4s.{Code, TypeInfo, coerce}

abstract class PNumeric extends PType {
  type NType <: PType

  def zero: Code[NType]

  def add(a: Code[_], b: Code[_]): Code[NType]

  def multiply(a: Code[_], b: Code[_]): Code[NType]
}
