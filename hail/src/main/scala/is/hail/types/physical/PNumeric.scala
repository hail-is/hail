package is.hail.types.physical

import is.hail.asm4s.{Code, Value}

abstract class PNumeric extends PType {
  type NType <: PType

  def zero: Value[NType]

  def add(a: Code[_], b: Code[_]): Code[NType]

  def multiply(a: Code[_], b: Code[_]): Code[NType]
}
