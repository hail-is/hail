package is.hail.expr.ir

import is.hail.asm4s.{Code, TypeInfo}
import is.hail.expr.types.physical.{PType, PVoid}

abstract class PSettable {
  def load(): PValue

  def store(v: PValue): Code[Unit]

  def :=(v: PValue): Code[Unit] = store(v)
}

object PValue {
  def _empty: PValue = PValue(PVoid, Code._empty)
}

case class PValue(
  pt: PType,
  code: Code[_]) {
  def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def typeInfo: TypeInfo[_] = typeToTypeInfo(pt)
}
