package is.hail.types

import is.hail.types.virtual._

object MapTypes {
  def apply(f: Type => Type)(typ: Type): Type = typ match {
    case TInterval(pointType) => TInterval(f(pointType))
    case TArray(elt) => TArray(f(elt))
    case TSet(elt) => TSet(f(elt))
    case TDict(kt, vt) => TDict(f(kt), f(vt))
    case t: TStruct => TStruct(t.fields.map(field => (field.name, f(field.typ))): _*)
    case t: TTuple => TTuple(t.types.map(f): _*)
    case _ => typ
  }

  def recur(f: Type => Type)(typ: Type): Type = {
    def recurF(t: Type): Type = f(apply(t => recurF(t))(t))
    recurF(typ)
  }

  def foreach(f: Type => Unit)(typ: Type): Unit =
    recur { t => f(t); t }(typ)
}
