package is.hail.expr.types

import scala.language.implicitConversions

package object physical {
  // FIXME required to compile
  implicit def toType(p: PType): Type = ???
  implicit def toTArray(p: PArray): TArray = ???
  implicit def toTBaseStruct(p: PBaseStruct): TBaseStruct = ???
  implicit def toTStruct(p: PStruct): TStruct = ???
  implicit def toTTuple(p: PTuple): TTuple = ???
  implicit def toTDict(p: PDict): TDict = ???
  implicit def toTSet(p: PSet): TSet  = ???
  implicit def toTInterval(p: PInterval): TInterval  = ???
}
