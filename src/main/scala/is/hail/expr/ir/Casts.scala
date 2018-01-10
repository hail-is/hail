package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.types._

import scala.language.existentials

object Casts {
  private val casts: Map[(Type, Type), (Code[T] => Code[_]) forSome {type T}] = Map(
    (TInt32(), TInt64()) -> ((x: Code[Int]) => x.toL),
    (TInt32(), TFloat32()) -> ((x: Code[Int]) => x.toF),
    (TInt32(), TFloat64()) -> ((x: Code[Int]) => x.toD),
    (TInt64(), TFloat32()) -> ((x: Code[Long]) => x.toF),
    (TInt64(), TFloat64()) -> ((x: Code[Long]) => x.toD),
    (TFloat32(), TFloat64()) -> ((x: Code[Float]) => x.toD)
)

  def get(from: Type, to: Type): Code[_] => Code[_] =
    casts(from -> to).asInstanceOf[Code[_] => Code[_]]

  def valid(from: Type, to: Type): Boolean =
    casts.contains(from -> to)
}
