package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types._
import is.hail.types.virtual._

import scala.language.existentials

object Casts {
  private val casts: Map[(Type, Type), ((Code[T], LineNumber) => Code[_]) forSome {type T}] = Map(
    (TInt32, TInt32) -> ((x: Code[Int], line: LineNumber) => x),
    (TInt32, TInt64) -> ((x: Code[Int], line: LineNumber) => x.toL(line)),
    (TInt32, TFloat32) -> ((x: Code[Int], line: LineNumber) => x.toF(line)),
    (TInt32, TFloat64) -> ((x: Code[Int], line: LineNumber) => x.toD(line)),
    (TInt64, TInt32) -> ((x: Code[Long], line: LineNumber) => x.toI(line)),
    (TInt64, TInt64) -> ((x: Code[Long], line: LineNumber) => x),
    (TInt64, TFloat32) -> ((x: Code[Long], line: LineNumber) => x.toF(line)),
    (TInt64, TFloat64) -> ((x: Code[Long], line: LineNumber) => x.toD(line)),
    (TFloat32, TInt32) -> ((x: Code[Float], line: LineNumber) => x.toI(line)),
    (TFloat32, TInt64) -> ((x: Code[Float], line: LineNumber) => x.toL(line)),
    (TFloat32, TFloat32) -> ((x: Code[Float], line: LineNumber) => x),
    (TFloat32, TFloat64) -> ((x: Code[Float], line: LineNumber) => x.toD(line)),
    (TFloat64, TInt32) -> ((x: Code[Double], line: LineNumber) => x.toI(line)),
    (TFloat64, TInt64) -> ((x: Code[Double], line: LineNumber) => x.toL(line)),
    (TFloat64, TFloat32) -> ((x: Code[Double], line: LineNumber) => x.toF(line)),
    (TFloat64, TFloat64) -> ((x: Code[Double], line: LineNumber) => x),
    (TInt32, TCall) -> ((x: Code[Int], line: LineNumber) => x))

  def get(from: Type, to: Type)(implicit line: LineNumber): Code[_] => Code[_] =
    v => casts(from -> to).asInstanceOf[(Code[_], LineNumber) => Code[_]](v, line)

  def valid(from: Type, to: Type): Boolean =
    casts.contains(from -> to)
}
