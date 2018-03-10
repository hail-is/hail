package is.hail.expr

import is.hail.asm4s.Code
import is.hail.expr.types._

sealed trait Fun {
  def captureType(): Fun = this

  def retType: Type

  def subst(): Fun
}

case class Transformation[T, U](f: T => U, fcode: Code[T] => CM[Code[U]]) extends (T => U) {
  def apply(t: T): U = f(t)
}

case class UnaryDependentFunCode[T, U](retType: Type, code: () => Code[T] => CM[Code[U]]) extends Fun {
  override def captureType() = UnaryFunCode(retType, code())

  def apply(ct: Code[T]): CM[Code[U]] =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class UnaryDependentFun[T, U](retType: Type, code: () => T => U) extends Fun {
  override def captureType() = UnaryFun(retType, code())

  def apply(t: T): U =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class UnaryFunCode[T, U](retType: Type, code: Code[T] => CM[Code[U]]) extends Fun with (Code[T] => CM[Code[U]]) {
  def apply(ct: Code[T]): CM[Code[U]] = code(ct)

  def subst() = UnaryFunCode[T, U](retType.subst(), code)
}

case class BinarySpecialCode[T, U, V](retType: Type, code: (Code[T], Code[U]) => CM[Code[V]])
  extends Fun with Serializable with ((Code[T], Code[U]) => CM[Code[V]]) {
  def apply(t: Code[T], u: Code[U]): CM[Code[V]] = code(t, u)

  def subst() = BinarySpecialCode(retType.subst(), code)
}

case class BinaryFunCode[T, U, V](retType: Type, code: (Code[T], Code[U]) => CM[Code[V]]) extends Fun with Serializable with ((Code[T], Code[U]) => CM[Code[V]]) {
  def apply(t: Code[T], u: Code[U]): CM[Code[V]] = code(t, u)

  def subst() = BinaryFunCode(retType.subst(), code)
}

case class BinaryDependentFun[T, U, V](retType: Type, code: () => (T, U) => V) extends Fun {
  override def captureType() = BinaryFun(retType, code())

  def apply(t: T, u: U): V =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity0Aggregator[T, U](retType: Type, ctor: () => TypedAggregator[U]) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst(), ctor)
}

case class Arity0DependentAggregator[T, U](retType: Type, ctor: () => (() => TypedAggregator[U])) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst(), ctor())
}

class TransformedAggregator[T](val prev: TypedAggregator[T], transform: (Any) => Any) extends TypedAggregator[T] {
  def seqOp(x: Any) = prev.seqOp(transform(x))

  def combOp(agg2: this.type) = prev.combOp(agg2.prev)

  def result = prev.result

  def copy() = new TransformedAggregator(prev.copy(), transform)
}

case class Arity1Aggregator[T, U, V](retType: Type, ctor: (U) => TypedAggregator[V]) extends Fun {
  def subst() = Arity1Aggregator[T, U, V](retType.subst(), ctor)
}

case class Arity1DependentAggregator[T, U, V](retType: Type, ctor: () => ((U) => TypedAggregator[V])) extends Fun {
  def subst() = Arity1Aggregator[T, U, V](retType.subst(), ctor())
}

case class Arity3Aggregator[T, U, V, W, X](retType: Type, ctor: (U, V, W) => TypedAggregator[X]) extends Fun {
  def subst() = Arity3Aggregator[T, U, V, W, X](retType.subst(), ctor)
}

case class UnaryLambdaAggregator[T, U, V](retType: Type, ctor: ((Any) => Any) => TypedAggregator[V]) extends Fun {
  def subst() = UnaryLambdaAggregator[T, U, V](retType.subst(), ctor)
}

case class BinaryLambdaAggregator[T, U, V, W](retType: Type, ctor: ((Any) => Any, V) => TypedAggregator[W]) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst(), ctor)
}

case class BinaryDependentLambdaAggregator[T, U, V, W](retType: Type, ctor: () => (((Any) => Any, V) => TypedAggregator[W])) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst(), ctor())
}

case class UnaryFun[T, U](retType: Type, f: (T) => U) extends Fun with Serializable with ((T) => U) {
  def apply(t: T): U = f(t)

  def subst() = UnaryFun(retType.subst(), f)
}

case class UnarySpecial[T, U](retType: Type, f: (() => Any) => U) extends Fun with Serializable with ((() => Any) => U) {
  def apply(t: () => Any): U = f(t)

  def subst() = UnarySpecial(retType.subst(), f)
}

case class BinaryFun[T, U, V](retType: Type, f: (T, U) => V) extends Fun with Serializable with ((T, U) => V) {
  def apply(t: T, u: U): V = f(t, u)

  def subst() = BinaryFun(retType.subst(), f)
}

case class BinarySpecial[T, U, V](retType: Type, f: (() => Any, () => Any) => V)
  extends Fun with Serializable with ((() => Any, () => Any) => V) {
  def apply(t: () => Any, u: () => Any): V = f(t, u)

  def subst() = BinarySpecial(retType.subst(), f)
}

case class BinaryLambdaFun[T, U, V](retType: Type, f: (T, (Any) => Any) => V)
  extends Fun with Serializable with ((T, (Any) => Any) => V) {
  def apply(t: T, u: (Any) => Any): V = f(t, u)

  def subst() = BinaryLambdaFun(retType.subst(), f)
}

case class Arity3LambdaMethod[T, U, V, W](retType: Type, f: (T, (Any) => Any, V) => W)
  extends Fun with Serializable with ((T, (Any) => Any, V) => W) {
  def apply(t: T, u: (Any) => Any, v: V): W = f(t, u, v)

  def subst() = Arity3LambdaMethod(retType.subst(), f)
}

case class Arity3LambdaFun[T, U, V, W](retType: Type, f: ((Any) => Any, U, V) => W)
  extends Fun with Serializable with (((Any) => Any, U, V) => W) {
  def apply(t: (Any) => Any, u: U, v: V): W = f(t, u, v)

  def subst() = Arity3LambdaFun(retType.subst(), f)
}

case class BinaryLambdaAggregatorTransformer[T, U, V](retType: Type, f: (CPS[Any], (Any) => Any) => CPS[V],
  fcode: (Code[AnyRef], Code[AnyRef] => CM[Code[AnyRef]]) => CMCodeCPS[AnyRef])
    extends Fun with Serializable with ((CPS[Any], (Any) => Any) => CPS[V]) {
  def apply(t: CPS[Any], u: (Any) => Any): CPS[V] = f(t, u)

  def subst() = BinaryLambdaAggregatorTransformer(retType.subst(), f, fcode)
}

case class Arity3Fun[T, U, V, W](retType: Type, f: (T, U, V) => W) extends Fun with Serializable with ((T, U, V) => W) {
  def apply(t: T, u: U, v: V): W = f(t, u, v)

  def subst() = Arity3Fun(retType.subst(), f)
}

case class Arity3Special[T, U, V, W](retType: Type, f: (() => Any, () => Any, () => Any) => W)
  extends Fun with Serializable with ((() => Any, () => Any, () => Any) => W) {
  def apply(t: () => Any, u: () => Any, v: () => Any): W = f(t, u, v)

  def subst() = Arity3Special(retType.subst(), f)
}

case class Arity3DependentFun[T, U, V, W](retType: Type, code: () => (T, U, V) => W) extends Fun {
  override def captureType() = Arity3Fun(retType, code())

  def apply(t: T, u: U, v: V): W =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity4Fun[T, U, V, W, X](retType: Type, f: (T, U, V, W) => X) extends Fun with Serializable with ((T, U, V, W) => X) {
  def apply(t: T, u: U, v: V, w: W): X = f(t, u, v, w)

  def subst() = Arity4Fun(retType.subst(), f)
}

case class Arity4DependentFun[T, U, V, W, X](retType: Type, code: () => (T, U, V, W) => X) extends Fun {
  override def captureType() = Arity4Fun(retType, code())

  def apply(t: T, u: U, v: V, w: W): X =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity5Fun[T, U, V, W, X, Y](retType: Type, f: (T, U, V, W, X) => Y) extends Fun with Serializable with ((T, U, V, W, X) => Y) {
  def apply(t: T, u: U, v: V, w: W, x:X): Y = f(t, u, v, w, x)

  def subst() = Arity5Fun(retType.subst(), f)
}

case class Arity5DependentFun[T, U, V, W, X, Y](retType: Type, code: () => (T, U, V, W, X) => Y) extends Fun {
  override def captureType() = Arity5Fun(retType, code())

  def apply(t: T, u: U, v: V, w: W, x: X): Y =
    throw new UnsupportedOperationException("must captureType first")

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity6Special[T, U, V, W, X, Y, Z](retType: Type, f: (() => Any, () => Any, () => Any, () => Any, () => Any, () => Any) => Z)
  extends Fun with Serializable with ((() => Any, () => Any, () => Any, () => Any, () => Any, () => Any) => Z) {
  def apply(t: () => Any, u: () => Any, v: () => Any, w: () => Any, x: () => Any, y: () => Any): Z = f(t, u, v, w, x, y)

  def subst() = Arity6Special(retType.subst(), f)
}
