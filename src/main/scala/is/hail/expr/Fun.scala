package is.hail.expr

import is.hail.asm4s.Code

sealed trait Fun {
  def retType: Type

  def subst(): Fun

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun
}

case class Transformation[T, U](f: T => U, fcode: Code[T] => CM[Code[U]]) extends (T => U) {
  def apply(t: T): U = f(t)
}

case class UnaryFunCode[T, U](retType: Type, code: Code[T] => CM[Code[U]]) extends Fun with (Code[T] => CM[Code[U]]) {
  def apply(ct: Code[T]): CM[Code[U]] = code(ct)

  def subst() = UnaryFunCode[T, U](retType.subst(), code)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 1)

    UnaryFunCode[Any, Any](retType, (a: Code[Any]) =>
      transformations(0).fcode(a).flatMap(ct => code(ct.asInstanceOf[Code[T]])))
  }
}

case class BinarySpecialCode[T, U, V](retType: Type, code: (Code[T], Code[U]) => CM[Code[V]])
  extends Fun with Serializable with ((Code[T], Code[U]) => CM[Code[V]]) {
  def apply(t: Code[T], u: Code[U]): CM[Code[V]] = code(t, u)

  def subst() = BinarySpecialCode(retType.subst(), code)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class BinaryFunCode[T, U, V](retType: Type, code: (Code[T], Code[U]) => CM[Code[V]]) extends Fun with Serializable with ((Code[T], Code[U]) => CM[Code[V]]) {
  def apply(t: Code[T], u: Code[U]): CM[Code[V]] = code(t, u)

  def subst() = BinaryFunCode(retType.subst(), code)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 2)

    BinaryFunCode[Any, Any, Any](retType, { (a, b) => for (
      ct <- transformations(0).fcode(a);
      cu <- transformations(1).fcode(b);
      ret <- code(ct.asInstanceOf[Code[T]], cu.asInstanceOf[Code[U]])
    ) yield ret
    })
  }
}

case class Arity0Aggregator[T, U](retType: Type, ctor: () => TypedAggregator[U]) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst(), ctor)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    assert(transformations.length == 1)

    Arity0Aggregator[T, U](retType, () => new TransformedAggregator(ctor(), transformations(0).f))
  }
}

object GroupByAggregatorFun {
  def apply[T](retType: Type): GroupByAggregatorFun[T] =
    GroupByAggregatorFun(retType,
      { (key: Any => Any, transformer: (Any, Any => Any) => Any, downstream: TypedAggregator[T]) =>
        new is.hail.methods.GroupByAggregator(key, transformer, downstream) })
}

case class GroupByAggregatorFun[T](retType: Type,
  ctor: (Any => Any, (Any, Any => Any) => Any, TypedAggregator[T]) => TypedAggregator[Map[Any, T]]) extends Fun {
  def subst() = GroupByAggregatorFun[T](retType.subst(), ctor)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    assert(transformations.length == 1)

    GroupByAggregatorFun[T](retType, (key: Any => Any, transformation: (Any, Any => Any) => Any, downstream: TypedAggregator[T]) =>
      new is.hail.methods.GroupByAggregator(key, (x, k)  => transformation(transformations(0).f(x), k), downstream))
  }
}

class TransformedAggregator[T](val prev: TypedAggregator[T], transform: (Any) => Any) extends TypedAggregator[T] {
  def seqOp(x: Any) = prev.seqOp(transform(x))

  def combOp(agg2: this.type) = prev.combOp(agg2.prev)

  def result = prev.result

  def copy() = new TransformedAggregator(prev.copy(), transform)
}

case class Arity1Aggregator[T, U, V](retType: Type, ctor: (U) => TypedAggregator[V]) extends Fun {
  def subst() = Arity1Aggregator[T, U, V](retType.subst(), ctor)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    assert(transformations.length == 2)

    Arity1Aggregator[T, U, V](retType, { (u) =>
      new TransformedAggregator(ctor(
        transformations(1).f(u).asInstanceOf[U]),
        transformations(0).f)
    })
  }
}

case class Arity3Aggregator[T, U, V, W, X](retType: Type, ctor: (U, V, W) => TypedAggregator[X]) extends Fun {
  def subst() = Arity3Aggregator[T, U, V, W, X](retType.subst(), ctor)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    assert(transformations.length == 4)

    Arity3Aggregator[T, U, V, W, X](retType, { (u, v, w) =>
      new TransformedAggregator(ctor(
        transformations(1).f(u).asInstanceOf[U],
        transformations(2).f(v).asInstanceOf[V],
        transformations(3).f(w).asInstanceOf[W]),
        transformations(0).f)
    })
  }
}

case class UnaryLambdaAggregator[T, U, V](retType: Type, ctor: ((Any) => Any) => TypedAggregator[V]) extends Fun {
  def subst() = UnaryLambdaAggregator[T, U, V](retType.subst(), ctor)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class BinaryLambdaAggregator[T, U, V, W](retType: Type, ctor: ((Any) => Any, V) => TypedAggregator[W]) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst(), ctor)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class UnaryFun[T, U](retType: Type, f: (T) => U) extends Fun with Serializable with ((T) => U) {
  def apply(t: T): U = f(t)

  def subst() = UnaryFun(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 1)

    UnaryFun[Any, Any](retType, (a: Any) => f(transformations(0).f(a).asInstanceOf[T]))
  }
}

case class UnarySpecial[T, U](retType: Type, f: (() => Any) => U) extends Fun with Serializable with ((() => Any) => U) {
  def apply(t: () => Any): U = f(t)

  def subst() = UnarySpecial(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class BinaryFun[T, U, V](retType: Type, f: (T, U) => V) extends Fun with Serializable with ((T, U) => V) {
  def apply(t: T, u: U): V = f(t, u)

  def subst() = BinaryFun(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 2)

    BinaryFun[Any, Any, Any](retType, (a, b) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U]))
  }
}

case class BinarySpecial[T, U, V](retType: Type, f: (() => Any, () => Any) => V)
  extends Fun with Serializable with ((() => Any, () => Any) => V) {
  def apply(t: () => Any, u: () => Any): V = f(t, u)

  def subst() = BinarySpecial(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class BinaryLambdaFun[T, U, V](retType: Type, f: (T, (Any) => Any) => V)
  extends Fun with Serializable with ((T, (Any) => Any) => V) {
  def apply(t: T, u: (Any) => Any): V = f(t, u)

  def subst() = BinaryLambdaFun(retType.subst(), f)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class Arity3LambdaFun[T, U, V, W](retType: Type, f: (T, (Any) => Any, V) => W)
  extends Fun with Serializable with ((T, (Any) => Any, V) => W) {
  def apply(t: T, u: (Any) => Any, v: V): W = f(t, u, v)

  def subst() = Arity3LambdaFun(retType.subst(), f)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class BinaryLambdaAggregatorTransformer[T, U, V](retType: Type, f: (CPS[Any], (Any) => Any) => CPS[V],
  fcode: (Code[AnyRef], Code[AnyRef] => CM[Code[AnyRef]]) => CMCodeCPS[AnyRef])
    extends Fun with Serializable with ((CPS[Any], (Any) => Any) => CPS[V]) {
  def apply(t: CPS[Any], u: (Any) => Any): CPS[V] = f(t, u)

  def subst() = BinaryLambdaAggregatorTransformer(retType.subst(), f, fcode)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = ???
}

case class Arity3Fun[T, U, V, W](retType: Type, f: (T, U, V) => W) extends Fun with Serializable with ((T, U, V) => W) {
  def apply(t: T, u: U, v: V): W = f(t, u, v)

  def subst() = Arity3Fun(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 3)

    Arity3Fun[Any, Any, Any, Any](retType, (a, b, c) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V]))
  }
}

case class Arity4Fun[T, U, V, W, X](retType: Type, f: (T, U, V, W) => X) extends Fun with Serializable with ((T, U, V, W) => X) {
  def apply(t: T, u: U, v: V, w: W): X = f(t, u, v, w)

  def subst() = Arity4Fun(retType.subst(), f)

  def convertArgs(transformations: Array[Transformation[Any, Any]]): Fun = {
    require(transformations.length == 4)

    Arity4Fun[Any, Any, Any, Any, Any](retType, (a, b, c, d) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V],
      transformations(3).f(d).asInstanceOf[W]))
  }
}
