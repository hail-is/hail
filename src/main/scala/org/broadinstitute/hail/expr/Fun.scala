package org.broadinstitute.hail.expr

sealed trait Fun {
  def retType: Type

  def subst(): Fun

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun
}

case class Arity0Aggregator[T, U](retType: Type, ctor: () => TypedAggregator[U]) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst(), ctor)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    assert(transformations.length == 1)

    Arity0Aggregator[T, U](retType, () => new TransformedAggregator(ctor(), transformations(0).f))
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

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    assert(transformations.length == 2)

    Arity1Aggregator[T, U, V](retType, { (u) =>
      new TransformedAggregator(ctor(
        transformations(1).f(u).asInstanceOf[U]),
        transformations(0))
    })
  }
}

case class Arity3Aggregator[T, U, V, W, X](retType: Type, ctor: (U, V, W) => TypedAggregator[X]) extends Fun {
  def subst() = Arity3Aggregator[T, U, V, W, X](retType.subst(), ctor)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    assert(transformations.length == 4)

    Arity3Aggregator[T, U, V, W, X](retType, { (u, v, w) =>
      new TransformedAggregator(ctor(
        transformations(1).f(u).asInstanceOf[U],
        transformations(2).f(v).asInstanceOf[V],
        transformations(3).f(w).asInstanceOf[W]),
        transformations(0))
    })
  }
}

case class UnaryLambdaAggregator[T, U, V](retType: Type, ctor: ((Any) => Any) => TypedAggregator[V]) extends Fun {
  def subst() = UnaryLambdaAggregator[T, U, V](retType.subst(), ctor)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = ???
}

case class BinaryLambdaAggregator[T, U, V, W](retType: Type, ctor: ((Any) => Any, V) => TypedAggregator[W]) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst(), ctor)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = ???
}

case class UnaryFun[T, U](retType: Type, f: (T) => U) extends Fun with Serializable with ((T) => U) {
  def apply(t: T): U = f(t)

  def subst() = UnaryFun(retType.subst(), f)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 1)

    UnaryFun[Any, Any](retType, (a: Any) => f(transformations(0).f(a).asInstanceOf[T]))
  }
}

case class OptionUnaryFun[T, U](retType: Type, f: (T) => Option[U]) extends Fun with Serializable with ((T) => Option[U]) {
  def apply(t: T): Option[U] = f(t)

  def subst() = OptionUnaryFun(retType.subst(), f)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 1)

    OptionUnaryFun[Any, Any](retType, (a) => f(transformations(0).f(a).asInstanceOf[T]))
  }
}

case class BinaryFun[T, U, V](retType: Type, f: (T, U) => V) extends Fun with Serializable with ((T, U) => V) {
  def apply(t: T, u: U): V = f(t, u)

  def subst() = BinaryFun(retType.subst(), f)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 2)

    BinaryFun[Any, Any, Any](retType, (a, b) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U]))
  }
}

case class BinaryLambdaFun[T, U, V](retType: Type, f: (T, (Any) => Any) => V)
  extends Fun with Serializable with ((T, (Any) => Any) => V) {
  def apply(t: T, u: (Any) => Any): V = f(t, u)

  def subst() = BinaryLambdaFun(retType.subst(), f)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = ???
}

case class BinaryLambdaSpecial[T, U, V](retType: Type, f: (() => Any, (Any) => Any) => V)
  extends Fun with Serializable with ((() => Any, (Any) => Any) => V) {
  def apply(t: () => Any, u: (Any) => Any): V = f(t, u)

  def subst() = BinaryLambdaSpecial(retType.subst(), f)

  // conversion can't apply to function type
  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = ???
}


case class Arity3Fun[T, U, V, W](retType: Type, f: (T, U, V) => W) extends Fun with Serializable with ((T, U, V) => W) {
  def apply(t: T, u: U, v: V): W = f(t, u, v)

  def subst() = Arity3Fun(retType.subst(), f)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 3)

    Arity3Fun[Any, Any, Any, Any](retType, (a, b, c) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V]))
  }
}

case class Arity4Fun[T, U, V, W, X](retType: Type, f: (T, U, V, W) => X) extends Fun with Serializable with ((T, U, V, W) => X) {
  def apply(t: T, u: U, v: V, w: W): X = f(t, u, v, w)

  def subst() = Arity4Fun(retType.subst(), f)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 4)

    Arity4Fun[Any, Any, Any, Any, Any](retType, (a, b, c, d) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V],
      transformations(3).f(d).asInstanceOf[W]))
  }
}