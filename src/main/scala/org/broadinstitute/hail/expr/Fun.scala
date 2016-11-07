package org.broadinstitute.hail.expr

sealed trait Fun {
  def retType: BaseType

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun
}

case class UnaryFun[T, U](retType: BaseType, f: (T) => U) extends Fun with Serializable with ((T) => U) {
  def apply(t: T): U = f(t)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 1)

    UnaryFun[Any, Any](retType, (a: Any) => f(transformations(0).f(a).asInstanceOf[T]))
  }
}

case class OptionUnaryFun[T, U](retType: BaseType, f: (T) => Option[U]) extends Fun with Serializable with ((T) => Option[U]) {
  def apply(t: T): Option[U] = f(t)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 1)

    OptionUnaryFun[Any, Any](retType, (a) => f(transformations(0).f(a).asInstanceOf[T]))
  }
}

case class BinaryFun[T, U, V](retType: BaseType, f: (T, U) => V) extends Fun with Serializable with ((T, U) => V) {
  def apply(t: T, u: U): V = f(t, u)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 2)

    BinaryFun[Any, Any, Any](retType, (a, b) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U]))
  }
}

case class Arity3Fun[T, U, V, W](retType: BaseType, f: (T, U, V) => W) extends Fun with Serializable with ((T, U, V) => W) {
  def apply(t: T, u: U, v: V): W = f(t, u, v)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 3)

    Arity3Fun[Any, Any, Any, Any](retType, (a, b, c) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V]))
  }
}

case class Arity4Fun[T, U, V, W, X](retType: BaseType, f: (T, U, V, W) => X) extends Fun with Serializable with ((T, U, V, W) => X) {
  def apply(t: T, u: U, v: V, w: W): X = f(t, u, v, w)

  def convertArgs(transformations: Array[UnaryFun[Any, Any]]): Fun = {
    require(transformations.length == 4)

    Arity4Fun[Any, Any, Any, Any, Any](retType, (a, b, c, d) => f(transformations(0).f(a).asInstanceOf[T],
      transformations(1).f(b).asInstanceOf[U],
      transformations(2).f(c).asInstanceOf[V],
      transformations(3).f(d).asInstanceOf[W]))
  }
}