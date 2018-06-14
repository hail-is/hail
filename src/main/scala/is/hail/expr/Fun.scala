package is.hail.expr

import is.hail.expr.types._

sealed trait Fun {
  def captureType(): Fun = this

  def retType: Type

  def subst(): Fun
}

case class UnaryDependentFunCode[T, U](retType: Type) extends Fun {
  override def captureType() = UnaryFunCode(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class UnaryDependentFun[T, U](retType: Type) extends Fun {
  override def captureType() = UnaryFun(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class UnaryFunCode[T, U](retType: Type) extends Fun {
  def subst() = UnaryFunCode[T, U](retType.subst())
}

case class BinarySpecialCode[T, U, V](retType: Type)
  extends Fun with Serializable {

  def subst() = BinarySpecialCode(retType.subst())
}

case class BinaryFunCode[T, U, V](retType: Type) extends Fun with Serializable {
  def subst() = BinaryFunCode(retType.subst())
}

case class BinaryDependentFun[T, U, V](retType: Type) extends Fun {
  override def captureType() = BinaryFun(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity0Aggregator[T, U](retType: Type) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst())
}

case class Arity0DependentAggregator[T, U](retType: Type) extends Fun {
  def subst() = Arity0Aggregator[T, U](retType.subst())
}

case class Arity1Aggregator[T, U, V](retType: Type) extends Fun {
  def subst() = Arity1Aggregator[T, U, V](retType.subst())
}

case class Arity1DependentAggregator[T, U, V](retType: Type) extends Fun {
  def subst() = Arity1Aggregator[T, U, V](retType.subst())
}

case class Arity3Aggregator[T, U, V, W, X](retType: Type) extends Fun {
  def subst() = Arity3Aggregator[T, U, V, W, X](retType.subst())
}

case class UnaryLambdaAggregator[T, U, V](retType: Type) extends Fun {
  def subst() = UnaryLambdaAggregator[T, U, V](retType.subst())
}

case class BinaryLambdaAggregator[T, U, V, W](retType: Type) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst())
}

case class BinaryDependentLambdaAggregator[T, U, V, W](retType: Type) extends Fun {
  def subst() = BinaryLambdaAggregator[T, U, V, W](retType.subst())
}

case class UnaryFun[T, U](retType: Type) extends Fun with Serializable {
  def subst() = UnaryFun(retType.subst())
}

case class UnarySpecial[T, U](retType: Type) extends Fun with Serializable {
  def subst() = UnarySpecial(retType.subst())
}

case class BinaryFun[T, U, V](retType: Type) extends Fun with Serializable {
  def subst() = BinaryFun(retType.subst())
}

case class BinarySpecial[T, U, V](retType: Type)
  extends Fun with Serializable {
  def subst() = BinarySpecial(retType.subst())
}

case class BinaryLambdaFun[T, U, V](retType: Type)
  extends Fun with Serializable {
  def subst() = BinaryLambdaFun(retType.subst())
}

case class Arity3LambdaMethod[T, U, V, W](retType: Type)
  extends Fun with Serializable {
  def subst() = Arity3LambdaMethod(retType.subst())
}

case class Arity3LambdaFun[T, U, V, W](retType: Type)
  extends Fun with Serializable {
  def subst() = Arity3LambdaFun(retType.subst())
}

case class BinaryLambdaAggregatorTransformer[T, U, V](retType: Type)
    extends Fun with Serializable {
  def subst() = BinaryLambdaAggregatorTransformer(retType.subst())
}

case class Arity3Fun[T, U, V, W](retType: Type) extends Fun with Serializable {
  def subst() = Arity3Fun(retType.subst())
}

case class Arity3Special[T, U, V, W](retType: Type)
  extends Fun with Serializable {

  def subst() = Arity3Special(retType.subst())
}

case class Arity3DependentFun[T, U, V, W](retType: Type) extends Fun {
  override def captureType() = Arity3Fun(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity4Fun[T, U, V, W, X](retType: Type) extends Fun with Serializable {
  def subst() = Arity4Fun(retType.subst())
}

case class Arity4DependentFun[T, U, V, W, X](retType: Type) extends Fun {
  override def captureType() = Arity4Fun(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity5Fun[T, U, V, W, X, Y](retType: Type) extends Fun with Serializable {
  def subst() = Arity5Fun(retType.subst())
}

case class Arity5DependentFun[T, U, V, W, X, Y](retType: Type) extends Fun {
  override def captureType() = Arity5Fun(retType)

  def subst() =
    throw new UnsupportedOperationException("must captureType first")
}

case class Arity6Special[T, U, V, W, X, Y, Z](retType: Type)
  extends Fun with Serializable {
  def subst() = Arity6Special(retType.subst())
}
