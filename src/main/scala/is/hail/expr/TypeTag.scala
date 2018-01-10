package is.hail.expr

import is.hail.expr.types._

object TypeTag

sealed trait TypeTag {
  def xs: Seq[Type]

  def clear() {
    xs.foreach(_.clear())
  }

  def unify(concrete: TypeTag): Boolean = {
    (this, concrete) match {
      case (MethodType(_@_*), MethodType(_@_*))
           | (FunType(_@_*), FunType(_@_*))
           | (FieldType(_@_*), FieldType(_@_*)) =>
        xs.length == concrete.xs.length &&
          (xs, concrete.xs).zipped.forall { case (x, cx) =>
            x.unify(cx)
          }

      case _ =>
        false
    }

  }

  def subst(): TypeTag

  override def toString: String = s"""(${ xs.mkString(", ") })"""
}

case class MethodType(xs: Type*) extends TypeTag {
  def subst(): TypeTag = MethodType(xs.map(_.subst()): _*)
}

case class FunType(xs: Type*) extends TypeTag {
  def subst(): TypeTag = FunType(xs.map(_.subst()): _*)
}

case class FieldType(xs: Type*) extends TypeTag {
  def subst(): TypeTag = FieldType(xs.map(_.subst()): _*)
}