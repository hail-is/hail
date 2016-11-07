package org.broadinstitute.hail.expr

object TypeTag

sealed trait TypeTag {
  def xs: Seq[BaseType]

  override def toString: String = s"""(${ xs.mkString(", ") })"""
}

case class MethodType(xs: BaseType*) extends TypeTag

case class FunType(xs: BaseType*) extends TypeTag

