package org.broadinstitute.hail.expr

object TypeTag

sealed trait TypeTag

case class MethodType(xs: BaseType*) extends TypeTag

case class FunType(xs: BaseType*) extends TypeTag

