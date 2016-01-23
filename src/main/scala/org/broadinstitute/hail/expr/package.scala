package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, Option[Any]]
  type TypeSymbolTable = Map[String, Type]

  implicit val toInt = IntNumericConversion
  implicit val toLong = LongNumericConversion
  implicit val toFloat = FloatNumericConversion
  implicit val toDouble = DoubleNumericConversion
}
