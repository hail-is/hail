package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, (Int, BaseType)]

  implicit val toInt = IntNumericConversion
  implicit val toLong = LongNumericConversion
  implicit val toFloat = FloatNumericConversion
  implicit val toDouble = DoubleNumericConversion
}
