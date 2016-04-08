package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, (Int, BaseType)]

  type Aggregator = (() => Any, (Any) => Any, (Any, Any) => Any, Int)

  implicit val toInt = IntNumericConversion
  implicit val toLong = LongNumericConversion
  implicit val toFloat = FloatNumericConversion
  implicit val toDouble = DoubleNumericConversion
}
