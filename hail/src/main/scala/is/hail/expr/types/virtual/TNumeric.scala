package is.hail.expr.types.virtual

abstract class TNumeric extends Type {
  override def canCompare(other: Type): Boolean = other.isInstanceOf[TNumeric]
}
