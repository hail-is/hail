package is.hail.expr.types.physical


abstract class PNumeric extends PType {
  override def canCompare(other: PType): Boolean = other.isInstanceOf[PNumeric]
}
