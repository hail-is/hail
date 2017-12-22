package is.hail.expr.typ

/**
  * Created by dking on 12/21/17.
  */
object TNumeric {
  def promoteNumeric(types: Set[TNumeric]): Type = {
    assert(types.forall(!_.required))
    if (types.size == 1)
      types.head
    else if (types(TFloat64Optional))
      TFloat64()
    else if (types(TFloat32Optional))
      TFloat32()
    else if (types(TInt64Optional))
      TInt64()
    else {
      assert(types(TInt32Optional))
      TInt32()
    }
  }
}

abstract class TNumeric extends Type {
  def conv: NumericConversion[_, _]

  override def canCompare(other: Type): Boolean = other.isInstanceOf[TNumeric]
}