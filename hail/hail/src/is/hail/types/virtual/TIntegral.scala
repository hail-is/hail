package is.hail.types.virtual

import org.apache.arrow.vector.types.pojo.ArrowType

abstract class TIntegral extends TNumeric {
  override def arrowType = new ArrowType.Int(bitWidth, /*isSigned=*/ true)
}
