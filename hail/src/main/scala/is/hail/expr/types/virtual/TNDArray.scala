package is.hail.expr.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.physical.PNDArray
import org.apache.spark.sql.Row

import scala.reflect.{ClassTag, classTag}

final case class TNDArray(elementType: Type, nDims: Type, override val required: Boolean = false) extends Type {
  lazy val physicalType: PNDArray = PNDArray(elementType.physicalType, nDims.physicalType, required)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("ndarray<")
    elementType.pyString(sb)
    sb.append(", ")
    nDims.pyString(sb)
    sb.append('>')
  }
  
  def _toPretty = s"NDArray[$elementType,$nDims]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("NDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(",")
    nDims.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TNDArray(et, nd, _) => elementType.unify(et) && nDims.unify(nd)
      case _ => false
    }
  }

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException

  val ordering: ExtendedOrdering = null
}
