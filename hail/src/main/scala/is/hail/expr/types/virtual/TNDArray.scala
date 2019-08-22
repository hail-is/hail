package is.hail.expr.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.{NatBase, Nat}
import is.hail.expr.types.physical.PNDArray
import org.apache.spark.sql.Row

import scala.reflect.{ClassTag, classTag}

object TNDArray {
  def matMulNDims(l: Int, r: Int): Int = {
    (l, r) match {
      case (1, 1) => 0
      case (1, n) => n - 1
      case (n, 1) => n - 1
      case (_, _) => l
    }
  }
}

final case class TNDArray(elementType: Type, nDimsBase: NatBase, override val required: Boolean = false) extends Type {
  lazy val nDims: Int = {
    assert(nDimsBase.isInstanceOf[Nat], s"Missing concrete number of dimensions.")
    nDimsBase.asInstanceOf[Nat].n
  }
  lazy val physicalType: PNDArray = PNDArray(elementType.physicalType, nDims, required)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("ndarray<")
    elementType.pyString(sb)
    sb.append(", ")
    sb.append(nDims)
    sb.append('>')
  }

  def _toPretty = s"NDArray[$elementType,$nDims]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("NDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(",")
    sb.append(nDims)
    sb.append("]")
  }

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TNDArray(cElementType, cNDims, _) => elementType.unify(cElementType) && nDimsBase.unify(cNDims)
      case _ => false
    }
  }

  override def clear(): Unit = {
    elementType.clear()
    nDimsBase.clear()
  }

  override def subst(): TNDArray = TNDArray(elementType.subst(), nDimsBase.subst(), required)

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException

  val ordering: ExtendedOrdering = null

  lazy val representation = TStruct(required = true,
    ("flags", TInt32Required),
    ("offset", TInt32Required),
    ("shape", TTuple(true, Array.tabulate(nDims)(_ => TInt64Required):_*)),
    ("strides", TTuple(true, Array.tabulate(nDims)(_ => TInt64Required):_*)),
    ("data", TArray(elementType, required = true))
  )
}
