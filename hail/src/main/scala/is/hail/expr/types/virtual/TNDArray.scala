package is.hail.expr.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering, UnsafeIndexedSeq}
import is.hail.expr.{Nat, NatBase}
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

final case class TNDArray(elementType: Type, nDimsBase: NatBase) extends Type {
  lazy val nDims: Int = {
    assert(nDimsBase.isInstanceOf[Nat], s"Missing concrete number of dimensions.")
    nDimsBase.asInstanceOf[Nat].n
  }

  override def fundamentalType: Type = representation.fundamentalType

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

  override def str(a: Annotation): String = {
    if (a == null) "NA" else {
      val a_row = a.asInstanceOf[Row]
      val shape = a_row(2).asInstanceOf[Row].toSeq.asInstanceOf[Seq[Long]].map(_.toInt)
      val data = a_row(4).asInstanceOf[UnsafeIndexedSeq]

      def dataToNestedString(data: Iterator[Annotation], shape: Seq[Int], sb: StringBuilder):Unit  = {
        if (shape.isEmpty) {
          sb.append(data.next().toString)
        }
        else {
          sb.append("[")
          val howMany = shape.head
          var repeat = 0
          while (repeat < howMany) {
            dataToNestedString(data, shape.tail, sb)
            if (repeat != howMany - 1) {
              sb.append(", ")
            }
            repeat += 1
          }
          sb.append("]")
        }
      }

      val stringBuilder = new StringBuilder("")
      dataToNestedString(data.iterator, shape, stringBuilder)
      val prettyData = stringBuilder.result()
      val prettyShape = "(" + shape.mkString(", ") + ")"

      s"ndarray{shape=${prettyShape}, data=${prettyData}}"
    }
  }

  override def unify(concrete: Type): Boolean = {
    concrete match {
      case TNDArray(cElementType, cNDims) => elementType.unify(cElementType) && nDimsBase.unify(cNDims)
      case _ => false
    }
  }

  override def clear(): Unit = {
    elementType.clear()
    nDimsBase.clear()
  }

  override def subst(): TNDArray = TNDArray(elementType.subst(), nDimsBase.subst())

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  def _typeCheck(a: Any): Boolean = throw new UnsupportedOperationException

  val ordering: ExtendedOrdering = null

  lazy val representation = TStruct(
    ("flags", TInt32),
    ("offset", TInt32),
    ("shape", TTuple(Array.tabulate(nDims)(_ => TInt64):_*)),
    ("strides", TTuple(Array.tabulate(nDims)(_ => TInt64):_*)),
    ("data", TArray(elementType))
  )
}
