package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering, NDArray}
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.expr.{Nat, NatBase}

import scala.reflect.{classTag, ClassTag}

import org.apache.spark.sql.Row

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

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean = {
    if (a1 == null || a2 == null) {
      a1 == a2
    } else {
      val aNd1 = a1.asInstanceOf[NDArray]
      val aNd2 = a2.asInstanceOf[NDArray]

      val sameShape = aNd1.shape == aNd2.shape
      val sameData = aNd1.getRowMajorElements().zip(aNd2.getRowMajorElements()).forall {
        case (e1, e2) => elementType.valuesSimilar(e1, e2, tolerance, absolute)
      }

      sameShape && sameData
    }
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("ndarray<")
    elementType.pyString(sb)
    sb.append(", ")
    sb.append(nDims)
    sb.append('>')
  }

  def _toPretty = s"NDArray[$elementType,$nDims]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb.append("NDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append(",")
    sb.append(nDims)
    sb.append("]")
  }

  override def str(a: Annotation): String = {
    if (a == null) "NA"
    else {
      val aNd = a.asInstanceOf[NDArray]
      val shape = aNd.shape
      val data = aNd.getRowMajorElements()

      def dataToNestedString(data: Iterator[Annotation], shape: Seq[Long], sb: StringBuilder)
        : Unit = {
        if (shape.isEmpty) {
          sb.append(data.next().toString)
        } else {
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

      s"ndarray{shape=$prettyShape, data=$prettyData}"
    }
  }

  override def unify(concrete: Type): Boolean =
    concrete match {
      case TNDArray(cElementType, cNDims) =>
        elementType.unify(cElementType) && nDimsBase.unify(cNDims)
      case _ => false
    }

  override def clear(): Unit = {
    elementType.clear()
    nDimsBase.clear()
  }

  override def subst(): TNDArray = TNDArray(elementType.subst(), nDimsBase.subst())

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  def _typeCheck(a: Annotation): Boolean = a match {
    case nd: NDArray => nd.forall(e => elementType.typeCheck(e))
    case _ => false
  }

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering = null

  lazy val shapeType: TTuple = TTuple(Array.fill(nDims)(TInt64): _*)
}
