package is.hail.expr.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.check.Gen
import is.hail.expr.types.physical.PNDArray
import is.hail.utils.NDArray
import org.apache.spark.sql.Row

import scala.reflect.{ClassTag, classTag}

final case class TNDArray(elementType: Type, override val required: Boolean = false) extends ComplexType {
  val representation: Type = TStruct(
    "flags" -> TInt64Required,           // encodes data layout
    "shape" -> TArray(TInt64Required),   // length is ndim
    "offset" ->  TInt64Required,         // offset into data
    "strides" -> TArray(TInt64Required), // stride in each dim, length is 1
    "data" -> TArray(elementType)
  )

  lazy val physicalType: PNDArray = PNDArray(elementType.physicalType, required)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("ndarray<")
    elementType.pyString(sb)
    sb.append('>')
  }
  
//  override def genNonmissingValue: Gen[Annotation] =
//    Gen.zip(Gen.b Gen.choose()
//    Gen.buildableOf[NDArray](elementType.genValue).map(x => ..., x: NDArray)
  
  def _toPretty = s"NDArray[$elementType]"

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  def _typeCheck(a: Any): Boolean = {
    a.isInstanceOf[Row] && {
      val r = a.asInstanceOf[Row]
      val s = representation.asInstanceOf[TStruct]
      r.length == s.size && s.types.zipWithIndex.forall {
        case (t, i) => t._typeCheck(r.get(i))
      }
    }
  }

  val ordering: ExtendedOrdering = TBaseStruct.getOrdering(
    representation.asInstanceOf[TStruct].types)
}
