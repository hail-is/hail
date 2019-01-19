package is.hail.utils

import org.apache.spark.sql.Row
import org.json4s.{JArray, JInt, JObject, JValue}

object NDArray {
  def fromRow(x: Row): NDArray =
    new NDArray(
      x.getLong(0),
      x.getAs[IndexedSeq[Int]](1),
      x.getLong(2),
      x.getAs[IndexedSeq[Int]](3),
      x.getAs[IndexedSeq[Any]](4))
}

case class NDArray(
  flags:Long, 
  shape: IndexedSeq[Int], 
  offset: Long, 
  strides: IndexedSeq[Int], 
  data: IndexedSeq[Any]) extends Serializable {
  
  def toRow = Row(flags, shape, offset, strides, data)

  def toJSON(f: (Any) => JValue): JValue = JObject(
    "flags" -> JInt(flags),
    "shape"-> JArray(shape.map(JInt(_)).toList),
    "offset" -> JInt(offset),
    "strides" -> JArray(strides.map(JInt(_)).toList),
    "data" -> JArray(data.map(f).toList))
}
