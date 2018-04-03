package is.hail.annotations

import is.hail.expr.types.{TArray, TBaseStruct, Type}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

abstract class BroadcastValue[T: ClassTag](value: T, t: Type, sc: SparkContext) {

  def safeValue: T

  lazy val broadcast: Broadcast[T] = sc.broadcast(value)

  def toRegion(region: Region): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.addAnnotation(t, value)
    rvb.end()
  }
}

case class BroadcastRow(value: Row,
  t: TBaseStruct,
  sc: SparkContext) extends BroadcastValue[Row](value, t, sc) {
  require(Annotation.isSafe(t, value))

  lazy val safeValue: Row = value
}

case class BroadcastIndexedSeq(value: IndexedSeq[Annotation],
  t: TArray,
  sc: SparkContext) extends BroadcastValue[IndexedSeq[Annotation]](value, t, sc) {
  require(Annotation.isSafe(t, value))

  lazy val safeValue: IndexedSeq[Annotation] = value
}
