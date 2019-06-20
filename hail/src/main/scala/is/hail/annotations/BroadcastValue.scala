package is.hail.annotations

import is.hail.HailContext
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.types.physical.PBaseStruct
import is.hail.expr.types.virtual.{TArray, TBaseStruct, TStruct}
import is.hail.expr.types.physical.{PInt64, PStruct, PType}
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row

object BroadcastRow {
  def empty(): BroadcastRow =
    BroadcastRow(Row.empty, PStruct.empty(), HailContext.backend)

  def apply(value: Row, t: PBaseStruct, sc: SparkContext): BroadcastRow =
    BroadcastRow(value, t, HailContext.backend)
}

case class BroadcastRow(value: Row,
  t: PBaseStruct,
  backend: Backend) {
  require(Annotation.isSafe(t, value))

  lazy val safeValue: Row = value

  lazy val broadcast: BroadcastValue[Row] = backend.broadcast(value)

  def toRegion(region: Region): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.addAnnotation(t, value)
    rvb.end()
  }
}

object BroadcastIndexedSeq {
  def apply(value: IndexedSeq[Annotation], t: TArray, sc: SparkContext): BroadcastIndexedSeq =
    BroadcastIndexedSeq(value, t, HailContext.backend)
}

case class BroadcastIndexedSeq(value: IndexedSeq[Annotation],
  t: TArray,
  backend: Backend) {
  require(Annotation.isSafe(t, value))

  lazy val safeValue: IndexedSeq[Annotation] = value

  lazy val broadcast: BroadcastValue[IndexedSeq[Annotation]] = backend.broadcast(value)

  def toRegion(region: Region): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(t.physicalType)
    rvb.addAnnotation(t, value)
    rvb.end()
  }
}
