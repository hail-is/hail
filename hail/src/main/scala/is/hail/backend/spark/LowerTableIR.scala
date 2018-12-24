package is.hail.backend.spark

import is.hail.annotations.RegionValue
import is.hail.backend.Binding
import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

case class SparkCollect(
  child: SparkStage)

case class SparkShuffle(
  child: SparkStage)

// FIXME How to refer to result of shuffle stage?  IR to refer to external RDD
case class SparkStage(
  globals: List[Binding],
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  rdds: Map[String, RDD[RegionValue]],
  contextType: Type, context: Array[Any],
  body: IR)

class LowerTableIR {
  def lower(tir: TableIR): SparkStage = tir match {
    case TableRange(n, nPartitions) =>
      val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
      val partCounts = partition(n, nPartitionsAdj)
      val partStarts = partCounts.scanLeft(0)(_ + _)

      val rvdType = RVDType(tir.typ.rowType.physicalType, Array("idx"))

      val contextType = TStruct(
        "start" -> TInt32(),
        "end" -> TInt32())

      val g = genUID()

      SparkStage(
        List(Binding(g, MakeStruct(Seq()))),
        rvdType,
        new RVDPartitioner(Array("idx"), tir.typ.rowType,
          Array.tabulate(nPartitionsAdj) { i =>
            val start = partStarts(i)
            val end = partStarts(i + 1)
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
        Map.empty,
        contextType,
        Array.tabulate(nPartitionsAdj) { i =>
          val start = partStarts(i)
          val end = partStarts(i + 1)
          Row(start, end)
        },
        ArrayRange(
          GetField(Ref("context", contextType), "start"),
          GetField(Ref("context", contextType), "end"),
          I32(1)))
  }
}
