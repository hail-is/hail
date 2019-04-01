package is.hail.backend.spark

import is.hail.annotations._
import is.hail.cxx
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

case class SparkShuffle(child: SparkStage)

case class SparkBinding(name: String, value: IR)

case class SparkStage(
  globals: SparkBinding,
  otherBroadcastVals: List[SparkBinding],
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  rdds: Map[String, RDD[RegionValue]],
  contextType: Type,
  contexts: IR,
  body: IR) {

  val broadcastVals: List[SparkBinding] = otherBroadcastVals :+ globals

  def toIR(bodyTransform: IR => IR): CollectDistributedArray = {
    val globalVals = MakeStruct(broadcastVals.map { case SparkBinding(n, v) => n -> v })
    val substEnv = Env[IR](broadcastVals.map(b => b.name -> GetField(Ref("global", globalVals.typ), b.name)): _*)
    val newBody = Subst(bodyTransform(body), BindingEnv(substEnv))
    CollectDistributedArray(contexts, globalVals, "context", "global", newBody)
  }
}

object LowerTableIR {

  def lower(ir: IR): IR = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      invoke("sum", stage.toIR(node => Cast(ArrayLen(node), TInt64())))

    case TableGetGlobals(child) =>
      lower(child).globals.value

    case TableCollect(child) =>
      val lowered = lower(child)
      assert(lowered.body.typ.isInstanceOf[TContainer])
      val elt = genUID()
      MakeStruct(FastIndexedSeq(
        "rows" -> ArrayFlatMap(lowered.toIR(x => x), elt, Ref(elt, lowered.body.typ)),
        "global" -> lowered.globals.value))

    case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
      throw new cxx.CXXUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")

    case node if node.children.exists( _.isInstanceOf[MatrixIR] ) =>
      throw new cxx.CXXUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case _: In =>
      throw new cxx.CXXUnsupportedOperation(s"`In` value IR node cannot be lowered in Spark backend.")

    case node =>
      Copy(node, ir.children.map { case c: IR => lower(c) })
  }

  // table globals should be stored in the first element of `globals` in SparkStage;
  // globals in SparkStage should have unique identifiers.
  def lower(tir: TableIR): SparkStage = tir match {
    case TableRange(n, nPartitions) =>
      val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
      val partCounts = partition(n, nPartitionsAdj)
      val partStarts = partCounts.scanLeft(0)(_ + _)

      val rvdType = RVDType(tir.typ.rowType.physicalType, Array("idx"))

      val contextType = TStruct(
        "start" -> TInt32(),
        "end" -> TInt32())

      val i = Ref(genUID(), TInt32())
      val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }

      SparkStage(
        SparkBinding(genUID(), MakeStruct(Seq())),
        List(),
        rvdType,
        new RVDPartitioner(Array("idx"), tir.typ.rowType,
          ranges.map { case (start, end) =>
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
        Map.empty,
        contextType,
        MakeArray(ranges.map { case (start, end) =>
          MakeStruct(FastIndexedSeq("start" -> start, "end" -> end)) },
          TArray(contextType)
        ),
        ArrayMap(ArrayRange(
          GetField(Ref("context", contextType), "start"),
          GetField(Ref("context", contextType), "end"),
          I32(1)), i.name, MakeStruct(FastSeq("idx" -> i))))

    case TableMapGlobals(child, newGlobals) =>
      val loweredChild = lower(child)
      val oldGlobals = loweredChild.globals
      val newBroadcastVals = loweredChild.otherBroadcastVals :+ oldGlobals
      loweredChild.copy(otherBroadcastVals = newBroadcastVals, globals = SparkBinding(genUID(), lower(Let("global", oldGlobals.value, newGlobals))))

    case TableFilter(child, cond) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val global = loweredChild.globals
      val env: Env[IR] = Env("row" -> row, "global" -> Ref(global.name, global.value.typ))
      loweredChild.copy(body = ArrayFilter(loweredChild.body, row.name, Subst(cond, BindingEnv(env))))

    case TableMapRows(child, newRow) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val global = loweredChild.globals
      val env: Env[IR] = Env("row" -> row, "global" -> Ref(global.name, global.value.typ))
      loweredChild.copy(body = ArrayMap(loweredChild.body, row.name, Subst(newRow, BindingEnv(env, scan = Some(env)))))

    case TableExplode(child, path) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)

      val fieldRef = path.foldLeft[IR](row) { case (expr, field) => GetField(expr, field) }
      val elt = Ref(genUID(), coerce[TContainer](fieldRef.typ).elementType)

      val refs = path.scanLeft(row)((struct, name) =>
        Ref(genUID(), coerce[TStruct](struct.typ).field(name).typ))
      val newRow = path.zip(refs).zipWithIndex.foldRight[IR](elt) {
        case (((field, ref), i), arg) =>
          InsertFields(ref, FastIndexedSeq(field ->
              Let(refs(i + 1).name, GetField(ref, field), arg)))
      }.asInstanceOf[InsertFields]

      loweredChild.copy(body = ArrayFlatMap(loweredChild.body, row.name, ArrayMap(fieldRef, elt.name, newRow)))

    case node =>
      throw new cxx.CXXUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
  }
}
