package is.hail.backend

import is.hail.HailContext
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

case class ShuffledStage(child: TableStage)

case class Binding(name: String, value: IR)

case class TableStage(
  broadcastVals: IR,
  globalsField: String,
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  contextType: Type,
  contexts: IR,
  body: IR) {

  def toIR(bodyTransform: IR => IR): CollectDistributedArray =
    CollectDistributedArray(contexts, broadcastVals, "context", "global", bodyTransform(body))

  def broadcastRef: IR = Ref("global", broadcastVals.typ)
  def contextRef: IR = Ref("context", contextType)
  def globals: IR = GetField(broadcastRef, globalsField)
}

object LowerTableIR {

  def apply(ir0: IR, timer: Option[ExecutionTimer], optimize: Boolean = true): IR = {

    def opt(context: String, ir: IR): IR =
      Optimize(ir, noisy = true, canGenerateLiterals = true,
        Some(timer.map(t => s"${ t.context }: $context")
          .getOrElse(context)))

    def time(context: String, ir: (String) => IR): IR =
      timer.map(t => t.time(ir(context), context))
        .getOrElse(ir(context))

    var ir = ir0

    if (optimize) { ir = time( "first pass", opt(_, ir)) }

    ir = time("lowering MatrixIR", _ => LowerMatrixIR(ir))

    if (optimize) { ir = time("after MatrixIR lowering", opt(_, ir)) }

    ir = time("lowering TableIR", _ => LowerTableIR.lower(ir))

    if (optimize) { ir = time("after TableIR lowering", opt(_, ir)) }
    ir
  }

  def lower(ir: IR): IR = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      invoke("sum", TInt64(), stage.toIR(node => Cast(ArrayLen(node), TInt64())))

    case TableGetGlobals(child) =>
      val stage = lower(child)
      GetField(stage.broadcastVals, stage.globalsField)

    case TableCollect(child) =>
      val lowered = lower(child)
      assert(lowered.body.typ.isInstanceOf[TContainer], s"${ lowered.body.typ }")
      val elt = genUID()
      MakeStruct(FastIndexedSeq(
        "rows" -> ArrayFlatMap(lowered.toIR(x => x), elt, Ref(elt, lowered.body.typ)),
        "global" -> GetField(lowered.broadcastVals, lowered.globalsField)))

    case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
      throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")

    case node if node.children.exists( _.isInstanceOf[MatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case node if node.children.exists( _.isInstanceOf[BlockMatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"BlockMatrixIR nodes are not supported: \n${ Pretty(node) }")

    case node =>
      Copy(node, ir.children.map { case c: IR => lower(c) })
  }

  // table globals should be stored in the first element of `globals` in TableStage;
  // globals in TableStage should have unique identifiers.
  def lower(tir: TableIR): TableStage = tir match {
    case TableRead(typ, dropRows, reader) =>
      val gType = typ.globalType
      val rowType = typ.rowType
      val rvdType = typ.canonicalRVDType
      val globalRef = genUID()

      reader match {
        case r@TableNativeReader(path, None, _) =>
          val globalsPath = r.spec.globalsComponent.absolutePath(path)
          val globalsSpec = AbstractRVDSpec.read(HailContext.get, globalsPath)
          val gPath = AbstractRVDSpec.partPath(globalsPath, globalsSpec.partFiles.head)
          val gSpec = globalsSpec.codecSpec2

          if (dropRows) {
            TableStage(
              MakeStruct(FastIndexedSeq(globalRef -> ArrayRef(ReadPartition(Str(gPath), gSpec, gType), 0))),
              globalRef,
              rvdType,
              RVDPartitioner.empty(rvdType),
              TStruct(),
              MakeArray(FastIndexedSeq(), TArray(TStruct())),
              MakeArray(FastIndexedSeq(), TArray(rvdType.rowType.virtualType)))
          } else {
            val rowsPath = r.spec.rowsComponent.absolutePath(path)
            val rowsSpec = AbstractRVDSpec.read(HailContext.get, rowsPath)
            val partitioner = rowsSpec.partitioner
            val rSpec = rowsSpec.codecSpec2
            val ctxType = TStruct("path" -> TString())

            if (rowsSpec.key startsWith typ.key) {
              TableStage(
                MakeStruct(FastIndexedSeq(globalRef -> ArrayRef(ToArray(ReadPartition(Str(gPath), gSpec, gType)), 0))),
                globalRef,
                rvdType,
                partitioner,
                ctxType,
                MakeArray(rowsSpec.partFiles.map(f => MakeStruct(FastIndexedSeq("path" -> Str(AbstractRVDSpec.partPath(rowsPath, f))))), TArray(ctxType)),
                ToArray(ReadPartition(GetField(Ref("context", ctxType), "path"), rSpec, rowType)))
            } else {
              throw new LowererUnsupportedOperation("can't lower a table if sort is needed after read.")
            }
          }
        case r =>
          throw new LowererUnsupportedOperation(s"can't lower a TableRead with reader $r.")
      }

    case TableRange(n, nPartitions) =>
      val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
      val partCounts = partition(n, nPartitionsAdj)
      val partStarts = partCounts.scanLeft(0)(_ + _)

      val rvdType = RVDType(PType.canonical(tir.typ.rowType).asInstanceOf[PStruct], Array("idx"))

      val contextType = TStruct(
        "start" -> TInt32(),
        "end" -> TInt32())

      val i = Ref(genUID(), TInt32())
      val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }
      val globalRef = genUID()

      TableStage(
        MakeStruct(FastIndexedSeq(globalRef -> MakeStruct(Seq()))),
        globalRef,
        rvdType,
        new RVDPartitioner(Array("idx"), tir.typ.rowType,
          ranges.map { case (start, end) =>
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
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
      val oldbroadcast = Ref(genUID(), loweredChild.broadcastVals.typ)
      val newGlobRef = genUID()
      val newBroadvastVals =
        Let(
          oldbroadcast.name,
          loweredChild.broadcastVals,
          InsertFields(oldbroadcast,
            FastIndexedSeq(newGlobRef ->
              Subst(lower(newGlobals),
                BindingEnv.eval("global" -> GetField(oldbroadcast, loweredChild.globalsField))))))

      loweredChild.copy(broadcastVals = newBroadvastVals, globalsField = newGlobRef)

    case TableFilter(child, cond) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
      loweredChild.copy(body = ArrayFilter(loweredChild.body, row.name, Subst(cond, BindingEnv(env))))

    case TableMapRows(child, newRow) =>
      if (ContainsScan(newRow))
        throw new LowererUnsupportedOperation(s"scans are not supported: \n${ Pretty(newRow) }")
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
      loweredChild.copy(body = ArrayMap(loweredChild.body, row.name, Subst(newRow, BindingEnv(env, scan = Some(env)))))

    case TableExplode(child, path) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)

      var fieldRef = path.foldLeft[IR](row) { case (expr, field) => GetField(expr, field) }
      if (!fieldRef.typ.isInstanceOf[TArray])
        fieldRef = ToArray(fieldRef)
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
      throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
  }
}
