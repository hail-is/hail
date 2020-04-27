package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.expr.ir._
import is.hail.expr.types
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
  letBindings: IndexedSeq[(String, IR)],
  broadcastVals: IndexedSeq[Ref],
  globalsField: String,
  partitioner: RVDPartitioner,
  contextType: Type,
  contexts: IR,
  body: IR) {

  val broadcastValsIR = MakeStruct(broadcastVals.map(ref => (ref.name, ref)))

  def toIR(bodyTransform: IR => IR): IR = {
    val transformedBody = bodyTransform(body)
    val bodyWrappedInBroadcastRefs = broadcastVals.foldRight(transformedBody) { case (ref, inner) =>
      Let(ref.name, GetField(broadcastRef, ref.name), inner)
    }
    CollectDistributedArray(contexts, broadcastValsIR, "context", "global", bodyWrappedInBroadcastRefs)
  }

  def broadcastRef: IR = Ref("global", broadcastValsIR.typ)

  def contextRef: IR = Ref("context", contextType)

  def globals: IR = GetField(broadcastRef, globalsField)
  def wrapInBindings(body: IR): IR = {
    letBindings.foldRight(body){case ((name, binding), soFar) =>
      Let(name, binding, soFar)
    }
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext): IR = {
    def lowerIR(ir: IR) = LowerIR.lower(ir, typesToLower, ctx)

    def lower(tir: TableIR): TableStage = {
      if (typesToLower == DArrayLowering.BMOnly)
        throw new LowererUnsupportedOperation("found TableIR in lowering; lowering only BlockMatrixIRs.")
      tir match {
        case TableRead(typ, dropRows, reader) =>
          val gType = typ.globalType
          val rowType = typ.rowType
          val globalId = genUID()

          reader match {
            case r: TableNativeReader =>
              val path = r.params.path
              val globalsPath = r.spec.globalsComponent.absolutePath(path)
              val globalsSpec = r.spec.globalsSpec
              val gPath = AbstractRVDSpec.partPath(globalsPath, globalsSpec.partFiles.head)
              val globals = ArrayRef(ToArray(ReadPartition(Str(gPath), gType, PartitionNativeReader(globalsSpec.typedCodecSpec))), 0)

              if (dropRows) {
                TableStage(
                  IndexedSeq[(String, IR)](
                    (globalId, globals)
                  ),
                  FastIndexedSeq(Ref(globalId, globals.typ)),
                  globalId,
                  RVDPartitioner.empty(typ.keyType),
                  TStruct.empty,
                  MakeStream(FastIndexedSeq(), TStream(TStruct.empty)),
                  MakeStream(FastIndexedSeq(), TStream(typ.rowType)))
              } else {
                val rowsPath = r.spec.rowsComponent.absolutePath(path)
                val rowsSpec = r.spec.rowsSpec
                val partitioner = rowsSpec.partitioner
                val rSpec = rowsSpec.typedCodecSpec
                val ctxType = TStruct("path" -> TString)

                if (rowsSpec.key startsWith typ.key) {
                  TableStage(
                    IndexedSeq[(String, IR)](
                      (globalId, globals)
                    ),
                    FastIndexedSeq(Ref(globalId, globals.typ)),
                    globalId,
                    partitioner,
                    ctxType,
                    MakeStream(rowsSpec.partFiles.map(f => MakeStruct(FastIndexedSeq("path" -> Str(AbstractRVDSpec.partPath(rowsPath, f))))), TStream(ctxType)),
                    ReadPartition(GetField(Ref("context", ctxType), "path"), rowType, PartitionNativeReader(rSpec)))
                } else {
                  throw new LowererUnsupportedOperation("can't lower a table if sort is needed after read.")
                }
              }
            case r =>
              throw new LowererUnsupportedOperation(s"can't lower a TableRead with reader $r.")
          }

        case TableParallelize(rowsAndGlobal, nPartitions) =>
          val nPartitionsAdj = nPartitions.getOrElse(16)
          val loweredRowsAndGlobal = lowerIR(rowsAndGlobal)
          val loweredRowsAndGlobalRef = Ref(genUID(), loweredRowsAndGlobal.typ)

          val contextType = TStruct(
            "elements" -> TArray(GetField(loweredRowsAndGlobalRef, "rows").typ.asInstanceOf[TArray].elementType)
          )
          val numRows = ArrayLen(GetField(loweredRowsAndGlobalRef, "rows"))

          val numNonEmptyPartitions = If(numRows < nPartitionsAdj, numRows, nPartitionsAdj)
          val numNonEmptyPartitionsRef = Ref(genUID(), numNonEmptyPartitions.typ)

          val q = numRows floorDiv numNonEmptyPartitionsRef
          val qRef = Ref(genUID(), q.typ)

          val remainder = numRows - qRef * numNonEmptyPartitionsRef
          val remainderRef = Ref(genUID(), remainder.typ)

          val context = MakeStream((0 until nPartitionsAdj).map { partIdx =>
            val length = (numRows - partIdx + nPartitionsAdj - 1) floorDiv nPartitionsAdj

            val start = If(numNonEmptyPartitionsRef >= partIdx,
              If(remainderRef > 0,
                If(remainderRef < partIdx, qRef * partIdx + remainderRef, (qRef + 1) * partIdx),
                qRef * partIdx
              ),
              0
            )

            val elements = bindIR(start) { startRef =>
              ToArray(mapIR(rangeIR(startRef, startRef + length)) { elt =>
                ArrayRef(GetField(loweredRowsAndGlobalRef, "rows"), elt)
              })
            }
            MakeStruct(FastIndexedSeq("elements" -> elements))
          }, TStream(contextType))

          val globalsIR = GetField(loweredRowsAndGlobalRef, "global")
          val globalsRef = Ref(genUID(), globalsIR.typ)

          TableStage(
            IndexedSeq[(String, IR)](
              (loweredRowsAndGlobalRef.name, loweredRowsAndGlobal),
              (globalsRef.name, globalsIR),
              (numNonEmptyPartitionsRef.name, numNonEmptyPartitions),
              (qRef.name, q),
              (remainderRef.name, remainder)
            ),
            FastIndexedSeq(globalsRef),
            globalsRef.name,
            RVDPartitioner.unkeyed(nPartitionsAdj),
            contextType,
            context,
            ToStream(GetField(Ref("context", contextType), "elements"))
          )

        case TableRange(n, nPartitions) =>
          val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
          val partCounts = partition(n, nPartitionsAdj)
          val partStarts = partCounts.scanLeft(0)(_ + _)

          val rvdType = RVDType(PType.canonical(tir.typ.rowType).setRequired(true).asInstanceOf[PStruct], Array("idx"))

          val contextType = TStruct(
            "start" -> TInt32,
            "end" -> TInt32)

          val i = Ref(genUID(), TInt32)
          val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }
          val globalId = genUID()
          val globalsIR = MakeStruct(Seq())

          TableStage(
            IndexedSeq[(String, IR)]((globalId, globalsIR)),
            FastIndexedSeq(Ref(globalId, globalsIR.typ)),
            globalId,
            new RVDPartitioner(Array("idx"), tir.typ.rowType,
              ranges.map { case (start, end) =>
                Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
              }),
            contextType,
            MakeStream(ranges.map { case (start, end) =>
              MakeStruct(FastIndexedSeq("start" -> start, "end" -> end)) },
              TStream(contextType)),
            StreamMap(StreamRange(
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
                  Subst(lowerIR(newGlobals),
                    BindingEnv.eval("global" -> GetField(oldbroadcast, loweredChild.globalsField))))))

          loweredChild.copy(broadcastVals = newBroadvastVals, globalsField = newGlobRef)

        case TableFilter(child, cond) =>
          val loweredChild = lower(child)
          val row = Ref(genUID(), child.typ.rowType)
          val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
          loweredChild.copy(body = StreamFilter(loweredChild.body, row.name, Subst(cond, BindingEnv(env))))

        case TableMapRows(child, newRow) =>
          if (ContainsScan(newRow))
            throw new LowererUnsupportedOperation(s"scans are not supported: \n${ Pretty(newRow) }")
          val loweredChild = lower(child)
          val row = Ref(genUID(), child.typ.rowType)
          val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
          loweredChild.copy(body = StreamMap(loweredChild.body, row.name, Subst(newRow, BindingEnv(env, scan = Some(env)))))

        case TableExplode(child, path) =>
          val loweredChild = lower(child)
          val row = Ref(genUID(), child.typ.rowType)

          var fieldRef = path.foldLeft[IR](row) { case (expr, field) => GetField(expr, field) }
          if (!fieldRef.typ.isInstanceOf[TArray])
            fieldRef = ToArray(fieldRef)
          val elt = Ref(genUID(), types.coerce[TContainer](fieldRef.typ).elementType)

          val refs = path.scanLeft(row)((struct, name) =>
            Ref(genUID(), types.coerce[TStruct](struct.typ).field(name).typ))
          val newRow = path.zip(refs).zipWithIndex.foldRight[IR](elt) {
            case (((field, ref), i), arg) =>
              InsertFields(ref, FastIndexedSeq(field ->
                Let(refs(i + 1).name, GetField(ref, field), arg)))
          }.asInstanceOf[InsertFields]

          loweredChild.copy(body = StreamFlatMap(loweredChild.body, row.name, StreamMap(fieldRef, elt.name, newRow)))

        case node =>
          throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
      }
    }

    ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        stage.wrapInBindings(invoke("sum", TInt64, stage.toIR(node => Cast(ArrayLen(ToArray(node)), TInt64))))

      case TableGetGlobals(child) =>
        val stage = lower(child)
        stage.wrapInBindings(GetField(stage.broadcastRef, stage.globalsField))

      case TableCollect(child) =>
        val lowered = lower(child)
        val elt = genUID()
        val cda = lowered.toIR(x => ToArray(x))
        lowered.wrapInBindings(MakeStruct(FastIndexedSeq(
          "rows" -> ToArray(StreamFlatMap(ToStream(cda), elt, ToStream(Ref(elt, cda.typ.asInstanceOf[TArray].elementType)))),
          "global" -> Ref(lowered.globalsField, child.typ.globalType))))

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")
    }
  }
}
