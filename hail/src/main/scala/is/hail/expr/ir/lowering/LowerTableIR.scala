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
  broadcastVals: IR,
  globalsField: String,
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
  def lower(ir: IR, typesToLower: DArrayLowering.Type): IR =
    new LowerTableIR(typesToLower).lower(ir)
}

class LowerTableIR(val typesToLower: DArrayLowering.Type) extends AnyVal {
  def lowerIR(ir: IR) = LowerIR.lower(ir, typesToLower)

  def lower(ir: IR): IR = ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        invoke("sum", TInt64, stage.toIR(node => Cast(ArrayLen(ToArray(node)), TInt64)))

      case TableGetGlobals(child) =>
        val stage = lower(child)
        GetField(stage.broadcastVals, stage.globalsField)

      case TableCollect(child) =>
        val lowered = lower(child)
        val elt = genUID()
        val cda = lowered.toIR(x => ToArray(x))
        MakeStruct(FastIndexedSeq(
          "rows" -> ToArray(StreamFlatMap(ToStream(cda), elt, ToStream(Ref(elt, cda.typ.asInstanceOf[TArray].elementType)))),
          "global" -> GetField(lowered.broadcastVals, lowered.globalsField)))

      case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")

      case node =>
        throw new LowererUnsupportedOperation(s"Value IRs with no TableIR children must be lowered through LowerIR: \n${ Pretty(node) }")
    }

  // table globals should be stored in the first element of `globals` in TableStage;
  // globals in TableStage should have unique identifiers.
  def lower(tir: TableIR): TableStage = {
    if (typesToLower == DArrayLowering.BMOnly)
      throw new LowererUnsupportedOperation("found TableIR in lowering; lowering only BlockMatrixIRs.")
    tir match {
      case TableRead(typ, dropRows, reader) =>
        val gType = typ.globalType
        val rowType = typ.rowType
        val globalRef = genUID()

        reader match {
          case r@TableNativeReader(path, None, _) =>
            val globalsPath = r.spec.globalsComponent.absolutePath(path)
            val globalsSpec = AbstractRVDSpec.read(HailContext.get, globalsPath)
            val gPath = AbstractRVDSpec.partPath(globalsPath, globalsSpec.partFiles.head)
            val globals = ArrayRef(ToArray(ReadPartition(Str(gPath), globalsSpec.typedCodecSpec, gType)), 0)

            if (dropRows) {
              TableStage(
                MakeStruct(FastIndexedSeq(globalRef -> globals)),
                globalRef,
                RVDPartitioner.empty(typ.keyType),
                TStruct.empty,
                MakeStream(FastIndexedSeq(), TStream(TStruct.empty)),
                MakeStream(FastIndexedSeq(), TStream(typ.rowType)))
            } else {
              val rowsPath = r.spec.rowsComponent.absolutePath(path)
              val rowsSpec = AbstractRVDSpec.read(HailContext.get, rowsPath)
              val partitioner = rowsSpec.partitioner
              val rSpec = rowsSpec.typedCodecSpec
              val ctxType = TStruct("path" -> TString)

              if (rowsSpec.key startsWith typ.key) {
                TableStage(
                  MakeStruct(FastIndexedSeq(globalRef -> globals)),
                  globalRef,
                  partitioner,
                  ctxType,
                  MakeStream(rowsSpec.partFiles.map(f => MakeStruct(FastIndexedSeq("path" -> Str(AbstractRVDSpec.partPath(rowsPath, f))))), TStream(ctxType)),
                  ReadPartition(GetField(Ref("context", ctxType), "path"), rSpec, rowType))
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
          "start" -> TInt32,
          "end" -> TInt32)

        val i = Ref(genUID(), TInt32)
        val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }
        val globalRef = genUID()

        TableStage(
          MakeStruct(FastIndexedSeq(globalRef -> MakeStruct(Seq()))),
          globalRef,
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
}
