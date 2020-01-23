package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.expr.ir._
import is.hail.expr.types
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

case class ShuffledStage(child: TableStage)

case class Binding(name: String, value: IR)

case class TableStage(
  broadcastStruct: IR,
  globalsField: String,
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  contexts: Map[String, (Type, Array[IR])],
  body: IR) {

  private lazy val bcType: TStruct = coerce[TStruct](broadcastStruct.typ)
  private lazy val broadcastFields: IndexedSeq[String] = bcType.fieldNames
  private lazy val bcRef: Ref = Ref(genUID(), bcType)
  def globalRef: Ref = Ref(globalsField, bcType.fieldType(globalsField))

  def collect(bodyTransform: IR => IR): CollectDistributedArray = {
    val nParts = partitioner.numPartitions
    val contextFields = contexts.mapValues(_._1).toArray
    val ctxType = TStruct(contextFields: _*)
    val ctxRef = Ref(genUID(), ctxType)

    val contextIR = MakeArray(
      Array.tabulate(nParts) { i =>
        MakeStruct(ctxType.fieldNames.map { name =>
          val contextVal = contexts(name)._2(i)
          assert(contextVal.typ == ctxType.fieldType(name))
          name -> contextVal
        })
      },
      TArray(ctxType))

    val b = contexts.keys.foldLeft(
      broadcastFields.foldLeft(
        bodyTransform(body)) {
        case (accum, fname) =>
          Let(fname, GetField(bcRef, fname), accum)
      }) {
      (accum, ctxF) => Let(ctxF, GetField(ctxRef, ctxF), accum)
    }
    CollectDistributedArray(contextIR, bcRef, ctxRef.name, bcRef.name, b)
  }

  def withGlobals(body: IR): IR =
    Let(bcRef.name, broadcastStruct, broadcastFields.foldLeft(body) {
      case (accum, fname) =>
        Let(fname, GetField(bcRef, fname), accum)
    })
}

object LowerTableIR {
  def lower(ir: IR): IR = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      stage.withGlobals(invoke("sum", TInt64(), stage.collect(node => Cast(ArrayLen(node), TInt64()))))

    case TableGetGlobals(child) =>
      val stage = lower(child)
      stage.withGlobals(stage.globalRef)

    case TableCollect(child) =>
      val lowered = lower(child)
      assert(lowered.body.typ.isInstanceOf[TContainer], s"${ lowered.body.typ }")
      val elt = genUID()
      lowered.withGlobals(MakeStruct(FastIndexedSeq(
        "rows" -> ArrayFlatMap(lowered.collect(x => x), elt, Ref(elt, lowered.body.typ)),
        "global" -> lowered.globalRef)))

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
          val globals = ArrayRef(ToArray(ReadPartition(Str(gPath), globalsSpec.typedCodecSpec, gType)), 0)
          val gRef = Ref(genUID(), gType)

          if (dropRows) {
            TableStage(
              MakeStruct(FastIndexedSeq(globalRef -> globals)),
              globalRef,
              rvdType,
              RVDPartitioner.empty(rvdType),
              Map(),
              MakeArray(FastIndexedSeq(), TArray(rvdType.rowType.virtualType)))
          } else {
            val rowsPath = r.spec.rowsComponent.absolutePath(path)
            val rowsSpec = AbstractRVDSpec.read(HailContext.get, rowsPath)
            val partitioner = rowsSpec.partitioner
            val rSpec = rowsSpec.typedCodecSpec
            val partPathRef = Ref(genUID(), TString())

            if (rowsSpec.key startsWith typ.key) {
              TableStage(
                MakeStruct(FastIndexedSeq(globalRef -> globals)),
                globalRef,
                rvdType,
                partitioner,
                Map(partPathRef.name -> (partPathRef.typ, rowsSpec.partFiles.map(f => Str(AbstractRVDSpec.partPath(rowsPath, f))))),
                ReadPartition(partPathRef, rSpec, rowType))
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

      val i = Ref(genUID(), TInt32())
      val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }
      val globalRef = Ref(genUID(), TStruct())

      val startRef = Ref(genUID(), TInt32())
      val endRef = Ref(genUID(), TInt32())

      TableStage(
        MakeStruct(FastIndexedSeq(globalRef.name -> MakeStruct(Seq()))),
        globalRef.name,
        rvdType,
        new RVDPartitioner(Array("idx"), tir.typ.rowType,
          ranges.map { case (start, end) =>
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
        Map(startRef.name -> (startRef.typ, ranges.map { case (start, _) => I32(start) }),
          endRef.name -> (endRef.typ, ranges.map { case (_, end) => I32(end) })),
        ArrayMap(ArrayRange(startRef, endRef,
          I32(1)), i.name, MakeStruct(FastSeq("idx" -> i))))

    case TableMapGlobals(child, newGlobals) =>
      val lowered = lower(child)
      val gRef = Ref(genUID(), newGlobals.typ)
      val newBroadcast = InsertFields(lowered.broadcastStruct, FastIndexedSeq(gRef.name -> Let("global", lowered.globalRef, lower(newGlobals))))
      lowered.copy(broadcastStruct = newBroadcast, globalsField = gRef.name)

    case TableFilter(child, cond) =>
      val lowered = lower(child)
      val filtered = Let("global", lowered.globalRef, ArrayFilter(lowered.body, "row", lower(cond)))
      lowered.copy(body = filtered)

    case TableMapRows(child, newRow) if !ContainsScan(newRow) =>
      val lowered = lower(child)
      val mapped = Let("global", lowered.globalRef, ArrayMap(lowered.body, "row", lower(newRow)))
      lowered.copy(body = mapped)

    case x@TableExplode(child, path) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)

      val fieldRef = ToArray(path.foldLeft[IR](row) { case (expr, field) => GetField(expr, field) })
      val elt = Ref(genUID(), types.coerce[TArray](fieldRef.typ).elementType)

      val refs = path.scanLeft(row)((struct, name) =>
        Ref(genUID(), types.coerce[TStruct](struct.typ).field(name).typ))
      val newRow = path.zip(refs).zipWithIndex.foldRight[IR](elt) {
        case (((field, ref), i), arg) =>
          InsertFields(ref, FastIndexedSeq(field ->
              Let(refs(i + 1).name, GetField(ref, field), arg)))
      }

      loweredChild.copy(body = ArrayFlatMap(loweredChild.body, row.name, ArrayMap(fieldRef, elt.name, newRow)))

    case node =>
      throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
  }
}
