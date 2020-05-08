package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.types
import is.hail.expr.types.virtual._
import is.hail.methods.NPartitionsTable
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner}
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

case class ShuffledStage(child: TableStage)

case class Binding(name: String, value: IR)

abstract class TableStage(
  val letBindings: IndexedSeq[(String, IR)],
  val broadcastVals: Set[String],
  val globals: IR,
  val partitioner: RVDPartitioner,
  val contexts: IR) { self =>

  def partition(ctxRef: Ref): IR

  def wrapInBindings(body: IR): IR = {
    letBindings.foldRight(body) { case ((name, binding), soFar) => Let(name, binding, soFar) }
  }

  def mapPartition(f: IR => IR): TableStage = {
    val outer = this
    new TableStage(letBindings, broadcastVals, globals, partitioner, contexts) {
      def partition(ctxRef: Ref): IR = f(outer.partition(ctxRef))
    }
  }

  def mapContexts(f: IR => IR): TableStage = {
    val outer = this
    new TableStage(letBindings, broadcastVals, globals, partitioner, f(contexts)) {
      override def partition(ctxRef: Ref): IR = outer.partition(ctxRef)
    }
  }

  def mapGlobals(f: IR => IR): TableStage = {
    val newGlobals = f(globals)
    val newID = genUID()
    val outer = this
    new TableStage(
      letBindings :+ newID -> newGlobals,
      broadcastVals + newID,
      Ref(newID, newGlobals.typ),
      partitioner, contexts) {
      def partition(ctxRef: Ref): IR = outer.partition(ctxRef)
    }
  }

  def collect(bind: Boolean = true): IR = {
    val ctx = Ref(genUID(), types.coerce[TStream](contexts.typ).elementType)
    val broadcastRefs = MakeStruct(letBindings.filter { case (name, _) => broadcastVals.contains(name) })
    val glob = Ref(genUID(), broadcastRefs.typ)
    val cda = CollectDistributedArray(contexts, broadcastRefs,
      ctx.name, glob.name,
      broadcastVals.foldLeft(partition(ctx))((accum, name) => Let(name, GetField(glob, name), accum)))
    if (bind) wrapInBindings(cda) else cda
  }

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage = {
    new TableStage(letBindings, broadcastVals, globals, newPartitioner, contexts) {
      def partition(ctxRef: Ref): IR = self.partition(ctxRef)
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
          val lowered = reader.lower(ctx, typ)
          val globalsID = genUID()

          if (dropRows) {
            new TableStage(
              FastIndexedSeq(globalsID -> lowered.globals),
              Set(globalsID),
              Ref(globalsID, lowered.globals.typ),
              RVDPartitioner.empty(typ.keyType),
              MakeStream(FastIndexedSeq(), TStream(TStruct.empty))) {
              def partition(ctxRef: Ref): IR = MakeStream(FastIndexedSeq(), TStream(typ.rowType))
            }
          } else {
            new TableStage(
              FastIndexedSeq(globalsID -> lowered.globals),
              Set(globalsID),
              Ref(globalsID, lowered.globals.typ),
              lowered.partitioner,
              lowered.contexts) {
              def partition(ctxRef: Ref): IR = lowered.body(ctxRef)
            }
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

          new TableStage(
            FastIndexedSeq[(String, IR)](
              (loweredRowsAndGlobalRef.name, loweredRowsAndGlobal),
              (globalsRef.name, globalsIR),
              (numNonEmptyPartitionsRef.name, numNonEmptyPartitions),
              (qRef.name, q),
              (remainderRef.name, remainder)
            ),
            Set(globalsRef.name),
            globalsRef,
            RVDPartitioner.unkeyed(nPartitionsAdj),
            context) {
            override def partition(ctxRef: Ref): IR = {
              ToStream(GetField(ctxRef, "elements"))
            }
          }

        case TableRange(n, nPartitions) =>
          val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
          val partCounts = partition(n, nPartitionsAdj)
          val partStarts = partCounts.scanLeft(0)(_ + _)

          val contextType = TStruct(
            "start" -> TInt32,
            "end" -> TInt32)

          val i = Ref(genUID(), TInt32)
          val ranges = Array.tabulate(nPartitionsAdj) { i => partStarts(i) -> partStarts(i + 1) }

          new TableStage(
            FastIndexedSeq.empty[(String, IR)],
            Set(),
            MakeStruct(FastSeq()),
            new RVDPartitioner(Array("idx"), tir.typ.rowType,
              ranges.map { case (start, end) =>
                Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
              }),
            MakeStream(ranges.map { case (start, end) =>
              MakeStruct(FastIndexedSeq("start" -> start, "end" -> end)) },
              TStream(contextType))) {
            override def partition(ctxRef: Ref): IR = {
              StreamMap(StreamRange(
                GetField(ctxRef, "start"),
                GetField(ctxRef, "end"),
                I32(1)), i.name, MakeStruct(FastSeq("idx" -> i)))
            }
          }

        case TableMapGlobals(child, newGlobals) =>
          lower(child).mapGlobals(old => Let("global", old, newGlobals))

        case TableFilter(child, cond) =>
          val row = Ref(genUID(), child.typ.rowType)
          val loweredChild = lower(child)
          val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
          loweredChild.mapPartition(rows => StreamFilter(rows, row.name, Subst(cond, BindingEnv(env))))

        case TableHead(child, targetNumRows) =>
          val loweredChild = lower(child)

          val partitionSizeArrayFunc = genUID()
          val howManyPartsToTry = Ref(genUID(), TInt32)
          val partitionSizeArray = TailLoop(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry.name -> 4),
            bindIR(loweredChild.mapContexts(ctxs => StreamTake(ctxs, howManyPartsToTry)).mapPartition(rows =>  ArrayLen(ToArray(rows))).collect()) { counts =>
              If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (ArrayLen(ToArray(loweredChild.contexts)) <= ArrayLen(counts)),
                counts,
                Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry * 4), TArray(TInt32))
              )
            }
          )
          val partitionSizeArrayRef = Ref(genUID(), partitionSizeArray.typ)

          val answerTuple = bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            val howManyPartsToKeep = genUID()
            val i = Ref(genUID(), TInt32)
            val numLeft = Ref(genUID(), TInt64)
            def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple(FastIndexedSeq((0, howManyParts), (1, howManyFromLast)))

            If(numPartitions ceq 0,
              makeAnswer(0, 0L),
              TailLoop(howManyPartsToKeep, FastIndexedSeq(i.name -> 0, numLeft.name -> targetNumRows),
                If((i ceq numPartitions - 1) || ((numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)) <= 0L),
                  makeAnswer(i + 1, numLeft),
                  Recur(howManyPartsToKeep, FastIndexedSeq(i + 1, numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)), TTuple(TInt32, TInt64))
                )
              )
            )
          }

          new TableStage(
            loweredChild.letBindings :+ partitionSizeArrayRef.name -> partitionSizeArray,
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            {
              val contexts = loweredChild.contexts
              val contextElementType = contexts.typ.asInstanceOf[TStream].elementType
              bindIR(answerTuple) { answerTupleRef =>
                val numParts = GetTupleElement(answerTupleRef, 0)
                val numElementsFromLastPart = GetTupleElement(answerTupleRef, 1)
                val onlyNeededPartitions = StreamTake(contexts, numParts)
                val howManyFromEachPart = mapIR(rangeIR(numParts)) { idxRef =>
                  If(idxRef ceq (numParts - 1),
                    Cast(numElementsFromLastPart, TInt32),
                    ArrayRef(partitionSizeArrayRef, idxRef)
                  )
                }
                StreamZip(FastIndexedSeq(onlyNeededPartitions, howManyFromEachPart), FastIndexedSeq("part", "howMany"),
                  MakeStruct(FastIndexedSeq("numberToTake" -> Ref("howMany", TInt32), "old" -> Ref("part", contextElementType))),
                  ArrayZipBehavior.AssumeSameLength
                )
              }
            }
          ) {
            override def partition(ctxRef: Ref): IR = {
              bindIR(GetField(ctxRef, "old")) { oldRef =>
                StreamTake(loweredChild.partition(oldRef), GetField(ctxRef, "numberToTake"))
              }
            }
          }

        case TableTail(child, targetNumRows) =>
          val loweredChild = lower(child)

          val partitionSizeArrayFunc = genUID()
          val howManyPartsToTry = Ref(genUID(), TInt32)

          val totalNumPartitions = ArrayLen(ToArray(loweredChild.contexts))
          val totalNumPartitionsRef = Ref(genUID(), TInt32)
          val partitionSizeArray =
            TailLoop(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry.name -> 4),
              bindIR(loweredChild.mapContexts(ctxs => StreamDrop(ctxs, maxIR(totalNumPartitionsRef - howManyPartsToTry, 0))).mapPartition(rows => ArrayLen(ToArray(rows))).collect()) { counts =>
                If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitionsRef <= ArrayLen(counts)),
                  counts,
                  Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry * 4), TArray(TInt32))
                )
              }
            )

          val partitionSizeArrayRef = Ref(genUID(), partitionSizeArray.typ)

          // First element is how many partitions to drop from partitionSizeArrayRef, second is how many to keep from first kept element.
          val answerTuple = bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            val howManyPartsToDrop = genUID()
            val i = Ref(genUID(), TInt32)
            val numLeft = Ref(genUID(), TInt64)
            def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple(FastIndexedSeq((0, howManyParts), (1, howManyFromLast)))

            If(numPartitions ceq 0,
              makeAnswer(0, 0L),
              TailLoop(howManyPartsToDrop, FastIndexedSeq(i.name -> numPartitions, numLeft.name -> targetNumRows),
                If((i ceq 1) || ((numLeft - Cast(ArrayRef(partitionSizeArrayRef, i - 1), TInt64)) <= 0L),
                  makeAnswer(i - 1, numLeft),
                  Recur(howManyPartsToDrop, FastIndexedSeq(i - 1, numLeft - Cast(ArrayRef(partitionSizeArrayRef, i - 1), TInt64)), TTuple(TInt32, TInt64))
                )
              )
            )
        }

          new TableStage(
            loweredChild.letBindings ++ FastIndexedSeq(
              totalNumPartitionsRef.name -> totalNumPartitions,
              partitionSizeArrayRef.name -> partitionSizeArray
            ),
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            {
              val contexts = loweredChild.contexts
              val contextElementType = contexts.typ.asInstanceOf[TStream].elementType
              bindIR(answerTuple) { answerTupleRef =>
                val numPartsToDropFromPartitionSizeArray = GetTupleElement(answerTupleRef, 0)
                val numElementsFromFirstPart = GetTupleElement(answerTupleRef, 1)
                val numPartsToDropFromTotal = numPartsToDropFromPartitionSizeArray + (totalNumPartitionsRef - ArrayLen(partitionSizeArrayRef))
                val onlyNeededPartitions = StreamDrop(contexts, numPartsToDropFromTotal)
                val howManyFromEachPart = mapIR(rangeIR(ArrayLen(ToArray(onlyNeededPartitions)))) { idxRef =>
                  If(idxRef ceq 0,
                    Cast(numElementsFromFirstPart, TInt32),
                    ArrayRef(partitionSizeArrayRef, idxRef)
                  )
                }
                StreamZip(FastIndexedSeq(onlyNeededPartitions, howManyFromEachPart, StreamDrop(ToStream(partitionSizeArrayRef), numPartsToDropFromPartitionSizeArray)), FastIndexedSeq("part", "howMany", "partLength"),
                  MakeStruct(FastIndexedSeq(
                    "numberToDrop" -> maxIR(0, Ref("partLength", TInt32) - Ref("howMany", TInt32)),
                    "old" -> Ref("part", contextElementType))),
                  ArrayZipBehavior.AssertSameLength
                )
              }
            }
          ) {
            override def partition(ctxRef: Ref): IR = {
              bindIR(GetField(ctxRef, "old")) { oldRef =>
                StreamDrop(loweredChild.partition(oldRef), GetField(ctxRef, "numberToDrop"))
              }
            }
          }

        case TableMapRows(child, newRow) =>
          if (ContainsScan(newRow))
            throw new LowererUnsupportedOperation(s"scans are not supported: \n${ Pretty(newRow) }")
          val loweredChild = lower(child)
          loweredChild.mapPartition(rows => mapIR(rows) { row =>
            val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
            Subst(newRow, BindingEnv(env, scan = Some(env)))
          })

        case TableGroupWithinPartitions(child, groupedStructName, n) =>
          val loweredChild = lower(child)
          val keyFields = FastIndexedSeq(child.typ.keyType.fieldNames: _*)
          loweredChild.mapPartition { part =>
            val grouped =  StreamGrouped(part, n)
            val groupedArrays = mapIR(grouped) (group => ToArray(group))
            val withKeys = mapIR(groupedArrays) {group =>
              bindIR(group) { groupRef =>
                bindIR(ArrayRef(groupRef, 0)) { firstElement =>
                  val firstElementKeys = keyFields.map(keyField => (keyField, GetField(firstElement, keyField)))
                  val rowStructFields = firstElementKeys ++ FastIndexedSeq(groupedStructName -> groupRef)
                  MakeStruct(rowStructFields)
                }
              }
            }
            withKeys
          }

        case t@TableKeyBy(child, newKey, isSorted: Boolean) =>
          val loweredChild = lower(child)
          val nPreservedFields = loweredChild.partitioner.kType.fieldNames
            .zip(newKey)
            .takeWhile { case (l, r) => l == r }
            .length

          if (nPreservedFields == newKey.length)
            loweredChild
          else if (isSorted) {
            val newPartitioner = loweredChild.partitioner
              .coarsen(nPreservedFields)
              .extendKey(t.typ.keyType)
            loweredChild.changePartitionerNoRepartition(newPartitioner)
          } else
            ctx.backend.lowerDistributedSort(loweredChild, newKey.map(k => SortField(k, Ascending)))

        case TableOrderBy(child, sortFields) =>
          val loweredChild = lower(child)
          if (TableOrderBy.isAlreadyOrdered(sortFields, loweredChild.partitioner.kType.fieldNames))
            loweredChild
          else
            ctx.backend.lowerDistributedSort(loweredChild, sortFields)

        case TableExplode(child, path) =>
          lower(child).mapPartition { rows =>
            flatMapIR(rows) { row: Ref =>
              val refs = Array.fill[Ref](path.length + 1)(null)
              val roots = Array.fill[IR](path.length)(null)
              var i = 0
              refs(0) = row
              while (i < path.length) {
                roots(i) = GetField(refs(i), path(i))
                refs(i + 1) = Ref(genUID(), roots(i).typ)
                i += 1
              }
              refs.tail.zip(roots).foldRight(
                mapIR(refs.last) { elt =>
                  path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
                    InsertFields(ref, FastSeq(p -> inserted))
                  }
                }) { case ((ref, root), accum) =>  Let(ref.name, root, accum) }
            }
          }

        case TableRename(child, rowMap, globalMap) =>
          val loweredChild = lower(child)
          val newGlobId = genUID()
          val newGlobals = CastRename(loweredChild.globals, loweredChild.globals.typ.asInstanceOf[TStruct].rename(globalMap))
          new TableStage(
            loweredChild.letBindings :+ newGlobId -> newGlobals,
            loweredChild.broadcastVals + newGlobId,
            Ref(newGlobId, newGlobals.typ),
            loweredChild.partitioner.copy(kType = loweredChild.partitioner.kType.rename(rowMap)),
            loweredChild.contexts
          ) {
            override def partition(ctxRef: Ref): IR = {
              val oldPartition = loweredChild.partition(ctxRef)
              mapIR(oldPartition) { row =>
                CastRename(row, row.typ.asInstanceOf[TStruct].rename(rowMap))
              }
            }
          }

        case node =>
          throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
      }
    }

    ir match {
      case TableCount(tableIR) =>
        invoke("sum", TInt64,
          lower(tableIR).mapPartition(rows => Cast(ArrayLen(ToArray(rows)), TInt64)).collect())

      case TableGetGlobals(child) =>
        val stage = lower(child)
        stage.wrapInBindings(stage.globals)

      case TableCollect(child) =>
        val lowered = lower(child).mapPartition(ToArray)
        lowered.wrapInBindings(
          MakeStruct(FastIndexedSeq(
            "rows" -> ToArray(flatMapIR(ToStream(lowered.collect(bind = false))) { elt => ToStream(elt) }),
            "global" -> lowered.globals)))

      case TableToValueApply(child, NPartitionsTable()) =>
        val lowered = lower(child)
        ArrayLen(ToArray(lowered.contexts))

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")
    }
  }
}
