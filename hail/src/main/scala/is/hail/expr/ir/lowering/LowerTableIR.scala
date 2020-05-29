package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.types
import is.hail.types.virtual._
import is.hail.methods.{ForceCountTable, NPartitionsTable}
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner}
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

case class ShuffledStage(child: TableStage)

case class Binding(name: String, value: IR)

object TableStage {
  def apply(
    globals: IR,
    partitioner: RVDPartitioner,
    contexts: IR,
    body: (IR) => IR): TableStage = {
    val globalsID = genUID()
    new TableStage(
      FastIndexedSeq(globalsID -> globals),
      Set(globalsID),
      Ref(globalsID, globals.typ),
      partitioner,
      contexts) {
      def partition(ctxRef: Ref): IR = body(ctxRef)
    }
  }
}

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

  def collectWithGlobals(): IR = {
    val mapped = mapPartition(ToArray)
    mapped.wrapInBindings(
      MakeStruct(FastIndexedSeq(
        "rows" -> ToArray(flatMapIR(ToStream(mapped.collect(bind = false))) { elt => ToStream(elt) }),
        "global" -> mapped.globals)))
  }

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage = {
    new TableStage(letBindings, broadcastVals, globals, newPartitioner, contexts) {
      def partition(ctxRef: Ref): IR = self.partition(ctxRef)
    }
  }

  def repartitionNoShuffle(newPartitioner: RVDPartitioner): TableStage = {
    require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    require(newPartitioner.kType.isPrefixOf(partitioner.kType))

    val boundType = RVDPartitioner.intervalIRRepresentation(newPartitioner.kType)
    val partitionMapping: IndexedSeq[Row] = newPartitioner.rangeBounds.map { i =>
      Row(RVDPartitioner.intervalToIRRepresentation(i, newPartitioner.kType.size), partitioner.queryInterval(i))
    }
    val partitionMappingType = TStruct(
      "partitionBound" -> boundType,
      "parentPartitions" -> TArray(TInt32)
    )

    val prevContextUID = genUID()
    val mappingUID = genUID()
    val idxUID = genUID()
    val newContexts = Let(
      prevContextUID,
      ToArray(contexts),
      StreamMap(
        ToStream(
          Literal(
            TArray(partitionMappingType),
            partitionMapping)),
        mappingUID,
        MakeStruct(
          FastSeq(
            "partitionBound" -> GetField(Ref(mappingUID, partitionMappingType), "partitionBound"),
            "oldContexts" -> ToArray(
              StreamMap(
                ToStream(GetField(Ref(mappingUID, partitionMappingType), "parentPartitions")),
                idxUID,
                ArrayRef(Ref(prevContextUID, TArray(contexts.typ.asInstanceOf[TStream].elementType)), Ref(idxUID, TInt32))
              ))
          )
        )
      )
    )

    val intervalUID = genUID()
    val eltUID = genUID()
    val prevContextUIDPartition = genUID()
    new TableStage(letBindings, broadcastVals, globals, newPartitioner, newContexts) {
      override def partition(ctxRef: Ref): IR = {
        val body = self.partition(Ref(prevContextUIDPartition, self.contexts.typ.asInstanceOf[TStream].elementType))
        Let(
          intervalUID,
          GetField(ctxRef, "partitionBound"),
          StreamFilter(
            StreamFlatMap(
              ToStream(GetField(ctxRef, "oldContexts")),
              prevContextUIDPartition,
              body
            ),
            eltUID,
            invoke("partitionIntervalContains",
              TBoolean,
              Ref(intervalUID, boundType),
              SelectFields(Ref(eltUID, body.typ.asInstanceOf[TStream].elementType), partitioner.kType.fieldNames)
            )
          ))
      }
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
          if (dropRows) {
            val globals = reader.lowerGlobals(ctx, typ.globalType)
            val globalsID = genUID()
            new TableStage(
              FastIndexedSeq(globalsID -> globals),
              Set(globalsID),
              Ref(globalsID, globals.typ),
              RVDPartitioner.empty(typ.keyType),
              MakeStream(FastIndexedSeq(), TStream(TStruct.empty))) {
              def partition(ctxRef: Ref): IR = MakeStream(FastIndexedSeq(), TStream(typ.rowType))
            }
          } else
            reader.lower(ctx, typ)

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
              MakeStruct(FastIndexedSeq[(String, IR)]("start" -> start, "end" -> end)) },
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

        case TableAggregateByKey(child, expr) =>
          val loweredChild = lower(child)

          loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
            .mapPartition { partition =>

              mapIR(StreamGroupByKey(partition, child.typ.key)) { groupRef =>
                StreamAgg(
                  groupRef,
                  "row",
                  bindIRs(ArrayRef(ApplyAggOp(FastSeq(I32(1)), FastSeq(SelectFields(Ref("row", child.typ.rowType), child.typ.key)),
                    AggSignature(Take(), FastSeq(TInt32), FastSeq(child.typ.keyType))), I32(0)), // FIXME: would prefer a First() agg op
                    expr) { case Seq(key, value) =>
                    MakeStruct(child.typ.key.map(k => (k, GetField(key, k))) ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                      (f, GetField(value, f))
                    })
                  }
                )
              }
            }

        case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
          val loweredChild = lower(child)

          // New Plan:
          // 1. Make new rows with all the keys contained in newKey, assume always that newKey is a struct
          //     Maybe also include all the fields in the row that aren't keys?
          // 2. Shuffle to all the new keys
          val newKeyType = newKey.typ.asInstanceOf[TStruct]
          val oldRowType = child.typ.rowType
          val oldRowFieldsNotInNewKey = oldRowType.fieldNames.filter(fieldName => !newKeyType.fieldNames.contains(fieldName))
          val shuffledRowType = newKeyType ++ oldRowType.filter(field => !newKeyType.fieldNames.contains(field.name))._1

          val withNewKeyFields = loweredChild.mapPartition { partition =>
            mapIR(partition) { partitionElement =>
              Let("row",
                partitionElement,
                bindIR(newKey) { newKeyRef =>
                  val getKeyFields = newKeyType.fieldNames.map(fieldName => fieldName -> GetField(newKeyRef, fieldName)).toIndexedSeq
                  val getOtherFields = oldRowFieldsNotInNewKey.map(fieldName => fieldName -> GetField(partitionElement, fieldName)).toIndexedSeq
                  MakeStruct( getKeyFields ++ getOtherFields)
                }
              )
            }
          }

          val sortFields = newKeyType.fieldNames.map(fieldName => SortField(fieldName, Ascending)).toIndexedSeq
          val shuffled = ctx.backend.lowerDistributedSort(ctx, withNewKeyFields, sortFields)

          shuffled.mapPartition { partition =>
            mapIR(StreamGroupByKey(partition, shuffled.partitioner.kType.fieldNames.toIndexedSeq)) { groupRef =>
              // I believe that `expr` is a going to make a struct of aggregator fields.
              StreamAgg(
                groupRef,
                "row",
                bindIRs(
                  ArrayRef(
                    ApplyAggOp(FastSeq(I32(1)),
                      FastSeq(SelectFields(Ref("row", shuffledRowType), newKeyType.fieldNames)),
                      AggSignature(Take(), FastSeq(TInt32), FastSeq(newKeyType))),
                    I32(0)),
                  expr) { case Seq(groupRep, value) =>

                  val keyIRs: IndexedSeq[(String, IR)] = newKeyType.fieldNames.map(keyName => keyName -> GetField(groupRep, keyName))
                  MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                    (f, GetField(value, f))
                  })
                }
              )
            }
          }

        case TableDistinct(child) =>
          val loweredChild = lower(child)

          loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
            .mapPartition { partition =>
              flatMapIR(StreamGroupByKey(partition, child.typ.key)) { groupRef =>
                StreamTake(groupRef, 1)
              }
            }

        case TableFilter(child, cond) =>
          val row = Ref(genUID(), child.typ.rowType)
          val loweredChild = lower(child)
          val env: Env[IR] = Env("row" -> row, "global" -> loweredChild.globals)
          loweredChild.mapPartition(rows => StreamFilter(rows, row.name, Subst(cond, BindingEnv(env))))

        case TableHead(child, targetNumRows) =>
          val loweredChild = lower(child)

          val partitionSizeArrayFunc = genUID()
          val howManyPartsToTry = Ref(genUID(), TInt32)
          val childContexts = Ref(genUID(), TArray(coerce[TStream](loweredChild.contexts.typ).elementType))
          val partitionSizeArray = TailLoop(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry.name -> 4),
            bindIR(loweredChild.mapContexts(_ => StreamTake(ToStream(childContexts), howManyPartsToTry)).mapPartition(rows =>  StreamLen(rows)).collect()) { counts =>
              If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (ArrayLen(childContexts) <= ArrayLen(counts)),
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
            loweredChild.letBindings
              :+ childContexts.name -> ToArray(loweredChild.contexts)
              :+ partitionSizeArrayRef.name -> partitionSizeArray,
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            {
              val contextElementType = coerce[TStream](loweredChild.contexts.typ).elementType
              bindIR(answerTuple) { answerTupleRef =>
                val numParts = GetTupleElement(answerTupleRef, 0)
                val numElementsFromLastPart = GetTupleElement(answerTupleRef, 1)
                val onlyNeededPartitions = StreamTake(ToStream(childContexts), numParts)
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
          val childContexts = Ref(genUID(), TArray(coerce[TStream](loweredChild.contexts.typ).elementType))

          val totalNumPartitions = Ref(genUID(), TInt32)
          val partitionSizeArray =
            TailLoop(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry.name -> 4),
              bindIR(loweredChild.mapContexts(_ => StreamDrop(ToStream(childContexts), maxIR(totalNumPartitions - howManyPartsToTry, 0))).mapPartition(rows => StreamLen(rows)).collect()) { counts =>
                If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
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
              childContexts.name -> ToArray(loweredChild.contexts),
              totalNumPartitions.name -> ArrayLen(childContexts),
              partitionSizeArrayRef.name -> partitionSizeArray
            ),
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            {
              val contextElementType = coerce[TStream](loweredChild.contexts.typ).elementType
              bindIR(answerTuple) { answerTupleRef =>
                val numPartsToDropFromPartitionSizeArray = GetTupleElement(answerTupleRef, 0)
                val numElementsFromFirstPart = GetTupleElement(answerTupleRef, 1)
                val numPartsToDropFromTotal = numPartsToDropFromPartitionSizeArray + (totalNumPartitions - ArrayLen(partitionSizeArrayRef))
                val onlyNeededPartitions = StreamDrop(ToStream(childContexts), numPartsToDropFromTotal)
                val howManyFromEachPart = mapIR(rangeIR(StreamLen(onlyNeededPartitions))) { idxRef =>
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
          } else {
            val sorted = ctx.backend.lowerDistributedSort(ctx, loweredChild, newKey.map(k => SortField(k, Ascending)))
            assert(sorted.partitioner.kType.fieldNames.sameElements(newKey))
            sorted
          }

        case TableLeftJoinRightDistinct(left, right, root) =>
          require(right.typ.keyType isPrefixOf left.typ.keyType)

          val commonKeyLength = right.typ.keyType.size
          val loweredLeft = lower(left)
          val leftKeyToRightKeyMap = left.typ.keyType.fieldNames.zip(right.typ.keyType.fieldNames).toMap
          val newRightPartitioner = loweredLeft.partitioner.coarsen(commonKeyLength)
            .rename(leftKeyToRightKeyMap)
          val loweredRight = lower(right).repartitionNoShuffle(newRightPartitioner)

          val leftCtxTyp = loweredLeft.contexts.typ.asInstanceOf[TStream].elementType
          val rightCtxTyp = loweredRight.contexts.typ.asInstanceOf[TStream].elementType
          val leftCtxRef = Ref(genUID(), leftCtxTyp)
          val rightCtxRef = Ref(genUID(), rightCtxTyp)

          val leftCtxStructField = genUID()
          val rightCtxStructField = genUID()

          new TableStage(
            loweredLeft.letBindings ++ loweredRight.letBindings,
            loweredLeft.broadcastVals ++ loweredRight.broadcastVals,
            loweredLeft.globals,
            loweredLeft.partitioner,
            StreamZip(
              FastIndexedSeq(loweredLeft.contexts, loweredRight.contexts), FastIndexedSeq(leftCtxRef.name, rightCtxRef.name),
              MakeStruct(FastIndexedSeq(leftCtxStructField -> leftCtxRef, rightCtxStructField -> rightCtxRef)), ArrayZipBehavior.AssertSameLength
            )
          ) {
            override def partition(ctxRef: Ref): IR = {
              bindIR(GetField(ctxRef, leftCtxStructField)) { leftCtxFieldRef =>
                bindIR(GetField(ctxRef, rightCtxStructField)) { rightCtxFieldRef =>
                  val leftPart = loweredLeft.partition(leftCtxFieldRef)
                  val rightPart = loweredRight.partition(rightCtxFieldRef)
                  val leftElementRef = Ref(genUID(), left.typ.rowType)
                  val rightElementRef = Ref(genUID(), right.typ.rowType)

                  val (typeOfRootStruct, _) = right.typ.rowType.filterSet(right.typ.key.toSet, false)
                  val rootStruct = SelectFields(rightElementRef, typeOfRootStruct.fieldNames.toIndexedSeq)
                  val joiningOp = InsertFields(leftElementRef, Seq(root -> rootStruct))
                  StreamJoinRightDistinct(leftPart, rightPart, left.typ.key.take(commonKeyLength), right.typ.key, leftElementRef.name, rightElementRef.name, joiningOp, "left")
                }

              }
            }
          }
        case TableOrderBy(child, sortFields) =>
          val loweredChild = lower(child)
          if (TableOrderBy.isAlreadyOrdered(sortFields, loweredChild.partitioner.kType.fieldNames))
            loweredChild
          else
            ctx.backend.lowerDistributedSort(ctx, loweredChild, sortFields)

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
          lower(tableIR).mapPartition(rows => Cast(StreamLen(rows), TInt64)).collect())

      case TableToValueApply(child, ForceCountTable()) =>
        invoke("sum", TInt64,
          lower(child).mapPartition(rows => Cast(StreamLen(mapIR(rows)(row => Consume(row))), TInt64)).collect())

      case TableGetGlobals(child) =>
        val stage = lower(child)
        stage.wrapInBindings(stage.globals)

      case TableCollect(child) =>
        lower(child).collectWithGlobals()

      case TableToValueApply(child, NPartitionsTable()) =>
        val lowered = lower(child)
        StreamLen(lowered.contexts)

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")
    }
  }
}
