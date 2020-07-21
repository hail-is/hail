package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.methods.{ForceCountTable, NPartitionsTable}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.virtual._
import is.hail.types.{RTable, TableType}
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
    body: (Ref) => IR
  ): TableStage = {
    val globalsRef = Ref(genUID(), globals.typ)
    TableStage(
      FastIndexedSeq(globalsRef.name -> globals),
      FastIndexedSeq(globalsRef.name -> globalsRef),
      globalsRef,
      partitioner,
      contexts,
      body)
  }

  def apply(
    letBindings: IndexedSeq[(String, IR)],
    broadcastVals: IndexedSeq[(String, IR)],
    globals: Ref,
    partitioner: RVDPartitioner,
    contexts: IR,
    partition: Ref => IR
  ): TableStage = {
    val ctxType = contexts.typ.asInstanceOf[TStream].elementType
    val ctxRef = Ref(genUID(), ctxType)

    new TableStage(letBindings, broadcastVals, globals, partitioner, contexts, ctxRef.name, partition(ctxRef))
  }
}

// Scope structure:
// * 'letBindings' are evaluated in scope of previous 'letBindings', and are
//   visible in 'broadcastVals' and 'contexts'.
// * 'broadcastVals' are evaluated in scope of 'letBindings', and are visible
//   in 'partitionIR'.
// * 'globals' must be bound in 'letBindings', and rebound in 'broadcastVals',
//   so 'globals' is visible both in later 'letBindings' and in 'partitionIR'.
class TableStage(
  val letBindings: IndexedSeq[(String, IR)],
  val broadcastVals: IndexedSeq[(String, IR)],
  val globals: Ref,
  val partitioner: RVDPartitioner,
  val contexts: IR,
  private val ctxRefName: String,
  private val partitionIR: IR) { self =>

  def ctxType: Type = contexts.typ.asInstanceOf[TStream].elementType
  def rowType: TStruct = partitionIR.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
  def kType: TStruct = partitioner.kType
  def key: IndexedSeq[String] = kType.fieldNames
  def globalType: TStruct = globals.typ.asInstanceOf[TStruct]

  assert(key.forall(f => rowType.hasField(f)))
  assert(kType.fields.forall(f => rowType.field(f.name).typ == f.typ))
  assert(broadcastVals.exists { case (name, value) => name == globals.name && value == globals})

  def copy(
    letBindings: IndexedSeq[(String, IR)] = letBindings,
    broadcastVals: IndexedSeq[(String, IR)] = broadcastVals,
    globals: Ref = globals,
    partitioner: RVDPartitioner = partitioner,
    contexts: IR = contexts,
    ctxRefName: String = ctxRefName,
    partitionIR: IR = partitionIR
  ): TableStage =
    new TableStage(letBindings, broadcastVals, globals, partitioner, contexts, ctxRefName, partitionIR)

  def partition(ctx: IR): IR = {
    require(ctx.typ == ctxType)
    Let(ctxRefName, ctx, partitionIR)
  }

  def numPartitions: Int = partitioner.numPartitions

  private def wrapInBindings(body: IR): IR = letBindings.foldRight[IR](body) {
    case ((name, value), body) => Let(name, value, body)
  }

  def mapPartition(f: IR => IR): TableStage =
    copy(partitionIR = f(partitionIR))

  def zipPartitions(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    val left = this
    val leftCtxTyp = left.ctxType
    val rightCtxTyp = right.ctxType

    val leftCtxRef = Ref(genUID(), leftCtxTyp)
    val rightCtxRef = Ref(genUID(), rightCtxTyp)

    val leftCtxStructField = genUID()
    val rightCtxStructField = genUID()

    val zippedCtxs = StreamZip(
      FastIndexedSeq(left.contexts, right.contexts),
      FastIndexedSeq(leftCtxRef.name, rightCtxRef.name),
      MakeStruct(FastIndexedSeq(leftCtxStructField -> leftCtxRef,
                                rightCtxStructField -> rightCtxRef)),
      ArrayZipBehavior.AssertSameLength)

    val globals = newGlobals(left.globals, right.globals)
    val globalsRef = Ref(genUID(), globals.typ)

    TableStage(
      left.letBindings ++ right.letBindings :+ (globalsRef.name -> globals),
      left.broadcastVals ++ right.broadcastVals :+ (globalsRef.name -> globalsRef),
      globalsRef,
      left.partitioner,
      zippedCtxs,
      (ctxRef: Ref) => {
        bindIR(left.partition(GetField(ctxRef, leftCtxStructField))) { lPart =>
          bindIR(right.partition(GetField(ctxRef, rightCtxStructField))) { rPart =>
            body(lPart, rPart)
          }
        }
      })
  }

  def mapPartitionWithContext(f: (IR, Ref) => IR): TableStage =
    copy(partitionIR = f(partitionIR, Ref(ctxRefName, ctxType)))

  def mapContexts(f: IR => IR)(getOldContext: IR => IR): TableStage = {
    val newContexts = f(contexts)
    TableStage(letBindings, broadcastVals, globals, partitioner, newContexts, ctxRef => bindIR(getOldContext(ctxRef))(partition))
  }

  def mapGlobals(f: IR => IR): TableStage = {
    val newGlobals = f(globals)
    val globalsRef = Ref(genUID(), newGlobals.typ)

    copy(
      letBindings = letBindings :+ globalsRef.name -> newGlobals,
      broadcastVals = broadcastVals :+ globalsRef.name -> globalsRef,
      globals = globalsRef)
  }

  def mapCollect(bindings: Seq[(String, Type)])(f: IR => IR): IR = {
    mapCollectWithGlobals(bindings)(f) { (parts, globals) => parts }
  }

  def mapCollectWithGlobals(bindings: Seq[(String, Type)])(mapF: IR => IR)(body: (IR, IR) => IR): IR =
    mapCollectWithContextsAndGlobals(bindings)((part, ctx) => mapF(part))(body)

  def mapCollectWithContextsAndGlobals(bindings: Seq[(String, Type)])(mapF: (IR, Ref) => IR)(body: (IR, IR) => IR): IR = {
    val allBroadcastVals = broadcastVals ++ bindings.map { case (name, t) => (name, Ref(name, t))}
    val broadcastRefs = MakeStruct(allBroadcastVals)
    val glob = Ref(genUID(), broadcastRefs.typ)

    val cda = CollectDistributedArray(
      contexts, broadcastRefs,
      ctxRefName, glob.name,
      allBroadcastVals.foldLeft(mapF(partitionIR, Ref(ctxRefName, ctxType))) { case (accum, (name, _)) =>
        Let(name, GetField(glob, name), accum)
      })

    wrapInBindings(body(cda, globals))
  }

  def collectWithGlobals(bindings: Seq[(String, Type)]): IR = mapCollectWithGlobals(bindings)(ToArray) { (parts, globals) =>
    MakeStruct(FastSeq(
      "rows" -> ToArray(flatMapIR(ToStream(parts))(ToStream)),
      "global" -> globals))
  }

  def getGlobals(): IR = wrapInBindings(globals)

  def getNumPartitions(): IR = wrapInBindings(StreamLen(contexts))

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage =
    copy(partitioner = newPartitioner)

  def repartitionNoShuffle(newPartitioner: RVDPartitioner): TableStage = {
    require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    require(newPartitioner.kType.isPrefixOf(kType))

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

    TableStage(letBindings, broadcastVals, globals, newPartitioner, newContexts,
      (ctxRef: Ref) => {
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
              SelectFields(Ref(eltUID, body.typ.asInstanceOf[TStream].elementType), newPartitioner.kType.fieldNames))))
      })
  }

  def extendKeyPreservesPartitioning(newKey: IndexedSeq[String]): TableStage = {
    require(newKey startsWith kType.fieldNames)
    require(newKey.forall(rowType.fieldNames.contains))

    val newKeyType = rowType.typeAfterSelectNames(newKey)
    if (RVDPartitioner.isValid(newKeyType, partitioner.rangeBounds)) {
      changePartitionerNoRepartition(partitioner.copy(kType = newKeyType))
    } else {
      val adjustedPartitioner = partitioner.strictify
      repartitionNoShuffle(adjustedPartitioner)
        .changePartitionerNoRepartition(adjustedPartitioner.copy(kType = newKeyType))
    }
  }

  def orderedJoin(
    right: TableStage,
    joinKey: Int,
    joinType: String,
    globalJoiner: (IR, IR) => IR,
    joiner: (Ref, Ref) => IR
  ): TableStage = {
    assert(this.kType.truncate(joinKey).isIsomorphicTo(right.kType.truncate(joinKey)))

    val newPartitioner = {
      def leftPart: RVDPartitioner = this.partitioner.strictify
      def rightPart: RVDPartitioner = right.partitioner.coarsen(joinKey).extendKey(this.kType)
      (joinType: @unchecked) match {
        case "left" => leftPart
        case "right" => rightPart
        case "inner" => leftPart.intersect(rightPart)
        case "outer" => RVDPartitioner.generate(
          this.kType.fieldNames.take(joinKey),
          this.kType,
          leftPart.rangeBounds ++ rightPart.rangeBounds)
      }
    }
    val repartitionedLeft: TableStage = this.repartitionNoShuffle(newPartitioner)

    val partitionJoiner: (IR, IR) => IR = (lPart, rPart) => {
      val lEltType = lPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
      val rEltType = rPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]

      val lKey = this.kType.fieldNames.take(joinKey)
      val rKey = right.kType.fieldNames.take(joinKey)

      val lEltRef = Ref(genUID(), lEltType)
      val rEltRef = Ref(genUID(), rEltType)

      StreamJoin(lPart, rPart, lKey, rKey, lEltRef.name, rEltRef.name, joiner(lEltRef, rEltRef), joinType)
    }

    val newKey = kType.fieldNames ++ right.kType.fieldNames.drop(joinKey)

    val leftKeyToRightKeyMap = kType.fieldNames.zip(right.kType.fieldNames).toMap
    val newRightPartitioner = newPartitioner.coarsen(joinKey).strictify.rename(leftKeyToRightKeyMap)
    val repartitionedRight = right.repartitionNoShuffle(newRightPartitioner)

    repartitionedLeft.zipPartitions(repartitionedRight, globalJoiner, partitionJoiner)
      .extendKeyPreservesPartitioning(newKey)
  }

  // 'joiner' must take all output key values from
  // left stream, and be monotonic on left stream (it can drop or duplicate
  // elements of left iterator, or insert new elements in order, but cannot
  // rearrange them), and output values must conform to 'newTyp'. The
  // partitioner of the result will be the left partitioner. Each partition will
  // be computed by 'joiner', with corresponding partition of 'this' as first
  // iterator, and with all rows of 'that' whose 'joinKey' might match something
  // in partition as the second iterator.
  def alignAndZipPartitions(
    right: TableStage,
    joinKey: Int,
    globalJoiner: (IR, IR) => IR,
    joiner: (IR, IR) => IR
  ): TableStage = {
    require(joinKey <= kType.size)
    require(joinKey <= right.kType.size)

    val leftKeyToRightKeyMap = kType.fieldNames.zip(right.kType.fieldNames).toMap
    val newRightPartitioner = partitioner.coarsen(joinKey).strictify.rename(leftKeyToRightKeyMap)
    val repartitionedRight = right.repartitionNoShuffle(newRightPartitioner)
    zipPartitions(repartitionedRight, globalJoiner, joiner)
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, r: RequirednessAnalysis, relationalLetsAbove: Seq[(String, Type)]): IR = {
    def lowerIR(ir: IR) = LowerToCDA.lower(ir, typesToLower, ctx, r, relationalLetsAbove)

    def lower(tir: TableIR): TableStage = {
      if (typesToLower == DArrayLowering.BMOnly)
        throw new LowererUnsupportedOperation("found TableIR in lowering; lowering only BlockMatrixIRs.")

      val typ: TableType = tir.typ

      val lowered = tir match {
        case TableRead(typ, dropRows, reader) =>
          if (dropRows) {
            val globals = reader.lowerGlobals(ctx, typ.globalType)

            TableStage(
              globals,
              RVDPartitioner.empty(typ.keyType),
              MakeStream(FastIndexedSeq(), TStream(TStruct.empty)),
              (_: Ref) => MakeStream(FastIndexedSeq(), TStream(typ.rowType)))
          } else
            reader.lower(ctx, typ)

        case TableParallelize(rowsAndGlobal, nPartitions) =>
          val nPartitionsAdj = nPartitions.getOrElse(16)

          val loweredRowsAndGlobal = lowerIR(rowsAndGlobal)
          val loweredRowsAndGlobalRef = Ref(genUID(), loweredRowsAndGlobal.typ)

          val context = bindIR(ArrayLen(GetField(loweredRowsAndGlobalRef, "rows"))) { numRowsRef =>
            bindIR(If(numRowsRef < nPartitionsAdj, numRowsRef, nPartitionsAdj)) { numNonEmptyPartitionsRef =>
              bindIR(numRowsRef floorDiv numNonEmptyPartitionsRef) { qRef =>
                bindIR(numRowsRef - qRef * numNonEmptyPartitionsRef) { remainderRef =>
                  mapIR(rangeIR(0, nPartitionsAdj)) { partIdx =>
                    val length =
                      (numRowsRef - partIdx + nPartitionsAdj - 1) floorDiv nPartitionsAdj

                    val start =
                      If(numNonEmptyPartitionsRef >= partIdx,
                         If(remainderRef > 0,
                            If(remainderRef < partIdx,
                               qRef * partIdx + remainderRef,
                               (qRef + 1) * partIdx),
                            qRef * partIdx),
                         0)

                    bindIR(start) { startRef =>
                      ToArray(mapIR(rangeIR(startRef, startRef + length)) { elt =>
                        ArrayRef(GetField(loweredRowsAndGlobalRef, "rows"), elt)
                      })
                    }
                  }
                }
              }
            }
          }

          val globalsRef = Ref(genUID(), typ.globalType)
          TableStage(
            FastIndexedSeq(loweredRowsAndGlobalRef.name -> loweredRowsAndGlobal,
                           globalsRef.name -> GetField(loweredRowsAndGlobalRef, "global")),
            FastIndexedSeq(globalsRef.name -> globalsRef),
            globalsRef,
            RVDPartitioner.unkeyed(nPartitionsAdj),
            context,
            ctxRef => ToStream(ctxRef))

        case TableRange(n, nPartitions) =>
          val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
          val partCounts = partition(n, nPartitionsAdj)
          val partStarts = partCounts.scanLeft(0)(_ + _)

          val contextType = TStruct("start" -> TInt32, "end" -> TInt32)

          val ranges = Array.tabulate(nPartitionsAdj) { i =>
            partStarts(i) -> partStarts(i + 1)
          }

          TableStage(
            MakeStruct(FastSeq()),
            new RVDPartitioner(Array("idx"), tir.typ.rowType,
              ranges.map { case (start, end) =>
                Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
              }),
            MakeStream(
              ranges.map { case (start, end) =>
                MakeStruct(FastIndexedSeq("start" -> start, "end" -> end))
              },
              TStream(contextType)),
            (ctxRef: Ref) => mapIR(rangeIR(GetField(ctxRef, "start"), GetField(ctxRef, "end"))) { i =>
              MakeStruct(FastSeq("idx" -> i))
            })

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

        // TODO: This ignores nPartitions and bufferSize
        case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
          val loweredChild = lower(child)
          val newKeyType = newKey.typ.asInstanceOf[TStruct]
          val oldRowType = child.typ.rowType
          val filteredOldRowType = oldRowType.filter(field => !newKeyType.fieldNames.contains(field.name))._1
          val shuffledRowType = newKeyType ++ filteredOldRowType

          val withNewKeyFields = loweredChild.mapPartition { partition =>
            mapIR(partition) { partitionElement =>
              Let("row",
                partitionElement,
                bindIR(newKey) { newKeyRef =>
                  val getKeyFields = newKeyType.fieldNames.map(fieldName => fieldName -> GetField(newKeyRef, fieldName)).toIndexedSeq
                  InsertFields(partitionElement, getKeyFields)
                }
              )
            }
          }

          val sortFields = newKeyType.fieldNames.map(fieldName => SortField(fieldName, Ascending)).toIndexedSeq
          val shuffled = ctx.backend.lowerDistributedSort(ctx, withNewKeyFields, sortFields, relationalLetsAbove)
          val repartitioned = shuffled.repartitionNoShuffle(shuffled.partitioner.strictify)

          repartitioned.mapPartition { partition =>
            mapIR(StreamGroupByKey(partition, newKeyType.fieldNames.toIndexedSeq)) { groupRef =>
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
          val loweredChild = lower(child)
          loweredChild.mapPartition { rows =>
            Let("global", loweredChild.globals,
                StreamFilter(rows, "row", cond))
          }

        case TableFilterIntervals(child, intervals, keep) =>
          val loweredChild = lower(child)
          val part = loweredChild.partitioner
          val kt = child.typ.keyType
          val ord = PartitionBoundOrdering(kt)
          val iord = ord.intervalEndpointOrdering
          val nPartitions = part.numPartitions


          val filterPartitioner = new RVDPartitioner(kt, Interval.union(intervals.toArray, ord.intervalEndpointOrdering))
          val boundsType = TArray(RVDPartitioner.intervalIRRepresentation(kt))
          val filterIntervalsRef = Ref(genUID(), boundsType)
          val filterIntervals: IndexedSeq[Row] = filterPartitioner.rangeBounds.map { i =>
            RVDPartitioner.intervalToIRRepresentation(i, kt.size)
          }

          val (newRangeBounds, includedIndices, startAndEndInterval, f) = if (keep) {
            val (newRangeBounds, includedIndices, startAndEndInterval) = part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
              if (filterPartitioner.overlaps(interval)) {
                Some((interval, i, (filterPartitioner.lowerBoundInterval(interval).min(nPartitions), filterPartitioner.upperBoundInterval(interval).min(nPartitions))))
              } else None
            }.unzip3

            val f: (IR, IR) => IR = {
              case (partitionIntervals, key) =>
                // FIXME: don't do a linear scan over intervals. Fine at first to get the plumbing right
                foldIR(ToStream(partitionIntervals), False()) { case (acc, elt) =>
                  acc || invoke("partitionIntervalContains",
                    TBoolean,
                    elt,
                    key)
                }
            }
            (newRangeBounds, includedIndices, startAndEndInterval, f)
          } else {
            // keep = False
            val (newRangeBounds, includedIndices, startAndEndInterval) = part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
              val lowerBound = filterPartitioner.lowerBoundInterval(interval)
              val upperBound = filterPartitioner.upperBoundInterval(interval)
              if ((lowerBound until upperBound).map(filterPartitioner.rangeBounds).exists { filterInterval =>
                iord.compareNonnull(filterInterval.left, interval.left) <= 0 && iord.compareNonnull(filterInterval.right, interval.right) >= 0
              })
                None
              else Some((interval, i, (lowerBound.min(nPartitions), upperBound.min(nPartitions))))
            }.unzip3

            val f: (IR, IR) => IR = {
              case (partitionIntervals, key) =>
                // FIXME: don't do a linear scan over intervals. Fine at first to get the plumbing right
                foldIR(ToStream(partitionIntervals), True()) { case (acc, elt) =>
                  acc && !invoke("partitionIntervalContains",
                    TBoolean,
                    elt,
                    key)
                }
            }
            (newRangeBounds, includedIndices, startAndEndInterval, f)
          }

          val newPart = new RVDPartitioner(kt, newRangeBounds)

          TableStage(
            letBindings = loweredChild.letBindings,
            broadcastVals = loweredChild.broadcastVals ++ FastIndexedSeq((filterIntervalsRef.name, Literal(boundsType, filterIntervals))),
            loweredChild.globals,
            newPart,
            contexts = bindIRs(
              ToArray(loweredChild.contexts),
              Literal(TArray(TTuple(TInt32, TInt32)), startAndEndInterval.map { case (start, end) => Row(start, end) }.toFastIndexedSeq)
            ) { case Seq(prevContexts, bounds) =>
              zip2(ToStream(Literal(TArray(TInt32), includedIndices.toFastIndexedSeq)), ToStream(bounds), ArrayZipBehavior.AssumeSameLength) { (idx, bound) =>
                MakeStruct(FastSeq(("prevContext", ArrayRef(prevContexts, idx)), ("bounds", bound)))
              }
            },
            { (part: Ref) =>
              val oldPart = loweredChild.partition(GetField(part, "prevContext"))
              bindIR(GetField(part, "bounds")) { bounds =>
                bindIRs(GetTupleElement(bounds, 0), GetTupleElement(bounds, 1)) { case Seq(startIntervalIdx, endIntervalIdx) =>
                  bindIR(ToArray(mapIR(rangeIR(startIntervalIdx, endIntervalIdx)) { i => ArrayRef(filterIntervalsRef, i) })) { partitionIntervals =>
                    filterIR(oldPart) { row =>
                      bindIR(SelectFields(row, child.typ.key)) { key =>
                        f(partitionIntervals, key)
                      }
                    }
                  }
                }
              }
            }
          )

        case TableHead(child, targetNumRows) =>
          val loweredChild = lower(child)

          def partitionSizeArray(childContexts: Ref): IR = {
            val partitionSizeArrayFunc = genUID()
            val howManyPartsToTry = Ref(genUID(), TInt32)

            TailLoop(
              partitionSizeArrayFunc,
              FastIndexedSeq(howManyPartsToTry.name -> 4),
              bindIR(loweredChild.mapContexts(_ => StreamTake(ToStream(childContexts), howManyPartsToTry)){ ctx: IR => ctx }
                                 .mapCollect(relationalLetsAbove)(StreamLen)) { counts =>
                If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (ArrayLen(childContexts) <= ArrayLen(counts)),
                  counts,
                  Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry * 4), TArray(TInt32)))
              })
          }

          def answerTuple(partitionSizeArrayRef: Ref): IR =
            bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
              val howManyPartsToKeep = genUID()
              val i = Ref(genUID(), TInt32)
              val numLeft = Ref(genUID(), TInt64)
              def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple(FastIndexedSeq((0, howManyParts), (1, howManyFromLast)))

              If(numPartitions ceq 0,
                makeAnswer(0, 0L),
                TailLoop(howManyPartsToKeep, FastIndexedSeq(i.name -> 0, numLeft.name -> targetNumRows),
                  If((i ceq numPartitions - 1) || ((numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)) <= 0L),
                    makeAnswer(i + 1, numLeft),
                    Recur(howManyPartsToKeep,
                          FastIndexedSeq(
                            i + 1,
                            numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)),
                          TTuple(TInt32, TInt64)))))
            }

          val newCtxs = bindIR(ToArray(loweredChild.contexts)) { childContexts =>
            bindIR(partitionSizeArray(childContexts)) { partitionSizeArrayRef =>
              bindIR(answerTuple(partitionSizeArrayRef)) { answerTupleRef =>
                val numParts = GetTupleElement(answerTupleRef, 0)
                val numElementsFromLastPart = GetTupleElement(answerTupleRef, 1)
                val onlyNeededPartitions = StreamTake(ToStream(childContexts), numParts)
                val howManyFromEachPart = mapIR(rangeIR(numParts)) { idxRef =>
                  If(idxRef ceq (numParts - 1),
                     Cast(numElementsFromLastPart, TInt32),
                     ArrayRef(partitionSizeArrayRef, idxRef))
                }

                StreamZip(
                  FastIndexedSeq(onlyNeededPartitions, howManyFromEachPart),
                  FastIndexedSeq("part", "howMany"),
                  MakeStruct(FastSeq("numberToTake" -> Ref("howMany", TInt32),
                                     "old" -> Ref("part", loweredChild.ctxType))),
                  ArrayZipBehavior.AssumeSameLength)
              }
            }
          }

          TableStage(
            loweredChild.letBindings,
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            newCtxs,
            (ctxRef: Ref) => StreamTake(
              loweredChild.partition(GetField(ctxRef, "old")),
              GetField(ctxRef, "numberToTake")))

        case TableTail(child, targetNumRows) =>
          val loweredChild = lower(child)

          def partitionSizeArray(childContexts: Ref, totalNumPartitions: Ref): IR = {
            val partitionSizeArrayFunc = genUID()
            val howManyPartsToTry = Ref(genUID(), TInt32)

            TailLoop(
              partitionSizeArrayFunc,
              FastIndexedSeq(howManyPartsToTry.name -> 4),
              bindIR(
                loweredChild
                  .mapContexts(_ => StreamDrop(ToStream(childContexts), maxIR(totalNumPartitions - howManyPartsToTry, 0))){ ctx: IR => ctx }
                  .mapCollect(relationalLetsAbove)(StreamLen)
              ) { counts =>
                If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                  counts,
                  Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTry * 4), TArray(TInt32)))
              })
          }

          // First element is how many partitions to drop from partitionSizeArrayRef, second is how many to keep from first kept element.
          def answerTuple(partitionSizeArrayRef: Ref): IR = {
            bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
              val howManyPartsToDrop = genUID()
              val i = Ref(genUID(), TInt32)
              val numLeft = Ref(genUID(), TInt64)
              def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple(FastIndexedSeq((0, howManyParts), (1, howManyFromLast)))

              If(numPartitions ceq 0,
                makeAnswer(0, 0L),
                TailLoop(
                  howManyPartsToDrop,
                  FastIndexedSeq(i.name -> numPartitions, numLeft.name -> targetNumRows),
                  If((i ceq 1) || ((numLeft - Cast(ArrayRef(partitionSizeArrayRef, i - 1), TInt64)) <= 0L),
                    makeAnswer(i - 1, numLeft),
                    Recur(
                      howManyPartsToDrop,
                      FastIndexedSeq(
                        i - 1,
                        numLeft - Cast(ArrayRef(partitionSizeArrayRef, i - 1), TInt64)),
                      TTuple(TInt32, TInt64)))))
            }
          }

          val newCtxs = bindIR(ToArray(loweredChild.contexts)) { childContexts =>
            bindIR(ArrayLen(childContexts)) { totalNumPartitions =>
              bindIR(partitionSizeArray(childContexts, totalNumPartitions)) { partitionSizeArrayRef =>
                bindIR(answerTuple(partitionSizeArrayRef)) { answerTupleRef =>
                  val numPartsToDropFromPartitionSizeArray = GetTupleElement(answerTupleRef, 0)
                  val numElementsFromFirstPart = GetTupleElement(answerTupleRef, 1)
                  val numPartsToDropFromTotal = numPartsToDropFromPartitionSizeArray + (totalNumPartitions - ArrayLen(partitionSizeArrayRef))
                  val onlyNeededPartitions = StreamDrop(ToStream(childContexts), numPartsToDropFromTotal)
                  val howManyFromEachPart = mapIR(rangeIR(StreamLen(onlyNeededPartitions))) { idxRef =>
                    If(idxRef ceq 0,
                       Cast(numElementsFromFirstPart, TInt32),
                       ArrayRef(partitionSizeArrayRef, idxRef))
                  }
                  StreamZip(
                    FastIndexedSeq(onlyNeededPartitions,
                                   howManyFromEachPart,
                                   StreamDrop(ToStream(partitionSizeArrayRef), numPartsToDropFromPartitionSizeArray)),
                    FastIndexedSeq("part", "howMany", "partLength"),
                    MakeStruct(FastIndexedSeq(
                      "numberToDrop" -> maxIR(0, Ref("partLength", TInt32) - Ref("howMany", TInt32)),
                      "old" -> Ref("part", loweredChild.ctxType))),
                    ArrayZipBehavior.AssertSameLength)
                }
              }
            }
          }

          TableStage(
            loweredChild.letBindings,
            loweredChild.broadcastVals,
            loweredChild.globals,
            loweredChild.partitioner,
            newCtxs,
            (ctxRef: Ref) => bindIR(GetField(ctxRef, "old")) { oldRef =>
              StreamDrop(loweredChild.partition(oldRef), GetField(ctxRef, "numberToDrop"))
            })

        case TableMapRows(child, newRow) =>
          if (ContainsScan(newRow))
            throw new LowererUnsupportedOperation(s"scans are not supported: \n${ Pretty(newRow) }")
          val loweredChild = lower(child)

          loweredChild.mapPartition { rows =>
            Let("global", loweredChild.globals,
              mapIR(rows)(row => Let("row", row, newRow)))
          }

        case TableGroupWithinPartitions(child, groupedStructName, n) =>
          val loweredChild = lower(child)
          val keyFields = FastIndexedSeq(child.typ.keyType.fieldNames: _*)
          loweredChild.mapPartition { part =>
            mapIR(StreamGrouped(part, n)) { group =>
              bindIR(ToArray(group)) { groupRef =>
                InsertFields(
                  SelectFields(ArrayRef(groupRef, 0), keyFields),
                  FastSeq(groupedStructName -> groupRef))
              }
            }
          }

        case TableKeyBy(child, newKey, isSorted: Boolean) =>
          val loweredChild = lower(child)

          val nPreservedFields = loweredChild.kType.fieldNames
            .zip(newKey)
            .takeWhile { case (l, r) => l == r }
            .length
          require(!isSorted || nPreservedFields > 0 || newKey.isEmpty)

          if (nPreservedFields == newKey.length || isSorted)
            // TODO: should this add a runtime check that keys are within the
            // partition bounds, like in RVD?
            loweredChild.changePartitionerNoRepartition(loweredChild.partitioner.coarsen(nPreservedFields))
              .extendKeyPreservesPartitioning(newKey)
          else {
            val sorted = ctx.backend.lowerDistributedSort(
              ctx, loweredChild, newKey.map(k => SortField(k, Ascending)), relationalLetsAbove)
            assert(sorted.kType.fieldNames.sameElements(newKey))
            sorted
          }

        case TableLeftJoinRightDistinct(left, right, root) =>
          val commonKeyLength = right.typ.keyType.size
          val loweredLeft = lower(left)
          val leftKeyToRightKeyMap = left.typ.keyType.fieldNames.zip(right.typ.keyType.fieldNames).toMap
          val newRightPartitioner = loweredLeft.partitioner.coarsen(commonKeyLength)
            .rename(leftKeyToRightKeyMap)
          val loweredRight = lower(right).repartitionNoShuffle(newRightPartitioner)

          loweredLeft.zipPartitions(
            loweredRight,
            (lGlobals, _) => lGlobals,
            (leftPart, rightPart) => {
              val leftElementRef = Ref(genUID(), left.typ.rowType)
              val rightElementRef = Ref(genUID(), right.typ.rowType)

              val (typeOfRootStruct, _) = right.typ.rowType.filterSet(right.typ.key.toSet, false)
              val rootStruct = SelectFields(rightElementRef, typeOfRootStruct.fieldNames.toIndexedSeq)
              val joiningOp = InsertFields(leftElementRef, Seq(root -> rootStruct))
              StreamJoinRightDistinct(
                leftPart, rightPart,
                left.typ.key.take(commonKeyLength), right.typ.key,
                leftElementRef.name, rightElementRef.name,
                joiningOp, "left")
            })

        case TableJoin(left, right, joinType, joinKey) =>
          val loweredLeft = lower(left)
          val loweredRight = lower(right)

          val lKeyFields = left.typ.key.take(joinKey)
          val lValueFields = left.typ.rowType.fieldNames.filter(f => !lKeyFields.contains(f))
          val rKeyFields = right.typ.key.take(joinKey)
          val rValueFields = right.typ.rowType.fieldNames.filter(f => !rKeyFields.contains(f))
          val lReq = r.lookup(left).asInstanceOf[RTable]
          val rReq = r.lookup(right).asInstanceOf[RTable]

          loweredLeft.orderedJoin(
            loweredRight, joinKey, joinType,
            (lGlobals, rGlobals) => {
              val rGlobalType = rGlobals.typ.asInstanceOf[TStruct]
              val rGlobalRef = Ref(genUID(), rGlobalType)
              Let(rGlobalRef.name, rGlobals,
                  InsertFields(lGlobals, rGlobalType.fieldNames.map(f => f -> GetField(rGlobalRef, f))))
            },
            (lEltRef, rEltRef) => {
            MakeStruct(
              (lKeyFields, rKeyFields).zipped.map { (lKey, rKey) =>
                if (joinType == "outer" && lReq.field(lKey).required && rReq.field(rKey).required)
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey), Die("TableJoin expected non-missing key", left.typ.rowType.fieldType(lKey))))
                else
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey)))
              }
                ++ lValueFields.map(f => f -> GetField(lEltRef, f))
                ++ rValueFields.map(f => f -> GetField(rEltRef, f)))
          })

        case x@TableUnion(children) =>
          val lowered = children.map(lower)
          val keyType = x.typ.keyType
          val newPartitioner = RVDPartitioner.generate(keyType, lowered.flatMap(_.partitioner.rangeBounds))
          val repartitioned = lowered.map(_.repartitionNoShuffle(newPartitioner))

          TableStage(
            repartitioned.flatMap(_.letBindings),
            repartitioned.flatMap(_.broadcastVals),
            repartitioned.head.globals,
            newPartitioner,
            zipIR(repartitioned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
              MakeTuple.ordered(ctxRefs)
            },
            ctxRef =>
              StreamMultiMerge(repartitioned.indices.map(i => repartitioned(i).partition(GetTupleElement(ctxRef, i))), keyType.fieldNames)
          )

        case x@TableMultiWayZipJoin(children, fieldName, globalName) =>
          val lowered = children.map(lower)
          val keyType = x.typ.keyType
          val newPartitioner = RVDPartitioner.generate(keyType, lowered.flatMap(_.partitioner.rangeBounds))
          val repartitioned = lowered.map(_.repartitionNoShuffle(newPartitioner))
          val newGlobals = MakeStruct(FastSeq(
            globalName -> MakeArray(lowered.map(_.globals), TArray(lowered.head.globalType))))
          val globalsRef = Ref(genUID(), newGlobals.typ)

          val keyRef = Ref(genUID(), keyType)
          val valsRef = Ref(genUID(), TArray(children.head.typ.rowType))
          val projectedVals = ToArray(mapIR(ToStream(valsRef)) { elt =>
            SelectFields(elt, children.head.typ.valueType.fieldNames)
          })

          TableStage(
            repartitioned.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals,
            repartitioned.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
            globalsRef,
            newPartitioner,
            zipIR(repartitioned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
              MakeTuple.ordered(ctxRefs)
            },
            ctxRef =>
              StreamZipJoin(
                repartitioned.indices.map(i => repartitioned(i).partition(GetTupleElement(ctxRef, i))),
                keyType.fieldNames,
                keyRef.name,
                valsRef.name,
                InsertFields(keyRef, FastSeq(fieldName -> projectedVals)))
            )

        case TableOrderBy(child, sortFields) =>
          val loweredChild = lower(child)
          if (TableOrderBy.isAlreadyOrdered(sortFields, loweredChild.partitioner.kType.fieldNames))
            loweredChild.changePartitionerNoRepartition(RVDPartitioner.unkeyed(loweredChild.partitioner.numPartitions))
          else
            ctx.backend.lowerDistributedSort(ctx, loweredChild, sortFields, relationalLetsAbove)

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
                mapIR(ToStream(refs.last)) { elt =>
                  path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
                    InsertFields(ref, FastSeq(p -> inserted))
                  }
                }) { case ((ref, root), accum) =>  Let(ref.name, root, accum) }
            }
          }

        case TableRename(child, rowMap, globalMap) =>
          val loweredChild = lower(child)
          val newGlobals =
            CastRename(loweredChild.globals, loweredChild.globals.typ.asInstanceOf[TStruct].rename(globalMap))
          val newGlobalsRef = Ref(genUID(), newGlobals.typ)

          TableStage(
            loweredChild.letBindings :+ newGlobalsRef.name -> newGlobals,
            loweredChild.broadcastVals :+ newGlobalsRef.name -> newGlobalsRef,
            newGlobalsRef,
            loweredChild.partitioner.copy(kType = loweredChild.kType.rename(rowMap)),
            loweredChild.contexts,
            (ctxRef: Ref) => mapIR(loweredChild.partition(ctxRef)) { row =>
              CastRename(row, row.typ.asInstanceOf[TStruct].rename(rowMap))
            })

        case node =>
          throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
      }

      assert(tir.typ.globalType == lowered.globalType, s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}")
      assert(tir.typ.rowType == lowered.rowType, s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}")
      assert(lowered.key startsWith tir.typ.keyType.fieldNames, s"\n  ir key: ${tir.typ.keyType.fieldNames}\n  lowered key: ${lowered.key}")

      lowered
    }

    val lowered = ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        invoke("sum", TInt64,
          stage.mapCollect(relationalLetsAbove)(rows => Cast(StreamLen(rows), TInt64)))

      case TableToValueApply(child, ForceCountTable()) =>
        val stage = lower(child)
        invoke("sum", TInt64,
          stage.mapCollect(relationalLetsAbove)(rows => Cast(StreamLen(mapIR(rows)(row => Consume(row))), TInt64))          )

      case TableGetGlobals(child) =>
        lower(child).getGlobals()

      case TableCollect(child) =>
        lower(child).collectWithGlobals(relationalLetsAbove)

      case TableAggregate(child, query) =>
        val resultUID = genUID()
        val aggs = agg.Extract(query, resultUID, r, false)
        val lc = lower(child)

        val initState =  RunAgg(
          aggs.init,
          MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
          aggs.states
        )
        val initStateRef = Ref(genUID(), initState.typ)
        val lcWithInitBinding = lc.copy(
          letBindings = lc.letBindings ++ FastIndexedSeq((initStateRef.name, initState)),
          broadcastVals = lc.broadcastVals ++ FastIndexedSeq((initStateRef.name, initStateRef)))

        val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
          InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state )})

        lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove)({ part: IR =>
          Let("global", lc.globals,
            RunAgg(
              Begin(FastIndexedSeq(
                initFromSerializedStates,
                StreamFor(part,
                  "row",
                  aggs.seqPerElt
                )
              )),
              MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
              aggs.states
            ))
        }) { case (collected, globals) =>
          Let("global",
            globals,
            RunAgg(
              Begin(FastIndexedSeq(
                initFromSerializedStates,
                forIR(ToStream(collected)) { state =>
                  Begin(aggs.aggs.zipWithIndex.map { case (sig, i) => CombOpValue(i, GetTupleElement(state, i), sig) })
                }
              )),
              Let(
                resultUID,
                ResultOp(0, aggs.aggs),
                aggs.postAggIR),
              aggs.states
            ))
        }

      case TableToValueApply(child, NPartitionsTable()) =>
        lower(child).getNumPartitions()

      case TableWrite(child, writer) =>
        writer.lower(ctx, lower(child), child, coerce[RTable](r.lookup(child)), relationalLetsAbove)

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")
    }
    lowered
  }
}
