package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.methods.{ForceCountTable, NPartitionsTable}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual._
import is.hail.types.{RField, RStruct, RTable, TableType}
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

case class ShuffledStage(child: TableStage)

case class Binding(name: String, value: IR)

object TableStage {
  def apply(
    globals: IR,
    partitioner: RVDPartitioner,
    dependency: TableStageDependency,
    contexts: IR,
    body: (Ref) => IR
  ): TableStage = {
    val globalsRef = Ref(genUID(), globals.typ)
    TableStage(
      FastIndexedSeq(globalsRef.name -> globals),
      FastIndexedSeq(globalsRef.name -> globalsRef),
      globalsRef,
      partitioner,
      dependency,
      contexts,
      body)
  }

  def apply(
    letBindings: IndexedSeq[(String, IR)],
    broadcastVals: IndexedSeq[(String, IR)],
    globals: Ref,
    partitioner: RVDPartitioner,
    dependency: TableStageDependency,
    contexts: IR,
    partition: Ref => IR
  ): TableStage = {
    val ctxType = contexts.typ.asInstanceOf[TStream].elementType
    val ctxRef = Ref(genUID(), ctxType)

    new TableStage(letBindings, broadcastVals, globals, partitioner, dependency, contexts, ctxRef.name, partition(ctxRef))
  }
  def wrapInBindings(body: IR, letBindings: IndexedSeq[(String, IR)]): IR = letBindings.foldRight[IR](body) {
    case ((name, value), body) => Let(name, value, body)
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
  val dependency: TableStageDependency,
  val contexts: IR,
  private val ctxRefName: String,
  private val partitionIR: IR) {
  self =>

  // useful for debugging, but should be disabled in production code due to N^2 complexity
  // typecheckPartition()

  def typecheckPartition(): Unit = {
    TypeCheck(partitionIR,
      BindingEnv(Env[Type](((letBindings ++ broadcastVals).map { case (s, x) => (s, x.typ) })
        ++ FastIndexedSeq[(String, Type)]((ctxRefName, contexts.typ.asInstanceOf[TStream].elementType)): _*)))

  }

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
    dependency: TableStageDependency = dependency,
    contexts: IR = contexts,
    ctxRefName: String = ctxRefName,
    partitionIR: IR = partitionIR
  ): TableStage =
    new TableStage(letBindings, broadcastVals, globals, partitioner, dependency, contexts, ctxRefName, partitionIR)

  def partition(ctx: IR): IR = {
    require(ctx.typ == ctxType)
    Let(ctxRefName, ctx, partitionIR)
  }

  def numPartitions: Int = partitioner.numPartitions

  def mapPartition(newKey: Option[IndexedSeq[String]])(f: IR => IR): TableStage = {
    val part = newKey match {
      case Some(k) =>
        if (!partitioner.kType.fieldNames.startsWith(k))
          throw new RuntimeException(s"cannot map partitions to new key!" +
            s"\n  prev key: ${ partitioner.kType.fieldNames.toSeq }" +
            s"\n  new key:  ${ k }")
        partitioner.coarsen(k.length)
      case None => partitioner
    }
    copy(partitionIR = f(partitionIR), partitioner = part)
  }

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
      left.dependency.union(right.dependency),
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
    TableStage(letBindings, broadcastVals, globals, partitioner, dependency, newContexts, ctxRef => bindIR(getOldContext(ctxRef))(partition))
  }

  def mapGlobals(f: IR => IR): TableStage = {
    val newGlobals = f(globals)
    val globalsRef = Ref(genUID(), newGlobals.typ)

    copy(
      letBindings = letBindings :+ globalsRef.name -> newGlobals,
      broadcastVals = broadcastVals :+ globalsRef.name -> globalsRef,
      globals = globalsRef)
  }

  def mapCollect(relationalBindings: Map[String, IR])(f: IR => IR): IR = {
    mapCollectWithGlobals(relationalBindings)(f) { (parts, globals) => parts }
  }

  def mapCollectWithGlobals(relationalBindings: Map[String, IR])(mapF: IR => IR)(body: (IR, IR) => IR): IR =
    mapCollectWithContextsAndGlobals(relationalBindings)((part, ctx) => mapF(part))(body)

  def mapCollectWithContextsAndGlobals(relationalBindings: Map[String, IR])(mapF: (IR, Ref) => IR)(body: (IR, IR) => IR): IR = {
    val broadcastRefs = MakeStruct(broadcastVals)
    val glob = Ref(genUID(), broadcastRefs.typ)

    val cda = CollectDistributedArray(
      contexts, broadcastRefs,
      ctxRefName, glob.name,
      broadcastVals.foldLeft(mapF(partitionIR, Ref(ctxRefName, ctxType))) { case (accum, (name, _)) =>
        Let(name, GetField(glob, name), accum)
      }, Some(dependency))

    LowerToCDA.substLets(TableStage.wrapInBindings(body(cda, globals), letBindings), relationalBindings)
  }

  def collectWithGlobals(relationalBindings: Map[String, IR]): IR =
    mapCollectWithGlobals(relationalBindings)(ToArray) { (parts, globals) =>
      MakeStruct(FastSeq(
        "rows" -> ToArray(flatMapIR(ToStream(parts))(ToStream(_))),
        "global" -> globals))
    }

  def getGlobals(): IR = TableStage.wrapInBindings(globals, letBindings)

  def getNumPartitions(): IR = TableStage.wrapInBindings(StreamLen(contexts), letBindings)

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage =
    copy(partitioner = newPartitioner)

  def strictify(): TableStage = {
    if (partitioner.satisfiesAllowedOverlap(kType.size - 1))
      this
    else
      repartitionNoShuffle(partitioner.strictify)
  }

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

    val prevContextUIDPartition = genUID()
    val newStage = TableStage(letBindings, broadcastVals, globals, newPartitioner, dependency, newContexts,
      (ctxRef: Ref) => {
        val body = self.partition(Ref(prevContextUIDPartition, self.contexts.typ.asInstanceOf[TStream].elementType))
        bindIR(GetField(ctxRef, "partitionBound")) { interval =>
          takeWhile(
            dropWhile(
              StreamFlatMap(
                ToStream(GetField(ctxRef, "oldContexts"), true),
                prevContextUIDPartition,
                body)) { elt =>
              !invoke("partitionIntervalEndpointGreaterThan", TBoolean,
                GetField(interval, "left"),
                SelectFields(elt, newPartitioner.kType.fieldNames),
                GetField(interval, "includesLeft"))

            }) { elt =>
            invoke("partitionIntervalEndpointLessThan", TBoolean,
              GetField(interval, "right"),
              SelectFields(elt, newPartitioner.kType.fieldNames),
              GetField(interval, "includesRight"))
          }
        }
      })

    assert(newStage.rowType == rowType,
      s"\n  repartitioned row type:     ${ newStage.rowType }" +
        s"\n  old row type: ${ rowType }")
    newStage
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

  // inserts a value/stage boundary that guarantees correct partition index seeding for directly
  // downstream operations
  def randomnessBoundary(ctx: ExecuteContext): TableStage = {
    new TableValueIntermediate(new TableStageIntermediate(this).asTableValue(ctx)).asTableStage(ctx)
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, r: RequirednessAnalysis, relationalLetsAbove: Map[String, IR]): IR = {
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
              TableStageDependency.none,
              MakeStream(FastIndexedSeq(), TStream(TStruct.empty)),
              (_: Ref) => MakeStream(FastIndexedSeq(), TStream(typ.rowType)))
          } else
            reader.lower(ctx, typ)

        case TableParallelize(rowsAndGlobal, nPartitions) =>
          val nPartitionsAdj = nPartitions.getOrElse(16)

          val loweredRowsAndGlobal = lowerIR(rowsAndGlobal)
          val loweredRowsAndGlobalRef = Ref(genUID(), loweredRowsAndGlobal.typ)

          val context = bindIR(ArrayLen(GetField(loweredRowsAndGlobalRef, "rows"))) { numRowsRef =>
            bindIR(If(numRowsRef < nPartitionsAdj, maxIR(numRowsRef, I32(1)), nPartitionsAdj)) { numNonEmptyPartitionsRef =>
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
            TableStageDependency.none,
            context,
            ctxRef => ToStream(ctxRef, true))

        case tr@TableRange(n, nPartitions) =>
          LowerTableIRHelpers.lowerTableRange(ctx, tr)

        case tmg@TableMapGlobals(child, newGlobals) =>
          LowerTableIRHelpers.lowerTableMapGlobals(ctx, tmg, lower(child))

        case TableAggregateByKey(child, expr) =>
          val loweredChild = lower(child)

          loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
            .mapPartition(Some(child.typ.key)) { partition =>

              Let("global", loweredChild.globals,
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
                })
            }

        // TODO: This ignores nPartitions and bufferSize
        case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
          val loweredChild = lower(child)
          val newKeyType = newKey.typ.asInstanceOf[TStruct]

          val fullRowUID = genUID()
          val withNewKeyFields = loweredChild.mapPartition(Some(FastIndexedSeq())) { partition =>
            Let("global", loweredChild.globals,
              mapIR(partition) { partitionElement =>
                Let("row",
                  partitionElement,
                  InsertFields(newKey, FastIndexedSeq((fullRowUID, partitionElement))))
              })
          }
          val shuffledRowType = withNewKeyFields.rowType

          val sortFields = newKeyType.fieldNames.map(fieldName => SortField(fieldName, Ascending)).toIndexedSeq
          val childRowRType = r.lookup(child).asInstanceOf[RTable].rowType
          val newKeyRType = r.lookup(newKey).asInstanceOf[RStruct]
          val withNewKeyRType = RStruct(
            newKeyRType.fields ++ Seq(RField(fullRowUID, childRowRType, newKeyRType.fields.length)))
          val shuffled = ctx.backend.lowerDistributedSort(
            ctx, withNewKeyFields, sortFields, relationalLetsAbove, withNewKeyRType)
          val repartitioned = shuffled.repartitionNoShuffle(shuffled.partitioner.strictify)

          repartitioned.mapPartition(None) { partition =>
            Let("global", repartitioned.globals,
              mapIR(StreamGroupByKey(partition, newKeyType.fieldNames.toIndexedSeq)) { groupRef =>
                StreamAgg(
                  groupRef,
                  "keyedRow",
                  bindIRs(
                    ArrayRef(
                      ApplyAggOp(FastSeq(I32(1)),
                        FastSeq(SelectFields(Ref("keyedRow", shuffledRowType), newKeyType.fieldNames)),
                        AggSignature(Take(), FastSeq(TInt32), FastSeq(newKeyType))),
                      I32(0)),
                    AggLet("row",
                      GetField(Ref("keyedRow", shuffledRowType), fullRowUID),
                      expr,
                      isScan = false)) { case Seq(groupRep, value) =>

                    val keyIRs: IndexedSeq[(String, IR)] = newKeyType.fieldNames.map(keyName => keyName -> GetField(groupRep, keyName))
                    MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                      (f, GetField(value, f))
                    })
                  }
                )
              })
          }

        case TableDistinct(child) =>
          val loweredChild = lower(child)

          loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
            .mapPartition(None) { partition =>
              flatMapIR(StreamGroupByKey(partition, child.typ.key)) { groupRef =>
                StreamTake(groupRef, 1)
              }
            }

        case tf@TableFilter(child, _) =>
          LowerTableIRHelpers.lowerTableFilter(ctx, tf, lower(child))

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
            loweredChild.dependency,
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

        case th@TableHead(child, targetNumRows) =>
          val loweredChild = lower(child)
          LowerTableIRHelpers.lowerTableHead(ctx, th, loweredChild, relationalLetsAbove)

        case tt@TableTail(child, targetNumRows) =>
          val loweredChild = lower(child)
          LowerTableIRHelpers.lowerTableTail(ctx, tt, loweredChild, relationalLetsAbove)

        case tmr@TableMapRows(child, _) =>
          LowerTableIRHelpers.lowerTableMapRows(ctx, tmr, lower(child), r, relationalLetsAbove)

        case TableKeyBy(child, newKey, isSorted: Boolean) =>
          val loweredChild = lower(child)

          val nPreservedFields = loweredChild.kType.fieldNames
            .zip(newKey)
            .takeWhile { case (l, r) => l == r }
            .length
          require(!isSorted || nPreservedFields > 0 || newKey.isEmpty,
            s"isSorted=${isSorted}, nPresFields=${nPreservedFields}, newKey=${newKey}, " +
              s"originalKey = ${loweredChild.kType.fieldNames.toSeq}, child key=${child.typ.keyType}")

          if (nPreservedFields == newKey.length || isSorted)
            // TODO: should this add a runtime check that keys are within the
            // partition bounds, like in RVD?
            loweredChild.changePartitionerNoRepartition(loweredChild.partitioner.coarsen(nPreservedFields))
              .extendKeyPreservesPartitioning(newKey)
          else {
            val rowRType = r.lookup(child).asInstanceOf[RTable].rowType
            val sorted = ctx.backend.lowerDistributedSort(
              ctx, loweredChild, newKey.map(k => SortField(k, Ascending)), relationalLetsAbove, rowRType)
            assert(sorted.kType.fieldNames.sameElements(newKey))
            sorted
          }

        case tj@TableLeftJoinRightDistinct(left, right, root) =>
          LowerTableIRHelpers.lowerTableLeftJoinRightDistinct(ctx, tj, lower(left), lower(right))

        case tj@TableJoin(left, right, joinType, joinKey) =>
          LowerTableIRHelpers.lowerTableJoin(ctx, tj, lower(left), lower(right), r)

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
            TableStageDependency.union(repartitioned.map(_.dependency)),
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
            TableStageDependency.union(repartitioned.map(_.dependency)),
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
          if (TableOrderBy.isAlreadyOrdered(sortFields, loweredChild.partitioner.kType.fieldNames)) {
            loweredChild.changePartitionerNoRepartition(RVDPartitioner.unkeyed(loweredChild.partitioner.numPartitions))
          } else {
            val rowRType = r.lookup(child).asInstanceOf[RTable].rowType
            ctx.backend.lowerDistributedSort(
              ctx, loweredChild, sortFields, relationalLetsAbove, rowRType)
          }

        case te@TableExplode(child, path) =>
          LowerTableIRHelpers.lowerTableExplode(ctx, te, lower(child))

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
            loweredChild.dependency,
            loweredChild.contexts,
            (ctxRef: Ref) => mapIR(loweredChild.partition(ctxRef)) { row =>
              CastRename(row, row.typ.asInstanceOf[TStruct].rename(rowMap))
            })

        case tmp@TableMapPartitions(child, globalName, partitionStreamName, body) =>
          val loweredChild = lower(child)
          LowerTableIRHelpers.lowerTableMapPartitions(ctx, tmp, loweredChild)

        case TableLiteral(typ, rvd, enc, encodedGlobals) =>
          RVDToTableStage(rvd, EncodedLiteral(enc, encodedGlobals))

        case bmtt@BlockMatrixToTable(bmir) =>
          val bmStage = LowerBlockMatrixIR.lower(bmir, typesToLower, ctx, r, relationalLetsAbove)
          val ts = LowerBlockMatrixIR.lowerToTableStage(bmir, typesToLower, ctx, r, relationalLetsAbove)
          // I now have an unkeyed table of (blockRow, blockCol, block).
          val entriesUnkeyed = ts.mapPartitionWithContext { (partition, ctxRef) =>
            flatMapIR(partition)(singleRowRef =>
              bindIR(GetField(singleRowRef, "block")) { singleNDRef =>
                bindIR(NDArrayShape(singleNDRef)) { shapeTupleRef =>
                  flatMapIR(rangeIR(Cast(GetTupleElement(shapeTupleRef, 0), TInt32))) { withinNDRowIdx =>
                    mapIR(rangeIR(Cast(GetTupleElement(shapeTupleRef, 1), TInt32))) { withinNDColIdx =>
                      val entry = NDArrayRef(singleNDRef, IndexedSeq(Cast(withinNDRowIdx, TInt64), Cast(withinNDColIdx, TInt64)), ErrorIDs.NO_ERROR)
                      val blockStartRow = GetField(singleRowRef, "blockRow") * bmir.typ.blockSize
                      val blockStartCol = GetField(singleRowRef, "blockCol") * bmir.typ.blockSize
                      makestruct("i" -> Cast(withinNDRowIdx + blockStartRow, TInt64), "j" -> Cast(withinNDColIdx + blockStartCol, TInt64), "entry" -> entry)
                    }
                  }
                }
              }
            )
          }

          val rowR = r.lookup(bmtt).asInstanceOf[RTable].rowType
          ctx.backend.lowerDistributedSort(ctx, entriesUnkeyed, IndexedSeq(SortField("i", Ascending), SortField("j", Ascending)), relationalLetsAbove, rowR)

        case node =>
          throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
      }

      assert(tir.typ.globalType == lowered.globalType, s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}")
      assert(tir.typ.rowType == lowered.rowType, s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}")
      assert(lowered.key startsWith tir.typ.keyType.fieldNames, s"\n  ir key: ${tir.typ.keyType.fieldNames.toSeq}\n  lowered key: ${lowered.key}")

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
          stage.mapCollect(relationalLetsAbove)(rows => foldIR(mapIR(rows)(row => Consume(row)), 0L)(_ + _)))

      case TableGetGlobals(child) =>
        lower(child).getGlobals()

      case TableCollect(child) =>
        lower(child).collectWithGlobals(relationalLetsAbove)

      case ta@TableAggregate(child, query) =>
        LowerTableIRHelpers.lowerTableAggregate(ctx, ta, lower(child), r, relationalLetsAbove)

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
