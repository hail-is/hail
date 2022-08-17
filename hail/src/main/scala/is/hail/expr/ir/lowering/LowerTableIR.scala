package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.TableCalculateNewPartitions
import is.hail.expr.ir.agg.{Aggs, Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.{agg, _}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.methods.{ForceCountTable, NPartitionsTable, TableFilterPartitions}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual._
import is.hail.types.{RField, RPrimitive, RStruct, RTable, TableType, TypeWithRequiredness}
import is.hail.types.{coerce => _, _}
import is.hail.utils.{partition, _}
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

  contexts.typ match {
    case TStream(t) if t.isRealizable =>
    case t => throw new IllegalArgumentException(s"TableStage constructed with illegal context type $t")
  }

  def typecheckPartition(ctx: ExecuteContext): Unit = {
    TypeCheck(
      ctx,
      partitionIR,
      BindingEnv(Env[Type](((letBindings ++ broadcastVals).map { case (s, x) => (s, x.typ) })
        ++ FastIndexedSeq[(String, Type)]((ctxRefName, contexts.typ.asInstanceOf[TStream].elementType)): _*)))

  }

  def ctxType: Type = contexts.typ.asInstanceOf[TStream].elementType
  def rowType: TStruct = partitionIR.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
  def kType: TStruct = partitioner.kType
  def key: IndexedSeq[String] = kType.fieldNames
  def globalType: TStruct = globals.typ.asInstanceOf[TStruct]

  assert(key.forall(f => rowType.hasField(f)), s"Key was ${key} \n kType was ${kType} \n rowType was ${rowType}")
  assert(kType.fields.forall(f => rowType.field(f.name).typ == f.typ), s"Key was ${key} \n, kType was ${kType} \n rowType was ${rowType}")
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

  def zipContextsWithIdx(): TableStage = {
    def getOldContext(ctx: IR) = GetField(ctx, "elt")
    mapContexts(zipWithIndex)(getOldContext)
  }

  def mapGlobals(f: IR => IR): TableStage = {
    val newGlobals = f(globals)
    val globalsRef = Ref(genUID(), newGlobals.typ)

    copy(
      letBindings = letBindings :+ globalsRef.name -> newGlobals,
      broadcastVals = broadcastVals :+ globalsRef.name -> globalsRef,
      globals = globalsRef)
  }

  def mapCollect(relationalBindings: Map[String, IR], staticID: String, dynamicID: IR = NA(TString))(f: IR => IR): IR = {
    mapCollectWithGlobals(relationalBindings, staticID, dynamicID)(f) { (parts, globals) => parts }
  }

  def mapCollectWithGlobals(relationalBindings: Map[String, IR], staticID: String, dynamicID: IR = NA(TString))(mapF: IR => IR)(body: (IR, IR) => IR): IR =
    mapCollectWithContextsAndGlobals(relationalBindings, staticID, dynamicID)((part, ctx) => mapF(part))(body)

  // mapf is (part, ctx) => ???, body is (parts, globals) => ???
  def mapCollectWithContextsAndGlobals(relationalBindings: Map[String, IR], staticID: String, dynamicID: IR = NA(TString))(mapF: (IR, Ref) => IR)(body: (IR, IR) => IR): IR = {
    val broadcastRefs = MakeStruct(broadcastVals)
    val glob = Ref(genUID(), broadcastRefs.typ)

    val cda = CollectDistributedArray(
      contexts, broadcastRefs,
      ctxRefName, glob.name,
      broadcastVals.foldLeft(mapF(partitionIR, Ref(ctxRefName, ctxType))) { case (accum, (name, _)) =>
        Let(name, GetField(glob, name), accum)
      }, dynamicID, staticID, Some(dependency))

    LowerToCDA.substLets(TableStage.wrapInBindings(bindIR(cda) { cdaRef => body(cdaRef, globals) }, letBindings), relationalBindings)
  }

  def collectWithGlobals(relationalBindings: Map[String, IR], staticID: String, dynamicID: IR = NA(TString)): IR =
    mapCollectWithGlobals(relationalBindings, staticID, dynamicID)(ToArray) { (parts, globals) =>
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

  def repartitionNoShuffle(newPartitioner: RVDPartitioner, allowDuplication: Boolean = false): TableStage = {
    if (newPartitioner == this.partitioner) {
      return this
    }

    if (!allowDuplication) {
      require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    }
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
              invoke("pointLessThanPartitionIntervalLeftEndpoint", TBoolean,
                SelectFields(elt, newPartitioner.kType.fieldNames),
                invoke("start", boundType.pointType, interval),
                invoke("includesStart", TBoolean, interval))

            }) { elt =>
            invoke("pointLessThanPartitionIntervalRightEndpoint", TBoolean,
              SelectFields(elt, newPartitioner.kType.fieldNames),
              invoke("end", boundType.pointType, interval),
              invoke("includesEnd", TBoolean, interval))
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
    joiner: (Ref, Ref) => IR,
    rightKeyIsDistinct: Boolean = false
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

      StreamJoin(lPart, rPart, lKey, rKey, lEltRef.name, rEltRef.name, joiner(lEltRef, rEltRef), joinType,
        requiresMemoryManagement = true, rightKeyIsDistinct = rightKeyIsDistinct)
    }

    val newKey = kType.fieldNames ++ right.kType.fieldNames.drop(joinKey)

    repartitionedLeft.alignAndZipPartitions(right, joinKey, globalJoiner, partitionJoiner)
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

    val leftKeyToRightKeyMap = (kType.fieldNames.take(joinKey), right.kType.fieldNames.take(joinKey)).zipped.toMap
    val newRightPartitioner = partitioner.coarsen(joinKey).rename(leftKeyToRightKeyMap)
    val repartitionedRight = right.repartitionNoShuffle(newRightPartitioner, allowDuplication = true)
    zipPartitions(repartitionedRight, globalJoiner, joiner)
  }

  // Like alignAndZipPartitions, when 'right' is keyed by intervals.
  // 'joiner' is called once for each partition of 'this', as in
  // alignAndZipPartitions, but now the second iterator will contain all rows
  // of 'that' whose key is an interval overlapping the range bounds of the
  // current partition of 'this', in standard interval ordering.
  def intervalAlignAndZipPartitions(
    ctx: ExecuteContext,
    right: TableStage,
    rightRowRType: RStruct,
    relationalLetsAbove: Map[String, IR],
    globalJoiner: (IR, IR) => IR,
    joiner: (IR, IR) => IR
  ): TableStage = {
    require(right.kType.size == 1)
    val rightKeyType = right.kType.fields.head.typ
    require(rightKeyType.isInstanceOf[TInterval])
    require(rightKeyType.asInstanceOf[TInterval].pointType == kType.types.head)

    val irPartitioner = partitioner.coarsen(1).partitionBoundsIRRepresentation

    val rightWithPartNums = right.mapPartition(None) { partStream =>
      flatMapIR(partStream) { row =>
        val interval = bindIR(GetField(row, right.key.head)) { interval =>
          invoke("Interval", TInterval(TTuple(kType.typeAfterSelect(Array(0)), TInt32)),
            MakeTuple.ordered(FastSeq(MakeStruct(FastSeq(kType.fieldNames.head -> invoke("start", kType.types.head, interval))), I32(1))),
            MakeTuple.ordered(FastSeq(MakeStruct(FastSeq(kType.fieldNames.head -> invoke("end", kType.types.head, interval))), I32(1))),
            invoke("includesStart", TBoolean, interval),
            invoke("includesEnd", TBoolean, interval)
          )
        }

        bindIR(invoke("partitionerFindIntervalRange", TTuple(TInt32, TInt32), irPartitioner, interval)) { range =>
          val rangeStream = StreamRange(GetTupleElement(range, 0), GetTupleElement(range, 1), I32(1), requiresMemoryManagementPerElement = true)
          mapIR(rangeStream) { partNum =>
            InsertFields(row, FastIndexedSeq("__partNum" -> partNum))
          }
        }
      }
    }
    val rightRowRTypeWithPartNum = RStruct(IndexedSeq(RField("__partNum", TypeWithRequiredness(TInt32), 0)) ++ rightRowRType.fields.map(rField => RField(rField.name, rField.typ, rField.index + 1)))
    val sorted = ctx.backend.lowerDistributedSort(ctx,
      rightWithPartNums,
      SortField("__partNum", Ascending) +: right.key.map(k => SortField(k, Ascending)),
      relationalLetsAbove,
      rightRowRTypeWithPartNum)
    assert(sorted.kType.fieldNames.sameElements("__partNum" +: right.key))
    val newRightPartitioner = new RVDPartitioner(
      Some(1),
      TStruct.concat(TStruct("__partNum" -> TInt32), right.kType),
      Array.tabulate[Interval](partitioner.numPartitions)(i => Interval(Row(i), Row(i), true, true))
      )
    val repartitioned = sorted.repartitionNoShuffle(newRightPartitioner)
      .changePartitionerNoRepartition(RVDPartitioner.unkeyed(newRightPartitioner.numPartitions))
      .mapPartition(None) { part =>
        mapIR(part) { row =>
          SelectFields(row, right.rowType.fieldNames)
        }
      }
    zipPartitions(repartitioned, globalJoiner, joiner)
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): IR = {
    def lower(tir: TableIR): TableStage = {
      this.applyTable(tir, typesToLower, ctx, analyses, relationalLetsAbove)
    }

    val lowered = ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        invoke("sum", TInt64,
          stage.mapCollect(relationalLetsAbove, "table_count")(rows => Cast(StreamLen(rows), TInt64)))

      case TableToValueApply(child, ForceCountTable()) =>
        val stage = lower(child)
        invoke("sum", TInt64,
          stage.mapCollect(relationalLetsAbove, "table_force_count")(rows => foldIR(mapIR(rows)(row => Consume(row)), 0L)(_ + _)))

      case TableToValueApply(child, TableCalculateNewPartitions(nPartitions)) =>
        val stage = lower(child)
        val sampleSize = math.min(nPartitions * 20, 1000000)
        val samplesPerPartition = sampleSize / math.max(1, stage.numPartitions)
        val keyType = child.typ.keyType
        val samplekey = AggSignature(TakeBy(),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, TFloat64))

        val minkey = AggSignature(TakeBy(),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))

        val maxkey = AggSignature(TakeBy(Descending),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))


        bindIR(flatten(stage.mapCollect(relationalLetsAbove, "table_calculate_new_partitions") { rows =>
          streamAggIR(rows) { elt =>
            ToArray(flatMapIR(ToStream(
              MakeArray(
                ApplyAggOp(
                  FastIndexedSeq(I32(samplesPerPartition)),
                  FastIndexedSeq(SelectFields(elt, keyType.fieldNames), invokeSeeded("rand_unif", 1, TFloat64, NA(TRNGState), F64(0.0), F64(1.0))),
                  samplekey),
                ApplyAggOp(
                  FastIndexedSeq(I32(1)),
                  FastIndexedSeq(elt, elt),
                  minkey),
                ApplyAggOp(
                  FastIndexedSeq(I32(1)),
                  FastIndexedSeq(elt, elt),
                  maxkey)
                )
            )) { inner => ToStream(inner) })
          }
        })) { partData =>

          val sorted = sortIR(partData) { (l, r) => ApplyComparisonOp(LT(keyType, keyType), l, r) }
          bindIR(ToArray(flatMapIR(StreamGroupByKey(ToStream(sorted), keyType.fieldNames, missingEqual = true)) { groupRef =>
            StreamTake(groupRef, 1)
          })) { boundsArray =>

            bindIR(ArrayLen(boundsArray)) { nBounds =>
              bindIR(minIR(nBounds, nPartitions)) { nParts =>
                If(nParts.ceq(0),
                  MakeArray(Seq(), TArray(TInterval(keyType))),
                  bindIR((nBounds + (nParts - 1)) floorDiv nParts) { stepSize =>
                    ToArray(mapIR(StreamRange(0, nBounds, stepSize)) { i =>
                      If((i + stepSize) < (nBounds - 1),
                        invoke("Interval", TInterval(keyType), ArrayRef(boundsArray, i), ArrayRef(boundsArray, i + stepSize), True(), False()),
                        invoke("Interval", TInterval(keyType), ArrayRef(boundsArray, i), ArrayRef(boundsArray, nBounds - 1), True(), True())
                      )})
                  }
                )
              }
            }
          }
        }

      case TableGetGlobals(child) =>
        lower(child).getGlobals()

      case TableCollect(child) =>
        lower(child).collectWithGlobals(relationalLetsAbove, "table_collect")

      case TableAggregate(child, query) =>
        val resultUID = genUID()
        val aggs = agg.Extract(query, resultUID, analyses.requirednessAnalysis, false)

        def results: IR = ResultOp.makeTuple(aggs.aggs)

        val lc = lower(child)

        val initState = Let("global", lc.globals,
          RunAgg(
            aggs.init,
            MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
            aggs.states
          ))
        val initStateRef = Ref(genUID(), initState.typ)
        val lcWithInitBinding = lc.copy(
          letBindings = lc.letBindings ++ FastIndexedSeq((initStateRef.name, initState)),
          broadcastVals = lc.broadcastVals ++ FastIndexedSeq((initStateRef.name, initStateRef)))

        val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
          InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state )})

        val useTreeAggregate = aggs.shouldTreeAggregate
        val isCommutative = aggs.isCommutative
        log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
        log.info(s"Aggregate: commutative=${ isCommutative }")

        if (useTreeAggregate) {
          val branchFactor = HailContext.get.branchingFactor
          val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

          val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
          lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove, "table_aggregate")({ part: IR =>
            Let("global", lc.globals,
              RunAgg(
                Begin(FastIndexedSeq(
                  initFromSerializedStates,
                  StreamFor(part,
                    "row",
                    aggs.seqPerElt
                  )
                )),
                WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                aggs.states
              ))
          }) { case (collected, globals) =>
            val treeAggFunction = genUID()
            val currentAggStates = Ref(genUID(), TArray(TString))
            val iterNumber = Ref(genUID(), TInt32)

            val distAggStatesRef = Ref(genUID(), TArray(TString))


            def combineGroup(partArrayRef: IR): IR = {
              Begin(FastIndexedSeq(
                bindIR(ReadValue(ArrayRef(partArrayRef, 0), codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                  Begin(
                    aggs.aggs.zipWithIndex.map { case (sig, i) =>
                      InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                    })
                },
                forIR(StreamRange(1, ArrayLen(partArrayRef), 1, requiresMemoryManagementPerElement = true)) { fileIdx =>

                  bindIR(ReadValue(ArrayRef(partArrayRef, fileIdx), codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        CombOpValue(i, GetTupleElement(serializedTuple, i), sig)
                      })
                  }
                }))
            }

            bindIR(TailLoop(treeAggFunction,
              FastIndexedSeq[(String, IR)](currentAggStates.name -> collected, iterNumber.name -> I32(0)),
              If(ArrayLen(currentAggStates) <= I32(branchFactor),
                currentAggStates,
                Recur(treeAggFunction, FastIndexedSeq(CollectDistributedArray(mapIR(StreamGrouped(ToStream(currentAggStates), I32(branchFactor)))(x => ToArray(x)),
                  MakeStruct(FastSeq()), distAggStatesRef.name, genUID(),
                  RunAgg(
                    combineGroup(distAggStatesRef),
                    WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                    aggs.states
                  ), strConcat(Str("iteration="), invoke("str", TString, iterNumber), Str(", n_states="), invoke("str", TString, ArrayLen(currentAggStates))),
                  "table_tree_aggregate"
                ), iterNumber + 1), currentAggStates.typ)))
            ) { finalParts =>
              RunAgg(
                combineGroup(finalParts),
                Let("global", globals,
                  Let(
                    resultUID,
                    results,
                    aggs.postAggIR)),
                aggs.states
              )
            }
          }
        }
        else {
          lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove, "table_aggregate_singlestage")({ part: IR =>
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
                  results,
                  aggs.postAggIR),
                aggs.states
              ))
          }
        }

      case TableToValueApply(child, NPartitionsTable()) =>
        lower(child).getNumPartitions()

      case TableWrite(child, writer) =>
        writer.lower(ctx, lower(child), child, coerce[RTable](analyses.requirednessAnalysis.lookup(child)), relationalLetsAbove)

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(ctx, node) }")
    }
    lowered
  }

  def applyTable(tir: TableIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: Analyses, relationalLetsAbove: Map[String, IR]): TableStage = {
    def lowerIR(ir: IR): IR = {
      LowerToCDA.lower(ir, typesToLower, ctx, analyses, relationalLetsAbove)
    }

    def lower(tir: TableIR): TableStage = {
      this.applyTable(tir, typesToLower, ctx, analyses, relationalLetsAbove)
    }

    if (typesToLower == DArrayLowering.BMOnly)
      throw new LowererUnsupportedOperation("found TableIR in lowering; lowering only BlockMatrixIRs.")

    val typ: TableType = tir.typ

    val lowered: TableStage = tir match {
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

      case TableRange(n, nPartitions) =>
        val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
        val partCounts = partition(n, nPartitionsAdj)
        val partStarts = partCounts.scanLeft(0)(_ + _)

        val contextType = TStruct("start" -> TInt32, "end" -> TInt32)

        val ranges = Array.tabulate(nPartitionsAdj) { i =>
          partStarts(i) -> partStarts(i + 1)
        }.toFastIndexedSeq

        TableStage(
          MakeStruct(FastSeq()),
          new RVDPartitioner(Array("idx"), tir.typ.rowType,
            ranges.map { case (start, end) =>
              Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
            }),
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType),
            ranges.map { case (start, end) => Row(start, end) })),
          (ctxRef: Ref) => mapIR(StreamRange(GetField(ctxRef, "start"), GetField(ctxRef, "end"), I32(1), true)) { i =>
            MakeStruct(FastSeq("idx" -> i))
          })

      case TableMapGlobals(child, newGlobals) =>
        lower(child).mapGlobals(old => Let("global", old, newGlobals))

      case TableAggregateByKey(child, expr) =>
        val loweredChild = lower(child)

        loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
          .mapPartition(Some(child.typ.key)) { partition =>

            Let("global", loweredChild.globals,
              mapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) { groupRef =>
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

      // This ignores nPartitions. The shuffler should handle repartitioning downstream
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val loweredChild = lower(child)
        val newKeyType = newKey.typ.asInstanceOf[TStruct]
        val resultUID = genUID()

        val aggs = Extract(expr, resultUID, analyses.requirednessAnalysis)
        val postAggIR = aggs.postAggIR
        val init = aggs.init
        val seq = aggs.seqPerElt
        val aggSigs = aggs.aggs

        val partiallyAggregated = loweredChild.mapPartition(Some(FastIndexedSeq())) { partition =>
          Let("global", loweredChild.globals,
            StreamBufferedAggregate(partition, init, newKey, seq, "row", aggSigs, bufferSize)
          )
        }
        val shuffledRowType = partiallyAggregated.rowType

        val sortFields = newKeyType.fieldNames.map(fieldName => SortField(fieldName, Ascending)).toIndexedSeq
        val withNewKeyRType = TypeWithRequiredness(shuffledRowType).asInstanceOf[RStruct]
        analyses.requirednessAnalysis.lookup(newKey).asInstanceOf[RStruct]
          .fields
          .foreach { f =>
            withNewKeyRType.field(f.name).unionFrom(f.typ)
          }

        val shuffled = ctx.backend.lowerDistributedSort(
          ctx, partiallyAggregated , sortFields, relationalLetsAbove, withNewKeyRType)
        val repartitioned = shuffled.repartitionNoShuffle(shuffled.partitioner.strictify)

        val takeVirtualSig = TakeStateSig(VirtualTypeWithReq(shuffledRowType, withNewKeyRType))
        val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)
        val aggStateSigsPlusTake = aggs.states ++ Array(takeVirtualSig)

        val postAggUID = genUID()
        val resultFromTakeUID = genUID()
        val result = ResultOp(aggs.aggs.length, takeAggSig)

        val aggSigsPlusTake = aggSigs ++ IndexedSeq(takeAggSig)
        repartitioned.mapPartition(None) { partition =>
          Let("global", repartitioned.globals,
            mapIR(StreamGroupByKey(partition, newKeyType.fieldNames.toIndexedSeq, missingEqual = true)) { groupRef =>
              RunAgg(
                forIR(zipWithIndex(groupRef)) { elemWithID =>
                  val idx = GetField(elemWithID, "idx")
                  val elem = GetField(elemWithID, "elt")
                  If(ApplyComparisonOp(EQ(TInt32, TInt32), idx, 0),
                    Begin((0 until aggSigs.length).map { aIdx =>
                      InitFromSerializedValue(aIdx, GetTupleElement(GetField(elem, "agg"), aIdx), aggSigsPlusTake(aIdx).state)
                    } ++ IndexedSeq(
                      InitOp(aggSigs.length, IndexedSeq(I32(1)), PhysicalAggSig(Take(), takeVirtualSig)),
                      SeqOp(aggSigs.length, IndexedSeq(elem), PhysicalAggSig(Take(), takeVirtualSig))
                    )),
                    Begin((0 until aggSigs.length).map { aIdx =>
                      CombOpValue(aIdx, GetTupleElement(GetField(elem, "agg"), aIdx), aggSigs(aIdx))
                    }))},

                Let(
                  resultUID,
                  ResultOp.makeTuple(aggs.aggs),
                  Let(postAggUID, postAggIR,
                    Let(resultFromTakeUID,
                      result, {
                      val keyIRs: IndexedSeq[(String, IR)] = newKeyType.fieldNames.map(keyName => keyName -> GetField(ArrayRef(Ref(resultFromTakeUID, result.typ), 0), keyName))
                      MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map {f => (f, GetField(Ref(postAggUID, postAggIR.typ), f))
                      })}
                    )
                  )
                ),
                aggStateSigsPlusTake)})}

      case TableDistinct(child) =>
        val loweredChild = lower(child)

        loweredChild.repartitionNoShuffle(loweredChild.partitioner.coarsen(child.typ.key.length).strictify)
          .mapPartition(None) { partition =>
            flatMapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) { groupRef =>
              StreamTake(groupRef, 1)
            }
          }

      case TableFilter(child, cond) =>
        val loweredChild = lower(child)
        loweredChild.mapPartition(None) { rows =>
          Let("global", loweredChild.globals,
            StreamFilter(rows, "row", cond))
        }

      case TableFilterIntervals(child, intervals, keep) =>
        val loweredChild = lower(child)
        val part = loweredChild.partitioner
        val kt = child.typ.keyType
        val ord = PartitionBoundOrdering(kt)
        val iord = ord.intervalEndpointOrdering

        val filterPartitioner = new RVDPartitioner(kt, Interval.union(intervals.toArray, ord.intervalEndpointOrdering))
        val boundsType = TArray(RVDPartitioner.intervalIRRepresentation(kt))
        val filterIntervalsRef = Ref(genUID(), boundsType)
        val filterIntervals: IndexedSeq[Interval] = filterPartitioner.rangeBounds.map { i =>
          RVDPartitioner.intervalToIRRepresentation(i, kt.size)
        }

        val (newRangeBounds, includedIndices, startAndEndInterval, f) = if (keep) {
          val (newRangeBounds, includedIndices, startAndEndInterval) = part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
            if (filterPartitioner.overlaps(interval)) {
              Some((interval, i, (filterPartitioner.lowerBoundInterval(interval), filterPartitioner.upperBoundInterval(interval))))
            } else None
          }.unzip3

          def f(partitionIntervals: IR, key: IR): IR =
            invoke("partitionerContains", TBoolean, partitionIntervals, key)
          (newRangeBounds, includedIndices, startAndEndInterval, f _)
        } else {
          // keep = False
          val (newRangeBounds, includedIndices, startAndEndInterval) = part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
            val lowerBound = filterPartitioner.lowerBoundInterval(interval)
            val upperBound = filterPartitioner.upperBoundInterval(interval)
            if ((lowerBound until upperBound).map(filterPartitioner.rangeBounds).exists { filterInterval =>
              iord.compareNonnull(filterInterval.left, interval.left) <= 0 && iord.compareNonnull(filterInterval.right, interval.right) >= 0
            })
              None
            else Some((interval, i, (lowerBound, upperBound)))
          }.unzip3

          def f(partitionIntervals: IR, key: IR): IR =
            !invoke("partitionerContains", TBoolean, partitionIntervals, key)
          (newRangeBounds, includedIndices, startAndEndInterval, f _)
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

      case TableHead(child, targetNumRows) =>
        val loweredChild = lower(child)

        def streamLenOrMax(a: IR): IR =
          if (targetNumRows <= Integer.MAX_VALUE)
            StreamLen(StreamTake(a, targetNumRows.toInt))
          else
            StreamLen(a)

        def partitionSizeArray(childContexts: Ref): IR = {
          child.partitionCounts match {
            case Some(partCounts) =>
              var idx = 0
              var sumSoFar = 0L
              while (idx < partCounts.length && sumSoFar < targetNumRows) {
                sumSoFar += partCounts(idx)
                idx += 1
              }
              val partsToKeep = partCounts.slice(0, idx)
              val finalParts = partsToKeep.map(partSize => partSize.toInt).toFastIndexedSeq
              Literal(TArray(TInt32), finalParts)
            case None =>
              val partitionSizeArrayFunc = genUID()
              val howManyPartsToTryRef = Ref(genUID(), TInt32)
              val howManyPartsToTry = if (targetNumRows == 1L) 1 else 4
              val iteration = Ref(genUID(), TInt32)

              TailLoop(
                partitionSizeArrayFunc,
                FastIndexedSeq(howManyPartsToTryRef.name -> howManyPartsToTry, iteration.name -> 0),
                bindIR(loweredChild.mapContexts(_ => StreamTake(ToStream(childContexts), howManyPartsToTryRef)){ ctx: IR => ctx }
                  .mapCollect(relationalLetsAbove, "table_head_recursive_count",
                    strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nParts="), invoke("str", TString, howManyPartsToTryRef))
                  )(streamLenOrMax)) { counts =>
                  If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (ArrayLen(childContexts) <= ArrayLen(counts)),
                    counts,
                    Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTryRef * 4, iteration + 1), TArray(TInt32)))
                })
          }
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

        val letBindNewCtx = TableStage.wrapInBindings(newCtxs, loweredChild.letBindings)
        val bindRelationLetsNewCtx = ToArray(LowerToCDA.substLets(letBindNewCtx, relationalLetsAbove))
        val newCtxSeq = CompileAndEvaluate(ctx, bindRelationLetsNewCtx).asInstanceOf[IndexedSeq[Any]]
        val numNewParts = newCtxSeq.length
        val newIntervals = loweredChild.partitioner.rangeBounds.slice(0,numNewParts)
        val newPartitioner = loweredChild.partitioner.copy(rangeBounds = newIntervals)

        TableStage(
          loweredChild.letBindings,
          loweredChild.broadcastVals,
          loweredChild.globals,
          newPartitioner,
          loweredChild.dependency,
          ToStream(Literal(bindRelationLetsNewCtx.typ, newCtxSeq)),
          (ctxRef: Ref) => StreamTake(
            loweredChild.partition(GetField(ctxRef, "old")),
            GetField(ctxRef, "numberToTake")))

      case TableTail(child, targetNumRows) =>
        val loweredChild = lower(child)

        def partitionSizeArray(childContexts: Ref, totalNumPartitions: Ref): IR = {
          child.partitionCounts match {
            case Some(partCounts) =>
              var idx = partCounts.length - 1
              var sumSoFar = 0L
              while(idx >= 0 && sumSoFar < targetNumRows) {
                sumSoFar += partCounts(idx)
                idx -= 1
              }
              val finalParts = (idx + 1 until partCounts.length).map{partIdx => partCounts(partIdx).toInt}.toFastIndexedSeq
              Literal(TArray(TInt32), finalParts)

            case None =>
              val partitionSizeArrayFunc = genUID()
              val howManyPartsToTryRef = Ref(genUID(), TInt32)
                val howManyPartsToTry = if (targetNumRows == 1L) 1 else 4

                val iteration = Ref(genUID(), TInt32)

              TailLoop(
                partitionSizeArrayFunc,
                FastIndexedSeq(howManyPartsToTryRef.name -> howManyPartsToTry, iteration.name -> 0),
                bindIR(
                  loweredChild
                    .mapContexts(_ => StreamDrop(ToStream(childContexts), maxIR(totalNumPartitions - howManyPartsToTryRef, 0))){ ctx: IR => ctx }
                    .mapCollect(relationalLetsAbove, "table_tail_recursive_count",
                      strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nParts="), invoke("str", TString, howManyPartsToTryRef)))(StreamLen)
                ) { counts =>
                  If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                    counts,
                    Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTryRef * 4, iteration + 1), TArray(TInt32)))
                })
          }
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

        val letBindNewCtx = ToArray(TableStage.wrapInBindings(newCtxs, loweredChild.letBindings))
        val newCtxSeq = CompileAndEvaluate(ctx, letBindNewCtx).asInstanceOf[IndexedSeq[Any]]
        val numNewParts = newCtxSeq.length
        val oldParts = loweredChild.partitioner.rangeBounds
        val newIntervals = oldParts.slice(oldParts.length - numNewParts, oldParts.length)
        val newPartitioner = loweredChild.partitioner.copy(rangeBounds = newIntervals)
        TableStage(
          loweredChild.letBindings,
          loweredChild.broadcastVals,
          loweredChild.globals,
          newPartitioner,
          loweredChild.dependency,
          ToStream(Literal(letBindNewCtx.typ, newCtxSeq)),
          (ctxRef: Ref) => bindIR(GetField(ctxRef, "old")) { oldRef =>
            StreamDrop(loweredChild.partition(oldRef), GetField(ctxRef, "numberToDrop"))
          })

      case TableMapRows(child, newRow) =>
        val lc = lower(child)
        if (!ContainsScan(newRow)) {
          lc.mapPartition(Some(child.typ.key)) { rows =>
            Let("global", lc.globals,
              mapIR(rows)(row => Let("row", row, newRow)))
          }
        } else {
          val resultUID = genUID()
          val aggs = agg.Extract(newRow, resultUID, analyses.requirednessAnalysis, isScan = true)

          val results: IR = ResultOp.makeTuple(aggs.aggs)
          val initState = RunAgg(
            Let("global", lc.globals, aggs.init),
            MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
            aggs.states
          )
          val initStateRef = Ref(genUID(), initState.typ)
          val lcWithInitBinding = lc.copy(
            letBindings = lc.letBindings ++ FastIndexedSeq((initStateRef.name, initState)),
            broadcastVals = lc.broadcastVals ++ FastIndexedSeq((initStateRef.name, initStateRef)))

          val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
            InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state)
          })
          val big = aggs.shouldTreeAggregate
          val (partitionPrefixSumValues, transformPrefixSum): (IR, IR => IR) = if (big) {
            val branchFactor = HailContext.get.branchingFactor
            val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

            val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
            val partitionPrefixSumFiles = lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove, "table_scan_write_prefix_sums")({ part: IR =>
              Let("global", lcWithInitBinding.globals,
                RunAgg(
                  Begin(FastIndexedSeq(
                    initFromSerializedStates,
                    StreamFor(part,
                      "row",
                      aggs.seqPerElt
                    )
                  )),
                  WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                  aggs.states
                ))
              // Collected is TArray of TString
            }) { case (collected, _) =>

              def combineGroup(partArrayRef: IR): IR = {
                Begin(FastIndexedSeq(
                  bindIR(ReadValue(ArrayRef(partArrayRef, 0), codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                      })
                  },
                  forIR(StreamRange(1, ArrayLen(partArrayRef), 1, requiresMemoryManagementPerElement = true)) { fileIdx =>

                    bindIR(ReadValue(ArrayRef(partArrayRef, fileIdx), codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                      Begin(
                        aggs.aggs.zipWithIndex.map { case (sig, i) =>
                          CombOpValue(i, GetTupleElement(serializedTuple, i), sig)
                        })
                    }
                  }))
              }

              // Return Array[Array[String]], length is log_b(num_partitions)
              // The upward pass starts with partial aggregations from each partition,
              // and aggregates these in a tree parameterized by the branching factor.
              // The tree ends when the number of partial aggregations is less than or
              // equal to the branching factor.

              // The upward pass returns the full tree of results as an array of arrays,
              // where the first element is partial aggregations per partition of the
              // input.
              def upPass(): IR = {
                val aggStack = Ref(genUID(), TArray(TArray(TString)))
                val iteration = Ref(genUID(), TInt32)
                val loopName = genUID()

                TailLoop(loopName, IndexedSeq((aggStack.name, MakeArray(collected)), (iteration.name, I32(0))),
                  bindIR(ArrayRef(aggStack, (ArrayLen(aggStack) - 1))) { states =>
                    bindIR(ArrayLen(states)) { statesLen =>
                      If(statesLen > branchFactor,
                        bindIR((statesLen + branchFactor - 1) floorDiv branchFactor) { nCombines =>
                          val contexts = mapIR(rangeIR(nCombines)) { outerIdxRef =>
                            sliceArrayIR(states, outerIdxRef * branchFactor, (outerIdxRef + 1) * branchFactor)
                          }
                          val cdaResult = cdaIR(contexts, MakeStruct(Seq()), "table_scan_up_pass",
                            strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nStates="), invoke("str", TString, statesLen))
                          ) { case (contexts, _) =>
                            RunAgg(
                              combineGroup(contexts),
                              WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                              aggs.states
                            )
                          }
                          Recur(loopName, IndexedSeq(invoke("extend", TArray(TArray(TString)), aggStack, MakeArray(cdaResult)), iteration + 1), TArray(TArray(TString)))
                        },
                        aggStack
                      )
                    }
                  }
                )
              }

              // The downward pass traverses the tree from root to leaves, computing partial scan
              // sums as it goes. The two pieces of state transmitted between iterations are the
              // level (an integer) referring to a position in the array `aggStack`, and `last`,
              // the partial sums from the last iteration. The starting state for `last` is an
              // array of a single empty aggregation state.
              bindIR(upPass()) { aggStack =>
                val downPassLoopName = genUID()
                val iteration = Ref(genUID(), TInt32)

                val level = Ref(genUID(), TInt32)
                val last = Ref(genUID(), TArray(TString))


                bindIR(WriteValue(initState, Str(tmpDir) + UUID4(), codecSpec)) { freshState =>
                  TailLoop(downPassLoopName, IndexedSeq((level.name, ArrayLen(aggStack) - 1), (last.name, MakeArray(freshState)), (iteration.name, I32(0))),
                    If(level < 0,
                      last,
                      bindIR(ArrayRef(aggStack, level)) { aggsArray =>

                        val groups = mapIR(zipWithIndex(mapIR(StreamGrouped(ToStream(aggsArray), I32(branchFactor)))(x => ToArray(x)))) { eltAndIdx =>
                          MakeStruct(FastSeq(
                            ("prev", ArrayRef(last, GetField(eltAndIdx, "idx"))),
                            ("partialSums", GetField(eltAndIdx, "elt"))
                          ))
                        }

                        val results = cdaIR(groups, MakeTuple.ordered(Seq()), "table_scan_down_pass",
                          strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", level="), invoke("str", TString, level))
                        ) { case (context, _) =>
                          bindIR(GetField(context, "prev")) { prev =>

                            val elt = Ref(genUID(), TString)
                            ToArray(RunAggScan(
                              ToStream(GetField(context, "partialSums"), requiresMemoryManagementPerElement = true),
                              elt.name,
                              bindIR(ReadValue(prev, codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                                Begin(
                                  aggs.aggs.zipWithIndex.map { case (sig, i) =>
                                    InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                                  })
                              },
                              bindIR(ReadValue(elt, codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                                Begin(
                                  aggs.aggs.zipWithIndex.map { case (sig, i) =>
                                    CombOpValue(i, GetTupleElement(serializedTuple, i), sig)
                                  })
                              },
                              WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                              aggs.states
                            ))
                          }
                        }
                        Recur(downPassLoopName,
                          IndexedSeq(
                            level - 1,
                            ToArray(flatten(ToStream(results))),
                            iteration + 1),
                          TArray(TString))
                      }
                    )
                  )
                }
              }
            }
            (partitionPrefixSumFiles, {(file: IR) => ReadValue(file, codecSpec, codecSpec.encodedVirtualType) })

          } else {
            val partitionAggs = lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove, "table_scan_prefix_sums_singlestage")({ part: IR =>
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
                ToArray(StreamTake({
                  val acc = Ref(genUID(), initStateRef.typ)
                  val value = Ref(genUID(), collected.typ.asInstanceOf[TArray].elementType)
                  StreamScan(
                    ToStream(collected),
                    initStateRef,
                    acc.name,
                    value.name,
                    RunAgg(
                      Begin(FastIndexedSeq(
                        Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
                          InitFromSerializedValue(i, GetTupleElement(acc, i), agg.state)
                        }),
                        Begin(aggs.aggs.zipWithIndex.map { case (sig, i) => CombOpValue(i, GetTupleElement(value, i), sig) }))),
                      MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
                      aggs.states
                    )
                  )
                }, ArrayLen(collected))))
            }
            (partitionAggs, identity[IR])
          }

          val partitionPrefixSumsRef = Ref(genUID(), partitionPrefixSumValues.typ)
          val zipOldContextRef = Ref(genUID(), lc.contexts.typ.asInstanceOf[TStream].elementType)
          val zipPartAggUID = Ref(genUID(), partitionPrefixSumValues.typ.asInstanceOf[TArray].elementType)
          TableStage.apply(
            letBindings = lc.letBindings ++ FastIndexedSeq((partitionPrefixSumsRef.name, partitionPrefixSumValues)),
            broadcastVals = lc.broadcastVals,
            partitioner = lc.partitioner,
            dependency = lc.dependency,
            globals = lc.globals,
            contexts = StreamZip(
              FastIndexedSeq(lc.contexts, ToStream(partitionPrefixSumsRef)),
              FastIndexedSeq(zipOldContextRef.name, zipPartAggUID.name),
              MakeStruct(FastSeq(("oldContext", zipOldContextRef), ("scanState", zipPartAggUID))),
              ArrayZipBehavior.AssertSameLength
            ),
            partition = { (partitionRef: Ref) =>
              bindIRs(GetField(partitionRef, "oldContext"), GetField(partitionRef, "scanState")) { case Seq(oldContext, rawPrefixSum) =>
                bindIR(transformPrefixSum(rawPrefixSum)) { scanState =>

                  Let("global", lc.globals,
                    RunAggScan(
                      lc.partition(oldContext),
                      "row",
                      Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
                        InitFromSerializedValue(i, GetTupleElement(scanState, i), agg.state)
                      }),
                      aggs.seqPerElt,
                      Let(
                        resultUID,
                        results,
                        aggs.postAggIR),
                      aggs.states
                    )
                  )
                }
              }
            }
          )
        }

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
          val rowRType = analyses.requirednessAnalysis.lookup(child).asInstanceOf[RTable].rowType
          val sorted = ctx.backend.lowerDistributedSort(
            ctx, loweredChild, newKey.map(k => SortField(k, Ascending)), relationalLetsAbove, rowRType)
          assert(sorted.kType.fieldNames.sameElements(newKey))
          sorted
        }

      case TableLeftJoinRightDistinct(left, right, root) =>
        val commonKeyLength = right.typ.keyType.size
        val loweredLeft = lower(left)
        val loweredRight = lower(right)

        loweredLeft.alignAndZipPartitions(
          loweredRight,
          commonKeyLength,
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

      case TableIntervalJoin(left, right, root, product) =>
        assert(!product)
        val loweredLeft = lower(left)
        val loweredRight = lower(right)

        def partitionJoiner(lPart: IR, rPart: IR): IR = {
          val lEltType = lPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
          val rEltType = rPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]

          val lKey = left.typ.key
          val rKey = right.typ.key

          val lEltRef = Ref(genUID(), lEltType)
          val rEltRef = Ref(genUID(), rEltType)

          StreamJoinRightDistinct(
            lPart, rPart,
            lKey, rKey,
            lEltRef.name, rEltRef.name,
            InsertFields(lEltRef, FastSeq(
              root -> SelectFields(rEltRef, right.typ.valueType.fieldNames))),
            "left")
        }

        loweredLeft.intervalAlignAndZipPartitions(ctx,
          loweredRight,
          analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable].rowType,
          relationalLetsAbove,
          (lGlobals, _) => lGlobals,
          partitionJoiner)

      case tj@TableJoin(left, right, joinType, joinKey) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)

        val lKeyFields = left.typ.key.take(joinKey)
        val lValueFields = left.typ.rowType.fieldNames.filter(f => !lKeyFields.contains(f))
        val rKeyFields = right.typ.key.take(joinKey)
        val rValueFields = right.typ.rowType.fieldNames.filter(f => !rKeyFields.contains(f))
        val lReq = analyses.requirednessAnalysis.lookup(left).asInstanceOf[RTable]
        val rReq = analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable]
        val rightKeyIsDistinct = analyses.distinctKeyedAnalysis.contains(right)

        val joinedStage = loweredLeft.orderedJoin(
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
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey), Die("TableJoin expected non-missing key", left.typ.rowType.fieldType(lKey), -1)))
                else
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey)))
              }
                ++ lValueFields.map(f => f -> GetField(lEltRef, f))
                ++ rValueFields.map(f => f -> GetField(rEltRef, f)))
          }, rightKeyIsDistinct)

        assert(joinedStage.rowType == tj.typ.rowType)
        joinedStage

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
          val rowRType = analyses.requirednessAnalysis.lookup(child).asInstanceOf[RTable].rowType
          ctx.backend.lowerDistributedSort(
            ctx, loweredChild, sortFields, relationalLetsAbove, rowRType)
        }

      case TableExplode(child, path) =>
        lower(child).mapPartition(Some(child.typ.key.takeWhile(k => k != path(0)))) { rows =>
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
              mapIR(ToStream(refs.last, true)) { elt =>
                path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
                  InsertFields(ref, FastSeq(p -> inserted))
                }
              }) { case ((ref, root), accum) =>  Let(ref.name, root, accum) }
          }
        }

      case TableRepartition(child, n, RepartitionStrategy.NAIVE_COALESCE) =>
        val lc = lower(child)
        val groupSize = (lc.numPartitions + n - 1) / n

        TableStage(
          letBindings = lc.letBindings,
          broadcastVals = lc.broadcastVals,
          globals = lc.globals,
          partitioner = lc.partitioner.copy(rangeBounds = lc.partitioner
            .rangeBounds
            .grouped(groupSize)
            .toArray
            .map(arr => Interval(arr.head.left, arr.last.right))),
          dependency = lc.dependency,
          contexts = mapIR(StreamGrouped(lc.contexts, groupSize)) { group => ToArray(group) },
          partition = (r: Ref) => flatMapIR(ToStream(r)) { prevCtx => lc.partition(prevCtx) }
        )

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

      case TableMapPartitions(child, globalName, partitionStreamName, body) =>
        val loweredChild = lower(child)
        loweredChild.mapPartition(Some(child.typ.key)) { part =>
          Let(globalName, loweredChild.globals, Let(partitionStreamName, part, body))
        }

      case TableLiteral(typ, rvd, enc, encodedGlobals) =>
        RVDToTableStage(rvd, EncodedLiteral(enc, encodedGlobals))

      case TableToTableApply(child, TableFilterPartitions(seq, keep)) =>
        val lc = lower(child)

        val arr = seq.sorted.toArray
        val keptSet = seq.toSet
        val lit = Literal(TSet(TInt32), keptSet)
        if (keep) {
          def lookupRangeBound(idx: Int): Interval = {
            try {
              lc.partitioner.rangeBounds(idx)
            } catch {
              case exc: ArrayIndexOutOfBoundsException =>
                fatal(s"_filter_partitions: no partition with index $idx", exc)
            }
          }

          lc.copy(
            partitioner = lc.partitioner.copy(rangeBounds = arr.map(lookupRangeBound)),
            contexts = mapIR(
              filterIR(
                zipWithIndex(lc.contexts)) { t =>
                invoke("contains", TBoolean, lit, GetField(t, "idx")) }) { t =>
              GetField(t, "elt") }
          )
        } else {
          lc.copy(
            partitioner = lc.partitioner.copy(rangeBounds = lc.partitioner.rangeBounds.zipWithIndex.filter { case (_, idx) => !keptSet.contains(idx) }.map(_._1)),
            contexts = mapIR(
              filterIR(
                zipWithIndex(lc.contexts)) { t =>
                !invoke("contains", TBoolean, lit, GetField(t, "idx")) }) { t =>
              GetField(t, "elt") }
          )
        }

      case bmtt@BlockMatrixToTable(bmir) =>
        val ts = LowerBlockMatrixIR.lowerToTableStage(bmir, typesToLower, ctx, analyses, relationalLetsAbove)
        // I now have an unkeyed table of (blockRow, blockCol, block).
        ts.mapPartitionWithContext { (partition, ctxRef) =>
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

      case node =>
        throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(ctx, node) }")
    }

    assert(tir.typ.globalType == lowered.globalType, s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}")
    assert(tir.typ.rowType == lowered.rowType, s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}")
    assert(lowered.key startsWith tir.typ.keyType.fieldNames, s"\n  ir key: ${tir.typ.keyType.fieldNames.toSeq}\n  lowered key: ${lowered.key}")

    lowered
  }
}
