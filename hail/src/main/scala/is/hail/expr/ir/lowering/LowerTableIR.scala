package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.ArrayZipBehavior.AssertSameLength
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToTableFunction}
import is.hail.expr.ir.{TableNativeWriter, agg, _}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.methods.{ForceCountTable, LocalLDPrune, NPartitionsTable, TableFilterPartitions}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types._
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual.TIterable.elementType
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

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
      FastSeq(globalsRef.name -> globals),
      FastSeq(globalsRef.name -> globalsRef),
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

  def concatenate(ctx: ExecuteContext, children: IndexedSeq[TableStage]): TableStage = {
    val keyType = children.head.kType
    assert(keyType.size == 0)
    assert(children.forall(_.kType == keyType))

    val ctxType = TTuple(children.map(_.ctxType): _*)
    val ctxArrays = children.view.zipWithIndex.map { case (child, idx) =>
      ToArray(mapIR(child.contexts) { ctx =>
        MakeTuple.ordered(children.indices.map { idx2 =>
          if (idx == idx2) ctx else NA(children(idx2).ctxType)
        })
      })
    }
    val ctxs = flatMapIR(MakeStream(ctxArrays.toFastSeq, TStream(TArray(ctxType)))) { ctxArray =>
      ToStream(ctxArray)
    }

    val newGlobals = children.head.globals
    val globalsRef = Ref(genUID(), newGlobals.typ)
    val newPartitioner = new RVDPartitioner(ctx.stateManager, keyType, children.flatMap(_.partitioner.rangeBounds))

    TableStage(
      children.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals,
      children.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
      globalsRef,
      newPartitioner,
      TableStageDependency.union(children.map(_.dependency)),
      ctxs,
      (ctxRef: Ref) => {
        StreamMultiMerge(
          children.indices.map { i =>
            bindIR(GetTupleElement(ctxRef, i)) { ctx =>
              If(IsNA(ctx),
                 MakeStream(IndexedSeq(), TStream(children(i).rowType)),
                 children(i).partition(ctx))
            }
          },
          IndexedSeq())
      })
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
  val ctxRefName: String,
  val partitionIR: IR) {
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
        ++ FastSeq[(String, Type)]((ctxRefName, contexts.typ.asInstanceOf[TStream].elementType)): _*)))

  }

  def upcast(ctx: ExecuteContext, newType: TableType): TableStage = {
    val newRowType = newType.rowType
    val newGlobalType = newType.globalType
    if (newRowType == rowType && newGlobalType == globalType)
      this
    else {
      changePartitionerNoRepartition(partitioner.coarsen(newType.key.length))
        .mapPartition(None)(PruneDeadFields.upcast(ctx, _, TStream(newRowType)))
        .mapGlobals(PruneDeadFields.upcast(ctx, _, newGlobalType))
    }
  }

  def ctxType: Type = contexts.typ.asInstanceOf[TStream].elementType
  def rowType: TStruct = partitionIR.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
  def kType: TStruct = partitioner.kType
  def key: IndexedSeq[String] = kType.fieldNames
  def globalType: TStruct = globals.typ.asInstanceOf[TStruct]
  def tableType: TableType =
    TableType(rowType, key, globalType)

  assert(kType.isSubsetOf(rowType), s"Key type $kType is not a subset of $rowType")
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
    Let(FastSeq(ctxRefName -> ctx), partitionIR)
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
      FastSeq(left.contexts, right.contexts),
      FastSeq(leftCtxRef.name, rightCtxRef.name),
      MakeStruct(FastSeq(leftCtxStructField -> leftCtxRef,
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

  def mapCollect(staticID: String, dynamicID: IR = NA(TString))(f: IR => IR): IR = {
    mapCollectWithGlobals(staticID, dynamicID)(f) { (parts, globals) => parts }
  }

  def mapCollectWithGlobals(staticID: String, dynamicID: IR = NA(TString))(mapF: IR => IR)(body: (IR, IR) => IR): IR =
    mapCollectWithContextsAndGlobals(staticID, dynamicID)((part, ctx) => mapF(part))(body)

  // mapf is (part, ctx) => ???, body is (parts, globals) => ???
  def mapCollectWithContextsAndGlobals(staticID: String, dynamicID: IR = NA(TString))(mapF: (IR, Ref) => IR)(body: (IR, IR) => IR): IR = {
    val broadcastRefs = MakeStruct(broadcastVals)
    val glob = Ref(genUID(), broadcastRefs.typ)

    val cda = CollectDistributedArray(
      contexts, broadcastRefs,
      ctxRefName, glob.name,
      Let(broadcastVals.map { case (name, _) => name -> GetField(glob, name) },
        mapF(partitionIR, Ref(ctxRefName, ctxType))
      ), dynamicID, staticID, Some(dependency))

    Let(letBindings, bindIR(cda) { cdaRef => body(cdaRef, globals) })
  }

  def collectWithGlobals(staticID: String, dynamicID: IR = NA(TString)): IR =
    mapCollectWithGlobals(staticID, dynamicID)(ToArray) { (parts, globals) =>
      MakeStruct(FastSeq(
        "rows" -> ToArray(flatMapIR(ToStream(parts))(ToStream(_))),
        "global" -> globals))
    }

  def countPerPartition(): IR = mapCollect("count_per_partition")(part => Cast(StreamLen(part), TInt64))

  def getGlobals(): IR =
    Let(letBindings, globals)

  def getNumPartitions(): IR =
    Let(letBindings, StreamLen(contexts))

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage = {
    require(partitioner.numPartitions == newPartitioner.numPartitions)
    copy(partitioner = newPartitioner)
  }

  def strictify(ec: ExecuteContext, allowedOverlap: Int = kType.size - 1): TableStage = {
    val newPart = partitioner.strictify(allowedOverlap)
    repartitionNoShuffle(ec, newPart)
  }

  def repartitionNoShuffle(ec: ExecuteContext,
                           newPartitioner: RVDPartitioner,
                           allowDuplication: Boolean = false,
                           dropEmptyPartitions: Boolean = false
                          ): TableStage = {

    if (newPartitioner == this.partitioner) {
      return this
    }

    if (!allowDuplication) {
      require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    }
    require(newPartitioner.kType.isPrefixOf(kType))

    val newStage = if (LowerTableIR.isRepartitioningCheap(partitioner, newPartitioner)) {
      val startAndEnd = partitioner.rangeBounds.map(newPartitioner.intervalRange).zipWithIndex
      if (startAndEnd.forall { case ((start, end), i) => start + 1 == end &&
        newPartitioner.rangeBounds(start).includes(newPartitioner.kord, partitioner.rangeBounds(i))
      }) {
        val newToOld = startAndEnd.groupBy(_._1._1).map { case (newIdx, values) =>
          (newIdx, values.map(_._2).sorted.toIndexedSeq)
        }

        val (oldPartIndices, newPartitionerFilt) =
          if (dropEmptyPartitions) {
            val indices = (0 until newPartitioner.numPartitions).filter(newToOld.contains)
            (indices.map(newToOld), newPartitioner.copy(rangeBounds = indices.map(newPartitioner.rangeBounds)))
          } else
            ((0 until newPartitioner.numPartitions).map(i => newToOld.getOrElse(i, FastSeq())), newPartitioner)

        log.info(
          "repartitionNoShuffle - fast path," +
            s" generated ${oldPartIndices.length} partitions from ${partitioner.numPartitions}" +
            s" (dropped ${newPartitioner.numPartitions - oldPartIndices.length} empty output parts)"
        )

        val newContexts = bindIR(ToArray(contexts)) { oldCtxs =>
          mapIR(ToStream(Literal(TArray(TArray(TInt32)), oldPartIndices))) { inds =>
            ToArray(mapIR(ToStream(inds)) { i => ArrayRef(oldCtxs, i) })
          }
        }

        return TableStage(letBindings, broadcastVals, globals, newPartitionerFilt, dependency, newContexts,
          (ctx: Ref) => flatMapIR(ToStream(ctx, true)) { oldCtx => partition(oldCtx) })
      }

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
      val newContexts = Let(FastSeq(prevContextUID -> ToArray(contexts)),
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
      TableStage(letBindings, broadcastVals, globals, newPartitioner, dependency, newContexts,
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
    } else {
      val location = ec.createTmpPath(genUID())
      CompileAndEvaluate(ec,
        TableNativeWriter(location).lower(ec, this, RTable.fromTableStage(ec, this))
      )

      val newTableType = TableType(rowType, newPartitioner.kType.fieldNames, globalType)
      val reader = TableNativeReader.read(ec.fs, location, Some(NativeReaderOptions(
        intervals = newPartitioner.rangeBounds,
        intervalPointType = newPartitioner.kType,
        filterIntervals = dropEmptyPartitions
      )))

      val table = TableRead(newTableType, dropRows = false, tr = reader)
      LowerTableIR.applyTable(table, DArrayLowering.All, ec, LoweringAnalyses.apply(table, ec))
    }

    assert(newStage.rowType == rowType,
      s"repartitioned row type: ${newStage.rowType}\n" +
        s"          old row type: $rowType")
    newStage
  }

  def extendKeyPreservesPartitioning(ec: ExecuteContext, newKey: IndexedSeq[String]): TableStage = {
    require(newKey startsWith kType.fieldNames)
    require(newKey.forall(rowType.fieldNames.contains))

    val newKeyType = rowType.typeAfterSelectNames(newKey)
    if (RVDPartitioner.isValid(partitioner.sm, newKeyType, partitioner.rangeBounds)) {
      changePartitionerNoRepartition(partitioner.copy(kType = newKeyType))
    } else {
      val adjustedPartitioner = partitioner.strictify()
      repartitionNoShuffle(ec, adjustedPartitioner)
        .changePartitionerNoRepartition(adjustedPartitioner.copy(kType = newKeyType))
    }
  }

  def orderedJoin(
    ec: ExecuteContext,
    right: TableStage,
    joinKey: Int,
    joinType: String,
    globalJoiner: (IR, IR) => IR,
    joiner: (Ref, Ref) => IR,
    rightKeyIsDistinct: Boolean = false
  ): TableStage = {
    assert(this.kType.truncate(joinKey).isIsomorphicTo(right.kType.truncate(joinKey)))

    val newPartitioner = {
      def leftPart: RVDPartitioner = this.partitioner.strictify()
      def rightPart: RVDPartitioner = right.partitioner.coarsen(joinKey).extendKey(this.kType)
      (joinType: @unchecked) match {
        case "left" => leftPart
        case "right" => rightPart
        case "inner" => leftPart.intersect(rightPart)
        case "outer" => RVDPartitioner.generate(
          partitioner.sm,
          this.kType.fieldNames.take(joinKey),
          this.kType,
          leftPart.rangeBounds ++ rightPart.rangeBounds)
      }
    }
    val repartitionedLeft: TableStage = repartitionNoShuffle(ec, newPartitioner)

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

    repartitionedLeft.alignAndZipPartitions(ec, right, joinKey, globalJoiner, partitionJoiner)
      .extendKeyPreservesPartitioning(ec, newKey)
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
    ec: ExecuteContext,
    right: TableStage,
    joinKey: Int,
    globalJoiner: (IR, IR) => IR,
    joiner: (IR, IR) => IR
  ): TableStage = {
    require(joinKey <= kType.size)
    require(joinKey <= right.kType.size)

    val leftKeyToRightKeyMap = (kType.fieldNames.take(joinKey), right.kType.fieldNames.take(joinKey)).zipped.toMap
    val newRightPartitioner = partitioner.coarsen(joinKey).rename(leftKeyToRightKeyMap)
    val repartitionedRight = right.repartitionNoShuffle(ec, newRightPartitioner, allowDuplication = true)
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
            InsertFields(row, FastSeq("__partNum" -> partNum))
          }
        }
      }
    }

    val rightRowRTypeWithPartNum = IndexedSeq("__partNum" -> TypeWithRequiredness(TInt32)) ++ rightRowRType.fields.map(rField => rField.name -> rField.typ)
    val rightTableRType = RTable(rightRowRTypeWithPartNum, FastSeq(), right.key)
    val sortedReader = ctx.backend.lowerDistributedSort(ctx,
      rightWithPartNums,
      SortField("__partNum", Ascending) +: right.key.map(k => SortField(k, Ascending)),
      rightTableRType)
    val sorted = sortedReader.lower(ctx, sortedReader.fullType)
    assert(sorted.kType.fieldNames.sameElements("__partNum" +: right.key))
    val newRightPartitioner = new RVDPartitioner(
      ctx.stateManager,
      Some(1),
      TStruct.concat(TStruct("__partNum" -> TInt32), right.kType),
      Array.tabulate[Interval](partitioner.numPartitions)(i => Interval(Row(i), Row(i), true, true))
      )
    val repartitioned = sorted.repartitionNoShuffle(ctx, newRightPartitioner)
      .changePartitionerNoRepartition(RVDPartitioner.unkeyed(ctx.stateManager, newRightPartitioner.numPartitions))
      .mapPartition(None) { part =>
        mapIR(part) { row =>
          SelectFields(row, right.rowType.fieldNames)
        }
      }
    zipPartitions(repartitioned, globalJoiner, joiner)
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): IR = {
    def lower(tir: TableIR): TableStage = {
      this.applyTable(tir, typesToLower, ctx, analyses)
    }

    val lowered = ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        invoke("sum", TInt64,
          stage.countPerPartition())

      case TableToValueApply(child, ForceCountTable()) =>
        val stage = lower(child)
        invoke("sum", TInt64,
          stage.mapCollect("table_force_count")(rows => foldIR(mapIR(rows)(row => Consume(row)), 0L)(_ + _)))

      case TableToValueApply(child, TableCalculateNewPartitions(nPartitions)) =>
        val stage = lower(child)
        val sampleSize = math.min((nPartitions * 20 + 256), 1000000)
        val samplesPerPartition = sampleSize / math.max(1, stage.numPartitions)
        val keyType = child.typ.keyType
        val samplekey = AggSignature(ReservoirSample(),
          FastSeq(TInt32),
          FastSeq(keyType))

        val minkey = AggSignature(TakeBy(),
          FastSeq(TInt32),
          FastSeq(keyType, keyType))

        val maxkey = AggSignature(TakeBy(Descending),
          FastSeq(TInt32),
          FastSeq(keyType, keyType))


        bindIR(flatten(stage.mapCollect("table_calculate_new_partitions") { rows =>
          streamAggIR(mapIR(rows) { row => SelectFields(row, keyType.fieldNames)}) { elt =>
            ToArray(flatMapIR(ToStream(
              MakeArray(
                ApplyAggOp(
                  FastSeq(I32(samplesPerPartition)),
                  FastSeq(elt),
                  samplekey),
                ApplyAggOp(
                  FastSeq(I32(1)),
                  FastSeq(elt, elt),
                  minkey),
                ApplyAggOp(
                  FastSeq(I32(1)),
                  FastSeq(elt, elt),
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
                  MakeArray(FastSeq(), TArray(TInterval(keyType))),
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
        lower(child).collectWithGlobals("table_collect")

      case TableAggregate(child, query) =>
        val resultUID = genUID()
        val aggs = agg.Extract(query, resultUID, analyses.requirednessAnalysis, false)

        def results: IR = ResultOp.makeTuple(aggs.aggs)

        val lc = lower(child)

        val initState = Let(FastSeq("global" -> lc.globals),
          RunAgg(
            aggs.init,
            MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
            aggs.states
          ))
        val initStateRef = Ref(genUID(), initState.typ)
        val lcWithInitBinding = lc.copy(
          letBindings = lc.letBindings ++ FastSeq((initStateRef.name, initState)),
          broadcastVals = lc.broadcastVals ++ FastSeq((initStateRef.name, initStateRef)))

        val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
          InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state )})

        val branchFactor = HailContext.get.branchingFactor
        val useTreeAggregate = aggs.shouldTreeAggregate && branchFactor < lc.numPartitions
        val isCommutative = aggs.isCommutative
        log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
        log.info(s"Aggregate: commutative=${ isCommutative }")

        if (useTreeAggregate) {
          val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

          val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
          val writer = ETypeValueWriter(codecSpec)
          val reader = ETypeValueReader(codecSpec)
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate")({ part: IR =>
            Let(FastSeq("global" -> lc.globals),
              RunAgg(
                Begin(FastSeq(
                  initFromSerializedStates,
                  StreamFor(part,
                    "row",
                    aggs.seqPerElt
                  )
                )),
                WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), writer),
                aggs.states
              ))
          }) { case (collected, globals) =>
            val treeAggFunction = genUID()
            val currentAggStates = Ref(genUID(), TArray(TString))
            val iterNumber = Ref(genUID(), TInt32)

            val distAggStatesRef = Ref(genUID(), TArray(TString))

            def combineGroup(partArrayRef: IR, useInitStates: Boolean): IR = {
              Begin(FastSeq(
                if (useInitStates) {
                  initFromSerializedStates
                } else {
                  bindIR(ReadValue(ArrayRef(partArrayRef, 0), reader, reader.spec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                      })
                  }
                },
                forIR(StreamRange(if (useInitStates) 0 else 1, ArrayLen(partArrayRef), 1, requiresMemoryManagementPerElement = true)) { fileIdx =>

                  bindIR(ReadValue(ArrayRef(partArrayRef, fileIdx), reader, reader.spec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        CombOpValue(i, GetTupleElement(serializedTuple, i), sig)
                      })
                  }
                }))
            }

            val loopBody = If(
              ArrayLen(currentAggStates) <= I32(branchFactor),
              currentAggStates,
              Recur(
                treeAggFunction,
                FastSeq(
                  CollectDistributedArray(
                    mapIR(StreamGrouped(ToStream(currentAggStates), I32(branchFactor)))(x => ToArray(x)),
                    MakeStruct(FastSeq()),
                    distAggStatesRef.name,
                    genUID(),
                    RunAgg(
                      combineGroup(distAggStatesRef, false),
                      WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), writer),
                      aggs.states),
                    strConcat(Str("iteration="), invoke("str", TString, iterNumber), Str(", n_states="), invoke("str", TString, ArrayLen(currentAggStates))),
                    "table_tree_aggregate"),
                  iterNumber + 1),
                currentAggStates.typ))
            bindIR(TailLoop(
              treeAggFunction,
              FastSeq[(String, IR)](currentAggStates.name -> collected, iterNumber.name -> I32(0)),
              loopBody.typ,
              loopBody
            )) { finalParts =>
              RunAgg(
                combineGroup(finalParts, true),
                Let(FastSeq("global" -> globals, resultUID -> results), aggs.postAggIR),
                aggs.states
              )
            }
          }
        }
        else {
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate_singlestage")({ part: IR =>
            Let(FastSeq("global" -> lc.globals),
              RunAgg(
                Begin(FastSeq(
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
            Let(FastSeq("global" -> globals),
              RunAgg(
                Begin(FastSeq(
                  initFromSerializedStates,
                  forIR(ToStream(collected, requiresMemoryManagementPerElement = true)) { state =>
                    Begin(aggs.aggs.zipWithIndex.map { case (sig, i) => CombOpValue(i, GetTupleElement(state, i), sig) })
                  }
                )),
                Let(FastSeq(resultUID -> results), aggs.postAggIR),
                aggs.states
              )
            )
          }
        }

      case TableToValueApply(child, NPartitionsTable()) =>
        lower(child).getNumPartitions()

      case TableWrite(child, writer) =>
        writer.lower(ctx, lower(child), tcoerce[RTable](analyses.requirednessAnalysis.lookup(child)))

      case TableMultiWrite(children, writer) =>
        writer.lower(ctx, children.map(child => (lower(child), tcoerce[RTable](analyses.requirednessAnalysis.lookup(child)))))

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(ctx, node) }")
    }
    lowered
  }

  def applyTable(tir: TableIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): TableStage = {
    def lowerIR(ir: IR): IR = {
      LowerToCDA.lower(ir, typesToLower, ctx, analyses)
    }

    def lower(tir: TableIR): TableStage = {
      this.applyTable(tir, typesToLower, ctx, analyses)
    }

    if (typesToLower == DArrayLowering.BMOnly)
      throw new LowererUnsupportedOperation("found TableIR in lowering; lowering only BlockMatrixIRs.")

    val typ: TableType = tir.typ

    val lowered: TableStage = tir match {
      case TableRead(typ, dropRows, reader) =>
        reader.lower(ctx, typ, dropRows)

      case TableParallelize(rowsAndGlobal, nPartitions) =>
        val nPartitionsAdj = nPartitions.getOrElse(ctx.backend.defaultParallelism)

        val loweredRowsAndGlobal = lowerIR(rowsAndGlobal)
        val loweredRowsAndGlobalRef = Ref(genUID(), loweredRowsAndGlobal.typ)

        val context = bindIR(ArrayLen(GetField(loweredRowsAndGlobalRef, "rows"))) { numRowsRef =>
          bindIR(invoke("extend", TArray(TInt32), ToArray(mapIR(rangeIR(nPartitionsAdj)) { partIdx =>
            (partIdx * numRowsRef) floorDiv nPartitionsAdj
          }),
            MakeArray((numRowsRef)))) { indicesArray =>
            bindIR(GetField(loweredRowsAndGlobalRef, "rows")) { rows =>
              mapIR(rangeIR(nPartitionsAdj)) { partIdx =>
                ToArray(mapIR(rangeIR(ArrayRef(indicesArray, partIdx), ArrayRef(indicesArray, partIdx + 1))) { rowIdx =>
                  ArrayRef(rows, rowIdx)
                })
              }
            }
          }
        }

        val globalsRef = Ref(genUID(), typ.globalType)
        TableStage(
          FastSeq(loweredRowsAndGlobalRef.name -> loweredRowsAndGlobal,
            globalsRef.name -> GetField(loweredRowsAndGlobalRef, "global")),
          FastSeq(globalsRef.name -> globalsRef),
          globalsRef,
          RVDPartitioner.unkeyed(ctx.stateManager, nPartitionsAdj),
          TableStageDependency.none,
          context,
          ctxRef => ToStream(ctxRef, true))

      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        val loweredGlobals = lowerIR(globals)
        TableStage(
          loweredGlobals,
          partitioner = partitioner,
          dependency = TableStageDependency.none,
          contexts = lowerIR {
            bindIR(ToArray(contexts)) { ref =>
              bindIR(ArrayLen(ref)) { len =>
                // Assert at runtime that the number of contexts matches the number of partitions
                val ctxs = ToStream(If(len ceq partitioner.numPartitions, ref, {
                  val dieMsg = strConcat(
                    s"TableGen: partitioner contains ${partitioner.numPartitions} partitions,",
                    " got ", len, " contexts."
                  )
                  Die(dieMsg, ref.typ, errorId)
                }))

                // [FOR KEYED TABLES ONLY]
                // AFAIK, there's no way to guarantee that the rows generated in the
                // body conform to their partition's range bounds at compile time so
                // assert this at runtime in the body before it wreaks havoc upon the world.
                val partitionIdx = StreamRange(I32(0), I32(partitioner.numPartitions), I32(1))
                val bounds = Literal(TArray(TInterval(partitioner.kType)), partitioner.rangeBounds.toIndexedSeq)
                zipIR(FastSeq(partitionIdx, ToStream(bounds), ctxs), AssertSameLength, errorId)(MakeTuple.ordered)
              }
            }
          },
          body = in => lowerIR {
            val rows = Let(FastSeq(cname -> GetTupleElement(in, 2), gname -> loweredGlobals), body)
            if (partitioner.kType.fields.isEmpty) rows
            else bindIR(GetTupleElement(in, 1)) { interval =>
              mapIR(rows) { row =>
                val key = SelectFields(row, partitioner.kType.fieldNames)
                If(invoke("contains", TBoolean, interval, key), row, {
                  val idx = GetTupleElement(in, 0)
                  val msg = strConcat(
                    "TableGen: Unexpected key in partition ", idx,
                    "\n\tRange bounds for partition ", idx, ": ", interval,
                    "\n\tInvalid key: ", key
                  )
                  Die(msg, row.typ, errorId)
                })
              }
            }
          }
        )

      case TableRange(n, nPartitions) =>
        val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
        val partCounts = partition(n, nPartitionsAdj)
        val partStarts = partCounts.scanLeft(0)(_ + _)

        val contextType = TStruct("start" -> TInt32, "end" -> TInt32)

        val ranges = Array.tabulate(nPartitionsAdj)(i => partStarts(i) -> partStarts(i + 1))

        TableStage(
          MakeStruct(FastSeq()),
          new RVDPartitioner(ctx.stateManager, Array("idx"), tir.typ.rowType, ranges.map {
            case (start, end) => Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), ranges.map(Row.fromTuple).toFastSeq)),
          (ctxRef: Ref) => mapIR(StreamRange(GetField(ctxRef, "start"), GetField(ctxRef, "end"), I32(1), true)) { i =>
            MakeStruct(FastSeq("idx" -> i))
          })

      case TableMapGlobals(child, newGlobals) =>
        lower(child).mapGlobals(old => Let(FastSeq("global" -> old), newGlobals))

      case TableAggregateByKey(child, expr) =>
        val loweredChild = lower(child)

        loweredChild.repartitionNoShuffle(ctx, loweredChild.partitioner.coarsen(child.typ.key.length).strictify())
          .mapPartition(Some(child.typ.key)) { partition =>

            Let(FastSeq("global" -> loweredChild.globals),
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

      case TableDistinct(child) =>
        val loweredChild = lower(child)

        if (analyses.distinctKeyedAnalysis.contains(child))
          loweredChild
        else
          loweredChild.repartitionNoShuffle(ctx, loweredChild.partitioner.coarsen(child.typ.key.length).strictify())
            .mapPartition(None) { partition =>
              flatMapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) { groupRef =>
                StreamTake(groupRef, 1)
              }
            }

      case TableFilter(child, cond) =>
        val loweredChild = lower(child)
        loweredChild.mapPartition(None) { rows =>
          Let(FastSeq("global" -> loweredChild.globals),
            StreamFilter(rows, "row", cond))
        }

      case TableFilterIntervals(child, intervals, keep) =>
        val loweredChild = lower(child)
        val part = loweredChild.partitioner
        val kt = child.typ.keyType
        val ord = PartitionBoundOrdering(ctx.stateManager, kt)
        val iord = ord.intervalEndpointOrdering

        val filterPartitioner = new RVDPartitioner(ctx.stateManager, kt, Interval.union(intervals, ord.intervalEndpointOrdering))
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

        val newPart = new RVDPartitioner(ctx.stateManager, kt, newRangeBounds)

        TableStage(
          letBindings = loweredChild.letBindings,
          broadcastVals = loweredChild.broadcastVals ++ FastSeq((filterIntervalsRef.name, Literal(boundsType, filterIntervals))),
          loweredChild.globals,
          newPart,
          loweredChild.dependency,
          contexts = bindIRs(
            ToArray(loweredChild.contexts),
            Literal(TArray(TTuple(TInt32, TInt32)), startAndEndInterval.map(Row.fromTuple).toFastSeq)
          ) { case Seq(prevContexts, bounds) =>
            zip2(ToStream(Literal(TArray(TInt32), includedIndices.toFastSeq)), ToStream(bounds), ArrayZipBehavior.AssumeSameLength) { (idx, bound) =>
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
              val finalParts = partsToKeep.map(partSize => partSize.toInt).toFastSeq
              Literal(TArray(TInt32), finalParts)
            case None =>
              val partitionSizeArrayFunc = genUID()
              val howManyPartsToTryRef = Ref(genUID(), TInt32)
              val howManyPartsToTry = if (targetNumRows == 1L) 1 else 4
              val iteration = Ref(genUID(), TInt32)

              val loopBody = bindIR(
                loweredChild
                  .mapContexts(_ => StreamTake(ToStream(childContexts), howManyPartsToTryRef)) { ctx: IR => ctx }
                  .mapCollect(
                    "table_head_recursive_count",
                    strConcat(Str("iteration="), invoke("str", TString, iteration), Str(",nParts="), invoke("str", TString, howManyPartsToTryRef))
                    )(streamLenOrMax)
                ) { counts =>
                  If(
                    (Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows)
                      || (ArrayLen(childContexts) <= ArrayLen(counts)),
                    counts,
                    Recur(partitionSizeArrayFunc, FastSeq(howManyPartsToTryRef * 4, iteration + 1), TArray(TInt32)))
              }
              TailLoop(
                partitionSizeArrayFunc,
                FastSeq(howManyPartsToTryRef.name -> howManyPartsToTry, iteration.name -> 0),
                loopBody.typ,
                loopBody)
          }
        }

        def answerTuple(partitionSizeArrayRef: Ref): IR =
          bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            val howManyPartsToKeep = genUID()
            val i = Ref(genUID(), TInt32)
            val numLeft = Ref(genUID(), TInt64)
            def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple(FastSeq((0, howManyParts), (1, howManyFromLast)))

            val loopBody = If(
              (i ceq numPartitions - 1) || ((numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)) <= 0L),
              makeAnswer(i + 1, numLeft),
              Recur(
                howManyPartsToKeep,
                FastSeq(
                  i + 1,
                  numLeft - Cast(ArrayRef(partitionSizeArrayRef, i), TInt64)),
                TTuple(TInt32, TInt64)))
            If(numPartitions ceq 0,
              makeAnswer(0, 0L),
              TailLoop(
                howManyPartsToKeep,
                FastSeq(i.name -> 0, numLeft.name -> targetNumRows),
                loopBody.typ,
                loopBody))
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
                FastSeq(onlyNeededPartitions, howManyFromEachPart),
                FastSeq("part", "howMany"),
                MakeStruct(FastSeq("numberToTake" -> Ref("howMany", TInt32),
                  "old" -> Ref("part", loweredChild.ctxType))),
                ArrayZipBehavior.AssumeSameLength)
            }
          }
        }

        val bindRelationLetsNewCtx = Let(loweredChild.letBindings, ToArray(newCtxs))
        val newCtxSeq = CompileAndEvaluate(ctx, bindRelationLetsNewCtx).asInstanceOf[IndexedSeq[Any]]
        val numNewParts = newCtxSeq.length
        val newIntervals = loweredChild.partitioner.rangeBounds.slice(0, numNewParts)
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
              while (idx >= 0 && sumSoFar < targetNumRows) {
                sumSoFar += partCounts(idx)
                idx -= 1
              }
              val finalParts = (idx + 1 until partCounts.length).map { partIdx => partCounts(partIdx).toInt }.toFastSeq
              Literal(TArray(TInt32), finalParts)

            case None =>
              val partitionSizeArrayFunc = genUID()
              val howManyPartsToTryRef = Ref(genUID(), TInt32)
              val howManyPartsToTry = if (targetNumRows == 1L) 1 else 4

              val iteration = Ref(genUID(), TInt32)

              val loopBody = bindIR(
                loweredChild
                  .mapContexts(_ => StreamDrop(ToStream(childContexts), maxIR(totalNumPartitions - howManyPartsToTryRef, 0))) { ctx: IR => ctx }
                  .mapCollect(
                    "table_tail_recursive_count",
                    strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nParts="), invoke("str", TString, howManyPartsToTryRef))
                    )(StreamLen)
                ) { counts =>
                If(
                  (Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                  counts,
                  Recur(partitionSizeArrayFunc, FastSeq(howManyPartsToTryRef * 4, iteration + 1), TArray(TInt32)))
              }
              TailLoop(
                partitionSizeArrayFunc,
                FastSeq(howManyPartsToTryRef.name -> howManyPartsToTry, iteration.name -> 0),
                loopBody.typ,
                loopBody)
          }
        }

        // First element is how many partitions to keep from the right partitionSizeArrayRef, second is how many to keep from first kept element.
        def answerTuple(partitionSizeArrayRef: Ref): IR = {
          bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            val howManyPartsToDrop = genUID()
            val i = Ref(genUID(), TInt32)
            val nRowsToRight = Ref(genUID(), TInt64)
            def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple.ordered(FastSeq(howManyParts, howManyFromLast))

            val loopBody = If(
              (i ceq numPartitions) || ((nRowsToRight + Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64)) >= targetNumRows),
              makeAnswer(i, maxIR(0L, Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64) - (I64(targetNumRows) - nRowsToRight)).toI),
              Recur(
                howManyPartsToDrop,
                FastSeq(
                  i + 1,
                  nRowsToRight + Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64)),
                TTuple(TInt32, TInt32)))
            If(numPartitions ceq 0,
              makeAnswer(0, 0),
              TailLoop(
                howManyPartsToDrop,
                FastSeq(i.name -> 1, nRowsToRight.name -> 0L),
                loopBody.typ,
                loopBody))
          }
        }

        val newCtxs = bindIR(ToArray(loweredChild.contexts)) { childContexts =>
          bindIR(ArrayLen(childContexts)) { totalNumPartitions =>
            bindIR(partitionSizeArray(childContexts, totalNumPartitions)) { partitionSizeArrayRef =>
              bindIR(answerTuple(partitionSizeArrayRef)) { answerTupleRef =>
                val numPartsToKeepFromRight = GetTupleElement(answerTupleRef, 0)
                val nToDropFromFirst = GetTupleElement(answerTupleRef, 1)
                bindIR(totalNumPartitions - numPartsToKeepFromRight) { startIdx =>
                  mapIR(rangeIR(numPartsToKeepFromRight)) { idx =>
                    MakeStruct(FastSeq(
                      "numberToDrop" -> If(idx ceq 0, nToDropFromFirst, 0),
                      "old" -> ArrayRef(childContexts, idx + startIdx)))
                  }
                }
              }
            }
          }
        }

        val letBindNewCtx = Let(loweredChild.letBindings, ToArray(newCtxs))
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
            Let(
              FastSeq("global" -> lc.globals),
              mapIR(rows) { row =>
                Let(FastSeq("row" -> row), newRow)
              }
            )
          }
        } else {
          val resultUID = genUID()
          val aggs = agg.Extract(newRow, resultUID, analyses.requirednessAnalysis, isScan = true)

          val results: IR = ResultOp.makeTuple(aggs.aggs)
          val initState = RunAgg(
            Let(FastSeq("global" -> lc.globals), aggs.init),
            MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
            aggs.states
          )
          val initStateRef = Ref(genUID(), initState.typ)
          val lcWithInitBinding = lc.copy(
            letBindings = lc.letBindings ++ FastSeq((initStateRef.name, initState)),
            broadcastVals = lc.broadcastVals ++ FastSeq((initStateRef.name, initStateRef)))

          val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
            InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state)
          })
          val branchFactor = HailContext.get.branchingFactor
          val big = aggs.shouldTreeAggregate && branchFactor < lc.numPartitions
          val (partitionPrefixSumValues, transformPrefixSum): (IR, IR => IR) = if (big) {
            val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

            val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
            val writer = ETypeValueWriter(codecSpec)
            val reader = ETypeValueReader(codecSpec)
            val partitionPrefixSumFiles = lcWithInitBinding.mapCollectWithGlobals("table_scan_write_prefix_sums")({ part: IR =>
              Let(FastSeq("global" -> lcWithInitBinding.globals),
                RunAgg(
                  Begin(FastSeq(
                    initFromSerializedStates,
                    StreamFor(part,
                      "row",
                      aggs.seqPerElt
                    )
                  )),
                  WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), writer),
                  aggs.states
                ))
              // Collected is TArray of TString
            }) { case (collected, _) =>

              def combineGroup(partArrayRef: IR): IR = {
                Begin(FastSeq(
                  bindIR(ReadValue(ArrayRef(partArrayRef, 0), reader, reader.spec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                      })
                  },
                  forIR(StreamRange(1, ArrayLen(partArrayRef), 1, requiresMemoryManagementPerElement = true)) { fileIdx =>

                    bindIR(ReadValue(ArrayRef(partArrayRef, fileIdx), reader, reader.spec.encodedVirtualType)) { serializedTuple =>
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

                val loopBody = bindIR(ArrayRef(aggStack, (ArrayLen(aggStack) - 1))) { states =>
                  bindIR(ArrayLen(states)) { statesLen =>
                    If(
                      statesLen > branchFactor,
                      bindIR((statesLen + branchFactor - 1) floorDiv branchFactor) { nCombines =>
                        val contexts = mapIR(rangeIR(nCombines)) { outerIdxRef =>
                          sliceArrayIR(states, outerIdxRef * branchFactor, (outerIdxRef + 1) * branchFactor)
                        }
                        val cdaResult = cdaIR(
                          contexts, MakeStruct(FastSeq()), "table_scan_up_pass",
                          strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nStates="), invoke("str", TString, statesLen))
                        ) { case (contexts, _) =>
                          RunAgg(
                            combineGroup(contexts),
                            WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), writer),
                            aggs.states)
                        }
                        Recur(loopName, IndexedSeq(invoke("extend", TArray(TArray(TString)), aggStack, MakeArray(cdaResult)), iteration + 1), TArray(TArray(TString)))
                      },
                      aggStack)
                  }
                }
                TailLoop(
                  loopName,
                  IndexedSeq((aggStack.name, MakeArray(collected)), (iteration.name, I32(0))),
                  loopBody.typ,
                  loopBody)
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


                bindIR(WriteValue(initState, Str(tmpDir) + UUID4(), writer)) { freshState =>
                  val loopBody = If(
                    level < 0,
                    last,
                    bindIR(ArrayRef(aggStack, level)) { aggsArray =>
                      val groups = mapIR(zipWithIndex(mapIR(StreamGrouped(ToStream(aggsArray), I32(branchFactor)))(x => ToArray(x)))) { eltAndIdx =>
                        MakeStruct(FastSeq(
                          ("prev", ArrayRef(last, GetField(eltAndIdx, "idx"))),
                          ("partialSums", GetField(eltAndIdx, "elt"))))
                      }

                      val results = cdaIR(
                        groups, MakeTuple.ordered(FastSeq()), "table_scan_down_pass",
                        strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", level="), invoke("str", TString, level))
                      ) { case (context, _) =>
                        bindIR(GetField(context, "prev")) { prev =>
                          val elt = Ref(genUID(), TString)
                          ToArray(RunAggScan(
                            ToStream(GetField(context, "partialSums"), requiresMemoryManagementPerElement = true),
                            elt.name,
                            bindIR(ReadValue(prev, reader, reader.spec.encodedVirtualType)) { serializedTuple =>
                              Begin(
                                aggs.aggs.zipWithIndex.map { case (sig, i) =>
                                  InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                                })
                            },
                            bindIR(ReadValue(elt, reader, reader.spec.encodedVirtualType)) { serializedTuple =>
                              Begin(
                                aggs.aggs.zipWithIndex.map { case (sig, i) =>
                                  CombOpValue(i, GetTupleElement(serializedTuple, i), sig)
                                })
                            },
                            WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), writer),
                            aggs.states))
                        }
                      }
                      Recur(
                        downPassLoopName,
                        IndexedSeq(
                          level - 1,
                          ToArray(flatten(ToStream(results))),
                          iteration + 1),
                        TArray(TString))
                    })
                  TailLoop(
                    downPassLoopName,
                    IndexedSeq((level.name, ArrayLen(aggStack) - 1), (last.name, MakeArray(freshState)), (iteration.name, I32(0))),
                    loopBody.typ,
                    loopBody)
                }
              }
            }
            (partitionPrefixSumFiles, { (file: IR) => ReadValue(file, reader, reader.spec.encodedVirtualType) })

          } else {
            val partitionAggs = lcWithInitBinding.mapCollectWithGlobals("table_scan_prefix_sums_singlestage")({ part: IR =>
              Let(FastSeq("global" -> lc.globals),
                RunAgg(
                  Begin(FastSeq(
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
              Let(FastSeq("global" -> globals),
                ToArray(StreamTake({
                  val acc = Ref(genUID(), initStateRef.typ)
                  val value = Ref(genUID(), collected.typ.asInstanceOf[TArray].elementType)
                  StreamScan(
                    ToStream(collected, requiresMemoryManagementPerElement = true),
                    initStateRef,
                    acc.name,
                    value.name,
                    RunAgg(
                      Begin(FastSeq(
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
            letBindings = lc.letBindings ++ FastSeq((partitionPrefixSumsRef.name, partitionPrefixSumValues)),
            broadcastVals = lc.broadcastVals,
            partitioner = lc.partitioner,
            dependency = lc.dependency,
            globals = lc.globals,
            contexts = StreamZip(
              FastSeq(lc.contexts, ToStream(partitionPrefixSumsRef)),
              FastSeq(zipOldContextRef.name, zipPartAggUID.name),
              MakeStruct(FastSeq(("oldContext", zipOldContextRef), ("scanState", zipPartAggUID))),
              ArrayZipBehavior.AssertSameLength
            ),
            partition = { (partitionRef: Ref) =>
              bindIRs(GetField(partitionRef, "oldContext"), GetField(partitionRef, "scanState")) { case Seq(oldContext, rawPrefixSum) =>
                bindIR(transformPrefixSum(rawPrefixSum)) { scanState =>
                  Let(FastSeq("global" -> lc.globals),
                    RunAggScan(
                      lc.partition(oldContext),
                      "row",
                      Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
                        InitFromSerializedValue(i, GetTupleElement(scanState, i), agg.state)
                      }),
                      aggs.seqPerElt,
                      Let(FastSeq(resultUID -> results), aggs.postAggIR),
                      aggs.states
                    )
                  )
                }
              }
            }
          )
        }

      case t@TableKeyBy(child, newKey, isSorted: Boolean) =>
        require(t.definitelyDoesNotShuffle)
        val loweredChild = lower(child)

        val nPreservedFields = loweredChild.kType.fieldNames
          .zip(newKey)
          .takeWhile { case (l, r) => l == r }
          .length

        loweredChild.changePartitionerNoRepartition(loweredChild.partitioner.coarsen(nPreservedFields))
          .extendKeyPreservesPartitioning(ctx, newKey)

      case TableLeftJoinRightDistinct(left, right, root) =>
        val commonKeyLength = right.typ.keyType.size
        val loweredLeft = lower(left)
        val loweredRight = lower(right)

        loweredLeft.alignAndZipPartitions(
          ctx,
          loweredRight,
          commonKeyLength,
          (lGlobals, _) => lGlobals,
          (leftPart, rightPart) => {
            val leftElementRef = Ref(genUID(), left.typ.rowType)
            val rightElementRef = Ref(genUID(), right.typ.rowType)

            val (typeOfRootStruct, _) = right.typ.rowType.filterSet(right.typ.key.toSet, false)
            val rootStruct = SelectFields(rightElementRef, typeOfRootStruct.fieldNames.toIndexedSeq)
            val joiningOp = InsertFields(leftElementRef, FastSeq(root -> rootStruct))
            StreamJoinRightDistinct(
              leftPart, rightPart,
              left.typ.key.take(commonKeyLength), right.typ.key,
              leftElementRef.name, rightElementRef.name,
              joiningOp, "left")
          })

      case TableIntervalJoin(left, right, root, product) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)

        loweredLeft.intervalAlignAndZipPartitions(ctx,
          loweredRight,
          analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable].rowType,
          (lGlobals, _) => lGlobals,
          { (lstream, rstream) =>
            val lref = Ref(genUID(), elementType(left.typ))
            if (product) {
              val rref = Ref(genUID(), TArray(elementType(right.typ)))
              StreamLeftIntervalJoin(
                lstream, rstream,
                left.typ.key,
                right.typ.keyType.fields.find(_.typ.isInstanceOf[TInterval]).map(_.name).getOrElse {
                  val msg = s"Right table key fields does not contain an interval type '${right.typ.keyType}'."
                  throw new UnsupportedOperationException(msg)
                },
                lref.name, rref.name,
                InsertFields(lref, FastSeq(root -> rref))
              )
            } else {
              val rref = Ref(genUID(), elementType(right.typ))
              StreamJoinRightDistinct(
                lstream, rstream,
                loweredLeft.key, loweredRight.key,
                lref.name, rref.name,
                InsertFields(lref, FastSeq(root -> SelectFields(rref, right.typ.valueType.fieldNames))),
                "left"
              )
            }
          }
        )

      case tj@TableJoin(left, right, joinType, joinKey) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)
        LowerTableIRHelpers.lowerTableJoin(ctx, analyses, tj, loweredLeft, loweredRight)

      case x@TableUnion(children) =>
        val lowered = children.map(lower)
        val keyType = x.typ.keyType

        if (keyType.size == 0) {
          TableStage.concatenate(ctx, lowered)
        } else {
          val newPartitioner = RVDPartitioner.generate(ctx.stateManager, keyType, lowered.flatMap(_.partitioner.rangeBounds))
          val repartitioned = lowered.map(_.repartitionNoShuffle(ctx, newPartitioner))

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
        }

      case x@TableMultiWayZipJoin(children, fieldName, globalName) =>
        val lowered = children.map(lower)
        val keyType = x.typ.keyType
        val newPartitioner = RVDPartitioner.generate(ctx.stateManager, keyType, lowered.flatMap(_.partitioner.rangeBounds))
        val repartitioned = lowered.map(_.repartitionNoShuffle(ctx, newPartitioner))
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

      case t@TableOrderBy(child, sortFields) =>
        require(t.definitelyDoesNotShuffle)
        val loweredChild = lower(child)
        loweredChild.changePartitionerNoRepartition(RVDPartitioner.unkeyed(ctx.stateManager, loweredChild.partitioner.numPartitions))

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
            Let(refs.tail.zip(roots).map { case (ref, root) => ref.name -> root },
              mapIR(ToStream(refs.last, true)) { elt =>
                path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
                  InsertFields(ref, FastSeq(p -> inserted))
                }
              })
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

      case TableMapPartitions(child, globalName, partitionStreamName, body, _, allowedOverlap) =>
        val loweredChild = lower(child).strictify(ctx, allowedOverlap)

        loweredChild.mapPartition(Some(child.typ.key)) { part =>
          Let(FastSeq(globalName -> loweredChild.globals, partitionStreamName -> part), body)
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

      case TableToTableApply(child, WrappedMatrixToTableFunction(localLDPrune: LocalLDPrune, colsFieldName, entriesFieldName, _)) =>
        val lc = lower(child)
        lc.mapPartition(Some(child.typ.key)) { rows =>
          localLDPrune.makeStream(rows, entriesFieldName, ArrayLen(GetField(lc.globals, colsFieldName)))
        }.mapGlobals(_ => makestruct())

      case bmtt@BlockMatrixToTable(bmir) =>
        val ts = LowerBlockMatrixIR.lowerToTableStage(bmir, typesToLower, ctx, analyses)
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
        throw new LowererUnsupportedOperation(s"undefined: \n${Pretty(ctx, node)}")
    }

    assert(tir.typ.globalType == lowered.globalType, s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}")
    assert(tir.typ.rowType == lowered.rowType, s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}")
    assert(tir.typ.keyType.isPrefixOf(lowered.kType), s"\n  ir key: ${tir.typ.key}\n  lowered key: ${lowered.key}")

    lowered
  }

  /* We have a couple of options when repartitioning a table:
   *  1. Send only the contexts needed to compute each new partition and
   *     take/drop the rows that fall in that partition.
   *  2. Compute the table with the old partitioner, write the table to cloud
   *     storage then read the new partitions from the index.
   *
   * We'd like to do 1 as keeping things in memory (with perhaps a bit of work
   * duplication) is generally less expensive than writing and reading a table
   * to and from cloud storage. There comes a cross-over point, however, where
   * it's cheaper to do the latter. One such example is as follows: consider a
   * repartitioning where the same context is used to compute multiple
   * partitions. The (parallel) computation of each partition involves at least
   * all of the work to compute the previous partition:
   *
   *                  *----------------------*
   *           in:    |                      |  ...
   *                  *----------------------*
   *                      /    |         \
   *                     /     |          \
   *                   *--*  *---*       *--*
   *          out:     |  |  |   |  ...  |  |
   *                   *--*  *---*       *--*
   *
   * We can estimate the relative cost of computing the new partitions vs
   * spilling as being proportional to the mean number of old partitions
   * used to compute new partitions.
   */
  def isRepartitioningCheap(original: RVDPartitioner, planned: RVDPartitioner): Boolean = {
    val cost =
      if (original.numPartitions == 0)
        0.0
      else
        (0.0167 / original.numPartitions) * planned
          .rangeBounds
          .map { intrvl => val (lo, hi) = original.intervalRange(intrvl); hi - lo }
          .sum

    log.info(s"repartition cost: $cost")
    cost <= 1.0
  }
}
