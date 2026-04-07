package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichArray, toRichIterable}
import is.hail.expr.ir._
import is.hail.expr.ir.agg.AggExecuteContextExtensions
import is.hail.expr.ir.analyses.PartitionCounts
import is.hail.expr.ir.defs._
import is.hail.expr.ir.defs.ArrayZipBehavior.AssertSameLength
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToTableFunction}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.methods.{ForceCountTable, LocalLDPrune, NPartitionsTable, TableFilterPartitions}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types._
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils._

import scala.collection.compat._

import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

object TableStage {
  def apply(
    globals: IR,
    partitioner: RVDPartitioner,
    dependency: TableStageDependency,
    contexts: IR,
    body: TrivialIR => IR,
  ): TableStage = {
    val globalsRef = Ref(freshName(), globals.typ)
    TableStage(
      FastSeq(globalsRef.name -> globals),
      FastSeq(globalsRef.name -> globalsRef),
      globalsRef.clone,
      partitioner,
      dependency,
      contexts,
      body,
    )
  }

  def apply(
    letBindings: IndexedSeq[(Name, IR)],
    broadcastVals: IndexedSeq[(Name, IR)],
    globals: TrivialIR,
    partitioner: RVDPartitioner,
    dependency: TableStageDependency,
    contexts: IR,
    partition: TrivialIR => IR,
  ): TableStage = {
    val ctxType = contexts.typ.asInstanceOf[TStream].elementType
    val ctxRef = Ref(freshName(), ctxType)

    new TableStage(
      letBindings,
      broadcastVals,
      globals,
      partitioner,
      dependency,
      contexts,
      ctxRef.name,
      partition(ctxRef),
    )
  }

  def concatenate(ctx: ExecuteContext, children: IndexedSeq[TableStage]): TableStage = {
    val keyType = children.head.kType
    assert(keyType.size == 0)
    assert(children.forall(_.kType == keyType))

    val ctxType = TTuple(children.map(_.ctxType): _*)
    val ctxArrays = children.view.zipWithIndex.map { case (child, idx) =>
      ToArray(mapIR(child.contexts) { ctx =>
        MakeTuple.ordered(children.indices.map { idx2 =>
          if (idx == idx2) ctx.clone else NA(children(idx2).ctxType)
        })
      })
    }.toFastSeq
    val ctxs = flatMapIR(MakeStream(ctxArrays, TStream(TArray(ctxType)))) { ctxArray =>
      ToStream(ctxArray)
    }

    val newGlobals = children.head.globals
    val globalsRef = Ref(freshName(), newGlobals.typ)
    val newPartitioner =
      new RVDPartitioner(ctx.stateManager, keyType, children.flatMap(_.partitioner.rangeBounds))

    TableStage(
      children.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals.clone,
      children.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
      globalsRef.clone,
      newPartitioner,
      TableStageDependency.union(children.map(_.dependency)),
      ctxs,
      ctxRef =>
        StreamMultiMerge(
          children.indices.map { i =>
            bindIR(GetTupleElement(ctxRef, i)) { ctx =>
              If(
                IsNA(ctx),
                MakeStream(IndexedSeq(), TStream(children(i).rowType)),
                children(i).partition(ctx),
              )
            }
          },
          IndexedSeq(),
        ),
    )
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
  val letBindings: IndexedSeq[(Name, IR)],
  val broadcastVals: IndexedSeq[(Name, IR)],
  val globals: TrivialIR,
  val partitioner: RVDPartitioner,
  val dependency: TableStageDependency,
  val contexts: IR,
  val ctxRefName: Name,
  val partitionIR: IR,
) extends Logging {
  self =>

  contexts.typ match {
    case TStream(t) if t.isRealizable =>
    case t =>
      throw new IllegalArgumentException(s"TableStage constructed with illegal context type $t")
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

  assert(globals match {
    case r: Ref => broadcastVals.exists { case (n, v) => n == r.name && v == r }
    case _ => false
  })

  def copy(
    letBindings: IndexedSeq[(Name, IR)] = letBindings,
    broadcastVals: IndexedSeq[(Name, IR)] = broadcastVals,
    globals: TrivialIR = globals,
    partitioner: RVDPartitioner = partitioner,
    dependency: TableStageDependency = dependency,
    contexts: IR = contexts,
    ctxRefName: Name = ctxRefName,
    partitionIR: IR = partitionIR,
  ): TableStage =
    new TableStage(letBindings, broadcastVals, globals, partitioner, dependency, contexts,
      ctxRefName, partitionIR)

  def partition(ctx: TrivialIR): IR = {
    require(ctx.typ == ctxType)
    Let(FastSeq(ctxRefName -> ctx), partitionIR)
  }

  def numPartitions: Int = partitioner.numPartitions

  def mapPartition(newKey: Option[IndexedSeq[String]])(f: TrivialIR => IR): TableStage = {
    val part = newKey match {
      case Some(k) =>
        if (!partitioner.kType.fieldNames.startsWith(k))
          throw new RuntimeException(s"cannot map partitions to new key!" +
            s"\n  prev key: ${partitioner.kType.fieldNames.toSeq}" +
            s"\n  new key:  $k")
        partitioner.coarsen(k.length)
      case None => partitioner
    }
    copy(partitionIR = bindIR(partitionIR)(f), partitioner = part)
  }

  def zipPartitions(
    right: TableStage,
    newGlobals: (TrivialIR, TrivialIR) => IR,
    body: (TrivialIR, TrivialIR) => IR,
  ): TableStage = {
    val left = this
    val leftCtxTyp = left.ctxType
    val rightCtxTyp = right.ctxType

    val leftCtxRef = Ref(freshName(), leftCtxTyp)
    val rightCtxRef = Ref(freshName(), rightCtxTyp)

    val leftCtxStructField = genUID()
    val rightCtxStructField = genUID()

    val zippedCtxs = StreamZip(
      FastSeq(left.contexts, right.contexts),
      FastSeq(leftCtxRef.name, rightCtxRef.name),
      MakeStruct(FastSeq(leftCtxStructField -> leftCtxRef, rightCtxStructField -> rightCtxRef)),
      ArrayZipBehavior.AssertSameLength,
    )

    val globals = newGlobals(left.globals, right.globals)
    val globalsRef = Ref(freshName(), globals.typ)

    TableStage(
      left.letBindings ++ right.letBindings :+ (globalsRef.name -> globals),
      left.broadcastVals ++ right.broadcastVals :+ (globalsRef.name -> globalsRef),
      globalsRef.clone,
      left.partitioner,
      left.dependency.union(right.dependency),
      zippedCtxs,
      ctxRef =>
        IRBuilder.scoped { b =>
          val lctx = b.memoize(GetField(ctxRef, leftCtxStructField))
          val lpart = b.memoize(left.partition(lctx))
          val rctx = b.memoize(GetField(ctxRef, rightCtxStructField))
          val rpart = b.memoize(right.partition(rctx))
          body(lpart, rpart)
        },
    )
  }

  def mapPartitionWithContext(f: (TrivialIR, TrivialIR) => IR): TableStage =
    copy(partitionIR = bindIR(partitionIR)(f(_, Ref(ctxRefName, ctxType))))

  def mapContexts(f: TrivialIR => IR)(getOldContext: TrivialIR => IR): TableStage = {
    val newContexts = bindIR(contexts)(f)
    val newCtxRef = Ref(freshName(), elementType(newContexts.typ))
    copy(
      contexts = newContexts,
      ctxRefName = newCtxRef.name,
      partitionIR = bindIR(getOldContext(newCtxRef))(partition),
    )
  }

  def mapGlobals(f: TrivialIR => IR): TableStage = {
    val newGlobals = f(globals)
    val globalsRef = Ref(freshName(), newGlobals.typ)

    copy(
      letBindings = letBindings :+ globalsRef.name -> newGlobals,
      broadcastVals = broadcastVals :+ globalsRef.name -> globalsRef,
      globals = globalsRef.clone,
    )
  }

  def mapCollect(staticID: String, dynamicID: IR = NA(TString))(f: TrivialIR => IR): IR =
    mapCollectWithGlobals(staticID, dynamicID)(f)((parts, _) => parts)

  def mapCollectWithGlobals(
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    mapF: TrivialIR => IR
  )(
    body: (TrivialIR, TrivialIR) => IR
  ): IR =
    mapCollectWithContextsAndGlobals(staticID, dynamicID)((part, _) => mapF(part))(body)

  // mapf is (part, ctx) => ???, body is (parts, globals) => ???
  def mapCollectWithContextsAndGlobals(
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    mapF: (TrivialIR, TrivialIR) => IR
  )(
    body: (TrivialIR, TrivialIR) => IR
  ): IR = {

    val broadcastRefs = MakeStruct(broadcastVals.map { case (n, ir) => n.str -> ir })

    val glob = Ref(freshName(), broadcastRefs.typ)
    val cda = CollectDistributedArray(
      contexts,
      broadcastRefs,
      ctxRefName,
      glob.name,
      Let(
        broadcastVals.map { case (name, _) => name -> GetField(glob.clone, name.str) },
        bindIR(partitionIR)(mapF(_, Ref(ctxRefName, ctxType))),
      ),
      dynamicID,
      staticID,
      Some(dependency),
    )

    Let(letBindings, bindIR(cda)(cdaRef => body(cdaRef, globals)))
  }

  def collectWithGlobals(staticID: String, dynamicID: IR = NA(TString)): IR =
    mapCollectWithGlobals(staticID, dynamicID)(ToArray(_)) { (parts, globals) =>
      MakeStruct(FastSeq(
        "rows" -> ToArray(flatMapIR(ToStream(parts))(ToStream(_))),
        "global" -> globals,
      ))
    }

  def countPerPartition(): IR =
    mapCollect("count_per_partition")(part => Cast(StreamLen(part), TInt64))

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

  def repartitionNoShuffle(
    ec: ExecuteContext,
    newPartitioner: RVDPartitioner,
    allowDuplication: Boolean = false,
    dropEmptyPartitions: Boolean = false,
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
      if (
        startAndEnd.forall { case ((start, end), i) =>
          start + 1 == end &&
          newPartitioner.rangeBounds(start).includes(
            newPartitioner.kord,
            partitioner.rangeBounds(i),
          )
        }
      ) {
        val newToOld = startAndEnd.groupBy(_._1._1).map { case (newIdx, values) =>
          (newIdx, values.map(_._2).sorted.toIndexedSeq)
        }

        val (oldPartIndices, newPartitionerFilt) =
          if (dropEmptyPartitions) {
            val indices = (0 until newPartitioner.numPartitions).filter(newToOld.contains)
            (
              indices.map(newToOld),
              newPartitioner.copy(rangeBounds = indices.map(newPartitioner.rangeBounds)),
            )
          } else
            (
              (0 until newPartitioner.numPartitions).map(i => newToOld.getOrElse(i, FastSeq())),
              newPartitioner,
            )

        logger.info(
          "repartitionNoShuffle - fast path," +
            s" generated ${oldPartIndices.length} partitions from ${partitioner.numPartitions}" +
            s" (dropped ${newPartitioner.numPartitions - oldPartIndices.length} empty output parts)"
        )

        val newContexts = bindIR(ToArray(contexts)) { oldCtxs =>
          mapIR(ToStream(Literal(TArray(TArray(TInt32)), oldPartIndices))) { inds =>
            ToArray(mapIR(ToStream(inds))(i => ArrayRef(oldCtxs, i)))
          }
        }

        return TableStage(
          letBindings,
          broadcastVals,
          globals,
          newPartitionerFilt,
          dependency,
          newContexts,
          ctx => flatMapIR(ToStream(ctx, true))(oldCtx => partition(oldCtx)),
        )
      }

      val boundType = RVDPartitioner.intervalIRRepresentation(newPartitioner.kType)
      val partitionMapping: IndexedSeq[Row] = newPartitioner.rangeBounds.map { i =>
        Row(
          RVDPartitioner.intervalToIRRepresentation(i, newPartitioner.kType.size),
          partitioner.queryInterval(i),
        )
      }
      val partitionMappingType = TStruct(
        "partitionBound" -> boundType,
        "parentPartitions" -> TArray(TInt32),
      )

      val newContexts =
        bindIR(ToArray(contexts)) { ctx =>
          mapIR(ToStream(Literal(TArray(partitionMappingType), partitionMapping))) { mapping =>
            makestruct(
              "partitionBound" -> GetField(mapping, "partitionBound"),
              "oldContexts" -> ToArray(
                mapIR(ToStream(GetField(mapping, "parentPartitions"))) {
                  idx => ArrayRef(ctx, idx)
                }
              ),
            )
          }
        }

      TableStage(
        letBindings,
        broadcastVals,
        globals,
        newPartitioner,
        dependency,
        newContexts,
        { ctxRef =>
          bindIR(GetField(ctxRef, "partitionBound")) { interval =>
            takeWhile(
              dropWhile(
                flatMapIR(ToStream(GetField(ctxRef, "oldContexts"), true))(partition)
              ) { elt =>
                invoke(
                  "pointLessThanPartitionIntervalLeftEndpoint",
                  TBoolean,
                  SelectFields(elt, newPartitioner.kType.fieldNames),
                  invoke("start", boundType.pointType, interval),
                  invoke("includesStart", TBoolean, interval),
                )
              }
            ) { elt =>
              invoke(
                "pointLessThanPartitionIntervalRightEndpoint",
                TBoolean,
                SelectFields(elt, newPartitioner.kType.fieldNames),
                invoke("end", boundType.pointType, interval),
                invoke("includesEnd", TBoolean, interval),
              )
            }
          }
        },
      )
    } else {
      val location = ec.createTmpPath(genUID())
      CompileAndEvaluate[Unit](
        ec,
        TableNativeWriter(location).lower(ec, this, RTable.fromTableStage(ec, this)),
      )

      val newTableType = TableType(rowType, newPartitioner.kType.fieldNames, globalType)
      val reader = TableNativeReader.read(
        ec.fs,
        location,
        Some(NativeReaderOptions(
          intervals = newPartitioner.rangeBounds,
          intervalPointType = newPartitioner.kType,
          filterIntervals = dropEmptyPartitions,
        )),
      )

      val table = TableRead(newTableType, dropRows = false, tr = reader)
      LowerTableIR.applyTable(table, DArrayLowering.All, ec, LoweringAnalyses.apply(table, ec))
    }

    assert(
      newStage.rowType == rowType,
      s"repartitioned row type: ${newStage.rowType}\n" +
        s"          old row type: $rowType",
    )
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
    globalJoiner: (TrivialIR, TrivialIR) => IR,
    joiner: (TrivialIR, TrivialIR) => IR,
    rightKeyIsDistinct: Boolean = false,
  ): TableStage = {
    assert(this.kType.truncate(joinKey).isJoinableWith(right.kType.truncate(joinKey)))

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
            leftPart.rangeBounds ++ rightPart.rangeBounds,
          )
      }
    }
    val repartitionedLeft: TableStage = repartitionNoShuffle(ec, newPartitioner)

    val partitionJoiner: (TrivialIR, TrivialIR) => IR = (lPart, rPart) => {
      val lEltType = lPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
      val rEltType = rPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]

      val lKey = this.kType.fieldNames.take(joinKey)
      val rKey = right.kType.fieldNames.take(joinKey)

      val lEltRef = Ref(freshName(), lEltType)
      val rEltRef = Ref(freshName(), rEltType)

      StreamJoin(
        lPart,
        rPart,
        lKey,
        rKey,
        lEltRef.name,
        rEltRef.name,
        joiner(lEltRef, rEltRef),
        joinType,
        requiresMemoryManagement = true,
        rightKeyIsDistinct = rightKeyIsDistinct,
      )
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
    globalJoiner: (TrivialIR, TrivialIR) => IR,
    joiner: (TrivialIR, TrivialIR) => IR,
  ): TableStage = {
    require(joinKey <= kType.size)
    require(joinKey <= right.kType.size)

    val leftKeyToRightKeyMap =
      (kType.fieldNames.take(joinKey) lazyZip right.kType.fieldNames.take(joinKey)).toMap
    val newRightPartitioner = partitioner.coarsen(joinKey).rename(leftKeyToRightKeyMap)
    val repartitionedRight =
      right.repartitionNoShuffle(ec, newRightPartitioner, allowDuplication = true)
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
    globalJoiner: (TrivialIR, TrivialIR) => IR,
    joiner: (TrivialIR, TrivialIR) => IR,
  ): TableStage = {
    require(right.kType.size == 1)
    val rightKeyType = right.kType.fields.head.typ
    require(rightKeyType.isInstanceOf[TInterval])
    require(rightKeyType.asInstanceOf[TInterval].pointType == kType.types.head)

    val irPartitioner = partitioner.coarsen(1).partitionBoundsIRRepresentation

    val rightWithPartNums = right.mapPartition(None) { partStream =>
      flatMapIR(partStream) { row =>
        val interval = bindIR(GetField(row, right.key.head)) { interval =>
          invoke(
            "Interval",
            TInterval(TTuple(kType.typeAfterSelect(ArraySeq(0)), TInt32)),
            MakeTuple.ordered(FastSeq(
              MakeStruct(FastSeq(kType.fieldNames.head -> invoke(
                "start",
                kType.types.head,
                interval,
              ))),
              I32(1),
            )),
            MakeTuple.ordered(FastSeq(
              MakeStruct(FastSeq(kType.fieldNames.head -> invoke(
                "end",
                kType.types.head,
                interval,
              ))),
              I32(1),
            )),
            invoke("includesStart", TBoolean, interval),
            invoke("includesEnd", TBoolean, interval),
          )
        }

        bindIR(invoke(
          "partitionerFindIntervalRange",
          TTuple(TInt32, TInt32),
          irPartitioner,
          interval,
        )) { range =>
          val rangeStream = StreamRange(
            GetTupleElement(range, 0),
            GetTupleElement(range, 1),
            I32(1),
            requiresMemoryManagementPerElement = true,
          )
          mapIR(rangeStream)(partNum => InsertFields(row, FastSeq("__partNum" -> partNum)))
        }
      }
    }

    val rightRowRTypeWithPartNum =
      IndexedSeq("__partNum" -> TypeWithRequiredness(TInt32)) ++ rightRowRType.fields.map(rField =>
        rField.name -> rField.typ
      )
    val rightTableRType = RTable(rightRowRTypeWithPartNum, FastSeq(), right.key)
    val sortedReader = ctx.backend.lowerDistributedSort(
      ctx,
      rightWithPartNums,
      SortField("__partNum", Ascending) +: right.key.map(k => SortField(k, Ascending)),
      rightTableRType,
    )
    val sorted = sortedReader.lower(ctx, sortedReader.fullType)
    assert(sorted.kType.fieldNames.sameElements("__partNum" +: right.key))
    val newRightPartitioner = new RVDPartitioner(
      ctx.stateManager,
      Some(1),
      TStruct.concat(TStruct("__partNum" -> TInt32), right.kType),
      ArraySeq.tabulate[Interval](partitioner.numPartitions)(i =>
        Interval(Row(i), Row(i), true, true)
      ),
    )
    val repartitioned = sorted.repartitionNoShuffle(ctx, newRightPartitioner)
      .changePartitionerNoRepartition(RVDPartitioner.unkeyed(
        ctx.stateManager,
        newRightPartitioner.numPartitions,
      ))
      .mapPartition(None) { part =>
        mapIR(part)(row => SelectFields(row, right.rowType.fieldNames))
      }
    zipPartitions(repartitioned, globalJoiner, joiner)
  }
}

object LowerTableIR extends Logging {
  def apply(
    ir: IR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): IR = {
    def lower(tir: TableIR): TableStage =
      this.applyTable(tir, typesToLower, ctx, analyses)

    val lowered = ir match {
      case TableCount(tableIR) =>
        val stage = lower(tableIR)
        invoke("sum", TInt64, stage.countPerPartition())

      case TableToValueApply(child, ForceCountTable()) =>
        val stage = lower(child)
        invoke(
          "sum",
          TInt64,
          stage.mapCollect("table_force_count")(rows =>
            foldIR(mapIR(rows)(row => Consume(row)), 0L)(_.clone + _)
          ),
        )

      case TableToValueApply(child, TableCalculateNewPartitions(nPartitions)) =>
        val stage = lower(child)
        val sampleSize = math.min((nPartitions * 20 + 256), 1000000)
        val samplesPerPartition = sampleSize / math.max(1, stage.numPartitions)
        val keyType = child.typ.keyType

        bindIR(flatten(stage.mapCollect("table_calculate_new_partitions") { rows =>
          streamAggIR(mapIR(rows)(row => SelectFields(row, keyType.fieldNames))) { elt =>
            ToArray(flatMapIR(ToStream(
              MakeArray(
                ApplyAggOp(FastSeq(I32(samplesPerPartition)), FastSeq(elt), ReservoirSample()),
                ApplyAggOp(FastSeq(I32(1)), FastSeq(elt, elt), TakeBy()),
                ApplyAggOp(FastSeq(I32(1)), FastSeq(elt, elt), TakeBy(Descending)),
              )
            ))(inner => ToStream(inner)))
          }
        })) { partData =>
          val sorted = sortIR(partData)((l, r) => ApplyComparisonOp(LT, l, r))
          bindIR(ToArray(flatMapIR(StreamGroupByKey(
            ToStream(sorted),
            keyType.fieldNames,
            missingEqual = true,
          ))(groupRef => StreamTake(groupRef, 1)))) { boundsArray =>
            bindIR(ArrayLen(boundsArray)) { nBounds =>
              bindIR(minIR(nBounds, nPartitions)) { nParts =>
                If(
                  nParts.ceq(0),
                  MakeArray(FastSeq(), TArray(TInterval(keyType))),
                  bindIR((nBounds.clone + nParts - 1) floorDiv nParts) { stepSize =>
                    ToArray(mapIR(StreamRange(0, nBounds, stepSize)) { i =>
                      If(
                        (i.clone + stepSize) < (nBounds.clone - 1),
                        invoke(
                          "Interval",
                          TInterval(keyType),
                          ArrayRef(boundsArray, i),
                          ArrayRef(boundsArray, i.clone + stepSize),
                          True(),
                          False(),
                        ),
                        invoke(
                          "Interval",
                          TInterval(keyType),
                          ArrayRef(boundsArray, i),
                          ArrayRef(boundsArray, nBounds.clone - 1),
                          True(),
                          True(),
                        ),
                      )
                    })
                  },
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
        val aggs = agg.Extract(ctx, query, analyses.requirednessAnalysis).independent
        val aggSigs = aggs.sigs

        val lc = lower(child)

        val initState = Let(
          FastSeq(TableIR.globalName -> lc.globals),
          RunAgg(aggs.init, aggSigs.valuesOp, aggSigs.states),
        )

        val initStateRef = Ref(freshName(), initState.typ)
        val lcWithInitBinding = lc.copy(
          letBindings = lc.letBindings ++ FastSeq(initStateRef.name -> initState),
          broadcastVals = lc.broadcastVals ++ FastSeq(initStateRef.name -> initStateRef),
        )

        def initFromSerializedStates = aggSigs.initFromSerializedValueOp(initStateRef)

        val branchFactor = ctx.branchingFactor
        val useTreeAggregate = aggSigs.shouldTreeAggregate && branchFactor < lc.numPartitions
        val isCommutative = aggSigs.isCommutative
        logger.info(s"Aggregate: useTreeAggregate=$useTreeAggregate")
        logger.info(s"Aggregate: commutative=$isCommutative")

        if (useTreeAggregate) {
          val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

          val codecSpec = TypedCodecSpec(
            ctx,
            PCanonicalTuple(true, Seq.fill(aggSigs.nAggs)(PCanonicalBinary(true)): _*),
            BufferSpec.wireSpec,
          )
          val writer = ETypeValueWriter(codecSpec)
          val reader = ETypeValueReader(codecSpec)
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate") { part =>
            Let(
              FastSeq(TableIR.globalName -> lc.globals),
              RunAgg(
                Begin(FastSeq(
                  initFromSerializedStates,
                  StreamFor(part, TableIR.rowName, aggs.seqPerElt),
                )),
                WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                aggSigs.states,
              ),
            )
          } { case (collected, globals) =>
            def combineGroup(partArrayRef: TrivialIR, useInitStates: Boolean): IR =
              Begin(FastSeq(
                if (useInitStates) initFromSerializedStates
                else {
                  bindIR(ReadValue(
                    ArrayRef(partArrayRef, 0),
                    reader,
                    reader.spec.encodedVirtualType,
                  ))(aggSigs.initFromSerializedValueOp)
                },
                forIR(StreamRange(
                  if (useInitStates) 0 else 1,
                  ArrayLen(partArrayRef),
                  1,
                  requiresMemoryManagementPerElement = true,
                )) { fileIdx =>
                  bindIR(ReadValue(
                    ArrayRef(partArrayRef, fileIdx),
                    reader,
                    reader.spec.encodedVirtualType,
                  ))(aggSigs.combOpValues)
                },
              ))

            val treeAggregation =
              tailLoop(TArray(TString), collected, 0) {
                case (recur, Seq(currentAggStates, iterNumber)) =>
                  If(
                    ArrayLen(currentAggStates) <= I32(branchFactor),
                    currentAggStates,
                    recur(
                      FastSeq(
                        cdaIR(
                          mapIR(StreamGrouped(ToStream(currentAggStates), I32(branchFactor)))(
                            ToArray(_)
                          ),
                          makestruct(),
                          "table_tree_aggregate",
                          strConcat(
                            Str("iteration="),
                            invoke("str", TString, iterNumber),
                            Str(", n_states="),
                            invoke("str", TString, ArrayLen(currentAggStates)),
                          ),
                        ) { (context, _) =>
                          RunAgg(
                            combineGroup(context, false),
                            WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                            aggSigs.states,
                          )
                        },
                        iterNumber.clone + 1,
                      )
                    ),
                  )
              }

            bindIR(treeAggregation) { finalParts =>
              RunAgg(
                combineGroup(finalParts, true),
                Let(
                  FastSeq(TableIR.globalName -> globals),
                  aggs.result,
                ),
                aggSigs.states,
              )
            }
          }
        } else {
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate_singlestage") { part =>
            Let(
              FastSeq(TableIR.globalName -> lc.globals),
              RunAgg(
                Begin(FastSeq(
                  initFromSerializedStates,
                  StreamFor(part, TableIR.rowName, aggs.seqPerElt),
                )),
                aggSigs.valuesOp,
                aggSigs.states,
              ),
            )
          } { case (collected, globals) =>
            Let(
              FastSeq(TableIR.globalName -> globals),
              RunAgg(
                Begin(FastSeq(
                  initFromSerializedStates,
                  forIR(ToStream(collected, requiresMemoryManagementPerElement = true))(
                    aggSigs.combOpValues
                  ),
                )),
                aggs.result,
                aggSigs.states,
              ),
            )
          }
        }

      case TableToValueApply(child, NPartitionsTable()) =>
        lower(child).getNumPartitions()

      case TableWrite(child, writer) =>
        writer.lower(
          ctx,
          lower(child),
          tcoerce[RTable](analyses.requirednessAnalysis.lookup(child)),
        )

      case TableMultiWrite(children, writer) =>
        writer.lower(
          ctx,
          children.map(child =>
            (lower(child), tcoerce[RTable](analyses.requirednessAnalysis.lookup(child)))
          ),
        )

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(
          s"IR nodes with TableIR children must be defined explicitly: \n${Pretty(ctx, node)}"
        )
    }

    NormalizeNames()(ctx, lowered)
  }

  def applyTable(
    tir: TableIR,
    typesToLower: DArrayLowering.Type,
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
  ): TableStage = {
    def lowerIR(ir: IR): IR =
      LowerToCDA.lower(ir, typesToLower, ctx, analyses)

    def lower(tir: TableIR): TableStage =
      this.applyTable(tir, typesToLower, ctx, analyses)

    if (typesToLower == DArrayLowering.BMOnly)
      throw new LowererUnsupportedOperation(
        "found TableIR in lowering; lowering only BlockMatrixIRs."
      )

    val typ: TableType = tir.typ

    val lowered: TableStage = tir match {
      case TableRead(typ, dropRows, reader) =>
        reader.lower(ctx, typ, dropRows)

      case TableParallelize(rowsAndGlobal, nPartitions) =>
        val nPartitionsAdj = nPartitions.getOrElse(ctx.backend.defaultParallelism)

        val loweredRowsAndGlobal = lowerIR(rowsAndGlobal)
        val loweredRowsAndGlobalRef = Ref(freshName(), loweredRowsAndGlobal.typ)

        val context =
          IRBuilder.scoped { b =>
            val rows = b.memoize(GetField(loweredRowsAndGlobalRef, "rows"))
            val numRowsRef = b.memoize(ArrayLen(rows))
            val indicesArray = b.memoize(
              invoke(
                "extend",
                TArray(TInt32),
                ToArray(mapIR(rangeIR(nPartitionsAdj)) { partIdx =>
                  (partIdx * numRowsRef) floorDiv nPartitionsAdj
                }),
                MakeArray(numRowsRef),
              )
            )

            mapIR(rangeIR(nPartitionsAdj)) { partIdx =>
              ToArray(mapIR(rangeIR(
                ArrayRef(indicesArray, partIdx),
                ArrayRef(indicesArray, partIdx.clone + 1),
              ))(rowIdx => ArrayRef(rows, rowIdx)))
            }
          }

        val globalsRef = Ref(freshName(), typ.globalType)
        TableStage(
          FastSeq(
            loweredRowsAndGlobalRef.name -> loweredRowsAndGlobal,
            globalsRef.name -> GetField(loweredRowsAndGlobalRef.clone, "global"),
          ),
          FastSeq(globalsRef.name -> globalsRef),
          globalsRef.clone,
          RVDPartitioner.unkeyed(ctx.stateManager, nPartitionsAdj),
          TableStageDependency.none,
          context,
          ctxRef => ToStream(ctxRef, true),
        )

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
                val ctxs = ToStream(If(
                  len ceq partitioner.numPartitions,
                  ref, {
                    val dieMsg = strConcat(
                      s"TableGen: partitioner contains ${partitioner.numPartitions} partitions,",
                      " got ",
                      len,
                      " contexts.",
                    )
                    Die(dieMsg, ref.typ, errorId)
                  },
                ))

                // [FOR KEYED TABLES ONLY]
                // AFAIK, there's no way to guarantee that the rows generated in the
                // body conform to their partition's range bounds at compile time so
                // assert this at runtime in the body before it wreaks havoc upon the world.
                val partitionIdx = StreamRange(I32(0), I32(partitioner.numPartitions), I32(1))
                val bounds = Literal(
                  TArray(TInterval(partitioner.kType)),
                  partitioner.rangeBounds,
                )
                zipIR(FastSeq(partitionIdx, ToStream(bounds), ctxs), AssertSameLength, errorId)(
                  elems => MakeTuple.ordered(elems.map(_.clone))
                )
              }
            }
          },
          body = in =>
            lowerIR {
              val rows =
                Let(FastSeq(cname -> GetTupleElement(in, 2), gname -> loweredGlobals), body)
              if (partitioner.kType.fields.isEmpty) rows
              else bindIR(GetTupleElement(in, 1)) { interval =>
                mapIR(rows) { row =>
                  val key = SelectFields(row, partitioner.kType.fieldNames)
                  If(
                    invoke("contains", TBoolean, interval, key),
                    row, {
                      val idx = GetTupleElement(in, 0)
                      val msg = strConcat(
                        "TableGen: Unexpected key in partition ", idx,
                        "\n\tRange bounds for partition ", idx, ": ", interval,
                        "\n\tInvalid key: ", key,
                      )
                      Die(msg, row.typ, errorId)
                    },
                  )
                }
              }
            },
        )

      case TableRange(n, nPartitions) =>
        val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
        val partCounts = partition(n, nPartitionsAdj)
        val partStarts = partCounts.scanLeft(0)(_ + _)

        val contextType = TStruct("start" -> TInt32, "end" -> TInt32)

        val ranges = ArraySeq.tabulate(nPartitionsAdj)(i => partStarts(i) -> partStarts(i + 1))

        TableStage(
          makestruct(),
          new RVDPartitioner(
            ctx.stateManager,
            Array("idx"),
            tir.typ.rowType,
            ranges.map { case (start, end) =>
              Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
            },
          ),
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), ranges.map(Row.fromTuple))),
          ctxRef =>
            mapIR(StreamRange(GetField(ctxRef, "start"), GetField(ctxRef, "end"), I32(1), true)) {
              i => makestruct("idx" -> i)
            },
        )

      case TableMapGlobals(child, newGlobals) =>
        lower(child).mapGlobals(old => Let(FastSeq(TableIR.globalName -> old), newGlobals))

      case TableAggregateByKey(child, expr) =>
        val loweredChild = lower(child)
        val repartitioned = loweredChild.repartitionNoShuffle(
          ctx,
          loweredChild.partitioner.coarsen(child.typ.key.length).strictify(),
        )

        repartitioned.mapPartition(Some(child.typ.key)) { partition =>
          Let(
            FastSeq(TableIR.globalName -> repartitioned.globals),
            mapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) { groupRef =>
              StreamAgg(
                groupRef,
                TableIR.rowName,
                bindIRs(
                  ArrayRef(
                    ApplyAggOp(
                      FastSeq(I32(1)),
                      FastSeq(SelectFields(Ref(TableIR.rowName, child.typ.rowType), child.typ.key)),
                      Take(),
                    ),
                    I32(0),
                  ), // FIXME: would prefer a First() agg op
                  expr,
                ) { case Seq(key, value) =>
                  MakeStruct(child.typ.key.map(k =>
                    (k, GetField(key, k))
                  ) ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                    (f, GetField(value, f))
                  })
                },
              )
            },
          )
        }

      case TableDistinct(child) =>
        val loweredChild = lower(child)

        if (analyses.distinctKeyedAnalysis.contains(child))
          loweredChild
        else
          loweredChild.repartitionNoShuffle(
            ctx,
            loweredChild.partitioner.coarsen(child.typ.key.length).strictify(),
          )
            .mapPartition(None) { partition =>
              flatMapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) {
                groupRef => StreamTake(groupRef, 1)
              }
            }

      case TableFilter(child, cond) =>
        val loweredChild = lower(child)
        loweredChild.mapPartition(None) { rows =>
          Let(
            FastSeq(TableIR.globalName -> loweredChild.globals),
            StreamFilter(rows, TableIR.rowName, cond),
          )
        }

      case TableFilterIntervals(child, intervals, keep) =>
        val loweredChild = lower(child)
        val part = loweredChild.partitioner
        val kt = child.typ.keyType
        val ord = PartitionBoundOrdering(ctx.stateManager, kt)
        val iord = ord.intervalEndpointOrdering

        val filterPartitioner = new RVDPartitioner(
          ctx.stateManager,
          kt,
          Interval.union(intervals, ord.intervalEndpointOrdering),
        )
        val boundsType = TArray(RVDPartitioner.intervalIRRepresentation(kt))
        val filterIntervalsRef = Ref(freshName(), boundsType)
        val filterIntervals: IndexedSeq[Interval] = filterPartitioner.rangeBounds.map { i =>
          RVDPartitioner.intervalToIRRepresentation(i, kt.size)
        }

        val (newRangeBounds, includedIndices, startAndEndInterval, f) = if (keep) {
          val (newRangeBounds, includedIndices, startAndEndInterval) =
            part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
              if (filterPartitioner.overlaps(interval)) {
                Some((
                  interval,
                  i,
                  (
                    filterPartitioner.lowerBoundInterval(interval),
                    filterPartitioner.upperBoundInterval(interval),
                  ),
                ))
              } else None
            }.unzip3

          def f(partitionIntervals: IR, key: IR): IR =
            invoke("partitionerContains", TBoolean, partitionIntervals, key)

          (newRangeBounds, includedIndices, startAndEndInterval, f _)
        } else {
          // keep = False
          val (newRangeBounds, includedIndices, startAndEndInterval) =
            part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
              val lowerBound = filterPartitioner.lowerBoundInterval(interval)
              val upperBound = filterPartitioner.upperBoundInterval(interval)
              if (
                (lowerBound until upperBound).map(filterPartitioner.rangeBounds).exists {
                  filterInterval =>
                    iord.compareNonnull(
                      filterInterval.left,
                      interval.left,
                    ) <= 0 && iord.compareNonnull(filterInterval.right, interval.right) >= 0
                }
              )
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
          broadcastVals = loweredChild.broadcastVals ++ FastSeq((
            filterIntervalsRef.name,
            Literal(boundsType, filterIntervals),
          )),
          loweredChild.globals,
          newPart,
          loweredChild.dependency,
          contexts = bindIRs(
            ToArray(loweredChild.contexts),
            Literal(
              TArray(TTuple(TInt32, TInt32)),
              startAndEndInterval.map(Row.fromTuple),
            ),
          ) { case Seq(prevContexts, bounds) =>
            zip2(
              ToStream(Literal(TArray(TInt32), includedIndices)),
              ToStream(bounds),
              ArrayZipBehavior.AssumeSameLength,
            ) { (idx, bound) =>
              makestruct("prevContext" -> ArrayRef(prevContexts, idx), "bounds" -> bound)
            }
          },
          { part =>
            IRBuilder.scoped { b =>
              val oldCtx = b.memoize(GetField(part, "prevContext"))
              val oldPart = loweredChild.partition(oldCtx)
              val bounds = b.memoize(GetField(part, "bounds"))
              val startIntervalIdx = b.memoize(GetTupleElement(bounds, 0))
              val endIntervalIdx = b.memoize(GetTupleElement(bounds, 1))
              val partitionIntervals = b.memoize(
                ToArray(mapIR(rangeIR(startIntervalIdx, endIntervalIdx)) { i =>
                  ArrayRef(filterIntervalsRef, i)
                })
              )

              filterIR(oldPart) { row =>
                bindIR(SelectFields(row, child.typ.key))(key => f(partitionIntervals, key))
              }
            }
          },
        )

      case TableHead(child, targetNumRows) =>
        val loweredChild = lower(child)

        def streamLenOrMax(a: TrivialIR): IR =
          if (targetNumRows <= Integer.MAX_VALUE)
            StreamLen(StreamTake(a, targetNumRows.toInt))
          else
            StreamLen(a)

        def partitionSizeArray(childContexts: TrivialIR): IR =
          PartitionCounts(child) match {
            case Some(partCounts) =>
              var idx = 0
              var sumSoFar = 0L
              while (idx < partCounts.length && sumSoFar < targetNumRows) {
                sumSoFar += partCounts(idx)
                idx += 1
              }
              val partsToKeep = partCounts.slice(0, idx)
              val finalParts = partsToKeep.map(partSize => partSize.toInt)
              Literal(TArray(TInt32), finalParts)
            case None =>
              tailLoop(TArray(TInt32), if (targetNumRows == 1L) 1 else 4, 0) {
                case (recur, Seq(numPartsToTry, iteration)) =>
                  bindIR(
                    loweredChild
                      .mapContexts(_ => StreamTake(ToStream(childContexts), numPartsToTry))(_.clone)
                      .mapCollect(
                        "table_head_recursive_count",
                        strConcat(
                          Str("iteration="),
                          invoke("str", TString, iteration),
                          Str(",nParts="),
                          invoke("str", TString, numPartsToTry),
                        ),
                      )(streamLenOrMax)
                  ) { counts =>
                    If(
                      (streamSumIR(ToStream(counts)).toL >= targetNumRows) || (ArrayLen(
                        childContexts
                      ) <= ArrayLen(counts)),
                      counts,
                      recur(FastSeq(numPartsToTry * 4, iteration.clone + 1)),
                    )
                  }
              }
          }

        def answerTuple(partitionSizeArrayRef: TrivialIR): IR =
          bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            If(
              numPartitions ceq 0,
              maketuple(0, 0L),
              tailLoop(TTuple(TInt32, TInt64), 0, targetNumRows) {
                case (recur, Seq(i, numLeft)) =>
                  If(
                    (i ceq numPartitions - 1) || ((numLeft - ArrayRef(
                      partitionSizeArrayRef,
                      i,
                    ).toL) <= 0L),
                    maketuple(i.clone + 1, numLeft),
                    recur(
                      FastSeq(
                        i.clone + 1,
                        numLeft - ArrayRef(partitionSizeArrayRef, i).toL,
                      )
                    ),
                  )
              },
            )
          }

        val newCtxs =
          IRBuilder.scoped { b =>
            val childContexts = b.memoize(ToArray(loweredChild.contexts))
            val partitionSizeArrayRef = b.memoize(partitionSizeArray(childContexts))
            val answerTupleRef = b.memoize(answerTuple(partitionSizeArrayRef))
            val numParts = b.memoize(GetTupleElement(answerTupleRef, 0))
            val numElementsFromLastPart = b.memoize(GetTupleElement(answerTupleRef, 1))

            val howManyFromEachPart =
              mapIR(rangeIR(numParts)) { idxRef =>
                If(
                  idxRef ceq (numParts - 1),
                  Cast(numElementsFromLastPart, TInt32),
                  ArrayRef(partitionSizeArrayRef, idxRef),
                )
              }

            zipIR(
              FastSeq(StreamTake(ToStream(childContexts), numParts), howManyFromEachPart),
              ArrayZipBehavior.AssumeSameLength,
            ) { case Seq(part, howMany) =>
              MakeStruct(FastSeq("numberToTake" -> howMany, "old" -> part))
            }
          }

        val bindRelationLetsNewCtx = Let(loweredChild.letBindings, ToArray(newCtxs))
        val newCtxSeq =
          CompileAndEvaluate[IndexedSeq[Any]](ctx, bindRelationLetsNewCtx)
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
          ctxRef =>
            bindIR(GetField(ctxRef, "old")) { old =>
              StreamTake(
                loweredChild.partition(old),
                GetField(ctxRef, "numberToTake"),
              )
            },
        )

      case TableTail(child, targetNumRows) =>
        val loweredChild = lower(child)

        def partitionSizeArray(childContexts: TrivialIR, totalNumPartitions: TrivialIR): IR =
          PartitionCounts(child) match {
            case Some(partCounts) =>
              var idx = partCounts.length
              var sumSoFar = 0L
              while (idx > 0 && sumSoFar < targetNumRows) {
                idx -= 1
                sumSoFar += partCounts(idx)
              }
              val finalParts = partCounts.view.drop(idx).map(_.toInt).to(ArraySeq)
              Literal(TArray(TInt32), finalParts)

            case None =>
              tailLoop(TArray(TInt32), if (targetNumRows == 1L) 1 else 4, 0) {
                case (recur, Seq(numPartsToTry, iteration)) =>
                  bindIR(
                    loweredChild
                      .mapContexts(_ =>
                        StreamDrop(
                          ToStream(childContexts),
                          maxIR(totalNumPartitions - numPartsToTry, 0),
                        )
                      )(_.clone)
                      .mapCollect(
                        "table_tail_recursive_count",
                        strConcat(
                          Str("iteration="),
                          invoke("str", TString, iteration),
                          Str(", nParts="),
                          invoke("str", TString, numPartsToTry),
                        ),
                      )(StreamLen(_))
                  ) { counts =>
                    If(
                      (streamSumIR(
                        ToStream(counts)
                      ).toL >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                      counts,
                      recur(FastSeq(numPartsToTry * 4, iteration.clone + 1)),
                    )
                  }
              }
          }

        // First element is how many partitions to keep from the right partitionSizeArray
        // Second element is how many to keep from first kept element.
        def answerTuple(partitionSizeArray: TrivialIR, nPartitions: TrivialIR): IR =
          If(
            nPartitions ceq 0,
            maketuple(0, 0),
            tailLoop(TTuple(TInt32, TInt32), 1, 0L) {
              case (recur, Seq(i, nRowsToRight)) =>
                bindIR(ArrayRef(partitionSizeArray, nPartitions - i).toL) { keep =>
                  If(
                    (i ceq nPartitions) || (nRowsToRight.clone + keep) >= targetNumRows,
                    maketuple(
                      i,
                      maxIR(0L, keep - (I64(targetNumRows) - nRowsToRight)).toI,
                    ),
                    recur(
                      FastSeq(
                        i.clone + 1,
                        nRowsToRight.clone + keep,
                      )
                    ),
                  )
                }
            },
          )

        val newCtxs =
          IRBuilder.scoped { b =>
            val childContexts = b.memoize(ToArray(loweredChild.contexts))
            val nContexts = b.memoize(ArrayLen(childContexts))

            val partitionSizeArrayRef = b.memoize(partitionSizeArray(childContexts, nContexts))
            val nPartitions = b.memoize(ArrayLen(partitionSizeArrayRef))
            val answerTupleRef = b.memoize(answerTuple(partitionSizeArrayRef, nPartitions))

            val numPartsToKeepFromRight = b.memoize(GetTupleElement(answerTupleRef, 0))
            val nToDropFromFirst = b.memoize(GetTupleElement(answerTupleRef, 1))
            val startIdx = b.memoize(nContexts - numPartsToKeepFromRight)
            mapIR(rangeIR(numPartsToKeepFromRight)) { idx =>
              makestruct(
                "numberToDrop" -> If(idx ceq 0, nToDropFromFirst, 0),
                "old" -> ArrayRef(childContexts, idx.clone + startIdx),
              )
            }
          }

        val letBindNewCtx = Let(loweredChild.letBindings, ToArray(newCtxs))
        val newCtxSeq = CompileAndEvaluate[IndexedSeq[Any]](ctx, letBindNewCtx)
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
          ctxRef =>
            bindIR(GetField(ctxRef, "old")) { oldRef =>
              StreamDrop(loweredChild.partition(oldRef), GetField(ctxRef, "numberToDrop"))
            },
        )

      case TableMapRows(child, newRow) =>
        val lc = lower(child)
        if (!ContainsScan(newRow)) {
          lc.mapPartition(Some(child.typ.key)) { rows =>
            Let(
              FastSeq(TableIR.globalName -> lc.globals),
              mapIR(rows)(row => Let(FastSeq(TableIR.rowName -> row), newRow)),
            )
          }
        } else {
          val aggs =
            agg.Extract(ctx, newRow, analyses.requirednessAnalysis, isScan = true).independent
          val aggSigs = aggs.sigs

          val initState = RunAgg(
            Let(FastSeq(TableIR.globalName -> lc.globals), aggs.init),
            aggSigs.valuesOp,
            aggSigs.states,
          )
          val initStateRef = Ref(freshName(), initState.typ)
          val lcWithInitBinding = lc.copy(
            letBindings = FastSeq(initStateRef.name -> initState),
            broadcastVals = lc.broadcastVals ++ FastSeq(initStateRef.name -> initStateRef),
          )

          def initFromSerializedStates = aggSigs.initFromSerializedValueOp(initStateRef)
          val branchFactor = ctx.branchingFactor
          val (partitionPrefixSumValues, transformPrefixSum): (IR, TrivialIR => IR) =
            if (aggSigs.shouldTreeAggregate && branchFactor < lc.numPartitions) {
              val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

              val codecSpec = TypedCodecSpec(
                ctx,
                PCanonicalTuple(true, Seq.fill(aggSigs.nAggs)(PCanonicalBinary(true)): _*),
                BufferSpec.wireSpec,
              )
              val writer = ETypeValueWriter(codecSpec)
              val reader = ETypeValueReader(codecSpec)
              val partitionPrefixSumFiles =
                lcWithInitBinding.mapCollectWithGlobals("table_scan_write_prefix_sums")({ part =>
                  Let(
                    FastSeq(TableIR.globalName -> lcWithInitBinding.globals),
                    RunAgg(
                      Begin(FastSeq(
                        initFromSerializedStates,
                        StreamFor(part, TableIR.rowName, aggs.seqPerElt),
                      )),
                      WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                      aggSigs.states,
                    ),
                  )
                  // Collected is TArray of TString
                }) { case (collected, _) =>
                  def combineGroup(partArrayRef: TrivialIR): IR = {
                    Begin(FastSeq(
                      bindIR(ReadValue(
                        ArrayRef(partArrayRef, 0),
                        reader,
                        reader.spec.encodedVirtualType,
                      ))(aggSigs.initFromSerializedValueOp),
                      forIR(StreamRange(
                        1,
                        ArrayLen(partArrayRef),
                        1,
                        requiresMemoryManagementPerElement = true,
                      )) { fileIdx =>
                        bindIR(ReadValue(
                          ArrayRef(partArrayRef, fileIdx),
                          reader,
                          reader.spec.encodedVirtualType,
                        ))(aggSigs.combOpValues)
                      },
                    ))
                  }

                  // Return Array[Array[String]], length is log_b(num_partitions)
                  // The upward pass starts with partial aggregations from each partition,
                  // and aggregates these in a tree parameterized by the branching factor.
                  // The tree ends when the number of partial aggregations is less than or
                  // equal to the branching factor.
                  // The upward pass returns the full tree of results as an array of arrays,
                  // where the first element is partial aggregations per partition of the
                  // input.
                  val upPass =
                    tailLoop(TArray(TArray(TString)), MakeArray(collected), 0) {
                      case (recur, Seq(aggStack, iteration)) =>
                        bindIR(ArrayRef(aggStack, ArrayLen(aggStack) - 1)) { states =>
                          bindIR(ArrayLen(states)) { statesLen =>
                            If(
                              statesLen > branchFactor, {
                                val nCombines =
                                  (statesLen.clone + branchFactor - 1) floorDiv branchFactor

                                val contexts =
                                  mapIR(rangeIR(nCombines)) { outerIdxRef =>
                                    sliceArrayIR(
                                      states,
                                      outerIdxRef * branchFactor,
                                      (outerIdxRef.clone + 1) * branchFactor,
                                    )
                                  }

                                val cdaResult =
                                  cdaIR(
                                    contexts,
                                    makestruct(),
                                    "table_scan_up_pass",
                                    strConcat(
                                      Str("iteration="),
                                      invoke("str", TString, iteration),
                                      Str(", nStates="),
                                      invoke("str", TString, statesLen),
                                    ),
                                  ) { case (contexts, _) =>
                                    RunAgg(
                                      combineGroup(contexts),
                                      WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                                      aggSigs.states,
                                    )
                                  }

                                recur(
                                  FastSeq(
                                    invoke(
                                      "extend",
                                      TArray(TArray(TString)),
                                      aggStack,
                                      MakeArray(cdaResult),
                                    ),
                                    iteration.clone + 1,
                                  )
                                )
                              },
                              aggStack,
                            )
                          }
                        }
                    }

                  // The downward pass traverses the tree from root to leaves, computing partial
                  // scan sums as it goes. The two pieces of state transmitted between iterations
                  // are:
                  // - the level (an integer) referring to a position in the array `aggStack`,
                  // - and `last`, the partial sums from the last iteration.
                  //
                  // The starting state for `last` is an array of a single empty aggregation state.
                  bindIR(upPass) { aggStack =>
                    val freshState = WriteValue(initState, Str(tmpDir) + UUID4(), writer)
                    tailLoop(TArray(TString), ArrayLen(aggStack) - 1, MakeArray(freshState), 0) {
                      case (recur, Seq(level, last, iteration)) =>
                        If(
                          level < 0,
                          last,
                          bindIR(ArrayRef(aggStack, level)) { aggsArray =>
                            val groups =
                              mapIR(
                                zipWithIndex(
                                  mapIR(StreamGrouped(ToStream(aggsArray), I32(branchFactor)))(
                                    ToArray(_)
                                  )
                                )
                              ) { eltAndIdx =>
                                makestruct(
                                  "prev" -> ArrayRef(last, GetField(eltAndIdx, "idx")),
                                  "partialSums" -> GetField(eltAndIdx, "elt"),
                                )
                              }

                            val results =
                              cdaIR(
                                groups,
                                maketuple(),
                                "table_scan_down_pass",
                                strConcat(
                                  Str("iteration="),
                                  invoke("str", TString, iteration),
                                  Str(", level="),
                                  invoke("str", TString, level),
                                ),
                              ) { case (context, _) =>
                                val elt = Ref(freshName(), TString)
                                ToArray(RunAggScan(
                                  ToStream(
                                    GetField(context, "partialSums"),
                                    requiresMemoryManagementPerElement = true,
                                  ),
                                  elt.name,
                                  bindIR(ReadValue(
                                    GetField(context, "prev"),
                                    reader,
                                    reader.spec.encodedVirtualType,
                                  ))(
                                    aggSigs.initFromSerializedValueOp
                                  ),
                                  bindIR(ReadValue(elt, reader, reader.spec.encodedVirtualType))(
                                    aggSigs.combOpValues
                                  ),
                                  WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                                  aggSigs.states,
                                ))
                              }

                            recur(
                              FastSeq(
                                level - 1,
                                ToArray(flatten(ToStream(results))),
                                iteration.clone + 1,
                              )
                            )
                          },
                        )
                    }
                  }
                }

              (
                partitionPrefixSumFiles,
                file => ReadValue(file, reader, reader.spec.encodedVirtualType),
              )
            } else {
              val partitionAggs =
                lcWithInitBinding.mapCollectWithGlobals("table_scan_prefix_sums_singlestage")({
                  part =>
                    Let(
                      FastSeq(TableIR.globalName -> lc.globals),
                      RunAgg(
                        Begin(FastSeq(
                          initFromSerializedStates,
                          StreamFor(part, TableIR.rowName, aggs.seqPerElt),
                        )),
                        aggSigs.valuesOp,
                        aggSigs.states,
                      ),
                    )
                }) { case (collected, globals) =>
                  Let(
                    FastSeq(TableIR.globalName -> globals),
                    ToArray(
                      StreamTake(
                        streamScanIR(
                          ToStream(collected, requiresMemoryManagementPerElement = true),
                          initStateRef.clone,
                        ) {
                          (acc, value) =>
                            RunAgg(
                              Begin(FastSeq(
                                aggSigs.initFromSerializedValueOp(acc),
                                aggSigs.combOpValues(value),
                              )),
                              aggSigs.valuesOp,
                              aggSigs.states,
                            )
                        },
                        ArrayLen(collected),
                      )
                    ),
                  )
                }

              (partitionAggs, identity(_))
            }

          val partitionPrefixSumsRef = Ref(freshName(), partitionPrefixSumValues.typ)
          TableStage.apply(
            letBindings =
              lc.letBindings ++ FastSeq(partitionPrefixSumsRef.name -> partitionPrefixSumValues),
            broadcastVals = lc.broadcastVals,
            partitioner = lc.partitioner,
            dependency = lc.dependency,
            globals = lc.globals,
            contexts = zipIR(
              FastSeq(lc.contexts.unsafeClone, ToStream(partitionPrefixSumsRef)),
              ArrayZipBehavior.AssertSameLength,
            ) { case Seq(oldContext, scanState) =>
              makestruct("oldContext" -> oldContext, "scanState" -> scanState)
            },
            partition = partitionRef =>
              IRBuilder.scoped { b =>
                val oldContext = b.memoize(GetField(partitionRef, "oldContext"))
                val rawPrefixSum = b.memoize(GetField(partitionRef, "scanState"))
                val scanState = b.memoize(transformPrefixSum(rawPrefixSum))
                b.strictMemoize(lc.globals, TableIR.globalName): Unit
                RunAggScan(
                  lc.partition(oldContext),
                  TableIR.rowName,
                  aggSigs.initFromSerializedValueOp(scanState),
                  aggs.seqPerElt,
                  aggs.result,
                  aggSigs.states,
                )
              },
          )
        }

      case t @ TableKeyBy(child, newKey, _: Boolean, _) =>
        require(t.definitelyDoesNotShuffle)
        val loweredChild = lower(child)

        val nPreservedFields = loweredChild.kType.fieldNames
          .zip(newKey)
          .takeWhile { case (l, r) => l == r }
          .length

        loweredChild.changePartitionerNoRepartition(
          loweredChild.partitioner.coarsen(nPreservedFields)
        )
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
            val leftElementRef = Ref(freshName(), left.typ.rowType)
            val rightElementRef = Ref(freshName(), right.typ.rowType)

            val (typeOfRootStruct, _) = right.typ.rowType.filterSet(right.typ.key.toSet, false)
            val rootStruct = SelectFields(rightElementRef, typeOfRootStruct.fieldNames.toIndexedSeq)
            val joiningOp = InsertFields(leftElementRef, FastSeq(root -> rootStruct))
            StreamJoinRightDistinct(
              leftPart,
              rightPart,
              left.typ.key.take(commonKeyLength),
              right.typ.key,
              leftElementRef.name,
              rightElementRef.name,
              joiningOp,
              "left",
            )
          },
        )

      case TableIntervalJoin(left, right, root, product) =>
        lower(left).intervalAlignAndZipPartitions(
          ctx,
          lower(right),
          analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable].rowType,
          (lGlobals, _) => lGlobals,
          { (lstream, rstream) =>
            val lref = Ref(freshName(), left.typ.rowType)
            if (product) {
              val rref = Ref(freshName(), TArray(right.typ.rowType))
              StreamLeftIntervalJoin(
                lstream,
                rstream,
                left.typ.key.head,
                right.typ.keyType.fields(0).name,
                lref.name,
                rref.name,
                InsertFields(
                  lref,
                  FastSeq(
                    root -> mapArray(rref)(SelectFields(_, right.typ.valueType.fieldNames))
                  ),
                ),
              )
            } else {
              val rref = Ref(freshName(), right.typ.rowType)
              StreamJoinRightDistinct(
                lstream,
                rstream,
                left.typ.key,
                right.typ.key,
                lref.name,
                rref.name,
                InsertFields(
                  lref,
                  FastSeq(root -> SelectFields(rref, right.typ.valueType.fieldNames)),
                ),
                "left",
              )
            }
          },
        )

      case tj @ TableJoin(left, right, _, _) =>
        val loweredLeft = lower(left)
        val loweredRight = lower(right)
        LowerTableIRHelpers.lowerTableJoin(ctx, analyses, tj, loweredLeft, loweredRight)

      case x @ TableUnion(children) =>
        val lowered = children.map(lower)
        val keyType = x.typ.keyType

        if (keyType.size == 0) {
          TableStage.concatenate(ctx, lowered)
        } else {
          val newPartitioner = RVDPartitioner.generate(
            ctx.stateManager,
            keyType,
            lowered.flatMap(_.partitioner.rangeBounds),
          )
          val repartitioned = lowered.map(_.repartitionNoShuffle(ctx, newPartitioner))

          TableStage(
            repartitioned.flatMap(_.letBindings),
            repartitioned.flatMap(_.broadcastVals),
            repartitioned.head.globals,
            newPartitioner,
            TableStageDependency.union(repartitioned.map(_.dependency)),
            zipIR(repartitioned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
              MakeTuple.ordered(ctxRefs.map(_.clone))
            },
            ctxRef =>
              StreamMultiMerge(
                repartitioned.indices.map(i =>
                  bindIR(GetTupleElement(ctxRef, i))(ctx => repartitioned(i).partition(ctx))
                ),
                keyType.fieldNames,
              ),
          )
        }

      case x @ TableMultiWayZipJoin(children, fieldName, globalName) =>
        val lowered = children.map(lower)
        val keyType = x.typ.keyType
        val newPartitioner = RVDPartitioner.generate(
          ctx.stateManager,
          keyType,
          lowered.flatMap(_.partitioner.rangeBounds),
        )
        val repartitioned = lowered.map(_.repartitionNoShuffle(ctx, newPartitioner))
        val newGlobals = MakeStruct(FastSeq(
          globalName -> MakeArray(
            repartitioned.map(_.globals.clone),
            TArray(repartitioned.head.globalType),
          )
        ))
        val globalsRef = Ref(freshName(), newGlobals.typ)

        val keyRef = Ref(freshName(), keyType)
        val valsRef = Ref(freshName(), TArray(children.head.typ.rowType))
        val projectedVals = ToArray(mapIR(ToStream(valsRef)) { elt =>
          SelectFields(elt, children.head.typ.valueType.fieldNames)
        })

        TableStage(
          repartitioned.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals,
          repartitioned.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
          globalsRef.clone,
          newPartitioner,
          TableStageDependency.union(repartitioned.map(_.dependency)),
          zipIR(repartitioned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
            MakeTuple.ordered(ctxRefs.map(_.clone))
          },
          ctxRef =>
            StreamZipJoin(
              repartitioned.indices.map(i =>
                bindIR(GetTupleElement(ctxRef, i))(ctx => repartitioned(i).partition(ctx))
              ),
              keyType.fieldNames,
              keyRef.name,
              valsRef.name,
              InsertFields(keyRef, FastSeq(fieldName -> projectedVals)),
            ),
        )

      case t @ TableOrderBy(child, _) =>
        require(t.definitelyDoesNotShuffle)
        val loweredChild = lower(child)
        loweredChild.changePartitionerNoRepartition(RVDPartitioner.unkeyed(
          ctx.stateManager,
          loweredChild.partitioner.numPartitions,
        ))

      case TableExplode(child, path) =>
        lower(child).mapPartition(Some(child.typ.key.takeWhile(k => k != path(0)))) { rows =>
          flatMapIR(rows) { row =>
            val N = path.length

            val bindings = new Array[Binding](N)
            val refs = new Array[TrivialIR](N)
            val last = (0 until N).foldLeft(row) { (ref, i) =>
              refs(i) = ref
              val root = GetField(ref, path(i))
              val next = Ref(freshName(), root.typ)
              bindings(i) = Binding(next.name, root)
              next
            }

            Block(
              bindings.unsafeToArraySeq,
              mapIR(ToStream(last, requiresMemoryManagementPerElement = true)) { elt =>
                path.zip(refs.unsafeToArraySeq).foldRight[IR](elt) { case ((p, ref), inserted) =>
                  InsertFields(ref, FastSeq(p -> inserted))
                }
              },
            )
          }
        }

      case TableRepartition(child, n, RepartitionStrategy.NAIVE_COALESCE) =>
        val lc = lower(child)
        val groupSize = (lc.numPartitions + n - 1) / n

        TableStage(
          letBindings = lc.letBindings,
          broadcastVals = lc.broadcastVals,
          globals = lc.globals,
          partitioner = lc.partitioner.copy(rangeBounds =
            lc.partitioner
              .rangeBounds
              .grouped(groupSize)
              .map(arr => Interval(arr.head.left, arr.last.right))
              .to(ArraySeq)
          ),
          dependency = lc.dependency,
          contexts = mapIR(StreamGrouped(lc.contexts, groupSize))(group => ToArray(group)),
          partition = r => flatMapIR(ToStream(r))(prevCtx => lc.partition(prevCtx)),
        )

      case TableRename(child, rowMap, globalMap) =>
        val loweredChild = lower(child)
        val newGlobals =
          CastRename(
            loweredChild.globals,
            loweredChild.globals.typ.asInstanceOf[TStruct].rename(globalMap),
          )
        val newGlobalsRef = Ref(freshName(), newGlobals.typ)

        TableStage(
          loweredChild.letBindings :+ newGlobalsRef.name -> newGlobals,
          loweredChild.broadcastVals :+ newGlobalsRef.name -> newGlobalsRef,
          newGlobalsRef.clone,
          loweredChild.partitioner.copy(kType = loweredChild.kType.rename(rowMap)),
          loweredChild.dependency,
          loweredChild.contexts,
          ctxRef =>
            mapIR(loweredChild.partition(ctxRef)) { row =>
              CastRename(row, row.typ.asInstanceOf[TStruct].rename(rowMap))
            },
        )

      case TableMapPartitions(child, globalName, partitionStreamName, body, _, allowedOverlap) =>
        val loweredChild = lower(child).strictify(ctx, allowedOverlap)

        loweredChild.mapPartition(Some(child.typ.key)) { part =>
          Let(FastSeq(globalName -> loweredChild.globals, partitionStreamName -> part), body)
        }

      case TableLiteral(_, rvd, enc, encodedGlobals) =>
        RVDToTableStage(rvd, EncodedLiteral(enc, encodedGlobals))

      case TableToTableApply(child, TableFilterPartitions(seq, keep)) =>
        val lc = lower(child)

        val arr = seq.sorted
        val keptSet = seq.toSet
        val lit = Literal(TSet(TInt32), keptSet)
        if (keep) {
          def lookupRangeBound(idx: Int): Interval = {
            try
              lc.partitioner.rangeBounds(idx)
            catch {
              case exc: ArrayIndexOutOfBoundsException =>
                fatal(s"_filter_partitions: no partition with index $idx", exc)
            }
          }

          lc.copy(
            partitioner = lc.partitioner.copy(rangeBounds = arr.map(lookupRangeBound)),
            contexts = mapIR(
              filterIR(
                zipWithIndex(lc.contexts)
              )(t => invoke("contains", TBoolean, lit, GetField(t, "idx")))
            )(t => GetField(t, "elt")),
          )
        } else {
          lc.copy(
            partitioner =
              lc.partitioner.copy(rangeBounds = lc.partitioner.rangeBounds.zipWithIndex.filter {
                case (_, idx) => !keptSet.contains(idx)
              }.map(_._1)),
            contexts = mapIR(
              filterIR(
                zipWithIndex(lc.contexts)
              )(t => !invoke("contains", TBoolean, lit, GetField(t, "idx")))
            )(t => GetField(t, "elt")),
          )
        }

      case TableToTableApply(
            child,
            WrappedMatrixToTableFunction(
              localLDPrune: LocalLDPrune,
              colsFieldName,
              entriesFieldName,
              _,
            ),
          ) =>
        val lc = lower(child)
        lc.mapPartition(Some(child.typ.key)) { rows =>
          localLDPrune.makeStream(
            rows,
            entriesFieldName,
            ArrayLen(GetField(lc.globals, colsFieldName)),
          )
        }.mapGlobals(_ => makestruct())

      case BlockMatrixToTable(bmir) =>
        val ts = LowerBlockMatrixIR.lowerToTableStage(bmir, typesToLower, ctx, analyses)
        // I now have an unkeyed table of (blockRow, blockCol, block).
        ts.mapPartitionWithContext { (partition, ctxRef) =>
          flatMapIR(partition)(singleRowRef =>
            bindIR(GetField(singleRowRef, "block")) { singleNDRef =>
              bindIR(NDArrayShape(singleNDRef)) { shapeTupleRef =>
                flatMapIR(rangeIR(Cast(GetTupleElement(shapeTupleRef, 0), TInt32))) {
                  withinNDRowIdx =>
                    mapIR(rangeIR(Cast(GetTupleElement(shapeTupleRef, 1), TInt32))) {
                      withinNDColIdx =>
                        val entry = NDArrayRef(
                          singleNDRef,
                          IndexedSeq(Cast(withinNDRowIdx, TInt64), Cast(withinNDColIdx, TInt64)),
                          ErrorIDs.NO_ERROR,
                        )
                        val blockStartRow = GetField(singleRowRef, "blockRow") * bmir.typ.blockSize
                        val blockStartCol = GetField(singleRowRef, "blockCol") * bmir.typ.blockSize
                        makestruct(
                          "i" -> (withinNDRowIdx.clone + blockStartRow).toL,
                          "j" -> (withinNDColIdx.clone + blockStartCol).toL,
                          "entry" -> entry,
                        )
                    }
                }
              }
            }
          )
        }

      case node =>
        throw new LowererUnsupportedOperation(s"undefined: \n${Pretty(ctx, node)}")
    }

    assert(
      tir.typ.globalType == lowered.globalType,
      s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}",
    )
    assert(
      tir.typ.rowType == lowered.rowType,
      s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}",
    )
    assert(
      tir.typ.keyType.isPrefixOf(lowered.kType),
      s"\n  ir key: ${tir.typ.key}\n  lowered key: ${lowered.key}",
    )

    lowered
  }

  // format: off

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

    logger.info(s"repartition cost: $cost")
    cost <= 1.0
  }

  // format: on
}
