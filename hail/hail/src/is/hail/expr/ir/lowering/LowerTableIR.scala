package is.hail.expr.ir.lowering

import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichArray
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.Scope.EVAL
import is.hail.expr.ir.agg.{AggExecuteContextExtensions, Extract}
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
    body: Atom => IR,
  ): TableStage = {
    val globalsRef = Ref(freshName(), globals.typ)
    TableStage(
      FastSeq(globalsRef.name -> globals),
      FastSeq(globalsRef.name -> globalsRef.ir),
      globalsRef.ir,
      partitioner,
      dependency,
      contexts,
      body,
    )
  }

  def apply(
    letBindings: IndexedSeq[(Name, IR)],
    broadcastVals: IndexedSeq[(Name, IR)],
    globals: Atom,
    partitioner: RVDPartitioner,
    dependency: TableStageDependency,
    contexts: IR,
    partition: Atom => IR,
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

    val ctxs = concatIR(children.indices.map { idx =>
      ToArray(mapIR(children(idx).contexts) { ctx =>
        MakeTuple.ordered(children.indices.map { idx2 =>
          if (idx == idx2) ctx.ir else NA(children(idx2).ctxType)
        })
      })
    }: _*)

    val newGlobals = children.head.globals
    val globalsRef = Ref(freshName(), newGlobals.typ)
    val newPartitioner =
      new RVDPartitioner(ctx.stateManager, keyType, children.flatMap(_.partitioner.rangeBounds))

    TableStage(
      children.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals.ir,
      children.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef.ir,
      globalsRef.ir,
      newPartitioner,
      TableStageDependency.union(children.map(_.dependency)),
      ctxs,
      ctxRef =>
        StreamMultiMerge(
          children.indices.map { i =>
            bindIR(ctxRef.get(i)) { ctx =>
              If(
                ctx.isNA,
                MakeStream.empty(children(i).rowType),
                children(i).partition(ctx),
              )
            }
          },
          FastSeq(),
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
  val globals: Atom,
  val partitioner: RVDPartitioner,
  val dependency: TableStageDependency,
  val contexts: IR,
  val ctxRefName: Name,
  val partitionIR: IR,
) extends Logging {
  self =>

  // useful for debugging, but should be disabled in production code due to N^2 complexity
  // typecheckPartition()

  contexts.typ match {
    case TStream(t) if t.isRealizable =>
    case t =>
      throw new IllegalArgumentException(s"TableStage constructed with illegal context type $t")
  }

  def typecheckPartition(ctx: ExecuteContext): Unit =
    TypeCheck(
      ctx,
      partitionIR,
      BindingEnv.eval(
        (letBindings ++ broadcastVals).map { case (s, x) => (s, x.typ) } ++
          FastSeq(ctxRefName -> TIterable.elementType(contexts.typ)): _*
      ),
    )

  def upcast(ctx: ExecuteContext, newType: TableType): TableStage = {
    val newRowType = newType.rowType
    val newGlobalType = newType.globalType
    if (newRowType == rowType && newGlobalType == globalType) this
    else changePartitionerNoRepartition(partitioner.coarsen(newType.key.length))
      .mapPartition(None)(PruneDeadFields.upcast(ctx, _, TStream(newRowType)))
      .mapGlobals(PruneDeadFields.upcast(ctx, _, newGlobalType))
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
    globals: Atom = globals,
    partitioner: RVDPartitioner = partitioner,
    dependency: TableStageDependency = dependency,
    contexts: IR = contexts,
    ctxRefName: Name = ctxRefName,
    partitionIR: IR = partitionIR,
  ): TableStage =
    new TableStage(letBindings, broadcastVals, globals, partitioner, dependency, contexts,
      ctxRefName, partitionIR)

  def deepCopy: TableStage =
    new TableStage(
      letBindings.map { case (n, r) => n -> r.deepCopy },
      broadcastVals.map { case (n, r) => n -> r.deepCopy },
      globals,
      partitioner,
      dependency,
      contexts.deepCopy,
      ctxRefName,
      partitionIR.deepCopy,
    )

  def partition(ctx: Atom): IR = {
    require(ctx.typ == ctxType)
    Let(FastSeq(ctxRefName -> ctx), partitionIR)
  }

  def numPartitions: Int = partitioner.numPartitions

  def mapPartition(newKey: Option[IndexedSeq[String]])(f: Atom => IR): TableStage = {
    val part = newKey match {
      case Some(k) =>
        if (!partitioner.kType.fieldNames.startsWith(k))
          throw new RuntimeException(s"cannot map partitions to new key!" +
            s"\n  prev key: ${partitioner.kType.fieldNames}" +
            s"\n  new key:  $k")
        partitioner.coarsen(k.length)
      case None => partitioner
    }
    copy(partitionIR = bindIR(partitionIR)(f), partitioner = part)
  }

  def zipPartitions(
    right: TableStage,
    newGlobals: (Atom, Atom) => IR,
    body: (Atom, Atom) => IR,
  ): TableStage = {
    val left = this

    val newGlobal = newGlobals(left.globals, right.globals)
    val globalName = freshName()
    val global = Ref(globalName, newGlobal.typ)

    TableStage(
      left.letBindings ++ right.letBindings :+ (globalName -> newGlobal),
      left.broadcastVals ++ right.broadcastVals :+ (globalName -> global),
      global,
      left.partitioner,
      left.dependency.union(right.dependency),
      zip2(left.contexts, right.contexts, AssertSameLength)(maketuple(_, _)),
      ctxRef =>
        M.eval {
          for {
            lpart <- ctxRef.get(0) map left.partition
            rpart <- ctxRef.get(1) map right.partition
          } yield body(lpart, rpart)
        },
    )
  }

  def mapPartitionWithContext(f: (Atom, Atom) => IR): TableStage =
    copy(partitionIR = bindIR(partitionIR)(f(_, Ref(ctxRefName, ctxType))))

  def mapContexts(f: Atom => IR)(getOldContext: Atom => IR): TableStage = {
    val newContexts = bindIR(contexts)(f)
    val newCtxRef = Ref(freshName(), TIterable.elementType(newContexts.typ))
    copy(
      contexts = newContexts,
      ctxRefName = newCtxRef.name,
      partitionIR = bindIR(getOldContext(newCtxRef))(partition),
    )
  }

  def mapGlobals(f: Atom => IR): TableStage = {
    val newGlobals = f(globals)
    val global = Ref(freshName(), newGlobals.typ)

    copy(
      letBindings = letBindings :+ global.name -> newGlobals,
      broadcastVals = broadcastVals :+ global.name -> global.ir,
      globals = global.ir,
    )
  }

  def mapCollect(staticID: String, dynamicID: IR = NA(TString))(f: Atom => IR): IR =
    mapCollectWithGlobals(staticID, dynamicID)(f)((parts, _) => parts)

  def mapCollectWithGlobals(
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    mapF: Atom => IR
  )(
    body: (Atom, Atom) => IR
  ): IR =
    mapCollectWithContextsAndGlobals(staticID, dynamicID)((part, _) => mapF(part))(body)

  // mapf is (part, ctx) => ???, body is (parts, globals) => ???
  def mapCollectWithContextsAndGlobals(
    staticID: String,
    dynamicID: IR = NA(TString),
  )(
    mapF: (Atom, Atom) => IR
  )(
    body: (Atom, Atom) => IR
  ): IR = {

    val broadcastRefs = MakeStruct(broadcastVals.map { case (n, ir) => n.str -> ir })

    val global = Ref(freshName(), broadcastRefs.typ)
    val cda = CollectDistributedArray(
      contexts,
      broadcastRefs,
      ctxRefName,
      global.name,
      Let(
        broadcastVals.map { case (n, _) => n -> global.ir.get(n.str) },
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
      makestruct(
        "rows" -> parts.stream.streamFlatten.toArray,
        "global" -> globals,
      )
    }

  def countPerPartition(): IR =
    mapCollect("count_per_partition")(_.len.toL)

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
          (newIdx, values.map(_._2).sorted)
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

        val newContexts =
          M.eval {
            for {
              oldCtxs <- ToArray(contexts)
              oldIdxs <- Literal(TArray(TArray(TInt32)), oldPartIndices)
            } yield oldIdxs.stream.streamMap(_.stream.streamMap(oldCtxs.at(_)).toArray)
          }

        return TableStage(
          letBindings,
          broadcastVals,
          globals,
          newPartitionerFilt,
          dependency,
          newContexts,
          ctx => flatMapIR(ToStream(ctx, true))(partition),
        )
      }

      val boundType = RVDPartitioner.intervalIRRepresentation(newPartitioner.kType)
      val partitionMapping: IndexedSeq[Row] = newPartitioner.rangeBounds.map { i =>
        RowSeq(
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
              "partitionBound" -> mapping.get("partitionBound"),
              "oldContexts" -> mapArray(mapping.get("parentPartitions"))(ctx.at(_)),
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
        ctxRef =>
          M.eval {
            for {
              interval <- ctxRef.get("partitionBound")
              start <- interval.invoke("start", boundType.pointType)
              includesStart <- interval.invoke("includesStart", TBoolean)
              end <- interval.invoke("end", boundType.pointType)
              includesEnd <- interval.invoke("includesEnd", TBoolean)
            } yield ToStream(ctxRef.get("oldContexts"), true)
              .streamFlatMap(partition)
              .dropWhile(_
                .select(newPartitioner.kType.fieldNames)
                .invoke(
                  "pointLessThanPartitionIntervalLeftEndpoint",
                  TBoolean,
                  start,
                  includesStart,
                ))
              .takeWhile(_
                .select(newPartitioner.kType.fieldNames)
                .invoke("pointLessThanPartitionIntervalRightEndpoint", TBoolean, end, includesEnd))
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
    globalJoiner: (Atom, Atom) => IR,
    joiner: (Atom, Atom) => IR,
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

    val partitionJoiner: (Atom, Atom) => IR = (lPart, rPart) =>
      joinIR(
        lPart,
        rPart,
        this.kType.fieldNames.take(joinKey),
        right.kType.fieldNames.take(joinKey),
        joinType,
        requiresMemoryManagement = true,
        rightKeyIsDistinct = rightKeyIsDistinct,
      )(joiner)

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
    globalJoiner: (Atom, Atom) => IR,
    joiner: (Atom, Atom) => IR,
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
    globalJoiner: (Atom, Atom) => IR,
    joiner: (Atom, Atom) => IR,
  ): TableStage = {
    require(right.kType.size == 1)
    val rightKeyType = right.kType.fields.head.typ
    require(rightKeyType.isInstanceOf[TInterval])
    require(rightKeyType.asInstanceOf[TInterval].pointType == kType.types.head)

    val irPartitioner = partitioner.coarsen(1).partitionBoundsIRRepresentation

    val rightWithPartNums =
      right.mapPartition(None)(_.streamFlatMap { row =>
        M.eval {
          for {
            interval <- row.get(right.key.head)
            f = kType.fields.head

            interval <-
              invoke(
                "Interval",
                TInterval(TTuple(kType.typeAfterSelect(ArraySeq(0)), TInt32)),
                maketuple(makestruct(f.name -> interval.invoke("start", f.typ)), 1),
                maketuple(makestruct(f.name -> interval.invoke("end", f.typ)), 1),
                interval.invoke("includesStart", TBoolean),
                interval.invoke("includesEnd", TBoolean),
              )

            range <-
              invoke(
                "partitionerFindIntervalRange",
                TTuple(TInt32, TInt32),
                irPartitioner,
                interval,
              )

          } yield StreamRange(range.get(0), range.get(1), 1, true)
            .streamMap(n => row.insert("__partNum" -> n))
        }
      })

    val rightRowRTypeWithPartNum =
      ("__partNum" -> TypeWithRequiredness(TInt32)) +:
        rightRowRType.fields.map(f => f.name -> f.typ)

    val rightTableRType = RTable(rightRowRTypeWithPartNum, FastSeq(), right.key)
    val sortedReader = ctx.backend.lowerDistributedSort(
      ctx,
      rightWithPartNums,
      SortField("__partNum", Ascending) +: right.key.map(k => SortField(k, Ascending)),
      rightTableRType,
    )
    val sorted = sortedReader.lower(ctx, sortedReader.fullType)
    assert(sorted.kType.fieldNames == "__partNum" +: right.key)
    val newRightPartitioner = new RVDPartitioner(
      ctx.stateManager,
      Some(1),
      TStruct.concat(TStruct("__partNum" -> TInt32), right.kType),
      ArraySeq.tabulate(partitioner.numPartitions)(i =>
        Interval(RowSeq(i), RowSeq(i), includesStart = true, includesEnd = true)
      ),
    )
    val repartitioned =
      sorted
        .repartitionNoShuffle(ctx, newRightPartitioner)
        .changePartitionerNoRepartition(
          RVDPartitioner.unkeyed(ctx.stateManager, newRightPartitioner.numPartitions)
        )
        .mapPartition(None)(_.streamMap(_.select(right.rowType.fieldNames)))

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
            foldIR(rows.streamMap(Consume(_)), 0L)(_ + _)
          ),
        )

      case TableToValueApply(child, TableCalculateNewPartitions(nPartitions)) =>
        val stage = lower(child)
        val sampleSize = math.min(nPartitions * 20 + 256, 1000000)
        val samplesPerPartition = sampleSize / math.max(1, stage.numPartitions)
        val keyType = child.typ.keyType

        M.eval {
          for {
            perPartData <-
              stage.mapCollect("table_calculate_new_partitions")(_
                .streamMap(_.select(keyType.fieldNames))
                .streamAgg { elt =>
                  ToArray(concatIR(
                    ApplyAggOp(ReservoirSample(), samplesPerPartition)(elt),
                    ApplyAggOp(TakeBy(Ascending), 1)(elt, elt),
                    ApplyAggOp(TakeBy(Descending), 1)(elt, elt),
                  ))
                })

            boundsArray <-
              perPartData
                .streamFlatten
                .sort(_ < _)
                .stream
                .groupedByKey(keyType.fieldNames, missingEqual = true)
                .streamFlatMap(_.take(1))
                .toArray

            nBounds <- boundsArray.len
            nParts <- minIR(nBounds, nPartitions)
          } yield If(
            nParts ceq 0,
            MakeArray.empty(TInterval(keyType)),
            bindIR((nBounds + nParts - 1) floorDiv nParts) { stepSize =>
              ToArray(mapIR(StreamRange(0, nBounds, stepSize)) { i =>
                bindIR((i + stepSize) < (nBounds - 1)) { closed =>
                  invoke(
                    "Interval",
                    TInterval(keyType),
                    ArrayRef(boundsArray, i),
                    ArrayRef(boundsArray, If(closed, i + stepSize, nBounds - 1)),
                    True(),
                    !closed,
                  )
                }
              })
            },
          )
        }

      case TableGetGlobals(child) =>
        lower(child).getGlobals()

      case TableCollect(child) =>
        lower(child).collectWithGlobals("table_collect")

      case TableAggregate(child, query) =>
        val aggs = Extract(ctx, query, analyses.requirednessAnalysis).independent
        val aggSigs = aggs.sigs

        val lc = lower(child)

        val initState = Let(
          FastSeq(TableIR.globalName -> lc.globals),
          RunAgg(aggs.init, aggSigs.valuesOp, aggSigs.states),
        )

        val initStateRef = Ref(freshName(), initState.typ)
        val lcWithInitBinding = lc.copy(
          letBindings = lc.letBindings ++ FastSeq(initStateRef.name -> initState),
          broadcastVals = lc.broadcastVals ++ FastSeq(initStateRef.name -> initStateRef.ir),
        )

        def initFromSerializedStates = aggSigs.initFromSerializedValueOp(initStateRef.ir)

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
            def combineGroup(partArrayRef: Atom, useInitStates: Boolean): IR =
              Begin(FastSeq(
                if (useInitStates) initFromSerializedStates
                else ReadValue(partArrayRef.at(0), reader, reader.spec.encodedVirtualType)
                  .bind(aggSigs.initFromSerializedValueOp),
                StreamRange(if (useInitStates) 0 else 1, partArrayRef.len, 1, true)
                  .streamFor { idx =>
                    ReadValue(partArrayRef.at(idx), reader, reader.spec.encodedVirtualType)
                      .bind(aggSigs.combOpValues)
                  },
              ))

            val treeAggregation =
              tailLoop(TArray(TString), collected, 0) {
                case (recur, Seq(currentAggStates, iterNumber)) =>
                  If(
                    currentAggStates.len <= branchFactor,
                    currentAggStates,
                    recur(
                      FastSeq(
                        cdaIR(
                          currentAggStates.stream.grouped(branchFactor).streamMap(ToArray(_)),
                          makestruct(),
                          "table_tree_aggregate",
                          strConcat("iteration=", iterNumber, ", n_states=", currentAggStates.len),
                        ) { (context, _) =>
                          RunAgg(
                            combineGroup(context, false),
                            WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                            aggSigs.states,
                          )
                        },
                        iterNumber + 1,
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

    if (lowered ne ir) NormalizeNames()(ctx, lowered)
    else ir
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

        val global = Ref(freshName(), typ.globalType)

        TableStage(
          FastSeq(
            loweredRowsAndGlobalRef.name -> loweredRowsAndGlobal,
            global.name -> loweredRowsAndGlobalRef.ir.get("global"),
          ),
          FastSeq(global.name -> global.ir),
          global.ir,
          RVDPartitioner.unkeyed(ctx.stateManager, nPartitionsAdj),
          TableStageDependency.none,
          M.eval {
            for {
              rows <- loweredRowsAndGlobalRef.ir.get("rows")
              numRows <- rows.len
              indicesArray <-
                invoke(
                  "extend",
                  TArray(TInt32),
                  rangeIR(nPartitionsAdj)
                    .streamMap(partIdx => (partIdx * numRows) floorDiv nPartitionsAdj)
                    .toArray,
                  MakeArray(numRows),
                )
            } yield rangeIR(nPartitionsAdj).streamMap { partIdx =>
              rangeIR(indicesArray.at(partIdx), indicesArray.at(partIdx + 1))
                .streamMap(rows.at(_))
                .toArray
            }
          },
          ToStream(_, true),
        )

      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        val loweredGlobals = lowerIR(globals)
        val global = Ref(gname, loweredGlobals.typ)
        TableStage(
          FastSeq(gname -> loweredGlobals),
          FastSeq(gname -> global.ir),
          global.ir,
          partitioner = partitioner,
          dependency = TableStageDependency.none,
          contexts = M.eval {
            for {
              contexts <- contexts.toArray
              length <- contexts.len
              // Assert at runtime that the number of contexts matches the number of partitions
              contexts <-
                If(
                  length ceq partitioner.numPartitions,
                  contexts, {
                    val dieMsg = strConcat(
                      s"TableGen: partitioner contains ${partitioner.numPartitions} partitions,",
                      " got ",
                      length,
                      " contexts.",
                    )
                    Die(dieMsg, contexts.typ, errorId)
                  },
                )

              // [FOR KEYED TABLES ONLY]
              // AFAIK, there's no way to guarantee that the rows generated in the
              // body conform to their partition's range bounds at compile time so
              // assert this at runtime in the body before it wreaks havoc upon the world.
              partIndices <- rangeIR(partitioner.numPartitions)
              bounds <- Literal(TArray(TInterval(partitioner.kType)), partitioner.rangeBounds)
            } yield zipIR(
              FastSeq(partIndices, bounds.stream, contexts.stream),
              AssertSameLength,
              errorId,
            )(elems => MakeTuple.ordered(elems.map(_.ir)))
          },
          partition = ctx =>
            M.eval {
              val rows: M[Scope.EVAL.type] =
                (cname -> ctx.get(2)) >> body

              if (partitioner.kType.fields.isEmpty) rows
              else ctx.get(1).flatMap { interval =>
                rows.map(_.streamMap { row =>
                  row.select(partitioner.kType.fieldNames).bind { key =>
                    If(
                      interval.invoke("contains", TBoolean, key),
                      row,
                      ctx.get(0).bind { idx =>
                        val msg = strConcat(
                          "TableGen: Unexpected key in partition ", idx,
                          "\n\tRange bounds for partition ", idx, ": ", interval,
                          "\n\tInvalid key: ", key,
                        )
                        Die(msg, row.typ, errorId)
                      },
                    )
                  }
                })
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
              Interval(RowSeq(start), RowSeq(end), includesStart = true, includesEnd = false)
            },
          ),
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), ranges.map(Row.fromTuple))),
          ctxRef =>
            StreamRange(ctxRef.get("start"), ctxRef.get("end"), 1, true)
              .streamMap(i => makestruct("idx" -> i)),
        )

      case TableMapGlobals(child, newGlobals) =>
        lower(child).mapGlobals(old => Let(FastSeq(TableIR.globalName -> old), newGlobals))

      case TableAggregateByKey(child, expr) =>
        val loweredChild = lower(child)
        val keyNames = child.typ.key

        val repartitioned = loweredChild.repartitionNoShuffle(
          ctx,
          loweredChild.partitioner.coarsen(keyNames.length).strictify(),
        )

        repartitioned.mapPartition(Some(keyNames)) { partition =>
          Let(
            FastSeq(TableIR.globalName -> repartitioned.globals),
            partition.groupedByKey(keyNames, missingEqual = true).streamMap { groupRef =>
              val newRowExpr =
                bindIRs(ApplyAggOp(Take(), 1)(child.row.select(keyNames)).at(0), expr) {
                  case Seq(key, value) =>
                    key.select(keyNames).insert(
                      expr.typ.asInstanceOf[TStruct].fieldNames.map(f => f -> value.get(f))
                    )
                }

              StreamAgg(groupRef, TableIR.rowName, newRowExpr)
            },
          )
        }

      case TableDistinct(child) =>
        val loweredChild = lower(child)

        if (analyses.distinctKeyedAnalysis.contains(child)) loweredChild
        else loweredChild
          .repartitionNoShuffle(
            ctx,
            loweredChild.partitioner.coarsen(child.typ.key.length).strictify(),
          )
          .mapPartition(None)(_
            .groupedByKey(child.typ.key, missingEqual = true)
            .streamFlatMap(_.take(1)))

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

        val ((newRangeBounds, includedIndices, startAndEndInterval), f) =
          if (keep)
            (
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
              }.unzip3,
              (intervals: IR, key: IR) =>
                invoke("partitionerContains", TBoolean, intervals, key),
            )
          else
            (
              part.rangeBounds.zipWithIndex.flatMap { case (interval, i) =>
                val lowerBound = filterPartitioner.lowerBoundInterval(interval)
                val upperBound = filterPartitioner.upperBoundInterval(interval)
                if (
                  (lowerBound until upperBound).exists { i =>
                    val filterInterval = filterPartitioner.rangeBounds(i)
                    iord.compareNonnull(filterInterval.left, interval.left) <= 0 &&
                    iord.compareNonnull(filterInterval.right, interval.right) >= 0
                  }
                )
                  None
                else Some((interval, i, (lowerBound, upperBound)))
              }.unzip3,
              (intervals: IR, key: IR) =>
                !invoke("partitionerContains", TBoolean, intervals, key),
            )

        val newPart = new RVDPartitioner(ctx.stateManager, kt, newRangeBounds)

        TableStage(
          letBindings = loweredChild.letBindings,
          broadcastVals = loweredChild.broadcastVals ++ FastSeq(
            filterIntervalsRef.name -> Literal(boundsType, filterIntervals)
          ),
          loweredChild.globals,
          newPart,
          loweredChild.dependency,
          contexts =
            bindIR(ToArray(loweredChild.contexts)) { prevContexts =>
              zip2(
                ToStream(Literal(TArray(TInt32), includedIndices)),
                ToStream(Literal(
                  TArray(TTuple(TInt32, TInt32)),
                  startAndEndInterval.map(RowSeq.fromTuple),
                )),
                ArrayZipBehavior.AssumeSameLength,
              ) { (idx, bound) =>
                makestruct("prevContext" -> prevContexts.at(idx), "bounds" -> bound)
              }
            },
          ctx =>
            M.eval {
              for {
                bounds <- ctx.get("bounds")
                partitionIntervals <-
                  rangeIR(bounds.get(0), bounds.get(1))
                    .streamMap(filterIntervalsRef.at(_))
                    .toArray
                rows <- ctx.get("prevContext") map loweredChild.partition
              } yield rows.filter(row => f(partitionIntervals, row.select(child.typ.key)))
            },
        )

      case TableHead(child, targetNumRows) =>
        val loweredChild = lower(child)

        val newContexts =
          M.eval {
            for {
              _ <- M.sequence(loweredChild.letBindings.map(M.let[EVAL.type]): _*)
              contexts <- loweredChild.contexts.toArray
              nPartitions <- contexts.len

              nPartsAndLastRowCount <-
                PartitionCounts(child) match {
                  case Some(sizes) =>
                    val (parts, rows) = nPartsAndRows(sizes, targetNumRows)((_, take) => take)
                    maketuple(parts, rows)
                  case None =>
                    nPartsAndRows(loweredChild, contexts, nPartitions, targetNumRows)(minIR(_, _))
                }

              nPartsToTake <- nPartsAndLastRowCount.get(0)
              nLastRows <- nPartsAndLastRowCount.get(1)
              end <- nPartsToTake - 1
            } yield rangeIR(0, nPartsToTake)
              .streamMap(i => maketuple(contexts.at(i), If(i ceq end, nLastRows, Int.MaxValue)))
              .toArray
          }

        val newCtxSeq = CompileAndEvaluate[IndexedSeq[Any]](ctx, newContexts)
        val numNewParts = newCtxSeq.length
        val newIntervals = loweredChild.partitioner.rangeBounds.slice(0, numNewParts)
        val newPartitioner = loweredChild.partitioner.copy(rangeBounds = newIntervals)

        TableStage(
          loweredChild.letBindings,
          loweredChild.broadcastVals,
          loweredChild.globals,
          newPartitioner,
          loweredChild.dependency,
          ToStream(Literal(newContexts.typ, newCtxSeq)),
          ctx => ctx.get(0).bind(old => loweredChild.partition(old).take(ctx.get(1))),
        )

      case TableTail(child, targetNumRows) =>
        val loweredChild = lower(child)

        val newContexts =
          M.eval {
            for {
              _ <- M.sequence(loweredChild.letBindings.map(M.let[EVAL.type]): _*)
              contexts <- loweredChild.contexts.toArray
              nPartitions <- contexts.len
              reverse <- StreamRange(nPartitions - 1, -1, -1).streamMap(contexts.at(_)).toArray

              nPartsAndLastRowCount <-
                PartitionCounts(child) match {
                  case Some(partCounts) =>
                    val (n, drop) = nPartsAndRows(partCounts.reverse, targetNumRows)(_ - _)
                    maketuple(n, math.max(0, drop))

                  case None =>
                    nPartsAndRows(loweredChild, reverse, nPartitions, targetNumRows) {
                      (takeRight, remainder) => maxIR(0L, takeRight - remainder)
                    }
                }

              nPartsToDrop <- nPartitions - nPartsAndLastRowCount.get(0)
              nRowsToDrop <- nPartsAndLastRowCount.get(1)

            } yield rangeIR(nPartsToDrop, nPartitions)
              .streamMap(i => maketuple(contexts.at(i), If(i ceq nPartsToDrop, nRowsToDrop, 0)))
              .toArray
          }

        val newCtxSeq = CompileAndEvaluate[IndexedSeq[Any]](ctx, newContexts)
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
          ToStream(Literal(newContexts.typ, newCtxSeq)),
          ctx => ctx.get(0).bind(old => loweredChild.partition(old).drop(ctx.get(1))),
        )

      case TableMapRows(child, newRow) =>
        val lc = lower(child)
        if (!ContainsScan(newRow)) {
          lc.mapPartition(Some(child.typ.key)) { rows =>
            Let(
              FastSeq(TableIR.globalName -> lc.globals),
              StreamMap(rows, TableIR.rowName, newRow),
            )
          }
        } else {

          val aggs = Extract(ctx, newRow, analyses.requirednessAnalysis, isScan = true).independent
          val aggSigs = aggs.sigs

          val initState = RunAgg(
            Let(FastSeq(TableIR.globalName -> lc.globals), aggs.init),
            aggSigs.valuesOp,
            aggSigs.states,
          )

          val initStateRef = Ref(freshName(), initState.typ)

          val lcWithInitBinding = {
            val lc_ = lc.deepCopy // no-sharing
            lc_
              .copy(
                letBindings = FastSeq(initStateRef.name -> initState),
                broadcastVals = lc_.broadcastVals ++ FastSeq(initStateRef.name -> initStateRef.ir),
              )
          }

          def initFromSerializedStates = aggSigs.initFromSerializedValueOp(initStateRef.ir)
          val branchFactor = ctx.branchingFactor

          val (partitionPrefixSumValues, transformPrefixSum): (IR, Atom => IR) =
            if (aggSigs.shouldTreeAggregate && branchFactor < lc.numPartitions) {
              val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

              val codecSpec =
                TypedCodecSpec(
                  ctx,
                  PCanonicalTuple(true, Seq.fill(aggSigs.nAggs)(PCanonicalBinary(true)): _*),
                  BufferSpec.wireSpec,
                )

              val reader = ETypeValueReader(codecSpec)
              val writer = ETypeValueWriter(codecSpec)

              val partitionPrefixSumFiles =
                lcWithInitBinding.mapCollectWithGlobals("table_scan_write_prefix_sums") { part =>
                  Let(
                    FastSeq(TableIR.globalName -> lcWithInitBinding.globals),
                    RunAgg(
                      Begin(FastSeq(
                        initFromSerializedStates,
                        StreamFor(part, TableIR.rowName, aggs.seqPerElt.deepCopy),
                      )),
                      WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                      aggSigs.states,
                    ),
                  )
                  // Collected is TArray of TString
                } { case (collected, _) =>
                  def combineGroup(partArrayRef: Atom): IR =
                    Begin(FastSeq(
                      bindIR(ReadValue(
                        partArrayRef.at(0),
                        reader,
                        reader.spec.encodedVirtualType,
                      ))(aggSigs.initFromSerializedValueOp),
                      forIR(StreamRange(1, partArrayRef.len, 1, true)) { fileIdx =>
                        bindIR(ReadValue(
                          partArrayRef.at(fileIdx),
                          reader,
                          reader.spec.encodedVirtualType,
                        ))(aggSigs.combOpValues)
                      },
                    ))

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
                        bindIR(aggStack.at(aggStack.len - 1)) { states =>
                          bindIR(states.len) { statesLen =>
                            If(
                              statesLen > branchFactor, {
                                val nCombines =
                                  (statesLen + branchFactor - 1) floorDiv branchFactor

                                val contexts =
                                  rangeIR(nCombines).streamMap { outerIdxRef =>
                                    states.slice(
                                      outerIdxRef * branchFactor,
                                      Some((outerIdxRef + 1) * branchFactor),
                                    )
                                  }

                                val cdaResult =
                                  cdaIR(
                                    contexts,
                                    makestruct(),
                                    "table_scan_up_pass",
                                    strConcat("iteration=", iteration, ", nStates=", statesLen),
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
                                    iteration + 1,
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
                    val freshState = WriteValue(initStateRef.ir, Str(tmpDir) + UUID4(), writer)
                    tailLoop(TArray(TString), aggStack.len - 1, MakeArray(freshState), 0) {
                      case (recur, Seq(level, last, iteration)) =>
                        If(
                          level < 0,
                          last,
                          aggStack.at(level).bind { aggsArray =>
                            val groups =
                              aggsArray
                                .stream
                                .grouped(branchFactor)
                                .zip(iota(0, 1), ArrayZipBehavior.TakeMinLength) { (group, idx) =>
                                  makestruct(
                                    "prev" -> last.at(idx),
                                    "partialSums" -> group.toArray,
                                  )
                                }

                            val results =
                              cdaIR(
                                groups,
                                maketuple(),
                                "table_scan_down_pass",
                                strConcat("iteration=", iteration, ", level=", level),
                              ) { case (ctx, _) =>
                                val elt = Ref(freshName(), TString)
                                ToArray(RunAggScan(
                                  ToStream(ctx.get("partialSums"), true),
                                  elt.name,
                                  ReadValue(ctx.get("prev"), reader, reader.spec.encodedVirtualType)
                                    .bind(aggSigs.initFromSerializedValueOp),
                                  ReadValue(elt, reader, reader.spec.encodedVirtualType)
                                    .bind(aggSigs.combOpValues),
                                  WriteValue(aggSigs.valuesOp, Str(tmpDir) + UUID4(), writer),
                                  aggSigs.states,
                                ))
                              }

                            recur(
                              FastSeq(
                                level - 1,
                                results.stream.streamFlatten.toArray,
                                iteration + 1,
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
                lcWithInitBinding.mapCollectWithGlobals("table_scan_prefix_sums_singlestage") {
                  rows =>
                    Let(
                      FastSeq(TableIR.globalName -> lc.globals),
                      RunAgg(
                        Begin(FastSeq(
                          initFromSerializedStates,
                          StreamFor(rows, TableIR.rowName, aggs.seqPerElt.deepCopy),
                        )),
                        aggSigs.valuesOp,
                        aggSigs.states,
                      ),
                    )
                } { case (collected, globals) =>
                  Let(
                    FastSeq(TableIR.globalName -> globals),
                    ToStream(collected, requiresMemoryManagementPerElement = true)
                      .streamScan(initStateRef) { (acc, value) =>
                        RunAgg(
                          Begin(FastSeq(
                            aggSigs.initFromSerializedValueOp(acc),
                            aggSigs.combOpValues(value),
                          )),
                          aggSigs.valuesOp,
                          aggSigs.states,
                        )
                      }
                      .take(collected.len)
                      .toArray,
                  )
                }

              (partitionAggs, identity(_))
            }

          val partitionPrefixSumsRef = Ref(freshName(), partitionPrefixSumValues.typ)
          TableStage(
            letBindings =
              lc.letBindings :+ (partitionPrefixSumsRef.name -> partitionPrefixSumValues),
            broadcastVals = lc.broadcastVals,
            partitioner = lc.partitioner,
            dependency = lc.dependency,
            globals = lc.globals,
            contexts = zipIR(
              FastSeq(lc.contexts, ToStream(partitionPrefixSumsRef)),
              ArrayZipBehavior.AssertSameLength,
            ) { case Seq(oldContext, scanState) =>
              makestruct("oldContext" -> oldContext, "scanState" -> scanState)
            },
            partition = ctx =>
              M.eval {
                for {
                  oldContext <- ctx.get("oldContext")
                  scanState <- ctx.get("scanState") map transformPrefixSum
                  _ <- TableIR.globalName -> lc.globals
                } yield RunAggScan(
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
            val (typeOfRootStruct, _) =
              right.typ.rowType.filterSet(right.typ.key.toSet, include = false)

            joinRightDistinctIR(
              leftPart,
              rightPart,
              left.typ.key.take(commonKeyLength),
              right.typ.key,
              "left",
            )((l, r) => l.insert(root -> r.select(typeOfRootStruct.fieldNames)))
          },
        )

      case TableIntervalJoin(left, right, root, product) =>
        lower(left).intervalAlignAndZipPartitions(
          ctx,
          lower(right),
          analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable].rowType,
          (lGlobals, _) => lGlobals,
          { (lstream, rstream) =>
            if (product)
              leftIntervalJoinIR(
                lstream,
                rstream,
                left.typ.key.head,
                right.typ.keyType.fields(0).name,
              )((l, r) => l.insert(root -> mapArray(r)(_.select(right.typ.valueType.fieldNames))))
            else
              joinRightDistinctIR(lstream, rstream, left.typ.key, right.typ.key, "left") { (l, r) =>
                l.insert(root -> r.select(right.typ.valueType.fieldNames))
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
              MakeTuple.ordered(ctxRefs.map(_.ir))
            },
            ctxRef =>
              StreamMultiMerge(
                repartitioned.indices.map(i =>
                  bindIR(ctxRef.get(i))(ctx => repartitioned(i).partition(ctx))
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
        val newGlobals = makestruct(
          globalName -> MakeArray(
            repartitioned.map(_.globals.ir),
            TArray(repartitioned.head.globalType),
          )
        )
        val globalsRef = Ref(freshName(), newGlobals.typ)

        TableStage(
          repartitioned.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals,
          repartitioned.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
          globalsRef,
          newPartitioner,
          TableStageDependency.union(repartitioned.map(_.dependency)),
          zipIR(repartitioned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
            MakeTuple.ordered(ctxRefs.map(_.ir))
          },
          ctx =>
            zipJoin2IR(
              repartitioned.indices.map(i =>
                bindIR(ctx.get(i))(ctx => repartitioned(i).partition(ctx))
              ),
              keyType.fieldNames,
            ) { (key, values) =>
              key.insert(
                fieldName ->
                  values
                    .stream
                    .streamMap(_.select(children.head.typ.valueType.fieldNames))
                    .toArray
              )
            },
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
            IRBuilder.scoped { ib =>
              val N = path.length

              val refs = new Array[Atom](N)
              val last = (0 until N).foldLeft(row) { (ref, i) =>
                refs(i) = ref
                ib.memoize(ref.get(path(i)))
              }

              mapIR(ToStream(last, requiresMemoryManagementPerElement = true)) { elt =>
                path.zip(refs.unsafeToArraySeq).foldRight[IR](elt) { case ((p, ref), inserted) =>
                  ref.insert(p -> inserted)
                }
              }
            }
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
          contexts = lc.contexts.grouped(groupSize).streamMap(ToArray(_)),
          partition = _.stream.streamFlatMap(lc.partition),
        )

      case TableRename(child, rowMap, globalMap) =>
        val loweredChild = lower(child)
        val newGlobals = loweredChild.globals.rename(_.rename(globalMap))
        val global = Ref(freshName(), newGlobals.typ)

        TableStage(
          loweredChild.letBindings :+ global.name -> newGlobals,
          loweredChild.broadcastVals :+ global.name -> global.ir,
          global.ir,
          loweredChild.partitioner.copy(kType = loweredChild.kType.rename(rowMap)),
          loweredChild.dependency,
          loweredChild.contexts,
          loweredChild.partition(_).streamMap(_.rename(_.rename(rowMap))),
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
        val keepSet = seq.toSet
        keepSet.foreach { p =>
          val N = lc.numPartitions
          if (p < 0 || p >= N)
            fatal(s"_filter_partitions: no partition with index $p (num partitions = $N)")
        }

        val keepSetRef = Ref(freshName(), TSet(TInt32))
        lc.copy(
          letBindings =
            lc.letBindings :+ keepSetRef.name -> Literal(TSet(TInt32), keepSet),
          partitioner =
            lc.partitioner.copy(
              rangeBounds =
                lc.partitioner
                  .rangeBounds
                  .zipWithIndex
                  .flatMap { case (interval, idx) =>
                    if (keep == keepSet.contains(idx)) Some(interval)
                    else None
                  }
            ),
          contexts =
            flatten(zip2(lc.contexts, iota(0, 1), ArrayZipBehavior.TakeMinLength) {
              (elt, idx) =>
                val `contains?` = invoke("contains", TBoolean, keepSetRef, idx)
                If(
                  if (keep) `contains?` else !`contains?`,
                  MakeStream(elt),
                  MakeStream.empty(elt.typ),
                )
            }),
        )

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
        ts.mapPartitionWithContext { (partition, _) =>
          flatMapIR(partition) { row =>
            M.eval {
              for {
                block <- row.get("block")
                rowOffset <- row.get("blockRow").toL * I64(bmir.typ.blockSize.toLong)
                colOffset <- row.get("blockCol").toL * I64(bmir.typ.blockSize.toLong)

                shape <- NDArrayShape(block)
                numRows <- shape.get(0).toI
                numCols <- shape.get(1).toI
              } yield rangeIR(numRows).streamMap(_.toL).streamFlatMap { rowIdx =>
                rangeIR(numCols).streamMap(_.toL).streamMap { colIdx =>
                  makestruct(
                    "i" -> (rowIdx + rowOffset),
                    "j" -> (colIdx + colOffset),
                    "entry" -> NDArrayRef(block, FastSeq(rowIdx, colIdx), ErrorIDs.NO_ERROR),
                  )
                }
              }
            }
          }
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

  private def nPartsAndRows(counts: IndexedSeq[Long], target: Long)(f: (Long, Long) => Long)
    : (Int, Int) = {
    val N = counts.length

    var p = 0
    var rows = 0L
    var remaining = target

    while (p < N && remaining > 0L) {
      val size = counts(p)
      val take = math.min(size, remaining)
      rows = f(size, take)
      remaining -= take
      p += 1
    }

    (p, rows.toInt)
  }

  private def nPartsAndRows(
    ts: TableStage,
    contexts: Atom,
    nPartitions: Atom,
    target: Long,
  )(
    f: (Atom, Atom) => IR
  ): IR = {
    val tresult = TTuple(TInt32, TInt32)
    val nPartsToRead = if (target == 1L) 1 else 4
    tailLoop(tresult, 0, target, 0, nPartsToRead) {
      case (outer, Seq(i, remainder, m, n)) =>
        M.eval {
          for {
            contexts <- contexts.stream.drop(m).take(n)

            counts <-
              ts
                .copy(letBindings = ArraySeq.empty, contexts = contexts)
                .mapCollect(
                  "table_head_or_tail_recursive_count",
                  strConcat("iteration=", i, ",nParts=", n),
                )(_.len)

            n <- counts.len
            p <- m + n
            eof <- p >= nPartitions

          } yield tailLoop(tresult, 0, remainder, 0L) {
            case (inner, Seq(j, remainder, count)) =>
              If(
                (remainder <= 0L) || ((j >= n) && eof),
                maketuple(m + j, count.toI),
                If(
                  j < n,
                  counts.at(j).toL.bind(c => inner(FastSeq(j + 1, remainder - c, f(c, remainder)))),
                  outer(FastSeq(i + 1, remainder, p, n * 3)),
                ),
              )
          }
        }
    }
  }

}
