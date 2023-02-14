package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.ArrayZipBehavior.AssertSameLength
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, WrappedMatrixToTableFunction}
import is.hail.expr.ir.{agg, _}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.methods.{ForceCountTable, LocalLDPrune, NPartitionsTable, TableFilterPartitions}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types._
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

class LowererUnsupportedOperation(msg: String = null) extends Exception(msg)

sealed trait PartitionSparsity {

  def includedIndices(nPartitions: Int): IndexedSeq[Int]
  def head(n: Int, nTot: Int): PartitionSparsity
  def tail(n: Int, nTot: Int): PartitionSparsity

  def intersect(other:PartitionSparsity): PartitionSparsity
  def union(other: PartitionSparsity): PartitionSparsity

  def subsetPartitioner(partitioner: RVDPartitioner): RVDPartitioner
}
object PartitionSparsity {
  case object Dense extends PartitionSparsity {

    def includedIndices(nPartitions: Int): IndexedSeq[Int] = 0 until nPartitions
    def head(n: Int, nTot: Int): PartitionSparsity = Sparse(Range(0, n))

    def tail(n: Int, nTot: Int): PartitionSparsity = Sparse(Range(nTot - n, nTot))

    override def intersect(other: PartitionSparsity): PartitionSparsity = other

    override def union(other: PartitionSparsity): PartitionSparsity = this

    def subsetPartitioner(partitioner: RVDPartitioner): RVDPartitioner = partitioner
  }
  case class Sparse(nonEmptyIndices: IndexedSeq[Int]) extends PartitionSparsity {

    def includedIndices(nPartitions: Int): IndexedSeq[Int] = nonEmptyIndices
    def head(n: Int, nTot: Int): PartitionSparsity = Sparse(nonEmptyIndices.take(n))

    def tail(n: Int, nTot: Int): PartitionSparsity = Sparse(nonEmptyIndices.takeRight(n))

    override def intersect(other: PartitionSparsity): PartitionSparsity = other match {
      case Dense => this
      case Sparse(inds2) =>
        val inds2Set = inds2.toSet
        Sparse(nonEmptyIndices.filter(inds2Set.contains))
    }

    override def union(other: PartitionSparsity): PartitionSparsity = other match {
      case Dense => Dense
      case Sparse(inds2) =>
        Sparse((nonEmptyIndices ++ inds2)
        .distinct
        .sorted)
    }

    def subsetPartitioner(partitioner: RVDPartitioner): RVDPartitioner = {
      partitioner.copy(rangeBounds = nonEmptyIndices.map(partitioner.rangeBounds(_)).toArray[Interval])
    }
  }
}

object TableStage {
  def apply(
    globals: IR,
    partitioner: RVDPartitioner,
    partitionSparsity: PartitionSparsity,
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
      partitionSparsity,
      dependency,
      contexts,
      body)
  }

  def apply(
    letBindings: IndexedSeq[(String, IR)],
    broadcastVals: IndexedSeq[(String, IR)],
    globals: Ref,
    partitioner: RVDPartitioner,
    partitionSparsity: PartitionSparsity,
    dependency: TableStageDependency,
    contexts: IR,
    partition: Ref => IR
  ): TableStage = {
    val ctxType = contexts.typ.asInstanceOf[TStream].elementType
    val ctxRef = Ref(genUID(), ctxType)

    new TableStage(letBindings, broadcastVals, globals, partitioner, partitionSparsity, dependency, contexts, ctxRef.name, partition(ctxRef))
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
case class TableStage private(
  val letBindings: IndexedSeq[(String, IR)],
  val broadcastVals: IndexedSeq[(String, IR)],
  val globals: Ref,
  val partitioner: RVDPartitioner,
  val partitionSparsity: PartitionSparsity,
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

  assert(kType.isSubsetOf(rowType), s"Key type $kType is not a subset of $rowType")
  assert(broadcastVals.exists { case (name, value) => name == globals.name && value == globals})

  def partition(ctx: IR): IR = {
    require(ctx.typ == ctxType)
    Let(ctxRefName, ctx, partitionIR)
  }

  def numPartitions: Int = partitioner.numPartitions

  def nContexts: Int = partitionSparsity match {
    case PartitionSparsity.Dense => numPartitions
    case PartitionSparsity.Sparse(inds) => inds.length
  }

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

  def forceToDense(): TableStage = partitionSparsity match {
    case PartitionSparsity.Dense =>
      this
    case PartitionSparsity.Sparse(inds) =>
      copy(
        partitionSparsity = PartitionSparsity.Dense,
        partitioner = partitioner.copy(rangeBounds = inds.map(partitioner.rangeBounds(_)).toArray[Interval]))
  }

  def changeSparsity(newSparsity: PartitionSparsity): TableStage = {
    if (newSparsity == partitionSparsity)
      return this

    (partitionSparsity, newSparsity) match {
      case (PartitionSparsity.Dense, PartitionSparsity.Sparse(inds)) =>
        val keptIndices = Literal(TArray(TInt32), inds)
        val newCtxs = bindIR(ToArray(contexts)) { oldCtxs => mapIR(ToStream(keptIndices)) { i => ArrayRef(oldCtxs, i) }}
        copy(contexts = newCtxs, partitionSparsity = newSparsity)

      case (PartitionSparsity.Sparse(inds), _) =>
        val globalIdxToInds1 = inds.zipWithIndex.toMap

        val includedGlobalPartitionIndices = newSparsity match {
          case PartitionSparsity.Dense => (0 until numPartitions)
          case PartitionSparsity.Sparse(inds2) => inds2
        }

        val inputsToEachFinalPart = includedGlobalPartitionIndices
          .map { i =>
            globalIdxToInds1
              .get(i)
              .map(FastIndexedSeq(_))
              .getOrElse(FastIndexedSeq())
          }

        val inputRangeForEachDensePart = Literal(TArray(TArray(TInt32)), inputsToEachFinalPart)

        // TStream(TArray(prevCtxType))
        val newCtxs = bindIR(ToArray(contexts)) { oldCtxs: Ref =>
          mapIR(ToStream(inputRangeForEachDensePart)) { range =>
            ToArray(mapIR(ToStream(range)) { i => ArrayRef(oldCtxs, i) }
            )
          }
        }

        TableStage(letBindings = letBindings,
          broadcastVals = broadcastVals,
          globals = globals,
          partitioner = partitioner,
          partitionSparsity = newSparsity,
          dependency = dependency,
          contexts = newCtxs,
          partition = (range: Ref) => flatMapIR(ToStream(range, requiresMemoryManagementPerElement = true)) { ctx => partition(ctx) })
    }
  }
  def zipPartitionsLeftJoin(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    zipPartitions(right.changeSparsity(this.partitionSparsity), newGlobals, body)
  }
  def zipPartitionsRightJoin(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    this.changeSparsity(right.partitionSparsity).zipPartitions(right, newGlobals, body)
  }

  def zipPartitionsIntersection(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    val newSparsity = this.partitionSparsity.intersect(right.partitionSparsity)
    this.changeSparsity(newSparsity).zipPartitions(right.changeSparsity(newSparsity), newGlobals, body)
  }
  def zipPartitionsUnion(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    val newSparsity = this.partitionSparsity.union(right.partitionSparsity)
    this.changeSparsity(newSparsity).zipPartitions(right.changeSparsity(newSparsity), newGlobals, body)
  }

  // 'body' must take all output key values from
  // left stream, and be monotonic on left stream (it can drop or duplicate
  // elements of left iterator, or insert new elements in order, but cannot
  // rearrange them)
  def zipPartitions(right: TableStage, newGlobals: (IR, IR) => IR, body: (IR, IR) => IR): TableStage = {
    val left = this
    val leftCtxTyp = left.ctxType
    val rightCtxTyp = right.ctxType

    assert(left.nContexts == right.nContexts)

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
      left.partitionSparsity,
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
    TableStage(letBindings, broadcastVals, globals, partitioner, partitionSparsity, dependency, newContexts, ctxRef => bindIR(getOldContext(ctxRef))(partition))
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
      broadcastVals.foldLeft(mapF(partitionIR, Ref(ctxRefName, ctxType))) { case (accum, (name, _)) =>
        Let(name, GetField(glob, name), accum)
      }, dynamicID, staticID, Some(dependency))

    TableStage.wrapInBindings(bindIR(cda) { cdaRef => body(cdaRef, globals) }, letBindings)
  }

  def collectWithGlobals(staticID: String, dynamicID: IR = NA(TString)): IR =
    mapCollectWithGlobals(staticID, dynamicID)(ToArray) { (parts, globals) =>
      MakeStruct(FastSeq(
        "rows" -> ToArray(flatMapIR(ToStream(parts))(ToStream(_))),
        "global" -> globals))
    }

  def countPerPartition(): IR = mapCollect("count_per_partition")(part => Cast(StreamLen(part), TInt64))

  def getGlobals(): IR = TableStage.wrapInBindings(globals, letBindings)

  def getNumPartitions(): IR = TableStage.wrapInBindings(StreamLen(contexts), letBindings)

  def changePartitionerNoRepartition(newPartitioner: RVDPartitioner): TableStage = {
    require(partitioner.numPartitions == newPartitioner.numPartitions)
    copy(partitioner = newPartitioner)
  }

  def strictify(allowedOverlap: Int = kType.size - 1): TableStage = {
    val newPart = partitioner.strictify(allowedOverlap)
    repartitionNoShuffle(newPart)
  }

  def repartitionNoShuffle(newPartitioner: RVDPartitioner, allowDuplication: Boolean = false, dropEmptyPartitions: Boolean = false): TableStage = {
    if (newPartitioner == this.partitioner) {
      return this
    }

    val oldSparsePartitioner = partitionSparsity match {
      case PartitionSparsity.Dense => partitioner
      case PartitionSparsity.Sparse(indices) => partitioner.copy(rangeBounds = indices.map(partitioner.rangeBounds).toArray[Interval])
    }

    if (!allowDuplication) {
      require(newPartitioner.satisfiesAllowedOverlap(newPartitioner.kType.size - 1))
    }
    require(newPartitioner.kType.isPrefixOf(kType))

    val startAndEnd = oldSparsePartitioner.rangeBounds.map(newPartitioner.intervalRange)
      .zipWithIndex
    val ord = PartitionBoundOrdering.apply(newPartitioner.sm, newPartitioner.kType)
    if (startAndEnd.forall { case ((start, end), index) =>
      start + 1 == end && newPartitioner.rangeBounds(start).includes(ord, oldSparsePartitioner.rangeBounds(index)) }) {
      val newToOld = startAndEnd
        .groupBy(_._1._1)
        .map { case (newIdx, values) => (newIdx, values.map(_._2).sorted.toFastIndexedSeq) }


      val (oldPartIndices, newPartitionerFilt) = if (dropEmptyPartitions) {
        val indices = (0 until newPartitioner.numPartitions).filter(newToOld.contains)
        (indices.map(i => newToOld(i)), newPartitioner.copy(rangeBounds = indices.toArray.map(i => newPartitioner.rangeBounds(i))))
      } else
        ((0 until newPartitioner.numPartitions).map(i => newToOld.getOrElse(i, FastIndexedSeq())), newPartitioner)
      log.info(s"repartitionNoShuffle - fast path, generated ${oldPartIndices.length} partitions from ${oldSparsePartitioner.numPartitions}" +
        s" (dropped ${newPartitioner.numPartitions - oldPartIndices.length} empty output parts)")

      val newContexts = bindIR(ToArray(contexts)) { oldCtxs =>
        mapIR(ToStream(Literal(TArray(TArray(TInt32)), oldPartIndices))) { inds =>
          ToArray(mapIR(ToStream(inds)) { i => ArrayRef(oldCtxs, i) })
        }
      }
      return TableStage(letBindings, broadcastVals, globals, newPartitionerFilt, PartitionSparsity.Dense, dependency, newContexts,
        (ctx: Ref) => flatMapIR(ToStream(ctx, true)) { oldCtx => partition(oldCtx) })
    }

    val boundType = RVDPartitioner.intervalIRRepresentation(newPartitioner.kType)
    val partitionMapping: IndexedSeq[Row] = newPartitioner.rangeBounds.map { i =>
      Row(RVDPartitioner.intervalToIRRepresentation(i, newPartitioner.kType.size), oldSparsePartitioner.queryInterval(i))
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
    val newStage = TableStage(letBindings, broadcastVals, globals, newPartitioner, PartitionSparsity.Dense, dependency, newContexts,
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
    if (RVDPartitioner.isValid(partitioner.sm, newKeyType, partitioner.rangeBounds)) {
      changePartitionerNoRepartition(partitioner.copy(kType = newKeyType))
    } else {
      val adjustedPartitioner = partitioner.strictify()
      repartitionNoShuffle(adjustedPartitioner)
        .changePartitionerNoRepartition(adjustedPartitioner.copy(kType = newKeyType))
    }
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
            InsertFields(row, FastIndexedSeq("__partNum" -> partNum))
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
    val repartitioned = sorted.repartitionNoShuffle(newRightPartitioner)
      .changePartitionerNoRepartition(RVDPartitioner.unkeyed(ctx.stateManager, newRightPartitioner.numPartitions))
      .mapPartition(None) { part =>
        mapIR(part) { row =>
          SelectFields(row, right.rowType.fieldNames)
        }
      }
    zipPartitions(repartitioned, globalJoiner, joiner)
  }

  def strictify(p: RequestedPartitioning, allowedOverlap: Int): TableStage = {
    p match {
      case UseThisPartitioning(p) =>
        assert(p.allowedOverlap <= allowedOverlap)
        assert(this.partitioner == p)
        this
      case UseTheDefaultPartitioning =>
        repartitionNoShuffle(this.partitioner.strictify(allowedOverlap))
    }
  }
}

object LowerTableIR {
  def apply(ir: IR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): IR = {
    def lower(tir: TableIR): TableStage = {
      val partitioner = PlanPartitioning.analyze(ctx, tir).chooseBest()
      this.lowerTable(tir, partitioner, typesToLower, ctx, analyses)
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
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType))

        val minkey = AggSignature(TakeBy(),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))

        val maxkey = AggSignature(TakeBy(Descending),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))


        bindIR(flatten(stage.mapCollect("table_calculate_new_partitions") { rows =>
          streamAggIR(mapIR(rows) { row => SelectFields(row, keyType.fieldNames)}) { elt =>
            ToArray(flatMapIR(ToStream(
              MakeArray(
                ApplyAggOp(
                  FastIndexedSeq(I32(samplesPerPartition)),
                  FastIndexedSeq(elt),
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
                  MakeArray(FastIndexedSeq(), TArray(TInterval(keyType))),
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

        val branchFactor = HailContext.get.branchingFactor
        val useTreeAggregate = aggs.shouldTreeAggregate && branchFactor < lc.numPartitions
        val isCommutative = aggs.isCommutative
        log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
        log.info(s"Aggregate: commutative=${ isCommutative }")

        if (useTreeAggregate) {
          val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

          val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate")({ part: IR =>
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

            def combineGroup(partArrayRef: IR, useInitStates: Boolean): IR = {
              Begin(FastIndexedSeq(
                if (useInitStates) {
                  initFromSerializedStates
                } else {
                  bindIR(ReadValue(ArrayRef(partArrayRef, 0), codecSpec, codecSpec.encodedVirtualType)) { serializedTuple =>
                    Begin(
                      aggs.aggs.zipWithIndex.map { case (sig, i) =>
                        InitFromSerializedValue(i, GetTupleElement(serializedTuple, i), sig.state)
                      })
                  }
                },
                forIR(StreamRange(if (useInitStates) 0 else 1, ArrayLen(partArrayRef), 1, requiresMemoryManagementPerElement = true)) { fileIdx =>

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
                Recur(treeAggFunction,
                  FastIndexedSeq(
                    CollectDistributedArray(
                      mapIR(StreamGrouped(ToStream(currentAggStates), I32(branchFactor)))(x => ToArray(x)),
                      MakeStruct(FastSeq()),
                      distAggStatesRef.name,
                      genUID(),
                      RunAgg(
                        combineGroup(distAggStatesRef, false),
                        WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                        aggs.states
                      ),
                      strConcat(Str("iteration="), invoke("str", TString, iterNumber), Str(", n_states="), invoke("str", TString, ArrayLen(currentAggStates))),
                      "table_tree_aggregate"),
                    iterNumber + 1),
                  currentAggStates.typ)))
            ) { finalParts =>
              RunAgg(
                combineGroup(finalParts, true),
                Let("global", globals,
                  Let(resultUID, results,
                    aggs.postAggIR)),
                aggs.states
              )
            }
          }
        }
        else {
          lcWithInitBinding.mapCollectWithGlobals("table_aggregate_singlestage")({ part: IR =>
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
                  forIR(ToStream(collected, requiresMemoryManagementPerElement = true)) { state =>
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
        writer.lower(ctx, lower(child), child, tcoerce[RTable](analyses.requirednessAnalysis.lookup(child)))

      case node if node.children.exists(_.isInstanceOf[TableIR]) =>
        throw new LowererUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(ctx, node) }")
    }
    lowered
  }

  def choosePartitioningAndLowerTable(tir: TableIR, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): TableStage = {
   lowerTable(tir, PlanPartitioning.analyze(ctx, tir).chooseBest(), typesToLower, ctx, analyses)
  }

  def lowerTable(tir: TableIR, requestedPartitioner: RequestedPartitioning, typesToLower: DArrayLowering.Type, ctx: ExecuteContext, analyses: LoweringAnalyses): TableStage = {
    def lowerIR(ir: IR): IR = {
      LowerToCDA.lower(ir, typesToLower, ctx, analyses)
    }

    def lower(tir: TableIR, requestedPartitioner: RequestedPartitioning = requestedPartitioner): TableStage = {
      this.lowerTable(tir, requestedPartitioner, typesToLower, ctx, analyses)
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
            RVDPartitioner.empty(ctx, typ.keyType),
            PartitionSparsity.Dense,
            TableStageDependency.none,
            MakeStream(FastIndexedSeq(), TStream(TStruct.empty)),
            (_: Ref) => MakeStream(FastIndexedSeq(), TStream(typ.rowType)))
        } else
          reader.lowerPartitioned(ctx, typ, requestedPartitioner)

      case TableParallelize(rowsAndGlobal, nPartitions) =>
        val nPartitionsAdj = nPartitions.getOrElse(16)

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
          FastIndexedSeq(loweredRowsAndGlobalRef.name -> loweredRowsAndGlobal,
            globalsRef.name -> GetField(loweredRowsAndGlobalRef, "global")),
          FastIndexedSeq(globalsRef.name -> globalsRef),
          globalsRef,
          RVDPartitioner.unkeyed(ctx.stateManager, nPartitionsAdj),
          PartitionSparsity.Dense,
          TableStageDependency.none,
          context,
          ctxRef => ToStream(ctxRef, true))

      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        val loweredGlobals = lowerIR(globals)
        TableStage(
          loweredGlobals,
          partitioner = partitioner,
          PartitionSparsity.Dense,
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
            val rows = Let(cname, GetTupleElement(in, 2), Let(gname, loweredGlobals, body))
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

        val stage = TableStage(
          MakeStruct(FastSeq()),
          new RVDPartitioner(ctx.stateManager, Array("idx"), tir.typ.rowType,
            ranges.map { case (start, end) =>
              Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
            }),
          PartitionSparsity.Dense,
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), ranges.map(Row.fromTuple).toFastIndexedSeq)),
          (ctxRef: Ref) => mapIR(StreamRange(GetField(ctxRef, "start"), GetField(ctxRef, "end"), I32(1), true)) { i =>
            MakeStruct(FastSeq("idx" -> i))
          })

        requestedPartitioner match {
          case UseThisPartitioning(p) => stage.repartitionNoShuffle(p)
          case UseTheDefaultPartitioning => stage
        }

      case TableMapGlobals(child, newGlobals) =>
        lower(child).mapGlobals(old => Let("global", old, newGlobals))

      case TableAggregateByKey(child, expr) =>
        val lc = lower(child)
        val loweredChild = requestedPartitioner match {
          case UseThisPartitioning(p) => lc
          case UseTheDefaultPartitioning =>
            lc.strictify(requestedPartitioner, child.typ.key.length - 1)
        }

        loweredChild.mapPartition(Some(child.typ.key)) { partition =>
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

      case TableDistinct(child) =>
        val lc = lower(child)
        val loweredChild = requestedPartitioner match {
          case UseThisPartitioning(p) => lc
          case UseTheDefaultPartitioning =>
            lc.strictify(requestedPartitioner, child.typ.key.length - 1)
        }

        loweredChild
          .mapPartition(None) { partition =>
            flatMapIR(StreamGroupByKey(partition, child.typ.key, missingEqual = true)) { groupRef =>
              StreamTake(groupRef, 1)
            }
          }

      case TableFilterIntervals(child, intervals, keep) =>
        val loweredChild = lower(child)
        val part = loweredChild.partitioner
        val kt = child.typ.keyType
        val ord = PartitionBoundOrdering(ctx.stateManager, kt)
        val iord = ord.intervalEndpointOrdering

        val filterPartitioner = new RVDPartitioner(ctx.stateManager, kt, Interval.union(intervals.toArray, ord.intervalEndpointOrdering))
        val boundsType = TArray(RVDPartitioner.intervalIRRepresentation(kt))
        val filterIntervalsRef = Ref(genUID(), boundsType)
        val filterIntervals: IndexedSeq[Interval] = filterPartitioner.rangeBounds.map { i =>
          RVDPartitioner.intervalToIRRepresentation(i, kt.size)
        }

        val includedPartsAndBounds = loweredChild.partitionSparsity.includedIndices(part.numPartitions)
          .zipWithIndex
          .map { case (globalIdx, localIdx) => (part.rangeBounds(globalIdx), (localIdx, globalIdx)) }


        val (newRangeBounds, includedIndices, startAndEndInterval, f) = if (keep) {
          val (newRangeBounds, includedIndices, startAndEndInterval) = includedPartsAndBounds.flatMap { case (interval, idxs) =>
            if (filterPartitioner.overlaps(interval)) {
              Some((interval, idxs, (filterPartitioner.lowerBoundInterval(interval), filterPartitioner.upperBoundInterval(interval))))
            } else None
          }.unzip3

          def f(partitionIntervals: IR, key: IR): IR =
            invoke("partitionerContains", TBoolean, partitionIntervals, key)
          (newRangeBounds, includedIndices, startAndEndInterval, f _)
        } else {
          // keep = False
          val (newRangeBounds, includedIndices, startAndEndInterval) = includedPartsAndBounds.flatMap { case (interval, idxs) =>
            val lowerBound = filterPartitioner.lowerBoundInterval(interval)
            val upperBound = filterPartitioner.upperBoundInterval(interval)
            if ((lowerBound until upperBound).map(filterPartitioner.rangeBounds).exists { filterInterval =>
              iord.compareNonnull(filterInterval.left, interval.left) <= 0 && iord.compareNonnull(filterInterval.right, interval.right) >= 0
            })
              None
            else Some((interval, idxs, (lowerBound, upperBound)))
          }.unzip3

          def f(partitionIntervals: IR, key: IR): IR =
            !invoke("partitionerContains", TBoolean, partitionIntervals, key)
          (newRangeBounds, includedIndices, startAndEndInterval, f _)
        }

        val (newPart, newSparsity) = requestedPartitioner match {
          case UseThisPartitioning(p) =>
            (p, PartitionSparsity.Sparse(includedIndices.map(_._2)))
          case UseTheDefaultPartitioning =>
            (new RVDPartitioner(ctx.stateManager, kt, newRangeBounds), PartitionSparsity.Dense)
        }

        TableStage(
          letBindings = loweredChild.letBindings,
          broadcastVals = loweredChild.broadcastVals ++ FastIndexedSeq((filterIntervalsRef.name, Literal(boundsType, filterIntervals))),
          loweredChild.globals,
          newPart,
          newSparsity,
          loweredChild.dependency,
          contexts = bindIRs(
            ToArray(loweredChild.contexts),
            Literal(TArray(TTuple(TInt32, TInt32)), startAndEndInterval.map(Row.fromTuple).toFastIndexedSeq)
          ) { case Seq(prevContexts, bounds) =>
            zip2(ToStream(Literal(TArray(TInt32), includedIndices.map(_._1).toFastIndexedSeq)), ToStream(bounds), ArrayZipBehavior.AssumeSameLength) { (idx, bound) =>
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

      case TableFilter(child, cond) =>
        val loweredChild = lower(child)
        loweredChild.mapPartition(None) { rows =>
          Let("global", loweredChild.globals,
            StreamFilter(rows, "row", cond))
        }

      case TableHead(child, targetNumRows) =>

        // figure out how many pattitions from child we need, and how many rows in each partition.
        // this requires a query!

        // the partitioner returned by TableHead can't be known without running that query
        // the partitioner of the TableStage returned by TableHead RIGHT NOW only contains intervals that are non-empty

        // right now, when we call lower(table: TableIR, UseThisPartitioning(p)), we assert that the partitioner of the
        // returned TableStage is EXACTLY p

        // TableCollect(TableHead(TableRead), 10)
        // if the first 10 rows are in the first partition of the table read, we want to only run one of the partition functions
        // instead of running all of the partition functions, most of which are returning no rows

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
                  .mapCollect("table_head_recursive_count",
                    strConcat(Str("iteration="), invoke("str", TString, iteration), Str(",nParts="), invoke("str", TString, howManyPartsToTryRef))
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
        val bindRelationLetsNewCtx = ToArray(letBindNewCtx)
        val newCtxSeq = CompileAndEvaluate(ctx, bindRelationLetsNewCtx).asInstanceOf[IndexedSeq[Any]]
        val numNewParts = newCtxSeq.length

        val (newPartitioner, newSparsity) = requestedPartitioner match {
          case UseThisPartitioning(p) =>
            (p, loweredChild.partitionSparsity.head(numNewParts, loweredChild.numPartitions))
          case UseTheDefaultPartitioning =>
            val newIntervals = loweredChild.partitioner.rangeBounds.slice(0,numNewParts)
            val newPartitioner = loweredChild.partitioner.copy(rangeBounds = newIntervals)
            (newPartitioner, PartitionSparsity.Dense)
        }

        TableStage(
          loweredChild.letBindings,
          loweredChild.broadcastVals,
          loweredChild.globals,
          newPartitioner,
          newSparsity,
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
                    .mapCollect("table_tail_recursive_count",
                      strConcat(Str("iteration="), invoke("str", TString, iteration), Str(", nParts="), invoke("str", TString, howManyPartsToTryRef)))(StreamLen)
                ) { counts =>
                  If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                    counts,
                    Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTryRef * 4, iteration + 1), TArray(TInt32)))
                })
          }
        }

        // First element is how many partitions to keep from the right partitionSizeArrayRef, second is how many to keep from first kept element.
        def answerTuple(partitionSizeArrayRef: Ref): IR = {
          bindIR(ArrayLen(partitionSizeArrayRef)) { numPartitions =>
            val howManyPartsToDrop = genUID()
            val i = Ref(genUID(), TInt32)
            val nRowsToRight = Ref(genUID(), TInt64)
            def makeAnswer(howManyParts: IR, howManyFromLast: IR) = MakeTuple.ordered(FastIndexedSeq(howManyParts, howManyFromLast))

            If(numPartitions ceq 0,
              makeAnswer(0, 0),
              TailLoop(
                howManyPartsToDrop,
                FastIndexedSeq(i.name -> 1, nRowsToRight.name -> 0L),
                If((i ceq numPartitions) || ((nRowsToRight + Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64)) >= targetNumRows),
                  makeAnswer(i, maxIR(0L, Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64) - (I64(targetNumRows) - nRowsToRight)).toI),
                  Recur(
                    howManyPartsToDrop,
                    FastIndexedSeq(
                      i + 1,
                      nRowsToRight + Cast(ArrayRef(partitionSizeArrayRef, numPartitions - i), TInt64)),
                    TTuple(TInt32, TInt32)))))
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
                    MakeStruct(FastIndexedSeq(
                      "numberToDrop" -> If(idx ceq 0, nToDropFromFirst, 0),
                      "old" -> ArrayRef(childContexts, idx + startIdx)))
                  }
                }
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
          loweredChild.partitionSparsity.tail(numNewParts, loweredChild.numPartitions),
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
          val branchFactor = HailContext.get.branchingFactor
          val big = aggs.shouldTreeAggregate && branchFactor < lc.numPartitions
          val (partitionPrefixSumValues, transformPrefixSum): (IR, IR => IR) = if (big) {
            val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

            val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
            val partitionPrefixSumFiles = lcWithInitBinding.mapCollectWithGlobals("table_scan_write_prefix_sums")({ part: IR =>
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
                          val cdaResult = cdaIR(contexts, MakeStruct(FastIndexedSeq()), "table_scan_up_pass",
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

                        val results = cdaIR(groups, MakeTuple.ordered(FastIndexedSeq()), "table_scan_down_pass",
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
            val partitionAggs = lcWithInitBinding.mapCollectWithGlobals("table_scan_prefix_sums_singlestage")({ part: IR =>
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
                    ToStream(collected, requiresMemoryManagementPerElement = true),
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
            partitionSparsity = lc.partitionSparsity,
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

      case t@TableKeyBy(child, newKey, isSorted: Boolean) =>
        val nPreservedFields = child.typ.keyType.fieldNames
          .zip(newKey)
          .takeWhile { case (l, r) => l == r }
          .length

        require(t.definitelyDoesNotShuffle)
        val loweredChild = if (newKey.isEmpty)
          choosePartitioningAndLowerTable(child, typesToLower, ctx, analyses)
        else
          lower(child)


        requestedPartitioner match {
          case UseThisPartitioning(p) => loweredChild.changePartitionerNoRepartition(p)
          case UseTheDefaultPartitioning =>
            loweredChild.changePartitionerNoRepartition(loweredChild.partitioner.coarsen(nPreservedFields))
              .extendKeyPreservesPartitioning(newKey)
        }

      case TableLeftJoinRightDistinct(left, right, root) =>
        val commonKeyLength = right.typ.keyType.size

        val (alignedLeft, alignedRight) = requestedPartitioner match {
          case UseThisPartitioning(p) => (lower(left), lower(right, UseThisPartitioning(p.coarsen(commonKeyLength))))
          case UseTheDefaultPartitioning =>
            val loweredLeft = lower(left)
            (loweredLeft, lower(right).repartitionNoShuffle(loweredLeft.partitioner
              .coarsen(commonKeyLength)
              .rename(left.typ.key.zip(right.typ.key).toMap)))
        }

        alignedLeft.zipPartitionsLeftJoin(
          alignedRight,
          (lGlobals, _) => lGlobals,
          (leftPart, rightPart) => {
            val leftElementRef = Ref(genUID(), left.typ.rowType)
            val rightElementRef = Ref(genUID(), right.typ.rowType)

            val (typeOfRootStruct, _) = right.typ.rowType.filterSet(right.typ.key.toSet, false)
            val rootStruct = SelectFields(rightElementRef, typeOfRootStruct.fieldNames.toIndexedSeq)
            val joiningOp = InsertFields(leftElementRef, FastIndexedSeq(root -> rootStruct))
            StreamJoinRightDistinct(
              leftPart, rightPart,
              left.typ.key.take(commonKeyLength), right.typ.key,
              leftElementRef.name, rightElementRef.name,
              joiningOp, "left")
          })

      case TableIntervalJoin(left, right, root, product) =>
        assert(!product)
        val loweredLeft = lower(left)
        val loweredRight = lower(right, requestedPartitioner = PlanPartitioning.analyze(ctx, right).chooseBest())

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
          (lGlobals, _) => lGlobals,
          partitionJoiner)

      case tj@TableJoin(left, right, joinType, joinKey) =>

        val lKeyFields = left.typ.key.take(joinKey)
        val lValueFields = left.typ.rowType.fieldNames.filter(f => !lKeyFields.contains(f))
        val rKeyFields = right.typ.key.take(joinKey)
        val rValueFields = right.typ.rowType.fieldNames.filter(f => !rKeyFields.contains(f))
        val lReq = analyses.requirednessAnalysis.lookup(left).asInstanceOf[RTable]
        val rReq = analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable]
        val rightKeyIsDistinct = analyses.distinctKeyedAnalysis.contains(right)

        val globalsJoiner: (IR, IR) => IR =  (lGlobals: IR, rGlobals: IR) => {
          val rGlobalType = rGlobals.typ.asInstanceOf[TStruct]
          val rGlobalRef = Ref(genUID(), rGlobalType)
          Let(rGlobalRef.name, rGlobals,
            InsertFields(lGlobals, rGlobalType.fieldNames.map(f => f -> GetField(rGlobalRef, f))))
        }

        val partitionJoiner: (IR, IR) => IR = (lPart, rPart) => {
          val lEltType = lPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
          val rEltType = rPart.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]

          val lKey = left.typ.keyType.fieldNames.take(joinKey)
          val rKey = right.typ.keyType.fieldNames.take(joinKey)

          val lEltRef = Ref(genUID(), lEltType)
          val rEltRef = Ref(genUID(), rEltType)

          StreamJoin(lPart, rPart, lKey, rKey, lEltRef.name, rEltRef.name, {
            MakeStruct(
              (lKeyFields, rKeyFields).zipped.map { (lKey, rKey) =>
                if (joinType == "outer" && lReq.field(lKey).required && rReq.field(rKey).required)
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey), Die("TableJoin expected non-missing key", left.typ.rowType.fieldType(lKey), -1)))
                else
                  lKey -> Coalesce(FastSeq(GetField(lEltRef, lKey), GetField(rEltRef, rKey)))
              }
                ++ lValueFields.map(f => f -> GetField(lEltRef, f))
                ++ rValueFields.map(f => f -> GetField(rEltRef, f)))
          }, joinType,
            requiresMemoryManagement = true, rightKeyIsDistinct = rightKeyIsDistinct)
        }

        val leftKeyToRightKeyMap = (left.typ.keyType.fieldNames.take(joinKey), right.typ.keyType.fieldNames.take(joinKey)).zipped.toMap

        val (alignedLeft, alignedRight) = requestedPartitioner match {
          case UseThisPartitioning(p) =>

            val leftRequestedPartitioning = p.coarsen(left.typ.key.length)
            val rightRequestedPartitioning = p.coarsen(joinKey).rename(leftKeyToRightKeyMap)
            (lower(left, UseThisPartitioning(leftRequestedPartitioning)), lower(right, UseThisPartitioning(rightRequestedPartitioning)))
          case UseTheDefaultPartitioning =>
            val loweredLeft = lower(left)
            val loweredRight = lower(right)
            val p1 = PartitionProposal.fromPartitioner(loweredLeft.partitioner)
            val p2 = PartitionProposal.fromPartitioner(loweredRight.partitioner)
            val joinedPartitioner = PlanPartitioning.joinedPlan(ctx, p1, p2, joinKey, joinType, tj.typ.keyType)
              .chooseBest().asInstanceOf[UseThisPartitioning].partitioner

            (loweredLeft.repartitionNoShuffle(joinedPartitioner.coarsen(left.typ.key.length)),
              loweredRight.repartitionNoShuffle(joinedPartitioner.coarsen(joinKey).rename(leftKeyToRightKeyMap)))
        }

        val joinedStage = joinType match {
          case "inner" => alignedLeft.zipPartitionsIntersection(alignedRight, globalsJoiner, partitionJoiner)
          case "outer" => alignedLeft.zipPartitionsUnion(alignedRight, globalsJoiner, partitionJoiner)
          case "left" => alignedLeft.zipPartitionsLeftJoin(alignedRight, globalsJoiner, partitionJoiner)
          case "right" => alignedLeft.zipPartitionsRightJoin(alignedRight, globalsJoiner, partitionJoiner)
        }
        assert(joinedStage.rowType == tj.typ.rowType)
        joinedStage

      case x@TableUnion(children) =>
        val lowered = children.map(lower(_))
        val keyType = x.typ.keyType
        val (alignedDiffSparsity, partitioner) = requestedPartitioner match {
          case UseThisPartitioning(part) => (lowered, part)
          case UseTheDefaultPartitioning =>
          val newPartitioner = RVDPartitioner.generate(ctx.stateManager, keyType, lowered.flatMap(_.partitioner.rangeBounds))
          (lowered.map(_.repartitionNoShuffle(newPartitioner)), newPartitioner)
        }
        val newSparsity = lowered.map(_.partitionSparsity).reduce(_ union _)
        val aligned = alignedDiffSparsity.map(_.changeSparsity(newSparsity))

        TableStage(
          aligned.flatMap(_.letBindings),
          aligned.flatMap(_.broadcastVals),
          aligned.head.globals,
          partitioner,
          newSparsity,
          TableStageDependency.union(aligned.map(_.dependency)),
          zipIR(aligned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
            MakeTuple.ordered(ctxRefs)
          },
          ctxRef =>
            StreamMultiMerge(aligned.indices.map(i => aligned(i).partition(GetTupleElement(ctxRef, i))), keyType.fieldNames)
        )

      case x@TableMultiWayZipJoin(children, fieldName, globalName) =>
        val lowered = children.map(lower(_))
        val keyType = x.typ.keyType
        val (alignedDiffSparsity, partitioner) = requestedPartitioner match {
          case UseThisPartitioning(part) => (lowered, part)
          case UseTheDefaultPartitioning =>
            val newPartitioner = RVDPartitioner.generate(ctx.stateManager, keyType, lowered.flatMap(_.partitioner.rangeBounds))
            (lowered.map(_.repartitionNoShuffle(newPartitioner)), newPartitioner)
        }

        val newSparsity = lowered.map(_.partitionSparsity).reduce(_ union _)
        val aligned = alignedDiffSparsity.map(_.changeSparsity(newSparsity))

        val newGlobals = MakeStruct(FastSeq(
          globalName -> MakeArray(lowered.map(_.globals), TArray(lowered.head.globalType))))
        val globalsRef = Ref(genUID(), newGlobals.typ)

        val keyRef = Ref(genUID(), keyType)
        val valsRef = Ref(genUID(), TArray(children.head.typ.rowType))
        val projectedVals = ToArray(mapIR(ToStream(valsRef)) { elt =>
          SelectFields(elt, children.head.typ.valueType.fieldNames)
        })

        TableStage(
          aligned.flatMap(_.letBindings) :+ globalsRef.name -> newGlobals,
          aligned.flatMap(_.broadcastVals) :+ globalsRef.name -> globalsRef,
          globalsRef,
          partitioner,
          newSparsity,
          TableStageDependency.union(aligned.map(_.dependency)),
          zipIR(aligned.map(_.contexts), ArrayZipBehavior.AssumeSameLength) { ctxRefs =>
            MakeTuple.ordered(ctxRefs)
          },
          ctxRef =>
            StreamZipJoin(
              aligned.indices.map(i => aligned(i).partition(GetTupleElement(ctxRef, i))),
              keyType.fieldNames,
              keyRef.name,
              valsRef.name,
              InsertFields(keyRef, FastSeq(fieldName -> projectedVals)))
        )

      case t@TableOrderBy(child, sortFields) =>
        require(t.definitelyDoesNotShuffle)
        assert(requestedPartitioner == UseTheDefaultPartitioning)

        val loweredChild = lower(child, requestedPartitioner = PlanPartitioning.analyze(ctx, child).chooseBest()).forceToDense()
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
            refs.tail.zip(roots).foldRight(
              mapIR(ToStream(refs.last, true)) { elt =>
                path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
                  InsertFields(ref, FastSeq(p -> inserted))
                }
              }) { case ((ref, root), accum) =>  Let(ref.name, root, accum) }
          }
        }

      case TableRepartition(child, n, RepartitionStrategy.NAIVE_COALESCE) =>
        requestedPartitioner match {
          case UseThisPartitioning(_) => lower(child)
          case UseTheDefaultPartitioning =>
            val lc = lower(child)
            val groupSize = (lc.numPartitions + n - 1) / n
            assert(lc.partitionSparsity == PartitionSparsity.Dense)
            TableStage(
              letBindings = lc.letBindings,
              broadcastVals = lc.broadcastVals,
              globals = lc.globals,
              partitioner = lc.partitioner.naiveCoalesce(groupSize),
              PartitionSparsity.Dense,
              dependency = lc.dependency,
              contexts = mapIR(StreamGrouped(lc.contexts, groupSize)) { group => ToArray(group) },
              partition = (r: Ref) => flatMapIR(ToStream(r)) { prevCtx => lc.partition(prevCtx) }
            )
        }

      case TableRename(child, rowMap, globalMap) =>
        val (loweredChild, partitioner) = requestedPartitioner match {
          case UseThisPartitioning(p) =>
            val revRowMap = rowMap.map { case (k, v) => (v, k) }
            val lc = lower(child, UseThisPartitioning(p.rename(revRowMap)))
            (lc, p)
          case UseTheDefaultPartitioning =>
            val lc = lower(child)
            (lc, lc.partitioner.rename(rowMap))
        }
        val newGlobals =
          CastRename(loweredChild.globals, loweredChild.globals.typ.asInstanceOf[TStruct].rename(globalMap))
        val newGlobalsRef = Ref(genUID(), newGlobals.typ)

        TableStage(
          loweredChild.letBindings :+ newGlobalsRef.name -> newGlobals,
          loweredChild.broadcastVals :+ newGlobalsRef.name -> newGlobalsRef,
          newGlobalsRef,
          partitioner,
          loweredChild.partitionSparsity,
          loweredChild.dependency,
          loweredChild.contexts,
          (ctxRef: Ref) => mapIR(loweredChild.partition(ctxRef)) { row =>
            CastRename(row, row.typ.asInstanceOf[TStruct].rename(rowMap))
          })

      case TableMapPartitions(child, globalName, partitionStreamName, body, _, allowedOverlap) =>
        val lc = lower(child)
        val loweredChild = requestedPartitioner match {
          case UseThisPartitioning(p) => lc
          case UseTheDefaultPartitioning => lc.strictify(allowedOverlap)
        }

        loweredChild.mapPartition(Some(child.typ.key)) { part =>
          Let(globalName, loweredChild.globals, Let(partitionStreamName, part, body))
        }

      case TableLiteral(typ, rvd, enc, encodedGlobals) =>
        val stage = RVDToTableStage(rvd, EncodedLiteral(enc, encodedGlobals))
        requestedPartitioner match {
          case UseThisPartitioning(p) => stage.repartitionNoShuffle(p)
          case UseTheDefaultPartitioning => stage
        }

      case TableToTableApply(child, WrappedMatrixToTableFunction(localLDPrune: LocalLDPrune, colsFieldName, entriesFieldName, _)) =>
        val lc = lower(child)
        lc.mapPartition(Some(child.typ.key)) { rows =>
          localLDPrune.makeStream(rows, entriesFieldName, ArrayLen(GetField(lc.globals, colsFieldName)))
        }.mapGlobals(_ => makestruct())

      case TableToTableApply(child, TableFilterPartitions(seq, keep)) =>

        val lc = lower(child)
        val filterSet = seq.toSet

        // TODO: implement the below in a way that makes it clear that we're doing set intersection and diff operations
        val filterSparsity = PartitionSparsity.Sparse(lc.partitionSparsity match {
          case PartitionSparsity.Sparse(indices) =>
            if (keep)
              indices.zipWithIndex.filter { case (_, sparseIdx) => filterSet.contains(sparseIdx) }.map(_._1)
            else
              indices.zipWithIndex.filter { case (_, sparseIdx) => !filterSet.contains(sparseIdx) }.map(_._1)
          case PartitionSparsity.Dense =>
            if (keep)
              (0 until lc.numPartitions).filter(filterSet.contains)
            else
              (0 until lc.numPartitions).filter(!filterSet.contains(_))
        })

        val filtered = lc.changeSparsity(filterSparsity)

        requestedPartitioner match {
          case UseThisPartitioning(p) => filtered
          case UseTheDefaultPartitioning => filtered.forceToDense()
        }

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
        throw new LowererUnsupportedOperation(s"undefined: \n${ Pretty(ctx, node) }")
    }

    requestedPartitioner match {
      case UseThisPartitioning(p) =>
        assert(p == lowered.partitioner, tir.getClass.getName)
      case UseTheDefaultPartitioning =>
        assert(lowered.partitionSparsity == PartitionSparsity.Dense, tir.getClass.getName)
    }

    assert(tir.typ.globalType == lowered.globalType, s"\n  ir global: ${tir.typ.globalType}\n  lowered global: ${lowered.globalType}")
    assert(tir.typ.rowType == lowered.rowType, s"\n  ir row: ${tir.typ.rowType}\n  lowered row: ${lowered.rowType}")
    assert(lowered.key startsWith tir.typ.keyType.fieldNames, s"\n  ir key: ${tir.typ.keyType.fieldNames.toSeq}\n  lowered key: ${lowered.key}")

    lowered
  }
}
