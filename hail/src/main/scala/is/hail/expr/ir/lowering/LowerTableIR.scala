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

  def mapCollect(relationalBindings: Map[String, IR])(f: IR => IR): IR = {
    mapCollectWithGlobals(relationalBindings)(f) { (parts, globals) => parts }
  }

  def mapCollectWithGlobals(relationalBindings: Map[String, IR])(mapF: IR => IR)(body: (IR, IR) => IR): IR =
    mapCollectWithContextsAndGlobals(relationalBindings)((part, ctx) => mapF(part))(body)

  // mapf is (part, ctx) => ???, body is (parts, globals) => ???
  def mapCollectWithContextsAndGlobals(relationalBindings: Map[String, IR])(mapF: (IR, Ref) => IR)(body: (IR, IR) => IR): IR = {
    val broadcastRefs = MakeStruct(broadcastVals)
    val glob = Ref(genUID(), broadcastRefs.typ)

    val cda = CollectDistributedArray(
      contexts, broadcastRefs,
      ctxRefName, glob.name,
      broadcastVals.foldLeft(mapF(partitionIR, Ref(ctxRefName, ctxType))) { case (accum, (name, _)) =>
        Let(name, GetField(glob, name), accum)
      }, Some(dependency))

    LowerToCDA.substLets(TableStage.wrapInBindings(bindIR(cda) { cdaRef => body(cdaRef, globals) }, letBindings), relationalBindings)
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


  // inserts a value/stage boundary that guarantees correct partition index seeding for directly
  // downstream operations
  def randomnessBoundary(ctx: ExecuteContext): TableStage = {
    new TableValueIntermediate(new TableStageIntermediate(this).asTableValue(ctx)).asTableStage(ctx)
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
          stage.mapCollect(relationalLetsAbove)(rows => Cast(StreamLen(rows), TInt64)))

      case TableToValueApply(child, ForceCountTable()) =>
        val stage = lower(child)
        invoke("sum", TInt64,
          stage.mapCollect(relationalLetsAbove)(rows => foldIR(mapIR(rows)(row => Consume(row)), 0L)(_ + _)))

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


        bindIR(flatten(stage.mapCollect(relationalLetsAbove) { rows =>
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
          bindIR(ToArray(flatMapIR(StreamGroupByKey(ToStream(sorted), keyType.fieldNames)) { groupRef =>
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
        lower(child).collectWithGlobals(relationalLetsAbove)

      case ta@TableAggregate(child, query) =>
        LowerTableIRHelpers.lowerTableAggregate(ctx, ta, lower(child), analyses.requirednessAnalysis, relationalLetsAbove)

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

      case tr: TableRange =>
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

      // This ignores nPartitions. The shuffler should handle repartitioning downstream
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val loweredChild = lower(child)
        val newKeyType = newKey.typ.asInstanceOf[TStruct]
        val resultUID = genUID()

        val aggs@Aggs(postAggIR, init, seq, aggSigs) = Extract(expr, resultUID, analyses.requirednessAnalysis)

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
            mapIR(StreamGroupByKey(partition, newKeyType.fieldNames.toIndexedSeq)) { groupRef =>
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

      case th@TableHead(child, targetNumRows) =>
        val loweredChild = lower(child)
        LowerTableIRHelpers.lowerTableHead(ctx, th, loweredChild, relationalLetsAbove)

      case tt@TableTail(child, targetNumRows) =>
        val loweredChild = lower(child)
        LowerTableIRHelpers.lowerTableTail(ctx, tt, loweredChild, relationalLetsAbove)

      case tmr@TableMapRows(child, newRow) =>
        LowerTableIRHelpers.lowerTableMapRows(ctx, tmr, lower(child), analyses.requirednessAnalysis, relationalLetsAbove)

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

      case tj@TableLeftJoinRightDistinct(left, right, root) =>
        LowerTableIRHelpers.lowerTableLeftJoinRightDistinct(ctx, tj, lower(left), lower(right))

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
        LowerTableIRHelpers.lowerTableJoin(ctx, tj, lower(left), lower(right), analyses)

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

      case te@TableExplode(child, path) =>
        LowerTableIRHelpers.lowerTableExplode(ctx, te, lower(child))

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

      case tmp@TableMapPartitions(child, globalName, partitionStreamName, body) =>
        val loweredChild = lower(child)
        LowerTableIRHelpers.lowerTableMapPartitions(ctx, tmp, loweredChild)

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
