package is.hail.expr.ir.lowering

import is.hail.annotations.{Annotation, ExtendedOrdering, Region, SafeRow, UnsafeRow}
import is.hail.asm4s.{AsmFunction1RegionLong, LongInfo, classInfo}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.types.physical.{PArray, PStruct, PTuple}
import is.hail.types.virtual.{TArray, TBoolean, TStream, TStruct, TTuple, Type}
import is.hail.rvd.RVDPartitioner
import is.hail.types.RStruct
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.utils._
import org.apache.spark.sql.Row

object LowerDistributedSort {
  def localSort(ctx: ExecuteContext, stage: TableStage, sortFields: IndexedSeq[SortField], relationalLetsAbove: Map[String, IR]): TableStage = {
    val numPartitions = stage.partitioner.numPartitions
    val collected = stage.collectWithGlobals(relationalLetsAbove)

    val (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) = ctx.timer.time("LowerDistributedSort.localSort.compile")(Compile[AsmFunction1RegionLong](ctx,
      FastIndexedSeq(),
      FastIndexedSeq(classInfo[Region]), LongInfo,
      collected,
      print = None,
      optimize = true))

    val fRunnable = ctx.timer.time("LowerDistributedSort.localSort.initialize")(f(ctx.fs, 0, ctx.r))
    val resultAddress = ctx.timer.time("LowerDistributedSort.localSort.run")(fRunnable(ctx.r))
    val rowsAndGlobal = ctx.timer.time("LowerDistributedSort.localSort.toJavaObject")(SafeRow.read(resultPType, resultAddress)).asInstanceOf[Row]

    val rowsType = resultPType.fieldType("rows").asInstanceOf[PArray]
    val rowType = rowsType.elementType.asInstanceOf[PStruct]

    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.virtualType.ordering
      if (so == Ascending) fo else fo.reverse
    }.toArray

    val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering
    val rows = rowsAndGlobal.getAs[IndexedSeq[Annotation]](0)
    val kType = TStruct(sortFields.map(f => (f.field, rowType.virtualType.fieldType(f.field))): _*)
    val kIndex = kType.fieldNames.map(f => rowType.fieldIdx(f))
    val sortedRows = ctx.timer.time("LowerDistributedSort.localSort.sort")(rows.sortBy{ a: Annotation =>
      a.asInstanceOf[Row].select(kIndex).asInstanceOf[Annotation]
    }(ord))
    val nPartitionsAdj = math.max(math.min(sortedRows.length, numPartitions), 1)
    val itemsPerPartition = (sortedRows.length.toDouble / nPartitionsAdj).ceil.toInt

    if (itemsPerPartition == 0)
      return TableStage(
        globals = Literal(resultPType.fieldType("global").virtualType, rowsAndGlobal.get(1)),
        partitioner = RVDPartitioner.empty(kType),
        TableStageDependency.none,
        MakeStream(FastSeq(), TStream(TStruct())),
        _ => MakeStream(FastSeq(), TStream(stage.rowType))
      )

    // partitioner needs keys to be ascending
    val partitionerKeyType = TStruct(sortFields.takeWhile(_.sortOrder == Ascending).map(f => (f.field, rowType.virtualType.fieldType(f.field))): _*)
    val partitionerKeyIndex = partitionerKeyType.fieldNames.map(f => rowType.fieldIdx(f))

    val partitioner = new RVDPartitioner(partitionerKeyType,
      sortedRows.grouped(itemsPerPartition).map { group =>
        val first = group.head.asInstanceOf[Row].select(partitionerKeyIndex)
        val last = group.last.asInstanceOf[Row].select(partitionerKeyIndex)
        Interval(first, last, includesStart = true, includesEnd = true)
      }.toIndexedSeq)

    TableStage(
      globals = Literal(resultPType.fieldType("global").virtualType, rowsAndGlobal.get(1)),
      partitioner = partitioner,
      TableStageDependency.none,
      contexts = mapIR(
        StreamGrouped(
          ToStream(Literal(rowsType.virtualType, sortedRows)),
          I32(itemsPerPartition))
        )(ToArray(_)),
      ctxRef => ToStream(ctxRef))
  }


  def distributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage = {
    // Array of struct("min", "max", "isSorted", "samples"),
    val perPartStats = stage.mapCollect(relationalBindings = relationalLetsAbove){ partitionStreamIR =>
      samplePartition(partitionStreamIR, ???, stage.key)
    }

    // TODO: Need to put perPartStats in the globals

    stage.mapCollect(relationalBindings = relationalLetsAbove){ partitionStreamIR =>
      // In here, need to distribute the keys into different buckets.
      val path = ctx.createTmpPath(ctx.tmpdir)
      StreamDistribute(partitionStreamIR, ???, path, ???)
    }

    ???
  }

  def howManySamplesPerPartition(rand: IRRandomness, totalNumberOfRecords: Int, initialNumSamplesToSelect: Int, partitionCounts: IndexedSeq[Int]): IndexedSeq[Int] = {
    var successStatesRemaining = initialNumSamplesToSelect
    var failureStatesRemaining = totalNumberOfRecords - successStatesRemaining

    val ans = new Array[Int](partitionCounts.size)

    var i = 0
    while (i < partitionCounts.size) {
      val numSuccesses = rand.rhyper(successStatesRemaining, failureStatesRemaining, partitionCounts(i)).toInt
      successStatesRemaining -= numSuccesses
      failureStatesRemaining -= (partitionCounts(i) - numSuccesses)
      ans(i) = numSuccesses
      i += 1
    }

    ans
  }

  def samplePartition(dataStream: IR, sampleIndices: IR, keyFields: IndexedSeq[String]): IR = {
    // Step 1: Join the dataStream zippedWithIdx on sampleIndices?
    // That requires sampleIndices to be a stream of structs
    val samplingIndexName = "samplingPartitionIndex"
    val structSampleIndices = mapIR(sampleIndices)(sampleIndex => MakeStruct(Seq((samplingIndexName, sampleIndex))))
    val dataWithIdx = zipWithIndex(dataStream)

    val leftName = genUID()
    val rightName = genUID()
    val leftRef = Ref(leftName, dataWithIdx.typ.asInstanceOf[TStream].elementType)
    val rightRef = Ref(rightName, structSampleIndices.typ.asInstanceOf[TStream].elementType)

    val joined = StreamJoin(dataWithIdx, structSampleIndices, IndexedSeq("idx"), IndexedSeq(samplingIndexName), leftName, rightName,
      MakeStruct(Seq(("elt", GetField(leftRef, "elt")), ("shouldKeep", ApplyUnaryPrimOp(Bang(), IsNA(rightRef))))),
      "left")

    // Step 2: Aggregate over joined, figure out how to collect only the rows that are marked "shouldKeep"
    val streamElementType = joined.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
    val streamElementName = genUID()
    val streamElementRef = Ref(streamElementName, streamElementType)
    val eltName = genUID()
    val eltType = dataStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
    val eltRef = Ref(eltName, eltType)

    val keyType = eltType.typeAfterSelectNames(keyFields)
    val minAndMaxZero = NA(keyType)

    // Folding for Min
    val aggFoldMinAccumName1 = genUID()
    val aggFoldMinAccumName2 = genUID()
    val aggFoldMinAccumRef1 = Ref(aggFoldMinAccumName1, keyType)
    val aggFoldMinAccumRef2 = Ref(aggFoldMinAccumName2, keyType)
    val minSeq = bindIR(SelectFields(eltRef, keyFields)) { keyOfCurElementRef =>
      If(IsNA(aggFoldMinAccumRef1),
        keyOfCurElementRef,
        If(ApplyComparisonOp(LT(keyType), aggFoldMinAccumRef1, keyOfCurElementRef), aggFoldMinAccumRef1, keyOfCurElementRef)
      )
    }
    val minComb =
      If(IsNA(aggFoldMinAccumRef1),
        aggFoldMinAccumRef2,
        If (ApplyComparisonOp(LT(keyType), aggFoldMinAccumRef1, aggFoldMinAccumRef2), aggFoldMinAccumRef1, aggFoldMinAccumRef2)
      )

    // Folding for Max
    val aggFoldMaxAccumName1 = genUID()
    val aggFoldMaxAccumName2 = genUID()
    val aggFoldMaxAccumRef1 = Ref(aggFoldMaxAccumName1, keyType)
    val aggFoldMaxAccumRef2 = Ref(aggFoldMaxAccumName2, keyType)
    val maxSeq = bindIR(SelectFields(eltRef, keyFields)) { keyOfCurElementRef =>
      If(IsNA(aggFoldMaxAccumRef1),
        keyOfCurElementRef,
        If(ApplyComparisonOp(GT(keyType), aggFoldMaxAccumRef1, keyOfCurElementRef), aggFoldMaxAccumRef1, keyOfCurElementRef)
      )
    }
    val maxComb =
      If(IsNA(aggFoldMaxAccumRef1),
        aggFoldMaxAccumRef2,
        If (ApplyComparisonOp(GT(keyType), aggFoldMaxAccumRef1, aggFoldMaxAccumRef2), aggFoldMaxAccumRef1, aggFoldMaxAccumRef2)
      )

    // Folding for isInternallySorted
    val aggFoldSortedAccumName1 = genUID()
    val aggFoldSortedAccumName2 = genUID()
    val isSortedStateType = TStruct("lastKeySeen" -> keyType, "sortedSoFar" -> TBoolean)
    val aggFoldSortedAccumRef1 = Ref(aggFoldSortedAccumName1, isSortedStateType)
    val aggFoldSortedAccumRef2 = Ref(aggFoldSortedAccumName2, isSortedStateType)
    val isSortedSeq = bindIR(SelectFields(eltRef, keyFields)) { keyOfCurElementRef =>
      bindIR(GetField(aggFoldSortedAccumRef1, "lastKeySeen")) { lastKeySeenRef =>
        If(IsNA(lastKeySeenRef),
          MakeStruct(Seq("lastKeySeen" -> keyOfCurElementRef, "sortedSoFar" -> true)),
          ???
        )
      }
    }



    StreamAgg(joined, streamElementName, {
      AggLet(eltName, GetField(streamElementRef, "elt"),
        MakeStruct(Seq(
          ("min", AggFold(minAndMaxZero, minSeq, minComb, aggFoldMinAccumName1, aggFoldMinAccumName2, false)),
          ("max", AggFold(minAndMaxZero, maxSeq, maxComb, aggFoldMaxAccumName1, aggFoldMaxAccumName2, false)),
          ("samples", AggFilter(GetField(streamElementRef, "shouldKeep"), ApplyAggOp(Collect())(eltRef), false)),
        )), false)
    })
  }

  // Given an IR of type TArray(TTuple(minKey, maxKey)), determine if there's any overlap between these closed intervals.
  def tuplesAreSorted(arrayOfTuples: IR, sortFields: IndexedSeq[SortField]): IR = {
    // assume for now array is sorted, could sort later.

    val intervalElementType = arrayOfTuples.typ.asInstanceOf[TArray].elementType.asInstanceOf[TTuple].types(0)

    // Make a code ordering:

    mapIR(rangeIR(1, ArrayLen(arrayOfTuples))) { idxOfTuple =>
      ApplyComparisonOp(LTEQ(intervalElementType), ArrayRef(arrayOfTuples, idxOfTuple - 1), ArrayRef(arrayOfTuples, idxOfTuple))
    }

    ???
  }
}
