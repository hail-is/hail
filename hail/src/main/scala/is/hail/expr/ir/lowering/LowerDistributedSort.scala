package is.hail.expr.ir.lowering

import is.hail.annotations.{Annotation, ExtendedOrdering, Region, RegionValueBuilder, SafeRow, UnsafeRow}
import is.hail.asm4s.{AsmFunction1RegionLong, AsmFunction2RegionLongLong, AsmFunction3RegionLongLongLong, LongInfo, classInfo}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.functions.{ArrayFunctions, IRRandomness}
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.types.physical.{PArray, PBaseStruct, PCanonicalArray, PStruct, PTuple, PType}
import is.hail.types.virtual.{TArray, TBoolean, TInt32, TStream, TString, TStruct, TTuple, Type}
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.RStruct
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.utils._
import org.apache.spark.sql.Row

import java.io.PrintWriter
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

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
    val rows = rowsAndGlobal.getAs[IndexedSeq[Annotation]](0)
    val kType = TStruct(sortFields.map(f => (f.field, rowType.virtualType.fieldType(f.field))): _*)

    val sortedRows = localAnnotationSort(ctx, rows, sortFields, rowType.virtualType)

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

  private def localAnnotationSort(
    ctx: ExecuteContext,
    annotations: IndexedSeq[Annotation],
    sortFields: IndexedSeq[SortField],
    rowType: TStruct
  ): IndexedSeq[Annotation] = {
    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.ordering
      if (so == Ascending) fo else fo.reverse
    }.toArray

    val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

    val kType = TStruct(sortFields.map(f => (f.field, rowType.fieldType(f.field))): _*)
    val kIndex = kType.fieldNames.map(f => rowType.fieldIdx(f))
    ctx.timer.time("LowerDistributedSort.localSort.sort")(annotations.sortBy{ a: Annotation =>
      a.asInstanceOf[Row].select(kIndex).asInstanceOf[Annotation]
    }(ord))
  }


  def distributedSort(
    ctx: ExecuteContext,
    inputStage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): IR = {

    val oversamplingNum = 1
    val seed = 7L
    val branchingFactor = 4

    val (newKType, _) = inputStage.rowType.select(sortFields.map(sf => sf.field))

    val spec = TypedCodecSpec(rowTypeRequiredness.canonicalPType(inputStage.rowType), BufferSpec.default)
    val reader = PartitionNativeReader(spec)
    val initialTmpPath = ctx.createTmpPath("hail_shuffle_temp_initial")
    val writer = PartitionNativeWriter(spec, initialTmpPath, None, None)
    val initialChunks = CompileAndEvaluate[Annotation](ctx, inputStage.mapCollect(relationalLetsAbove) { part =>
      WritePartition(part, UUID4(), writer)
    }).asInstanceOf[IndexedSeq[Row]].map(row => Chunk(initialTmpPath + row(0).asInstanceOf[String], row(1).asInstanceOf[Long].toInt))
    val initialSegment = SegmentResult(IndexedSeq(0), inputStage.partitioner.range.get, initialChunks)

    val totalNumberOfRows = initialChunks.map(_.size).sum
    val idealNumberOfRowsPerPart = totalNumberOfRows / inputStage.numPartitions

    var loopState = LoopState(IndexedSeq(initialSegment), IndexedSeq.empty[SegmentResult])

    // Output of streamDistribute: ListOfSegmentResult: Interval, then per partition info (array of file names and counts)
    // Let's just make this Scala.
    //var partitionCountsPerSegment: IndexedSeq[IndexedSeq[Int]] = loopState.largeSegments.map(sr => sr.chunks.map(_.size))
    var i = 0
    val rand = new IRRandomness((math.random() * 1000).toInt)

    while (!loopState.largeSegments.isEmpty) {
      println(s"Loop iteration ${i}")

      // 1. I have a loop state with some large segments in it. I know how many samples are in each chunk of each segment
      // 2. I take those segments, and I identify a good partitioning of them.
      // 3. Using this partitioning, I figure out the per segment sampling rules.
      // 4. I execute the sampling as a CDA

      val partitionData = segmentsToPartitionData(loopState.largeSegments, idealNumberOfRowsPerPart)

      //TODO: Dumb and temporary
      val partitionCountsPerSegment = partitionData.groupBy(_._1.last).toIndexedSeq.sortBy(_._1).map(_._2).map(oneSegment => oneSegment.map(_._3))
      println(s"partitionCountsPerSegment = ${partitionCountsPerSegment}")
      println(s"How many large segments? ${loopState.largeSegments.size}")
      assert(partitionCountsPerSegment.size == loopState.largeSegments.size)
      println(s"How many partitions did we come up with? ${partitionData.size}")

      val numSamplesPerPartitionPerSegment = partitionCountsPerSegment.map { partitionCountsForOneSegment =>
        val recordsInSegment = partitionCountsForOneSegment.sum
        howManySamplesPerPartition(rand, recordsInSegment, Math.min(recordsInSegment, (branchingFactor * oversamplingNum) - 1), partitionCountsForOneSegment)
      }

      val numSamplesPerPartition = numSamplesPerPartitionPerSegment.flatten

      // Ctx is file to read, numSamples to take
      val perPartStatsCDAContextData = partitionData.zip(numSamplesPerPartition).map { case (partData, numSamples) => Row(partData._1.last, partData._2, partData._3, numSamples)}
      val perPartStatsCDAContexts = ToStream(Literal(TArray(TStruct("segmentIdx" -> TInt32, "files" -> TArray(TString), "sizeOfPartition" -> TInt32, "numSamples" -> TInt32)), perPartStatsCDAContextData))
      val perPartStatsIR = cdaIR(perPartStatsCDAContexts, MakeStruct(Seq())){ (ctxRef, _) =>
        val filenames = GetField(ctxRef, "files")
        val samples = SeqSample(GetField(ctxRef, "sizeOfPartition"), GetField(ctxRef, "numSamples"), false)
        val partitionStream = flatMapIR(ToStream(filenames)) { fileName =>
          ReadPartition(fileName, spec._vType, reader)
        }
        MakeTuple.ordered(IndexedSeq(GetField(ctxRef, "segmentIdx"), samplePartition(partitionStream, samples, newKType.fields.map(_.name))))
      }

      // Going to check now if it's fully sorted, as well as collect and sort all the samples.
      val (Some(PTypeReferenceSingleCodeType(perPartStatsType: PArray)), fPerPartStats) = ctx.timer.time("LowerDistributedSort.distributedSort.compilePerPartStats")(Compile[AsmFunction1RegionLong](ctx,
        FastIndexedSeq(),
        FastIndexedSeq(classInfo[Region]), LongInfo,
        perPartStatsIR,
        print = None,
        optimize = true))

      val fPerPartStatsRunnable = ctx.timer.time("LowerDistributedSort.localSort.initialize")(fPerPartStats(ctx.fs, 0, ctx.r))
      val perPartStatsAddress = ctx.timer.time("LowerDistributedSort.localSort.run")(fPerPartStatsRunnable(ctx.r))
      val perPartStatsA = ctx.timer.time("LowerDistributedSort.localSort.toJavaObject")(SafeRow.read(perPartStatsType, perPartStatsAddress)).asInstanceOf[IndexedSeq[Row]]

      println(s"per part stats = ${perPartStatsA}")
      val groupedBySegmentNumber = perPartStatsA.groupBy(r => r(0).asInstanceOf[Int]).toIndexedSeq.sortBy(_._1)

      // These are the pivots for each segment number
      val pivotsWithEndpointsGroupedBySegmentNumber = groupedBySegmentNumber.map { case (segmentNumber, perPartStatsA) =>
        val perPartStatsMins = perPartStatsA.map(r => r(1).asInstanceOf[Row](0))
        val perPartStatsMaxes = perPartStatsA.map(r => r(1).asInstanceOf[Row](1))
        val perPartStatsSamples = perPartStatsA.map(r => r(1).asInstanceOf[Row](2).asInstanceOf[IndexedSeq[Annotation]]).flatten

        val sorted = localAnnotationSort(ctx, perPartStatsSamples, sortFields, newKType)
        // TODO: Clearly a terrible way to find min and max
        val min = localAnnotationSort(ctx, perPartStatsMins, sortFields, newKType)(0)
        val max = localAnnotationSort(ctx, perPartStatsMaxes, sortFields, newKType).last

        val pivotsWithEndpoints = IndexedSeq(min) ++ sorted ++ IndexedSeq(max)
        pivotsWithEndpoints
      }

      println(s"Pivots with endpoints groupedBySegmentNumber = ${pivotsWithEndpointsGroupedBySegmentNumber}")
      println(s"Segment indices = ${groupedBySegmentNumber.map(_._1)}")

      val pivotsWithEndpointsGroupedBySegmentNumberLiteral = Literal(TArray(TArray(newKType)), pivotsWithEndpointsGroupedBySegmentNumber)

      val tmpPath = ctx.createTmpPath("hail_shuffle_temp")

      val distributeContextsData = partitionData.zipWithIndex.map { case (part, partIdx) => Row(part._1.last, part._2, partIdx, part._4) }
      val distributeContexts = ToStream(Literal(TArray(TStruct("segmentIdx" -> TInt32, "files" -> TArray(TString), "partIdx" -> TInt32, "largeSegmentIdx" -> TInt32)), distributeContextsData))
      val distributeGlobals = MakeStruct(IndexedSeq("pivotsWithEndpointsGroupedBySegmentIdx" -> pivotsWithEndpointsGroupedBySegmentNumberLiteral))

      val distribute = cdaIR(distributeContexts, distributeGlobals) { (ctxRef, globalsRef) =>
        val segmentIdx = GetField(ctxRef, "segmentIdx")
        val largeSegmentIdx = GetField(ctxRef, "largeSegmentIdx")
        val pivotsWithEndpointsGroupedBySegmentIdx = GetField(globalsRef, "pivotsWithEndpointsGroupedBySegmentIdx")
        val path = invoke("concat", TString, Str(tmpPath + "_"), invoke("str", TString, GetField(ctxRef, "partIdx")))
        val filenames = GetField(ctxRef, "files")
        val partitionStream = flatMapIR(ToStream(filenames)) { fileName =>
          ReadPartition(fileName, spec._vType, reader)
        }
        MakeTuple.ordered(IndexedSeq(segmentIdx, StreamDistribute(partitionStream, ArrayRef(pivotsWithEndpointsGroupedBySegmentIdx, largeSegmentIdx), path, spec)))
      }

      // Now, execute distribute
      val (Some(PTypeReferenceSingleCodeType(resultPType)), f) = ctx.timer.time("LowerDistributedSort.localSort.compile")(Compile[AsmFunction1RegionLong](ctx,
        FastIndexedSeq(),
        FastIndexedSeq(classInfo[Region]), LongInfo,
        distribute,
        print = None,
        optimize = true))

      val fRunnable = ctx.timer.time("LowerDistributedSort.localSort.initialize")(f(ctx.fs, 0, ctx.r))
      val resultAddress = ctx.timer.time("LowerDistributedSort.localSort.run")(fRunnable(ctx.r))
      val distributeResult = ctx.timer.time("LowerDistributedSort.localSort.toJavaObject")(SafeRow.read(resultPType, resultAddress))
        .asInstanceOf[IndexedSeq[Row]].map(row => (
        row(0).asInstanceOf[Int],
        row(1).asInstanceOf[IndexedSeq[Row]].map(innerRow => (
          innerRow(0).asInstanceOf[Interval],
          innerRow(1).asInstanceOf[String],
          innerRow(2).asInstanceOf[Int]))))
      println("Distributed successfully")

      // distributeResult is a numPartitions length array of arrays, where each inner array tells me what
      // files were written to for each partition, as well as the number of entries in that file.

      // I need to:
      // 1. Group the partitions by their origin segment.
      // 2. Within an origin segment, I can trust that all inner arrays are the same length. So do a transpose within an origin segment to get things grouped by their new segments.

      println(distributeResult)

      val protoDataPerSegment = distributeResult
        .groupBy{ case (originSegment, _) => originSegment }.toIndexedSeq.sortBy { case (originSegment, _) => originSegment}.map { case (_, seqOfChunkData) => seqOfChunkData.map(_._2)}

      val transposedIntoNewSegments = protoDataPerSegment.map { oneOldSegment =>
        val headLen = oneOldSegment.head.length
        assert(oneOldSegment.forall(x => x.length == headLen))
        (0 until headLen).map(colIdx => oneOldSegment.map(row => row(colIdx)))
      }.flatten

      val dataPerSegment = transposedIntoNewSegments.zipWithIndex.map { case (chunksWithSameInterval, newSegmentIdx) =>
        val interval = chunksWithSameInterval.head._1
        val chunks = chunksWithSameInterval.map(chunk => Chunk(chunk._2, chunk._3))
        SegmentResult(IndexedSeq(newSegmentIdx), interval, chunks)
      }

      // Now I need to figure out how many partitions to allocate to each segment.
      val (newBigSegments, newSmallSegments) = dataPerSegment.partition(sr => sr.chunks.map(_.size).sum > 6)
      loopState = LoopState(newBigSegments, loopState.smallSegments ++ newSmallSegments)
      println(s"LoopState: Big = ${loopState.largeSegments.size}, small = ${loopState.smallSegments.size}")

      i = i + 1
    }

    // Ok, now all the small partitions from loop state can be dealt with.
    val unsortedSegments = loopState.smallSegments
    val keyOrdering = PartitionBoundOrdering.apply(newKType)
    val sortedSegments = unsortedSegments.sortWith((sr1, sr2) => sr1.interval.isBelow(keyOrdering, sr2.interval))

    val contextData = sortedSegments.map(segment => Row(segment.chunks.map(chunk => chunk.filename)))
    val contexts = ToStream(Literal(TArray(TStruct("files" -> TArray(TString))), contextData))
    val partitioner = new RVDPartitioner(newKType, sortedSegments.map(_.interval))
    val finalTs = TableStage(MakeStruct(Seq()), partitioner, TableStageDependency.none, contexts, { ctxRef =>
      val filenames = GetField(ctxRef, "files")
      val unsortedPartitionStream = flatMapIR(ToStream(filenames)) { fileName =>
        ReadPartition(fileName, spec._vType, reader)
      }
      ToStream(sortIR(unsortedPartitionStream) { (refLeft, refRight) =>
        ApplyComparisonOp(LT(newKType), SelectFields(refLeft, newKType.fields.map(_.name)), SelectFields(refRight, newKType.fields.map(_.name)))
      })
    })

    finalTs.mapCollect(Map.empty[String, IR])(x => ToArray(x))
  }

  def segmentsToPartitionData(segments: IndexedSeq[SegmentResult], idealNumberOfRowsPerPart: Int): IndexedSeq[(IndexedSeq[Int], IndexedSeq[String], Int, Int)] = {
    segments.zipWithIndex.flatMap { case (sr, largeSegmentIdx) =>
      val chunkDataSizes = sr.chunks.map(_.size)
      val segmentSize = chunkDataSizes.sum
      val numParts = (segmentSize + idealNumberOfRowsPerPart - 1) / idealNumberOfRowsPerPart
      var currentPartSize = 0
      val groupedIntoParts = new ArrayBuffer[(IndexedSeq[Int], IndexedSeq[String], Int, Int)](numParts)
      val currentFiles = new ArrayBuffer[String]()
      sr.chunks.foreach { chunk =>
        if (chunk.size > 0) {
          currentFiles.append(chunk.filename)
          currentPartSize += chunk.size
          if (currentPartSize > idealNumberOfRowsPerPart) {
            groupedIntoParts.append((sr.indices, currentFiles.result().toIndexedSeq, currentPartSize, largeSegmentIdx))
            currentFiles.clear()
            currentPartSize = 0
          }
        }
      }
      if (!currentFiles.isEmpty) {
        groupedIntoParts.append((sr.indices, currentFiles.result().toIndexedSeq, currentPartSize, largeSegmentIdx))
      }
      groupedIntoParts.result()
    }
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

  // TODO: Probably wrong for this to not take SortOrder?
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
    val isSortedStateType = TStruct("lastKeySeen" -> keyType, "sortedSoFar" -> TBoolean, "haveSeenAny" -> TBoolean)
    val aggFoldSortedAccumRef1 = Ref(aggFoldSortedAccumName1, isSortedStateType)
    val aggFoldSortedAccumRef2 = Ref(aggFoldSortedAccumName2, isSortedStateType)
    val isSortedSeq = bindIR(SelectFields(eltRef, keyFields)) { keyOfCurElementRef =>
      bindIR(GetField(aggFoldSortedAccumRef1, "lastKeySeen")) { lastKeySeenRef =>
        If(!GetField(aggFoldSortedAccumRef1, "haveSeenAny"),
          MakeStruct(Seq("lastKeySeen" -> keyOfCurElementRef, "sortedSoFar" -> true, "haveSeenAny" -> true)),
          If (ApplyComparisonOp(LTEQ(keyType), keyOfCurElementRef, lastKeySeenRef),
            MakeStruct(Seq("lastKeySeen" -> keyOfCurElementRef, "sortedSoFar" -> GetField(aggFoldSortedAccumRef1, "sortedSoFar"), "haveSeenAny" -> true)),
            MakeStruct(Seq("lastKeySeen" -> keyOfCurElementRef, "sortedSoFar" -> false, "haveSeenAny" -> true))
          )
        )
      }
    }


    StreamAgg(joined, streamElementName, {
      AggLet(eltName, GetField(streamElementRef, "elt"),
        MakeStruct(Seq(
          ("min", AggFold(minAndMaxZero, minSeq, minComb, aggFoldMinAccumName1, aggFoldMinAccumName2, false)),
          ("max", AggFold(minAndMaxZero, maxSeq, maxComb, aggFoldMaxAccumName1, aggFoldMaxAccumName2, false)),
          ("samples", AggFilter(GetField(streamElementRef, "shouldKeep"), ApplyAggOp(Collect())(SelectFields(eltRef, keyFields)), false)),
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

  def eval(x: IR): Any = ExecuteContext.scoped(){ ctx =>
    eval(x, Env.empty, FastIndexedSeq(), None, None, true, ctx)
  }

  def eval(x: IR,
           env: Env[(Any, Type)],
           args: IndexedSeq[(Any, Type)],
           agg: Option[(IndexedSeq[Row], TStruct)],
           bytecodePrinter: Option[PrintWriter] = None,
           optimize: Boolean = true,
           ctx: ExecuteContext
          ): Any = {
    val inputTypesB = new BoxedArrayBuilder[Type]()
    val inputsB = new mutable.ArrayBuffer[Any]()

    args.foreach { case (v, t) =>
      inputsB += v
      inputTypesB += t
    }

    env.m.foreach { case (name, (v, t)) =>
      inputsB += v
      inputTypesB += t
    }

    val argsType = TTuple(inputTypesB.result(): _*)
    val resultType = TTuple(x.typ)
    val argsVar = genUID()

    val (_, substEnv) = env.m.foldLeft((args.length, Env.empty[IR])) { case ((i, env), (name, (v, t))) =>
      (i + 1, env.bind(name, GetTupleElement(Ref(argsVar, argsType), i)))
    }

    def rewrite(x: IR): IR = {
      x match {
        case In(i, t) =>
          GetTupleElement(Ref(argsVar, argsType), i)
        case _ =>
          MapIR(rewrite)(x)
      }
    }

    val argsPType = PType.canonical(argsType).setRequired(true)
    agg match {
      case Some((aggElements, aggType)) =>
        val aggElementVar = genUID()
        val aggArrayVar = genUID()
        val aggPType = PType.canonical(aggType)
        val aggArrayPType = PCanonicalArray(aggPType, required = true)

        val substAggEnv = aggType.fields.foldLeft(Env.empty[IR]) { case (env, f) =>
          env.bind(f.name, GetField(Ref(aggElementVar, aggType), f.name))
        }
        val aggIR = StreamAgg(ToStream(Ref(aggArrayVar, aggArrayPType.virtualType)),
          aggElementVar,
          MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(eval = substEnv, agg = Some(substAggEnv)))))))

        val (Some(PTypeReferenceSingleCodeType(resultType2)), f) = Compile[AsmFunction3RegionLongLongLong](ctx,
          FastIndexedSeq((argsVar, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType))),
            (aggArrayVar, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(aggArrayPType)))),
          FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
          aggIR,
          print = bytecodePrinter,
          optimize = optimize)
        assert(resultType2.virtualType == resultType)

        ctx.r.pool.scopedRegion { region =>
          val rvb = new RegionValueBuilder(region)
          rvb.start(argsPType)
          rvb.startTuple()
          var i = 0
          while (i < inputsB.length) {
            rvb.addAnnotation(inputTypesB(i), inputsB(i))
            i += 1
          }
          rvb.endTuple()
          val argsOff = rvb.end()

          rvb.start(aggArrayPType)
          rvb.startArray(aggElements.length)
          aggElements.foreach { r =>
            rvb.addAnnotation(aggType, r)
          }
          rvb.endArray()
          val aggOff = rvb.end()

          val resultOff = f(ctx.fs, 0, region)(region, argsOff, aggOff)
          SafeRow(resultType2.asInstanceOf[PBaseStruct], resultOff).get(0)
        }

      case None =>
        val (Some(PTypeReferenceSingleCodeType(resultType2)), f) = Compile[AsmFunction2RegionLongLong](ctx,
          FastIndexedSeq((argsVar, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType)))),
          FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
          MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(substEnv))))),
          optimize = optimize,
          print = bytecodePrinter)
        assert(resultType2.virtualType == resultType)

        ctx.r.pool.scopedRegion { region =>
          val rvb = new RegionValueBuilder(region)
          rvb.start(argsPType)
          rvb.startTuple()
          var i = 0
          while (i < inputsB.length) {
            rvb.addAnnotation(inputTypesB(i), inputsB(i))
            i += 1
          }
          rvb.endTuple()
          val argsOff = rvb.end()

          val resultOff = f(ctx.fs, 0, region)(region, argsOff)
          SafeRow(resultType2.asInstanceOf[PBaseStruct], resultOff).get(0)
        }
    }
  }
}

case class Chunk(filename: String, size: Int)
case class SegmentResult(indices: IndexedSeq[Int], interval: Interval, chunks: IndexedSeq[Chunk])
case class LoopState(largeSegments: IndexedSeq[SegmentResult], smallSegments: IndexedSeq[SegmentResult])