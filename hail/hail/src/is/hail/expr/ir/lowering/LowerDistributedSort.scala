package is.hail.expr.ir.lowering

import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.defs._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.RVDPartitioner
import is.hail.types.{tcoerce, RTable, VirtualTypeWithReq}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.JsonAST.JString

object LowerDistributedSort extends Logging {

  def distributedSort(
    ctx: ExecuteContext,
    inputStage: TableStage,
    sortFields: IndexedSeq[SortField],
    tableRequiredness: RTable,
    optTargetNumPartitions: Option[Int] = None,
  ): TableReader = {

    val oversamplingNum = 3
    val maxBranchingFactor = ctx.getFlag("shuffle_max_branch_factor").toInt
    val defaultBranchingFactor =
      if (inputStage.numPartitions < maxBranchingFactor) Math.max(2, inputStage.numPartitions)
      else maxBranchingFactor

    val rowTypeRequiredness = tableRequiredness.rowType
    val sizeCutoff = ctx.getFlag("shuffle_cutoff_to_local_sort").toInt

    val (keyToSortBy, _) = inputStage.rowType.select(sortFields.map(sf => sf.field))

    val spec =
      TypedCodecSpec(
        ctx,
        rowTypeRequiredness.canonicalPType(inputStage.rowType),
        BufferSpec.wireSpec,
      )
    val reader = PartitionNativeReader(spec, "__dummy_uid")
    val initialTmpPath = ctx.createTmpPath("hail_shuffle_temp_initial")
    val writer = PartitionNativeWriter(
      spec,
      keyToSortBy.fieldNames,
      initialTmpPath,
      None,
      None,
      trackTotalBytes = true,
    )

    logger.info("DISTRIBUTED SORT: PHASE 1: WRITE DATA")
    val initialStageData =
      CompileAndEvaluate[Row](
        ctx,
        inputStage.mapCollectWithGlobals("shuffle_initial_write")(
          WritePartition(_, UUID4(), writer)
        ) {
          (part, globals) =>
            maketuple(
              part,
              globals,
              streamAggIR(ToStream(part)) { elem =>
                M.agg {
                  for {
                    fst <- GetField(elem, "firstKey")
                    lst <- GetField(elem, "lastKey")
                  } yield makestruct(
                    "min" -> AggFold.min(fst, sortFields),
                    "max" -> AggFold.max(lst, sortFields),
                  )
                }
              },
            )
        },
      )

    val initialChunks =
      initialStageData(0)
        .asInstanceOf[IndexedSeq[Row]]
        .map { row =>
          Chunk(
            initialTmpPath + row(0).asInstanceOf[String],
            row(1).asInstanceOf[Long],
            row.getLong(5),
          )
        }

    val initialGlobal = Literal(inputStage.globalType, initialStageData(1).asInstanceOf[Row])

    val intervalRange = initialStageData(2).asInstanceOf[Row]
    val initialInterval = Interval(intervalRange(0), intervalRange(1), true, true)
    val initialSegment = SegmentResult(FastSeq(0), initialInterval, initialChunks)

    val totalNumberOfRows = initialChunks.map(_.size).sum

    val targetNumPartitions =
      optTargetNumPartitions
        .fold(inputStage.numPartitions) { i =>
          assert(i >= 1, s"Must request positive number of partitions. Requested $i")
          i
        }

    val idealNumberOfRowsPerPart: Long =
      if (targetNumPartitions == 0) 1L
      else Math.max(1L, totalNumberOfRows / targetNumPartitions)

    var loopState = LoopState(
      ArraySeq(initialSegment),
      ArraySeq.empty[SegmentResult],
      ArraySeq.empty[OutputPartition],
    )

    var i = 0
    val rand = ThreefryRandomEngine()

    // Loop state keeps track of three things:
    // - largeSegments are too big to sort locally so have to broken up.
    // - smallSegments are small enough to be sorted locally.
    // - readyOutputParts are any partitions that were already sorted.
    //
    // Loop continues until there are no largeSegments left.
    // Finally, the small segments are sorted and combined with readyOutputParts.

    while (loopState.largeSegments.nonEmpty) {
      val partitionDataPerSegment =
        segmentsToPartitionData(loopState.largeSegments, idealNumberOfRowsPerPart)
      assert(partitionDataPerSegment.size == loopState.largeSegments.size)

      val numSamplesPerPartitionPerSegment = partitionDataPerSegment.map { partData =>
        val partitionCountsForOneSegment = partData.map(_.currentPartSize)
        val recordsInSegment = partitionCountsForOneSegment.sum
        val branchingFactor = math.min(recordsInSegment, defaultBranchingFactor.toLong)
        howManySamplesPerPartition(
          rand,
          recordsInSegment,
          Math.min(recordsInSegment, (branchingFactor * oversamplingNum) - 1).toInt,
          partitionCountsForOneSegment,
        )
      }

      val numSamplesPerPartition = numSamplesPerPartitionPerSegment.flatten

      val perPartStatsCDAContextData =
        partitionDataPerSegment.flatten.zip(numSamplesPerPartition).map {
          case (partData, numSamples) =>
            Row(
              partData.indices.last,
              partData.files,
              coerceToInt(partData.currentPartSize),
              numSamples,
              partData.currentPartByteSize,
            )
        }

      val partStatsContexts =
        ToStream(Literal(
          TArray(TStruct(
            "segmentIdx" -> TInt32,
            "files" -> TArray(TString),
            "sizeOfPartition" -> TInt32,
            "numSamples" -> TInt32,
            "byteSize" -> TInt64,
          )),
          perPartStatsCDAContextData,
        ))

      val partitionStatistics =
        cdaIR(partStatsContexts, makestruct(), s"shuffle_part_stats_iteration_$i") { (ctx, _) =>
          val filenames = GetField(ctx, "files")
          val samples = SeqSample(
            GetField(ctx, "sizeOfPartition"),
            GetField(ctx, "numSamples"),
            NA(TRNGState),
          )

          val partitions =
            flatMapIR(ToStream(filenames)) { fileName =>
              mapIR(
                ReadPartition(
                  MakeStruct(ArraySeq("partitionIndex" -> I64(0), "partitionPath" -> fileName)),
                  tcoerce[TStruct](spec._vType),
                  reader,
                )
              )(SelectFields(_, keyToSortBy.fields.map(_.name)))
            }

          makestruct(
            "segmentIdx" -> GetField(ctx, "segmentIdx"),
            "byteSize" -> GetField(ctx, "byteSize"),
            "partData" -> samplePartition(partitions, samples, sortFields),
          )
        }

      // Aggregate over the segments, to compute
      // - the pivots
      // - whether it's already sorted, and
      // - the key interval is contained in that segment.
      //
      // Also get the min and max of each individual partition.
      // That way if it's sorted already, we know the partitioning to use.
      val pivotsPerSegmentAndSortedCheck =
        ToArray(mapIR(StreamGroupByKey(
          ToStream(partitionStatistics),
          ArraySeq("segmentIdx"),
          missingEqual = true,
        )) { oneGroup =>
          M.eval {
            for {
              aggResults <-
                streamAggIR(oneGroup) { elem =>
                  M.agg {
                    for {
                      data <- GetField(elem, "partData")
                      min <- GetField(data, "min")
                      max <- GetField(data, "max")
                      isSorted <- GetField(data, "isSorted")
                    } yield makestruct(
                      "byteSize" -> ApplyAggOp(Sum())(GetField(elem, "byteSize")),
                      "min" -> AggFold.min(min, sortFields), // Min of the mins
                      "max" -> AggFold.max(max, sortFields), // Max of the maxes
                      "perPartMins" -> ApplyAggOp(Collect())(min), // All the mins
                      "perPartMaxes" -> ApplyAggOp(Collect())(max), // All the maxes
                      "samples" -> ApplyAggOp(Collect())(GetField(data, "samples")),
                      "eachPartSorted" -> AggFold.all(isSorted),
                      "perPartIntervalTuples" -> ApplyAggOp(Collect())(maketuple(min, max)),
                    )
                  }
                }

              sortedOversampling <-
                sortIR(flatten(ToStream(GetField(aggResults, "samples"))))(
                  ApplyComparisonOp(StructLT(sortFields), _, _)
                )

              numSamples <- ArrayLen(sortedOversampling)
              /* calculate a 'good' branch factor based on part sizes */
              branchingFactor <-
                maxIR(
                  2,
                  minIR(
                    minIR(numSamples, defaultBranchingFactor),
                    (I64(2L) * (GetField(aggResults, "byteSize") floorDiv sizeCutoff.toLong)).toI,
                  ),
                )

              branchingFactor <-
                If(
                  numSamples ceq 0,
                  Die(
                    strConcat("aggresults=", aggResults, ", sortedOversampling=",
                      sortedOversampling, ", numSamples=", numSamples, ", branchingFactor=",
                      branchingFactor),
                    TInt32,
                    -1,
                  ),
                  branchingFactor,
                )

              min <- GetField(aggResults, "min")
              max <- GetField(aggResults, "max")
              sortedSampling <-
                ToArray(mapIR(StreamRange(1, branchingFactor, 1)) { idx =>
                  ArrayRef(
                    sortedOversampling,
                    Apply(
                      "floor",
                      FastSeq(),
                      FastSeq(idx.toD * ((numSamples + 1) / branchingFactor)),
                      TFloat64,
                      ErrorIDs.NO_ERROR,
                    ).toI - 1,
                  )
                })

              perPartIntervalTuples <- GetField(aggResults, "perPartIntervalTuples")
            } yield makestruct(
              "pivotsWithEndpoints" ->
                concatIR(MakeArray(min), sortedSampling, MakeArray(max)).toArray,
              "isSorted" ->
                (GetField(aggResults, "eachPartSorted") &&
                  tuplesAreSorted(perPartIntervalTuples, sortFields)),
              "intervalTuple" -> maketuple(min, max),
              "perPartMins" -> GetField(aggResults, "perPartMins"),
              "perPartMaxes" -> GetField(aggResults, "perPartMaxes"),
            )
          }
        })

      logger.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: STAGE 1: SAMPLE VALUES FROM PARTITIONS")
      // Going to check now if it's fully sorted, as well as collect and sort all the samples.
      val pivotsWithEndpointsAndInfoGroupedBySegmentNumber =
        CompileAndEvaluate[IndexedSeq[Row]](ctx, pivotsPerSegmentAndSortedCheck)
          .map(x =>
            (
              x(0).asInstanceOf[IndexedSeq[Row]],
              x(1).asInstanceOf[Boolean],
              x(2).asInstanceOf[Row],
              x(3).asInstanceOf[IndexedSeq[Row]],
              x(4).asInstanceOf[IndexedSeq[Row]],
            )
          )

      def pivotCounts =
        pivotsWithEndpointsAndInfoGroupedBySegmentNumber
          .map(_._1.length)
          .groupBy(identity)
          .toArray
          .sortBy(_._1)
          .map { case (nPivots, segments) => s"$nPivots pivots: ${segments.length}" }
          .mkString("\n  ")

      logger.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: pivot counts:\n  $pivotCounts")

      val (sortedSegmentsTuples, unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber) =
        pivotsWithEndpointsAndInfoGroupedBySegmentNumber
          .zipWithIndex
          .partition { case ((_, isSorted, _, _, _), _) => isSorted }

      val outputPartitions =
        sortedSegmentsTuples.flatMap { case ((_, _, _, partMins, partMaxes), originalSegmentIdx) =>
          val segmentToBreakUp = loopState.largeSegments(originalSegmentIdx)
          val currentSegmentPartitionData = partitionDataPerSegment(originalSegmentIdx)
          val partRanges = partMins.zip(partMaxes)
          assert(partRanges.size == currentSegmentPartitionData.size)

          currentSegmentPartitionData.zip(partRanges).zipWithIndex.map {
            case ((pi, (intervalStart, intervalEnd)), idx) =>
              OutputPartition(
                segmentToBreakUp.indices :+ idx,
                Interval(intervalStart, intervalEnd, true, true),
                pi.files,
              )
          }
        }

      val remainingUnsortedSegments =
        unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.map {
          case (_, idx) => loopState.largeSegments(idx)
        }

      val (newBigUnsortedSegments, newSmallSegments) =
        if (unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.isEmpty)
          (ArraySeq.empty, ArraySeq.empty)
        else {
          val distributeContexts =
            Literal(
              TArray(TStruct(
                "segmentIdx" -> TInt32,
                "files" -> TArray(TString),
                "partIdx" -> TInt32,
                "indexIntoPivotsArray" -> TInt32,
              )),
              unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber
                .view
                .zipWithIndex
                .foldLeft((0, ArraySeq.newBuilder[Row])) {
                  case (s, ((_, pivotIdx), indexIntoPivotsArray)) =>
                    partitionDataPerSegment(pivotIdx).foldLeft(s) { case ((partIdx, b), x) =>
                      val r = Row(x.indices.last, x.files, partIdx, indexIntoPivotsArray)
                      (partIdx + 1, b += r)
                    }
                }
                ._2
                .result(),
            )

          val distributeGlobals =
            makestruct(
              "pivotsWithEndpointsGroupedBySegmentIdx" ->
                Literal(
                  TArray(TArray(keyToSortBy)),
                  unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.map(_._1._1),
                )
            )

          val tmpPath = ctx.createTmpPath("hail_shuffle_temp")

          val distribute =
            cdaIR(
              ToStream(distributeContexts),
              distributeGlobals,
              s"shuffle_distribute_iteration_$i",
            ) {
              (ctx, global) =>
                val segmentIdx = GetField(ctx, "segmentIdx")
                val indexIntoPivotsArray = GetField(ctx, "indexIntoPivotsArray")
                val pivotsWithEndpointsGroupedBySegmentIdx =
                  GetField(global, "pivotsWithEndpointsGroupedBySegmentIdx")

                val path = strConcat(tmpPath + "_", GetField(ctx, "partIdx"))

                val partitionStream =
                  flatMapIR(ToStream(GetField(ctx, "files"))) { fileName =>
                    ReadPartition(
                      makestruct("partitionIndex" -> I64(0), "partitionPath" -> fileName),
                      tcoerce[TStruct](spec._vType),
                      reader,
                    )
                  }

                maketuple(
                  segmentIdx,
                  StreamDistribute(
                    partitionStream,
                    ArrayRef(pivotsWithEndpointsGroupedBySegmentIdx, indexIntoPivotsArray),
                    path,
                    StructCompare(sortFields),
                    spec,
                  ),
                )
            }

          logger.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: STAGE 2: DISTRIBUTE")

          // distributeResult is a numPartitions length array of arrays, where
          // each inner array tells me which files were written for each partition,
          // as well as the number of entries in that file.
          val distributeResult =
            CompileAndEvaluate[IndexedSeq[Row]](ctx, distribute)
              .map(row =>
                (
                  row(0).asInstanceOf[Int],
                  row(1).asInstanceOf[IndexedSeq[Row]].map(innerRow =>
                    (
                      innerRow(0).asInstanceOf[Interval],
                      innerRow(1).asInstanceOf[String],
                      innerRow(2).asInstanceOf[Int],
                      innerRow(3).asInstanceOf[Long],
                    )
                  ),
                )
              )

          val protoDataPerSegment =
            orderedGroupBy[(Int, IndexedSeq[(Interval, String, Int, Long)]), Int](
              distributeResult,
              _._1,
            )
              .map { case (_, seqOfChunkData) => seqOfChunkData.map(_._2) }

          val transposedIntoNewSegments =
            protoDataPerSegment
              .zip(remainingUnsortedSegments.map(_.indices))
              .flatMap { case (oneOldSegment, priorIndices) =>
                val headLen = oneOldSegment.head.length
                assert(oneOldSegment.forall(_.length == headLen))
                ArraySeq.tabulate(headLen) { colIdx =>
                  (oneOldSegment.map(_(colIdx)), priorIndices)
                }
              }

          val dataPerSegment =
            transposedIntoNewSegments
              .zipWithIndex
              .map { case ((chunksWithSameInterval, priorIndices), newIndex) =>
                val interval = chunksWithSameInterval.head._1
                val chunks = chunksWithSameInterval.map { case (_, filename, numRows, numBytes) =>
                  Chunk(filename, numRows.toLong, numBytes)
                }
                val newSegmentIndices = priorIndices :+ newIndex
                SegmentResult(newSegmentIndices, interval, chunks)
              }

          // Decide whether a segment is small enough to be removed from consideration.
          dataPerSegment.partition { sr =>
            val isBig = sr.chunks.map(_.byteSize).sum > sizeCutoff
            // Need to call it "small" if it can't be further subdivided.
            isBig &&
            (sr.interval.left.point != sr.interval.right.point) &&
            (sr.chunks.map(_.size).sum > 1)
          }
        }

      logger.info(
        s"DISTRIBUTED SORT: PHASE ${i + 1}: ${newSmallSegments.length}/${newSmallSegments.length + newBigUnsortedSegments.length} segments can be locally sorted"
      )

      loopState = LoopState(
        newBigUnsortedSegments,
        loopState.smallSegments ++ newSmallSegments,
        loopState.readyOutputParts ++ outputPartitions,
      )
      i = i + 1
    }

    val needSortingFilenames = loopState.smallSegments.map(_.chunks.map(_.filename))
    val needSortingFilenamesContext = Literal(TArray(TArray(TString)), needSortingFilenames)

    val sortedFilenamesIR =
      cdaIR(ToStream(needSortingFilenamesContext), makestruct(), "shuffle_local_sort") {
        case (filenames, _) =>
          val partitionInputStream =
            flatMapIR(ToStream(filenames)) { fileName =>
              ReadPartition(
                makestruct("partitionIndex" -> I64(0), "partitionPath" -> fileName),
                tcoerce[TStruct](spec._vType),
                reader,
              )
            }

          val newKeyFieldNames = keyToSortBy.fields.map(_.name)
          val sortedStream =
            ToStream(sortIR(partitionInputStream) { (refLeft, refRight) =>
              ApplyComparisonOp(
                StructLT(sortFields),
                SelectFields(refLeft, newKeyFieldNames),
                SelectFields(refRight, newKeyFieldNames),
              )
            })

          WritePartition(sortedStream, UUID4(), writer)
      }

    logger.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: LOCALLY SORT FILES")
    logger.info(s"DISTRIBUTED_SORT: ${needSortingFilenames.length} segments to sort")
    val sortedFilenames =
      CompileAndEvaluate[IndexedSeq[Row]](ctx, sortedFilenamesIR)
        .map(_(0).asInstanceOf[String])

    val newlySortedSegments =
      loopState.smallSegments.zip(sortedFilenames).map { case (sr, newFilename) =>
        OutputPartition(sr.indices, sr.interval, FastSeq(initialTmpPath + newFilename))
      }

    val unorderedOutputPartitions = newlySortedSegments ++ loopState.readyOutputParts
    val orderedOutputPartitions = unorderedOutputPartitions.sortWith { (srt1, srt2) =>
      lessThanForSegmentIndices(srt1.indices, srt2.indices)
    }

    val keyed = sortFields.forall(sf => sf.sortOrder == Ascending)
    DistributionSortReader(
      keyToSortBy,
      keyed,
      spec,
      orderedOutputPartitions,
      initialGlobal,
      spec._vType.asInstanceOf[TStruct],
      tableRequiredness,
    )
  }

  private def orderedGroupBy[T: ClassTag, U](is: IndexedSeq[T], func: T => U)
    : IndexedSeq[(U, IndexedSeq[T])] = {
    val result = ArraySeq.newBuilder[(U, IndexedSeq[T])]
    result.sizeHint(is)
    val currentGroup = new ArrayBuffer[T]()
    var lastKeySeen: Option[U] = None
    is.foreach { element =>
      val key = func(element)
      if (currentGroup.isEmpty) {
        currentGroup.append(element)
        lastKeySeen = Some(key)
      } else if (lastKeySeen.contains(key)) {
        currentGroup.append(element)
      } else {
        result += lastKeySeen.get -> currentGroup.to(ArraySeq)
        currentGroup.clear()
        currentGroup.append(element)
        lastKeySeen = Some(key)
      }
    }
    if (currentGroup.nonEmpty) {
      result += lastKeySeen.get -> currentGroup.to(ArraySeq)
    }
    result.result()
  }

  private def lessThanForSegmentIndices(i1: IndexedSeq[Int], i2: IndexedSeq[Int]): Boolean = {
    var idx = 0
    val minLength = math.min(i1.length, i2.length)
    while (idx < minLength) {
      if (i1(idx) != i2(idx)) {
        return i1(idx) < i2(idx)
      }
      idx += 1
    }
    /* For there to be no difference at this point, they had to be equal whole way. Assert that
     * they're same length. */
    assert(i1.length == i2.length)
    false
  }

  case class PartitionInfo(
    indices: IndexedSeq[Int],
    files: IndexedSeq[String],
    currentPartSize: Long,
    currentPartByteSize: Long,
  )

  private def segmentsToPartitionData(
    segments: IndexedSeq[SegmentResult],
    idealNumberOfRowsPerPart: Long,
  ): IndexedSeq[IndexedSeq[PartitionInfo]] = {
    segments.map { sr =>
      val chunkDataSizes = sr.chunks.map(_.size)
      val segmentSize = chunkDataSizes.sum
      val numParts =
        coerceToInt((segmentSize + idealNumberOfRowsPerPart - 1) / idealNumberOfRowsPerPart)
      var currentPartSize = 0L
      var currentPartByteSize = 0L
      val groupedIntoParts = ArraySeq.newBuilder[PartitionInfo]
      groupedIntoParts.sizeHint(numParts)
      val currentFiles = new ArrayBuffer[String]()
      sr.chunks.foreach { chunk =>
        if (chunk.size > 0) {
          currentFiles += chunk.filename
          currentPartSize += chunk.size
          currentPartByteSize += chunk.byteSize
          if (currentPartSize >= idealNumberOfRowsPerPart) {
            groupedIntoParts += PartitionInfo(
              sr.indices,
              currentFiles.to(ArraySeq),
              currentPartSize,
              currentPartByteSize,
            )
            currentFiles.clear()
            currentPartSize = 0
            currentPartByteSize = 0L
          }
        }
      }
      if (currentFiles.nonEmpty) {
        groupedIntoParts += PartitionInfo(
          sr.indices,
          currentFiles.to(ArraySeq),
          currentPartSize,
          currentPartByteSize,
        )
      }
      groupedIntoParts.result()
    }
  }

  private def howManySamplesPerPartition(
    rand: ThreefryRandomEngine,
    totalNumberOfRecords: Long,
    initialNumSamplesToSelect: Int,
    partitionCounts: IndexedSeq[Long],
  ): IndexedSeq[Int] = {
    var successStatesRemaining = initialNumSamplesToSelect.toDouble
    var failureStatesRemaining = totalNumberOfRecords.toDouble - successStatesRemaining

    partitionCounts.map { n =>
      val numSuccesses = rand.rhyper(
        successStatesRemaining,
        failureStatesRemaining,
        n.toDouble,
      ).toInt
      successStatesRemaining -= numSuccesses
      failureStatesRemaining -= (n - numSuccesses)
      numSuccesses
    }
  }

  def samplePartition(dataStream: IR, sampleIndices: IR, sortFields: IndexedSeq[SortField]): IR = {

    // Step 1: Join the dataStream zippedWithIdx on sampleIndices?
    // That requires sampleIndices to be a stream of structs
    val structSampleIndices =
      mapIR(sampleIndices)(sampleIndex => makestruct("idx" -> sampleIndex))

    val joined =
      joinIR(
        zipWithIndex(dataStream),
        structSampleIndices,
        ArraySeq("idx"),
        ArraySeq("idx"),
        "left",
        requiresMemoryManagement = true,
      ) {
        (l, r) =>
          makestruct(
            "elt" -> GetField(l, "elt"),
            "shouldKeep" -> !IsNA(r),
          )
      }

    // Step 2: Aggregate over joined, figure out how to collect only the rows
    // that are marked "shouldKeep"
    val Zero =
      makestruct(
        "lastKeySeen" -> NA(TIterable.elementType(dataStream.typ)),
        "haveSeenAny" -> false,
        "sortedSoFar" -> true,
      )

    bindIR(Zero) { zero =>
      streamAggIR(joined) { elem =>
        aggBindIR(GetField(elem, "elt")) { elt =>
          makestruct(
            "min" -> AggFold.min(elt, sortFields),
            "max" -> AggFold.max(elt, sortFields),
            "samples" ->
              AggFilter(
                GetField(elem, "shouldKeep"),
                ApplyAggOp(Collect())(elt),
                isScan = false,
              ),
            "isSorted" ->
              GetField(
                aggFoldIR(zero) { accum =>
                  makestruct(
                    "lastKeySeen" -> elt,
                    "haveSeenAny" -> true,
                    "sortedSoFar" -> (
                      !GetField(accum, "haveSeenAny") || (
                        GetField(accum, "sortedSoFar") &&
                          ApplyComparisonOp(
                            StructLTEQ(sortFields),
                            GetField(accum, "lastKeySeen"),
                            elt,
                          )
                      )
                    ),
                  )
                }((a, _) => Die("combOp is never called in StreamAgg", a.typ)),
                "sortedSoFar",
              ),
          )
        }
      }
    }
  }

  // Given an IR of type TArray(TTuple(minKey, maxKey)), determine if there's any
  // overlap between these closed intervals.
  private def tuplesAreSorted(arrayOfTuples: Atom, sortFields: IndexedSeq[SortField]): IR =
    bindIR(ArrayLen(arrayOfTuples)) { len =>
      tailLoop(TBoolean, 1) { case (recur, Seq(idx)) =>
        val `sorted?` =
          ApplyComparisonOp(
            StructLTEQ(sortFields),
            GetTupleElement(ArrayRef(arrayOfTuples, idx - 1), 1),
            GetTupleElement(ArrayRef(arrayOfTuples, idx), 0),
          )

        If(idx >= len, True(), If(`sorted?`, recur(FastSeq(idx + 1)), False()))
      }
    }
}

/** a "Chunk" is a file resulting from any StreamDistribute. Chunks are internally unsorted but
  * contain data between two pivots.
  */
case class Chunk(filename: String, size: Long, byteSize: Long)

/** A SegmentResult is the set of chunks from various StreamDistribute tasks working on the same
  * segment of a previous iteration.
  */
case class SegmentResult(indices: IndexedSeq[Int], interval: Interval, chunks: IndexedSeq[Chunk])
case class OutputPartition(indices: IndexedSeq[Int], interval: Interval, files: IndexedSeq[String])

case class LoopState(
  largeSegments: IndexedSeq[SegmentResult],
  smallSegments: IndexedSeq[SegmentResult],
  readyOutputParts: IndexedSeq[OutputPartition],
)

case class DistributionSortReader(
  key: TStruct,
  keyed: Boolean,
  spec: TypedCodecSpec,
  orderedOutputPartitions: IndexedSeq[OutputPartition],
  globals: IR,
  rowType: TStruct,
  rt: RTable,
) extends TableReader {
  lazy val fullType: TableType = TableType(
    rowType,
    if (keyed) key.fieldNames else FastSeq(),
    globals.typ.asInstanceOf[TStruct],
  )

  override def pathsUsed: Seq[String] = FastSeq()

  def defaultPartitioning(sm: HailStateManager): RVDPartitioner = {
    val (partitionerKey, intervals) = if (keyed) {
      (key, orderedOutputPartitions.map(segment => segment.interval))
    } else {
      (TStruct(), orderedOutputPartitions.map(_ => Interval(Row(), Row(), true, false)))
    }

    new RVDPartitioner(sm, partitionerKey, intervals)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def isDistinctlyKeyed: Boolean = false // FIXME: No default value

  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq.subset(requestedType.rowType, rt.rowType)

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq.subset(requestedType.globalType, rt.globalType)

  override def toJValue: JValue = JString("DistributionSortReader")

  override def renderShort(): String = "DistributionSortReader"

  override def defaultRender(): String = "DistributionSortReader"

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    PruneDeadFields.upcast(ctx, globals, requestedGlobalsType)

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    var filesCount: Long = 0
    val contexts =
      orderedOutputPartitions.map { segment =>
        segment.files.map { partPath =>
          val partIdx = filesCount
          filesCount += 1
          Row(partIdx, partPath)
        }
      }

    val partReader = PartitionNativeReader(spec, "__dummy_uid")

    TableStage(
      PruneDeadFields.upcast(ctx, globals, requestedType.globalType),
      defaultPartitioning(ctx.stateManager).coarsen(requestedType.key.length),
      TableStageDependency.none,
      ToStream(Literal(TArray(TArray(partReader.contextType)), contexts)),
      ctxRef =>
        flatMapIR(ToStream(ctxRef)) { fileInfo =>
          ReadPartition(fileInfo, requestedType.rowType, partReader)
        },
    )
  }
}
