package is.hail.expr.ir.lowering

import cats.mtl.Ask
import cats.syntax.all._
import is.hail.annotations.{Annotation, ExtendedOrdering, Region, SafeRow}
import is.hail.asm4s.{AsmFunction1RegionLong, LongInfo, classInfo}
import utils._
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.expr.ir._
import is.hail.expr.ir.functions.{ArrayFunctions, IRRandomness, UtilFunctions}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.physical.{PArray, PStruct}
import is.hail.types.virtual._
import is.hail.types.{RTable, TableType, VirtualTypeWithReq, tcoerce}
import is.hail.utils.{time => _, _}
import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.JsonAST.JString

import scala.collection.mutable.ArrayBuffer
import scala.language.higherKinds

object LowerDistributedSort {
  def localSort[M[_]](inputStage: TableStage, sortFields: IndexedSeq[SortField], rt: RTable)
                     (implicit M: MonadLower[M]): M[TableReader] =
    for {
      (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) <-
        timeM("LowerDistributedSort.localSort.compile") {
          Compile[M, AsmFunction1RegionLong](
            FastIndexedSeq(),
            FastIndexedSeq(classInfo[Region]), LongInfo,
            inputStage.collectWithGlobals("shuffle_local_sort"),
            print = None,
            optimize = true
          )
        }

      numPartitions = inputStage.partitioner.numPartitions

      rowsAndGlobal <- scopedExecution { case (hcl, fs, htc, r) =>
        for {
          fRunnable <- time("LowerDistributedSort.localSort.initialize")(f(hcl, fs, htc, r))
          resultAddress <- time("LowerDistributedSort.localSort.run")(fRunnable(r))
          row <- time("LowerDistributedSort.localSort.toJavaObject")(SafeRow.read(resultPType, resultAddress))
        } yield row.asInstanceOf[Row]
      }

      rowsType = resultPType.fieldType("rows").asInstanceOf[PArray]
      rowType = rowsType.elementType.asInstanceOf[PStruct]
      rows = rowsAndGlobal.getAs[IndexedSeq[Annotation]](0)
      kType = TStruct(sortFields.map(f => (f.field, rowType.virtualType.fieldType(f.field))): _*)

      sortedRows <- localAnnotationSort(rows, sortFields, rowType.virtualType)

      nPartitionsAdj = math.max(math.min(sortedRows.length, numPartitions), 1)
      itemsPerPartition = math.max((sortedRows.length.toDouble / nPartitionsAdj).ceil.toInt, 1)

      // partitioner needs keys to be ascending
      partitionerKeyType = TStruct(sortFields.takeWhile(_.sortOrder == Ascending).map(f => (f.field, rowType.virtualType.fieldType(f.field))): _*)
      partitionerKeyIndex = partitionerKeyType.fieldNames.map(f => rowType.fieldIdx(f))

      stateManager <- M.reader(_.stateManager)
      partitioner = new RVDPartitioner(stateManager, partitionerKeyType,
        sortedRows.grouped(itemsPerPartition).map { group =>
          val first = group.head.asInstanceOf[Row].select(partitionerKeyIndex)
          val last = group.last.asInstanceOf[Row].select(partitionerKeyIndex)
          Interval(first, last, includesStart = true, includesEnd = true)
        }.toIndexedSeq)

      globalsIR = Literal(resultPType.fieldType("global").virtualType, rowsAndGlobal.get(1))
    } yield LocalSortReader(sortedRows, rowType.virtualType, globalsIR, partitioner, itemsPerPartition, rt)


  case class LocalSortReader(sortedRows: IndexedSeq[Annotation], rowType: TStruct, globals: IR, partitioner: RVDPartitioner, itemsPerPartition: Int, rt: RTable) extends TableReader {
    lazy val fullType: TableType = TableType(rowType, partitioner.kType.fieldNames, globals.typ.asInstanceOf[TStruct])

    override def pathsUsed: Seq[String] = FastIndexedSeq()

    override def partitionCounts: Option[IndexedSeq[Long]] = None

    override def apply[M[_]: MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue] =
      for {
        _ <- assertA(!dropRows);
        ts <- lower(requestedType);
        tv <- TableStageIntermediate(ts).asTableValue
      } yield tv

    override def isDistinctlyKeyed: Boolean = false // FIXME: No default value

    def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
      VirtualTypeWithReq.subset(requestedType.rowType, rt.rowType)
    }

    def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
      VirtualTypeWithReq.subset(requestedType.globalType, rt.globalType)
    }

    override def toJValue: JValue = JString("LocalSortReader")

    def renderShort(): String = "LocalSortReader"

    override def defaultRender(): String = "LocalSortReader"

    override def lowerGlobals[M[_] : MonadLower](requestedGlobalsType: TStruct): M[IR] =
      MonadLower[M].reader { ctx =>
        PruneDeadFields.upcast(ctx, globals, requestedGlobalsType)
      }

    override def lower[M[_]](requestedType: TableType)
                            (implicit M: MonadLower[M]): M[TableStage] =
      TableStage(
        globals = globals,
        partitioner = partitioner.coarsen(requestedType.key.length),
        TableStageDependency.none,
        contexts = mapIR(
          StreamGrouped(
            ToStream(Literal(TArray(rowType), sortedRows)),
            I32(itemsPerPartition))
        )(ToArray(_)),
        ToStream(_)
      ).upcast(requestedType)(M)
  }

  private def localAnnotationSort[M[_]](
    annotations: IndexedSeq[Annotation],
    sortFields: IndexedSeq[SortField],
    rowType: TStruct
  )(implicit M: Ask[M, ExecuteContext]): M[IndexedSeq[Annotation]] =
    M.reader { ctx =>
      val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
        val i = rowType.fieldIdx(n)
        val f = rowType.fields(i)
        val fo = f.typ.ordering(ctx.stateManager)
        if (so == Ascending) fo else fo.reverse
      }.toArray

      val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

      val kType = TStruct(sortFields.map(f => (f.field, rowType.fieldType(f.field))): _*)
      val kIndex = kType.fieldNames.map(f => rowType.fieldIdx(f))
      ctx.timer.time("LowerDistributedSort.localSort.sort")(annotations.sortBy { a: Annotation =>
        a.asInstanceOf[Row].select(kIndex).asInstanceOf[Annotation]
      }(ord))
    }

  def distributedSort[M[_]](inputStage: TableStage,
                            sortFields: IndexedSeq[SortField],
                            tableRequiredness: RTable,
                            optTargetNumPartitions: Option[Int] = None
                           )(implicit M: MonadLower[M]): M[TableReader] =
    for {
      defaultBranchingFactor <- readFlag("shuffle_max_branch_factor").map { maxBranchingFactor =>
        Math.min(maxBranchingFactor.toInt, Math.max(2, inputStage.numPartitions))
      }
      sizeCutoff <- readFlag("shuffle_cutoff_to_local_sort").map(_.toInt)

      initialTmpPath <- newTmpPath("hail_shuffle_temp_initial")
      spec = TypedCodecSpec(tableRequiredness.rowType.canonicalPType(inputStage.rowType), BufferSpec.wireSpec)
      (keyToSortBy, _) = inputStage.rowType.select(sortFields.map(_.field))
      writer = PartitionNativeWriter(spec, keyToSortBy.fieldNames, initialTmpPath, None, None, trackTotalBytes = true)

      _ = log.info("DISTRIBUTED SORT: PHASE 1: WRITE DATA")
      (initialPartInfo, initialGlobals, intervalRange) <-
        CompileAndEvaluate[M, Annotation](
          inputStage.mapCollectWithGlobals("shuffle_initial_write")(WritePartition(_, UUID4(), writer)) {
            case (part, globals) =>
              val streamElement = Ref(genUID(), part.typ.asInstanceOf[TArray].elementType)
              bindIR(StreamAgg(ToStream(part), streamElement.name, MakeStruct(FastSeq(
                "min" -> AggFold.min(GetField(streamElement, "firstKey"), sortFields),
                "max" -> AggFold.max(GetField(streamElement, "lastKey"), sortFields)
              )))) { intervalRange => MakeTuple.ordered(FastIndexedSeq(part, globals, intervalRange)) }
          }
        ).map { case initialStageDataRow: Row =>
          (
            initialStageDataRow(0).asInstanceOf[IndexedSeq[Row]],
            initialStageDataRow(1).asInstanceOf[Row],
            initialStageDataRow(2).asInstanceOf[Row]
          )
        }

      initialGlobalsLiteral = Literal(inputStage.globalType, initialGlobals)
      initialChunks = initialPartInfo.map(row => Chunk(initialTmpPath + row(0).asInstanceOf[String], row(1).asInstanceOf[Long], row.getLong(5)))

      initialInterval = Interval(intervalRange(0), intervalRange(1), true, true)
      initialSegment = SegmentResult(IndexedSeq(0), initialInterval, initialChunks)

      totalNumberOfRows = initialChunks.map(_.size).sum
      _ = optTargetNumPartitions.foreach(i => assert(i >= 1, s"Must request positive number of partitions. Requested ${i}"))
      targetNumPartitions = optTargetNumPartitions.getOrElse(inputStage.numPartitions)

      idealNumberOfRowsPerPart: Long = if (targetNumPartitions == 0) 1 else {
        Math.max(1L, totalNumberOfRows / targetNumPartitions)
      }

      rand = new IRRandomness(7L)
      oversamplingNum = 3
      reader = PartitionNativeReader(spec, "__dummy_uid")

      /*
      Loop state keeps track of three things. largeSegments are too big to sort locally so have to broken up.
      smallSegments are small enough to be sorted locally. readyOutputParts are any partitions that we noticed were
      sorted already during course of the recursion. Loop continues until there are no largeSegments left. Then we
      sort the small segments and combine them with readyOutputParts to get the final table.
       */
      init = (LoopState(FastSeq(initialSegment), IndexedSeq.empty, IndexedSeq.empty), 0)
      (loopState, i) <- M.iterateWhileM(init) { case (loopState, i) =>
        val partitionDataPerSegment = segmentsToPartitionData(loopState.largeSegments, idealNumberOfRowsPerPart)
        assert(partitionDataPerSegment.size == loopState.largeSegments.size)

        val numSamplesPerPartitionPerSegment = partitionDataPerSegment.map { partData =>
          val partitionCountsForOneSegment = partData.map(_.currentPartSize)
          val recordsInSegment = partitionCountsForOneSegment.sum
          val branchingFactor = math.min(recordsInSegment, defaultBranchingFactor)
          howManySamplesPerPartition(rand, recordsInSegment, Math.min(recordsInSegment, (branchingFactor * oversamplingNum) - 1).toInt, partitionCountsForOneSegment)
        }

        val numSamplesPerPartition = numSamplesPerPartitionPerSegment.flatten

        val perPartStatsCDAContextData = partitionDataPerSegment.flatten.zip(numSamplesPerPartition).map { case (partData, numSamples) =>
          Row(partData.indices.last, partData.files, coerceToInt(partData.currentPartSize), numSamples, partData.currentPartByteSize)
        }
        val perPartStatsCDAContexts = ToStream(Literal(TArray(TStruct(
          "segmentIdx" -> TInt32,
          "files" -> TArray(TString),
          "sizeOfPartition" -> TInt32,
          "numSamples" -> TInt32,
          "byteSize" -> TInt64)), perPartStatsCDAContextData
        ))

        val perPartStatsIR =
          cdaIR(perPartStatsCDAContexts, MakeStruct(FastIndexedSeq()), s"shuffle_part_stats_iteration_$i") { (ctxRef, _) =>
            val filenames = GetField(ctxRef, "files")
            val samples = SeqSample(GetField(ctxRef, "sizeOfPartition"), GetField(ctxRef, "numSamples"), NA(TRNGState), false)
            val partitionStream = flatMapIR(ToStream(filenames)) { fileName =>
              mapIR(
                ReadPartition(
                  MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> fileName)),
                  tcoerce[TStruct](spec._vType),
                  reader
                )
              )(SelectFields(_, keyToSortBy.fields.map(_.name)))
            }
            MakeStruct(IndexedSeq(
              "segmentIdx" -> GetField(ctxRef, "segmentIdx"),
              "byteSize" -> GetField(ctxRef, "byteSize"),
              "partData" -> samplePartition(partitionStream, samples, sortFields)
            ))
          }

        /*
        Aggregate over the segments, to compute the pivots, whether it's already sorted, and what key interval is contained in that segment.
        Also get the min and max of each individual partition. That way if it's sorted already, we know the partitioning to use.
         */
        val pivotsPerSegmentAndSortedCheck = ToArray(bindIR(perPartStatsIR) { perPartStats =>
          mapIR(StreamGroupByKey(ToStream(perPartStats), IndexedSeq("segmentIdx"), missingEqual = true)) { oneGroup =>
            val streamElementRef = Ref(genUID(), oneGroup.typ.asInstanceOf[TIterable].elementType)
            val dataRef = Ref(genUID(), streamElementRef.typ.asInstanceOf[TStruct].fieldType("partData"))
            val sizeRef = Ref(genUID(), streamElementRef.typ.asInstanceOf[TStruct].fieldType("byteSize"))
            bindIR(StreamAgg(oneGroup, streamElementRef.name, {
              AggLet(dataRef.name, GetField(streamElementRef, "partData"), AggLet(sizeRef.name, GetField(streamElementRef, "byteSize"),
                MakeStruct(FastIndexedSeq(
                  ("byteSize", ApplyAggOp(Sum())(sizeRef)),
                  ("min", AggFold.min(GetField(dataRef, "min"), sortFields)), // Min of the mins
                  ("max", AggFold.max(GetField(dataRef, "max"), sortFields)), // Max of the maxes
                  ("perPartMins", ApplyAggOp(Collect())(GetField(dataRef, "min"))), // All the mins
                  ("perPartMaxes", ApplyAggOp(Collect())(GetField(dataRef, "max"))), // All the maxes
                  ("samples", ApplyAggOp(Collect())(GetField(dataRef, "samples"))),
                  ("eachPartSorted", AggFold.all(GetField(dataRef, "isSorted"))),
                  ("perPartIntervalTuples", ApplyAggOp(Collect())(MakeTuple.ordered(FastIndexedSeq(GetField(dataRef, "min"), GetField(dataRef, "max")))))
                )), false), false)
            })) { aggResults =>
              val sortedOversampling = sortIR(flatMapIR(ToStream(GetField(aggResults, "samples"))) { onePartCollectedArray => ToStream(onePartCollectedArray) }) { case (left, right) =>
                ApplyComparisonOp(StructLT(keyToSortBy, sortFields), left, right)
              }
              val minArray = MakeArray(GetField(aggResults, "min"))
              val maxArray = MakeArray(GetField(aggResults, "max"))
              val tuplesInSortedOrder = tuplesAreSorted(GetField(aggResults, "perPartIntervalTuples"), sortFields)
              bindIR(sortedOversampling) { sortedOversampling =>
                bindIR(ArrayLen(sortedOversampling)) { numSamples =>
                  val sortedSampling = bindIR(
                    /* calculate a 'good' branch factor based on part sizes */
                    UtilFunctions.intMax(I32(2),
                      UtilFunctions.intMin(
                        UtilFunctions.intMin(numSamples, I32(defaultBranchingFactor)),
                        (I64(2L) * (GetField(aggResults, "byteSize").floorDiv(I64(sizeCutoff)))).toI))
                  ) { branchingFactor =>
                    ToArray(mapIR(StreamRange(I32(1), branchingFactor, I32(1))) { idx =>
                      If(ArrayLen(sortedOversampling) ceq 0,
                        Die(strConcat("aggresults=", aggResults, ", idx=", idx, ", sortedOversampling=", sortedOversampling, ", numSamples=", numSamples, ", branchingFactor=", branchingFactor), sortedOversampling.typ.asInstanceOf[TArray].elementType, -1),
                        ArrayRef(sortedOversampling, Apply("floor", FastIndexedSeq(), IndexedSeq(idx.toD * ((numSamples + 1) / branchingFactor)), TFloat64, ErrorIDs.NO_ERROR).toI - 1))

                    })
                  }
                  MakeStruct(FastIndexedSeq(
                    "pivotsWithEndpoints" -> ArrayFunctions.extend(ArrayFunctions.extend(minArray, sortedSampling), maxArray),
                    "isSorted" -> ApplySpecial("land", Seq.empty[Type], FastIndexedSeq(GetField(aggResults, "eachPartSorted"), tuplesInSortedOrder), TBoolean, ErrorIDs.NO_ERROR),
                    "intervalTuple" -> MakeTuple.ordered(FastIndexedSeq(GetField(aggResults, "min"), GetField(aggResults, "max"))),
                    "perPartMins" -> GetField(aggResults, "perPartMins"),
                    "perPartMaxes" -> GetField(aggResults, "perPartMaxes")
                  ))
                }
              }
            }
          }
        })

        log.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: STAGE 1: SAMPLE VALUES FROM PARTITIONS")
        // Going to check now if it's fully sorted, as well as collect and sort all the samples.
        for {
          pivotsWithEndpointsAndInfoGroupedBySegmentNumber <-
            for {r <- CompileAndEvaluate[M, Annotation](pivotsPerSegmentAndSortedCheck)}
              yield r.asInstanceOf[IndexedSeq[Row]].map { x =>
                (
                  x(0).asInstanceOf[IndexedSeq[Row]],
                  x(1).asInstanceOf[Boolean],
                  x(2).asInstanceOf[Row],
                  x(3).asInstanceOf[IndexedSeq[Row]],
                  x(4).asInstanceOf[IndexedSeq[Row]]
                )
              }

          pivotCounts = pivotsWithEndpointsAndInfoGroupedBySegmentNumber
            .map(_._1.length)
            .groupBy(identity)
            .toArray
            .map { case (i, values) => (i, values.length) }
            .sortBy(_._1)
            .map { case (nPivots, nSegments) => s"$nPivots pivots: $nSegments" }

          _ = log.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: pivot counts:\n  ${pivotCounts.mkString("\n  ")}")
          (sortedSegmentsTuples, unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber) =
            pivotsWithEndpointsAndInfoGroupedBySegmentNumber.zipWithIndex.partition { case ((_, isSorted, _, _, _), _) => isSorted }

          outputPartitions = sortedSegmentsTuples.flatMap { case ((_, _, _, partMins, partMaxes), originalSegmentIdx) =>
            val segmentToBreakUp = loopState.largeSegments(originalSegmentIdx)
            val currentSegmentPartitionData = partitionDataPerSegment(originalSegmentIdx)
            val partRanges = partMins.zip(partMaxes)
            assert(partRanges.size == currentSegmentPartitionData.size)

            currentSegmentPartitionData.zip(partRanges).zipWithIndex.map { case ((pi, (intervalStart, intervalEnd)), idx) =>
              OutputPartition(segmentToBreakUp.indices :+ idx, Interval(intervalStart, intervalEnd, true, true), pi.files)
            }
          }

          remainingUnsortedSegments = unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.map { case (_, idx) => loopState.largeSegments(idx) }

          (newBigUnsortedSegments, newSmallSegments) <-
            if (unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.nonEmpty) {
              val pivotsWithEndpointsGroupedBySegmentNumber = unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.map { case (r, _) => r._1 }
              val pivotsWithEndpointsGroupedBySegmentNumberLiteral = Literal(TArray(TArray(keyToSortBy)), pivotsWithEndpointsGroupedBySegmentNumber)
              val unsortedPartitionDataPerSegment = unsortedPivotsWithEndpointsAndInfoGroupedBySegmentNumber.map { case (_, idx) => partitionDataPerSegment(idx) }

              val partitionDataPerSegmentWithPivotIndex = unsortedPartitionDataPerSegment.zipWithIndex.map { case (partitionDataForOneSegment, indexIntoPivotsArray) =>
                partitionDataForOneSegment.map(x => (x.indices, x.files, x.currentPartSize, indexIntoPivotsArray))
              }

              val distributeContextsData = partitionDataPerSegmentWithPivotIndex.flatten.zipWithIndex.map { case (part, partIdx) => Row(part._1.last, part._2, partIdx, part._4) }
              val distributeContexts = ToStream(Literal(TArray(TStruct("segmentIdx" -> TInt32, "files" -> TArray(TString), "partIdx" -> TInt32, "indexIntoPivotsArray" -> TInt32)), distributeContextsData))
              val distributeGlobals = MakeStruct(IndexedSeq("pivotsWithEndpointsGroupedBySegmentIdx" -> pivotsWithEndpointsGroupedBySegmentNumberLiteral))
              for {
                tmpPath <- newTmpPath("hail_shuffle_temp")
                distribute =
                  cdaIR(distributeContexts, distributeGlobals, s"shuffle_distribute_iteration_$i") { (ctxRef, globalsRef) =>
                    val segmentIdx = GetField(ctxRef, "segmentIdx")
                    val indexIntoPivotsArray = GetField(ctxRef, "indexIntoPivotsArray")
                    val pivotsWithEndpointsGroupedBySegmentIdx = GetField(globalsRef, "pivotsWithEndpointsGroupedBySegmentIdx")
                    val path = invoke("concat", TString, Str(tmpPath + "_"), invoke("str", TString, GetField(ctxRef, "partIdx")))
                    val filenames = GetField(ctxRef, "files")
                    val partitionStream = flatMapIR(ToStream(filenames)) { fileName =>
                      ReadPartition(MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> fileName)), tcoerce[TStruct](spec._vType), reader)
                    }
                    MakeTuple.ordered(IndexedSeq(segmentIdx, StreamDistribute(partitionStream, ArrayRef(pivotsWithEndpointsGroupedBySegmentIdx, indexIntoPivotsArray), path, StructCompare(keyToSortBy, keyToSortBy, sortFields.toArray), spec)))
                  }

                _ = log.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: STAGE 2: DISTRIBUTE")

                distributeResult <-
                  for {distributeResult <- CompileAndEvaluate[M, Annotation](distribute)}
                    yield distributeResult.asInstanceOf[IndexedSeq[Row]].map { row =>
                      (row(0).asInstanceOf[Int], row(1).asInstanceOf[IndexedSeq[Row]].map { innerRow =>
                        (
                          innerRow(0).asInstanceOf[Interval],
                          innerRow(1).asInstanceOf[String],
                          innerRow(2).asInstanceOf[Int],
                          innerRow(3).asInstanceOf[Long]
                        )
                      })
                    }

                // distributeResult is a numPartitions length array of arrays, where each inner array tells me what
                // files were written to for each partition, as well as the number of entries in that file.
                protoDataPerSegment = orderedGroupBy[(Int, IndexedSeq[(Interval, String, Int, Long)]), Int](distributeResult, x => x._1)
                  .map { case (_, seqOfChunkData) => seqOfChunkData.map(_._2) }

                transposedIntoNewSegments =
                  protoDataPerSegment
                    .zip(remainingUnsortedSegments.map(_.indices))
                    .flatMap { case (oneOldSegment, priorIndices) =>
                      val headLen = oneOldSegment.head.length
                      assert(oneOldSegment.forall(x => x.length == headLen))
                      (0 until headLen).map(colIdx => (oneOldSegment.map(row => row(colIdx)), priorIndices))
                    }

                dataPerSegment = transposedIntoNewSegments.zipWithIndex.map { case ((chunksWithSameInterval, priorIndices), newIndex) =>
                  val interval = chunksWithSameInterval.head._1
                  val chunks = chunksWithSameInterval.map { case (_, filename, numRows, numBytes) => Chunk(filename, numRows, numBytes) }
                  val newSegmentIndices = priorIndices :+ newIndex
                  SegmentResult(newSegmentIndices, interval, chunks)
                }

                // Decide whether a segment is small enough to be removed from consideration.
              } yield dataPerSegment.partition { sr =>
                val isBig = sr.chunks.map(_.byteSize).sum > sizeCutoff
                // Need to call it "small" if it can't be further subdivided.
                isBig && (sr.interval.left.point != sr.interval.right.point) && (sr.chunks.map(_.size).sum > 1)
              }
            } else {
              M.pure((IndexedSeq.empty[SegmentResult], IndexedSeq.empty[SegmentResult]))
            }
          _ = log.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: ${newSmallSegments.length}/${newSmallSegments.length + newBigUnsortedSegments.length} segments can be locally sorted")
        } yield (
          LoopState(newBigUnsortedSegments, loopState.smallSegments ++ newSmallSegments, loopState.readyOutputParts ++ outputPartitions),
          i + 1
        )
      }(_._1.largeSegments.nonEmpty)

      needSortingFilenames = loopState.smallSegments.map(_.chunks.map(_.filename))
      needSortingFilenamesContext = Literal(TArray(TArray(TString)), needSortingFilenames)

      sortedFilenamesIR =
        cdaIR(ToStream(needSortingFilenamesContext), MakeStruct(FastIndexedSeq()), "shuffle_local_sort") { case (ctxRef, _) =>
          val filenames = ctxRef
          val partitionInputStream = flatMapIR(ToStream(filenames)) { fileName =>
            ReadPartition(MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> fileName)), tcoerce[TStruct](spec._vType), reader)
          }
          val newKeyFieldNames = keyToSortBy.fields.map(_.name)
          val sortedStream = ToStream(sortIR(partitionInputStream) { (refLeft, refRight) =>
            ApplyComparisonOp(StructLT(keyToSortBy, sortFields), SelectFields(refLeft, newKeyFieldNames), SelectFields(refRight, newKeyFieldNames))
          })
          WritePartition(sortedStream, UUID4(), writer)
        }

      _ = log.info(s"DISTRIBUTED SORT: PHASE ${i + 1}: LOCALLY SORT FILES")
      _ = log.info(s"DISTRIBUTED_SORT: ${needSortingFilenames.length} segments to sort")

      sortedFilenames <-
        for {r <- CompileAndEvaluate[M, Annotation](sortedFilenamesIR)}
          yield r.asInstanceOf[IndexedSeq[Row]].map(_(0).asInstanceOf[String])

      newlySortedSegments = loopState.smallSegments.zip(sortedFilenames).map { case (sr, newFilename) =>
        OutputPartition(sr.indices, sr.interval, IndexedSeq(initialTmpPath + newFilename))
      }

      unorderedOutputPartitions = newlySortedSegments ++ loopState.readyOutputParts
      orderedOutputPartitions = unorderedOutputPartitions.sortWith { (srt1, srt2) => lessThanForSegmentIndices(srt1.indices, srt2.indices) }
      keyed = sortFields.forall(sf => sf.sortOrder == Ascending)
    } yield DistributionSortReader(keyToSortBy, keyed, spec, orderedOutputPartitions, initialGlobalsLiteral, spec._vType.asInstanceOf[TStruct], tableRequiredness)

    def orderedGroupBy[T, U](is: IndexedSeq[T], func: T => U): IndexedSeq[(U, IndexedSeq[T])] = {
      val result = new ArrayBuffer[(U, IndexedSeq[T])](is.size)
      val currentGroup = new ArrayBuffer[T]()
      var lastKeySeen: Option[U] = None
      is.foreach { element =>
        val key = func(element)
        if (currentGroup.isEmpty) {
          currentGroup.append(element)
          lastKeySeen = Some(key)
        } else if (lastKeySeen.map(lastKey => lastKey == key).getOrElse(false)) {
          currentGroup.append(element)
        } else {
          result.append((lastKeySeen.get, currentGroup.result().toIndexedSeq))
          currentGroup.clear()
          currentGroup.append(element)
          lastKeySeen = Some(key)
        }
      }
      if (!currentGroup.isEmpty) {
        result.append((lastKeySeen.get, currentGroup))
      }
      result.result().toIndexedSeq
    }

  def lessThanForSegmentIndices(i1: IndexedSeq[Int], i2: IndexedSeq[Int]): Boolean = {
    var idx = 0
    val minLength = math.min(i1.length, i2.length)
    while (idx < minLength) {
      if (i1(idx) != i2(idx)) {
        return i1(idx) < i2(idx)
      }
      idx += 1
    }
    // For there to be no difference at this point, they had to be equal whole way. Assert that they're same length.
    assert(i1.length == i2.length)
    false
  }

  case class PartitionInfo(indices: IndexedSeq[Int], files: IndexedSeq[String], currentPartSize: Long, currentPartByteSize: Long)

  def segmentsToPartitionData(segments: IndexedSeq[SegmentResult], idealNumberOfRowsPerPart: Long): IndexedSeq[IndexedSeq[PartitionInfo]] = {
    segments.map { sr =>
      val chunkDataSizes = sr.chunks.map(_.size)
      val segmentSize = chunkDataSizes.sum
      val numParts = coerceToInt((segmentSize + idealNumberOfRowsPerPart - 1) / idealNumberOfRowsPerPart)
      var currentPartSize = 0L
      var currentPartByteSize = 0L
      val groupedIntoParts = new ArrayBuffer[PartitionInfo](numParts)
      val currentFiles = new ArrayBuffer[String]()
      sr.chunks.foreach { chunk =>
        if (chunk.size > 0) {
          currentFiles.append(chunk.filename)
          currentPartSize += chunk.size
          currentPartByteSize += chunk.byteSize
          if (currentPartSize >= idealNumberOfRowsPerPart) {
            groupedIntoParts.append(PartitionInfo(sr.indices, currentFiles.result().toIndexedSeq, currentPartSize, currentPartByteSize))
            currentFiles.clear()
            currentPartSize = 0
            currentPartByteSize = 0L
          }
        }
      }
      if (!currentFiles.isEmpty) {
        groupedIntoParts.append(PartitionInfo(sr.indices, currentFiles.result().toIndexedSeq, currentPartSize, currentPartByteSize))
      }
      groupedIntoParts.result()
    }
  }

  def howManySamplesPerPartition(rand: IRRandomness, totalNumberOfRecords: Long, initialNumSamplesToSelect: Int, partitionCounts: IndexedSeq[Long]): IndexedSeq[Int] = {
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

  def samplePartition(dataStream: IR, sampleIndices: IR, sortFields: IndexedSeq[SortField]): IR = {
    // Step 1: Join the dataStream zippedWithIdx on sampleIndices?
    // That requires sampleIndices to be a stream of structs
    val samplingIndexName = "samplingPartitionIndex"
    val structSampleIndices = mapIR(sampleIndices)(sampleIndex => MakeStruct(FastIndexedSeq((samplingIndexName, sampleIndex))))
    val dataWithIdx = zipWithIndex(dataStream)

    val leftName = genUID()
    val rightName = genUID()
    val leftRef = Ref(leftName, dataWithIdx.typ.asInstanceOf[TStream].elementType)
    val rightRef = Ref(rightName, structSampleIndices.typ.asInstanceOf[TStream].elementType)

    val joined = StreamJoin(dataWithIdx, structSampleIndices, IndexedSeq("idx"), IndexedSeq(samplingIndexName), leftName, rightName,
      MakeStruct(FastIndexedSeq(("elt", GetField(leftRef, "elt")), ("shouldKeep", ApplyUnaryPrimOp(Bang(), IsNA(rightRef))))),
      "left", requiresMemoryManagement = true)

    // Step 2: Aggregate over joined, figure out how to collect only the rows that are marked "shouldKeep"
    val streamElementType = joined.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
    val streamElementName = genUID()
    val streamElementRef = Ref(streamElementName, streamElementType)
    val eltName = genUID()
    val eltType = dataStream.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
    val eltRef = Ref(eltName, eltType)

    // Folding for isInternallySorted
    val aggFoldSortedZero = MakeStruct(FastIndexedSeq("lastKeySeen" -> NA(eltType), "sortedSoFar" -> true, "haveSeenAny" -> false))
    val aggFoldSortedAccumName1 = genUID()
    val aggFoldSortedAccumName2 = genUID()
    val isSortedStateType = TStruct("lastKeySeen" -> eltType, "sortedSoFar" -> TBoolean, "haveSeenAny" -> TBoolean)
    val aggFoldSortedAccumRef1 = Ref(aggFoldSortedAccumName1, isSortedStateType)
    val isSortedSeq =
      bindIR(GetField(aggFoldSortedAccumRef1, "lastKeySeen")) { lastKeySeenRef =>
        If(!GetField(aggFoldSortedAccumRef1, "haveSeenAny"),
          MakeStruct(FastIndexedSeq("lastKeySeen" -> eltRef, "sortedSoFar" -> true, "haveSeenAny" -> true)),
          If (ApplyComparisonOp(StructLTEQ(eltType, sortFields), lastKeySeenRef, eltRef),
            MakeStruct(FastIndexedSeq("lastKeySeen" -> eltRef, "sortedSoFar" -> GetField(aggFoldSortedAccumRef1, "sortedSoFar"), "haveSeenAny" -> true)),
            MakeStruct(FastIndexedSeq("lastKeySeen" -> eltRef, "sortedSoFar" -> false, "haveSeenAny" -> true))
          )
        )
      }
    val isSortedComb = aggFoldSortedAccumRef1 // Do nothing, as this will never be called in a StreamAgg


    StreamAgg(joined, streamElementName, {
      AggLet(eltName, GetField(streamElementRef, "elt"),
        MakeStruct(FastIndexedSeq(
          ("min", AggFold.min(eltRef, sortFields)),
          ("max", AggFold.max(eltRef, sortFields)),
          ("samples", AggFilter(GetField(streamElementRef, "shouldKeep"), ApplyAggOp(Collect())(eltRef), false)),
          ("isSorted", GetField(AggFold(aggFoldSortedZero, isSortedSeq, isSortedComb, aggFoldSortedAccumName1, aggFoldSortedAccumName2, false), "sortedSoFar"))
        )), false)
    })
  }

  // Given an IR of type TArray(TTuple(minKey, maxKey)), determine if there's any overlap between these closed intervals.
  def tuplesAreSorted(arrayOfTuples: IR, sortFields: IndexedSeq[SortField]): IR = {
    val intervalElementType = arrayOfTuples.typ.asInstanceOf[TArray].elementType.asInstanceOf[TTuple].types(0)

    foldIR(mapIR(rangeIR(1, ArrayLen(arrayOfTuples))) { idxOfTuple =>
      ApplyComparisonOp(StructLTEQ(intervalElementType, sortFields), GetTupleElement(ArrayRef(arrayOfTuples, idxOfTuple - 1), 1), GetTupleElement(ArrayRef(arrayOfTuples, idxOfTuple), 0))
    }, True()) { case (accum, elt) =>
      ApplySpecial("land", Seq.empty[Type], FastIndexedSeq(accum, elt), TBoolean, ErrorIDs.NO_ERROR)
    }
  }
}

/**
  * a "Chunk" is a file resulting from any StreamDistribute. Chunks are internally unsorted but contain
  * data between two pivots.
  */
case class Chunk(filename: String, size: Long, byteSize: Long)
/**
  * A SegmentResult is the set of chunks from various StreamDistribute tasks working on the same segment
  * of a previous iteration.
  */
case class SegmentResult(indices: IndexedSeq[Int], interval: Interval, chunks: IndexedSeq[Chunk])
case class OutputPartition(indices: IndexedSeq[Int], interval: Interval, files: IndexedSeq[String])
case class LoopState(largeSegments: IndexedSeq[SegmentResult], smallSegments: IndexedSeq[SegmentResult], readyOutputParts: IndexedSeq[OutputPartition])

case class DistributionSortReader(key: TStruct, keyed: Boolean, spec: TypedCodecSpec, orderedOutputPartitions: IndexedSeq[OutputPartition], globals: IR, rowType: TStruct, rt: RTable) extends TableReader {
  lazy val fullType: TableType = TableType(
    rowType,
    if (keyed) key.fieldNames else FastIndexedSeq(),
    globals.typ.asInstanceOf[TStruct]
  )

  override def pathsUsed: Seq[String] = FastIndexedSeq()

  def defaultPartitioning(sm: HailStateManager): RVDPartitioner = {
    val (partitionerKey, intervals) = if (keyed) {
      (key, orderedOutputPartitions.map { segment => segment.interval })
    } else {
      (TStruct(), orderedOutputPartitions.map { _ => Interval(Row(), Row(), true, false) })
    }

    new RVDPartitioner(sm, partitionerKey, intervals)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def apply[M[_] : MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue] =
    for {
      _ <- assertA(!dropRows);
      ts <- lower(requestedType)
      tv <- TableStageIntermediate(ts).asTableValue
    } yield tv



  override def isDistinctlyKeyed: Boolean = false // FIXME: No default value

  def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    VirtualTypeWithReq.subset(requestedType.rowType, rt.rowType)
  }

  def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    VirtualTypeWithReq.subset(requestedType.globalType, rt.globalType)
  }

  override def toJValue: JValue = JString("DistributionSortReader")

  def renderShort(): String = "DistributionSortReader"

  override def defaultRender(): String = "DistributionSortReader"

  override def lowerGlobals[M[_]](requestedGlobalsType: TStruct)(implicit M: MonadLower[M]): M[IR] =
    M.reader { ctx => PruneDeadFields.upcast(ctx, globals, requestedGlobalsType) }

  override def lower[M[_]](requestedType: TableType)(implicit M: MonadLower[M]): M[TableStage] =
    M.reader { ctx =>
      val contextData = {
        var filesCount: Long = 0
        for (segment <- orderedOutputPartitions) yield {
          val filesWithNums = segment.files.zipWithIndex.map { case (file, i) =>
            Row(i + filesCount, file)
          }
          filesCount += segment.files.length
          Row(filesWithNums)
        }
      }
      val contexts =
        ToStream(
          Literal(
            TArray(TStruct("files" -> TArray(TStruct("partitionIndex" -> TInt64, "partitionPath" -> TString)))),
            contextData
          )
        )

      val partitioner = defaultPartitioning(ctx.stateManager)

      TableStage(
        PruneDeadFields.upcast(ctx, globals, requestedType.globalType),
        partitioner.coarsen(requestedType.key.length),
        TableStageDependency.none,
        contexts,
        { ctxRef =>
          val files = GetField(ctxRef, "files")
          val partitionInputStream = flatMapIR(ToStream(files)) { fileInfo =>
            ReadPartition(fileInfo, requestedType.rowType, PartitionNativeReader(spec, "__dummy_uid"))
          }
          partitionInputStream
        }
      )
    }
}
