package is.hail.expr.ir.lowering

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{agg, _}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.RVDPartitioner
import is.hail.types.RTable
import is.hail.types.physical.{PCanonicalBinary, PCanonicalTuple}
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq, Interval, partition, _}
import org.apache.spark.sql.Row


object LowerTableIRHelpers {
  def lowerTableJoin(ctx: ExecuteContext, tj: TableJoin, loweredLeft: TableStage, loweredRight: TableStage, analyses: Analyses): TableStage = {
    val TableJoin(left, right, joinType, joinKey) = tj

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

  }

  def lowerTableLeftJoinRightDistinct(ctx: ExecuteContext, tj: TableLeftJoinRightDistinct, loweredLeftUnstrict: TableStage, loweredRight: TableStage): TableStage = {
    val TableLeftJoinRightDistinct(left, right, root) = tj
    val commonKeyLength = right.typ.keyType.size
    val loweredLeft = loweredLeftUnstrict.strictify()
    val leftKeyToRightKeyMap = left.typ.keyType.fieldNames.zip(right.typ.keyType.fieldNames).toMap
    val newRightPartitioner = loweredLeft.partitioner.coarsen(commonKeyLength)
      .rename(leftKeyToRightKeyMap)
    val repartitionedRight = loweredRight.repartitionNoShuffle(newRightPartitioner, allowDuplication = true)

    loweredLeft.zipPartitions(
      repartitionedRight,
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
  }

  def lowerTableExplode(ctx: ExecuteContext, te: TableExplode, lowered: TableStage): TableStage = {
    val TableExplode(child, path) = te
    lowered.mapPartition(Some(child.typ.key.takeWhile(k => k != path(0)))) { rows =>
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
  }

  def lowerTableFilter(ctx: ExecuteContext, tf: TableFilter, lowered: TableStage): TableStage = {
    val TableFilter(_, cond) = tf
    lowered.mapPartition(None) { rows =>
      Let("global", lowered.globals,
        StreamFilter(rows, "row", cond))
    }
  }

  def lowerTableMapRows(ctx: ExecuteContext, tmr: TableMapRows, lc: TableStage, r: RequirednessAnalysis, relationalLetsAbove: Map[String, IR]): TableStage = {
    val TableMapRows(child, newRow) = tmr
    if (!ContainsScan(newRow)) {
      lc.mapPartition(Some(child.typ.key)) { rows =>
        Let("global", lc.globals,
          mapIR(rows)(row => Let("row", row, newRow)))
      }
    } else {
      val resultUID = genUID()
      val aggs = agg.Extract(newRow, resultUID, r, isScan = true)

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
        val partitionPrefixSumFiles = lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove)({ part: IR =>
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
            val loopName = genUID()

            TailLoop(loopName, IndexedSeq((aggStack.name, MakeArray(collected))),
              bindIR(ArrayRef(aggStack, (ArrayLen(aggStack) - 1))) { states =>
                bindIR(ArrayLen(states)) { statesLen =>
                  If(statesLen > branchFactor,
                    bindIR((statesLen + branchFactor - 1) floorDiv branchFactor) { nCombines =>
                      val contexts = mapIR(rangeIR(nCombines)) { outerIdxRef =>
                        sliceArrayIR(states, outerIdxRef * branchFactor, (outerIdxRef + 1) * branchFactor)
                      }
                      val cdaResult = cdaIR(contexts, MakeStruct(Seq())) { case (contexts, _) =>
                        RunAgg(
                          combineGroup(contexts),
                          WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                          aggs.states
                        )
                      }
                      Recur(loopName, IndexedSeq(invoke("extend", TArray(TArray(TString)), aggStack, MakeArray(cdaResult))), TArray(TArray(TString)))
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
            val level = Ref(genUID(), TInt32)
            val last = Ref(genUID(), TArray(TString))


            bindIR(WriteValue(initState, Str(tmpDir) + UUID4(), codecSpec)) { freshState =>
              TailLoop(downPassLoopName, IndexedSeq((level.name, ArrayLen(aggStack) - 1), (last.name, MakeArray(freshState))),
                If(level < 0,
                  last,
                  bindIR(ArrayRef(aggStack, level)) { aggsArray =>

                    val groups = mapIR(zipWithIndex(mapIR(StreamGrouped(ToStream(aggsArray), I32(branchFactor)))(x => ToArray(x)))) { eltAndIdx =>
                      MakeStruct(FastSeq(
                        ("prev", ArrayRef(last, GetField(eltAndIdx, "idx"))),
                        ("partialSums", GetField(eltAndIdx, "elt"))
                      ))
                    }

                    val results = cdaIR(groups, MakeTuple.ordered(Seq())) { case (context, _) =>
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
                        ToArray(flatten(ToStream(results)))),
                      TArray(TString))
                  }
                )
              )
            }
          }
        }
        (partitionPrefixSumFiles, {(file: IR) => ReadValue(file, codecSpec, codecSpec.encodedVirtualType) })

      } else {
        val partitionAggs = lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove)({ part: IR =>
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
        partitioner = lc.partitioner.coarsen(tmr.typ.key.length),
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
  }

  def lowerTableMapGlobals(ctx: ExecuteContext, tmg: TableMapGlobals, child: TableStage): TableStage = {
    child.mapGlobals(old => Let("global", old, tmg.newGlobals))
  }

  def lowerTableMapPartitions(ctx: ExecuteContext, tmp: TableMapPartitions, loweredChild: TableStage): TableStage = {
    val TableMapPartitions(child, globalName, partitionStreamName, body) = tmp
    loweredChild.mapPartition(Some(child.typ.key)) { part =>
      Let(globalName, loweredChild.globals, Let(partitionStreamName, part, body))
    }
  }

  def lowerTableHead(ctx: ExecuteContext, th: TableHead, loweredChild: TableStage, relationalLetsAbove: Map[String, IR]): TableStage = {
    val TableHead(child, targetNumRows) = th
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

          TailLoop(
            partitionSizeArrayFunc,
            FastIndexedSeq(howManyPartsToTryRef.name -> howManyPartsToTry),
            bindIR(loweredChild.mapContexts(_ => StreamTake(ToStream(childContexts), howManyPartsToTryRef)){ ctx: IR => ctx }
              .mapCollect(relationalLetsAbove)(streamLenOrMax)) { counts =>
              If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (ArrayLen(childContexts) <= ArrayLen(counts)),
                counts,
                Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTryRef * 4), TArray(TInt32)))
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
  }

  def lowerTableTail(ctx: ExecuteContext, tt: TableTail, loweredChild: TableStage, relationalLetsAbove: Map[String, IR]): TableStage = {
    val TableTail(child, targetNumRows) = tt
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

          TailLoop(
            partitionSizeArrayFunc,
            FastIndexedSeq(howManyPartsToTryRef.name -> howManyPartsToTry),
            bindIR(
              loweredChild
                .mapContexts(_ => StreamDrop(ToStream(childContexts), maxIR(totalNumPartitions - howManyPartsToTryRef, 0))){ ctx: IR => ctx }
                .mapCollect(relationalLetsAbove)(StreamLen)
            ) { counts =>
              If((Cast(streamSumIR(ToStream(counts)), TInt64) >= targetNumRows) || (totalNumPartitions <= ArrayLen(counts)),
                counts,
                Recur(partitionSizeArrayFunc, FastIndexedSeq(howManyPartsToTryRef * 4), TArray(TInt32)))
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
  }

  def lowerTableRange(ctx: ExecuteContext, tr: TableRange): TableStage = {
    val TableRange(n, nPartitions) = tr
    val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
    val partCounts = partition(n, nPartitionsAdj)
    val partStarts = partCounts.scanLeft(0)(_ + _)

    val contextType = TStruct("start" -> TInt32, "end" -> TInt32)

    val ranges = Array.tabulate(nPartitionsAdj) { i =>
      partStarts(i) -> partStarts(i + 1)
    }.toFastIndexedSeq

    TableStage(
      MakeStruct(FastSeq()),
      new RVDPartitioner(Array("idx"), tr.typ.rowType,
        ranges.map { case (start, end) =>
          Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
        }),
      TableStageDependency.none,
      ToStream(Literal(TArray(contextType),
        ranges.map { case (start, end) => Row(start, end) })),
      (ctxRef: Ref) => mapIR(StreamRange(GetField(ctxRef, "start"), GetField(ctxRef, "end"), I32(1), true)) { i =>
        MakeStruct(FastSeq("idx" -> i))
      })
  }

  def lowerTableAggregate(ctx: ExecuteContext, ta: TableAggregate, loweredChild: TableStage, r: RequirednessAnalysis, relationalLetsAbove: Map[String, IR]): IR = {
    val TableAggregate(_, query) = ta
    val resultUID = genUID()
    val aggs = agg.Extract(query, resultUID, r, false)

    def results: IR = ResultOp.makeTuple(aggs.aggs)

    val initState = Let("global", loweredChild.globals,
      RunAgg(
        aggs.init,
        MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }),
        aggs.states
      ))
    val initStateRef = Ref(genUID(), initState.typ)
    val lcWithInitBinding = loweredChild.copy(
      letBindings = loweredChild.letBindings ++ FastIndexedSeq((initStateRef.name, initState)),
      broadcastVals = loweredChild.broadcastVals ++ FastIndexedSeq((initStateRef.name, initStateRef)))

    val initFromSerializedStates = Begin(aggs.aggs.zipWithIndex.map { case (agg, i) =>
      InitFromSerializedValue(i, GetTupleElement(initStateRef, i), agg.state)
    })

    val useTreeAggregate = aggs.shouldTreeAggregate
    val isCommutative = aggs.isCommutative
    log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
    log.info(s"Aggregate: commutative=${ isCommutative }")

    if (useTreeAggregate) {
      val branchFactor = HailContext.get.branchingFactor
      val tmpDir = ctx.createTmpPath("aggregate_intermediates/")

      val codecSpec = TypedCodecSpec(PCanonicalTuple(true, aggs.aggs.map(_ => PCanonicalBinary(true)): _*), BufferSpec.wireSpec)
      lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove)({ part: IR =>
        Let("global", loweredChild.globals,
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
          FastIndexedSeq((currentAggStates.name -> collected)),
          If(ArrayLen(currentAggStates) <= I32(branchFactor),
            currentAggStates,
            Recur(treeAggFunction, FastIndexedSeq(CollectDistributedArray(mapIR(StreamGrouped(ToStream(currentAggStates), I32(branchFactor)))(x => ToArray(x)),
              MakeStruct(FastSeq()), distAggStatesRef.name, genUID(),
              RunAgg(
                combineGroup(distAggStatesRef),
                WriteValue(MakeTuple.ordered(aggs.aggs.zipWithIndex.map { case (sig, i) => AggStateValue(i, sig.state) }), Str(tmpDir) + UUID4(), codecSpec),
                aggs.states
              )
            )), currentAggStates.typ)))
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
      lcWithInitBinding.mapCollectWithGlobals(relationalLetsAbove)({ part: IR =>
        Let("global", loweredChild.globals,
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
  }
}
