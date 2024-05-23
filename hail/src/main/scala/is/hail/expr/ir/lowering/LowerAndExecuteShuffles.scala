package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{Requiredness, _}
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object LowerAndExecuteShuffles {

  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR = {
    RewriteBottomUp(
      ir,
      {
        case t @ TableKeyBy(child, key, _) if !t.definitelyDoesNotShuffle =>
          val r = Requiredness(child, ctx)
          val reader = ctx.backend.lowerDistributedSort(
            ctx,
            child,
            key.map(k => SortField(k, Ascending)),
            r.lookup(child).asInstanceOf[RTable],
          )
          Some(TableRead(t.typ, false, reader))

        case t @ TableOrderBy(child, sortFields) if !t.definitelyDoesNotShuffle =>
          val r = Requiredness(child, ctx)
          val reader = ctx.backend.lowerDistributedSort(
            ctx,
            child,
            sortFields,
            r.lookup(child).asInstanceOf[RTable],
          )
          Some(TableRead(t.typ, false, reader))

        case t @ TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
          val newKeyType = newKey.typ.asInstanceOf[TStruct]
          val resultUID = genUID()

          val req = Requiredness(t, ctx)

          val aggs = Extract(expr, resultUID, req)
          val postAggIR = aggs.postAggIR
          val init = aggs.init
          val seq = aggs.seqPerElt
          val aggSigs = aggs.aggs

          val streamName = genUID()
          val streamTyp = TStream(child.typ.rowType)
          var ts = child

          val origGlobalTyp = ts.typ.globalType
          ts = TableKeyBy(child, IndexedSeq())
          ts = TableMapGlobals(
            ts,
            MakeStruct(FastSeq(
              ("oldGlobals", Ref("global", origGlobalTyp)),
              (
                "__initState",
                RunAgg(
                  init,
                  MakeTuple.ordered(aggSigs.indices.map { aIdx =>
                    AggStateValue(aIdx, aggSigs(aIdx).state)
                  }),
                  aggSigs.map(_.state),
                ),
              ),
            )),
          )

          val insGlobName = genUID()
          def insGlob = Ref(insGlobName, ts.typ.globalType)
          val partiallyAggregated =
            TableMapPartitions(
              ts,
              insGlob.name,
              streamName,
              Let(
                FastSeq("global" -> GetField(insGlob, "oldGlobals")),
                StreamBufferedAggregate(
                  Ref(streamName, streamTyp),
                  bindIR(GetField(insGlob, "__initState")) { states =>
                    Begin(aggSigs.indices.map { aIdx =>
                      InitFromSerializedValue(
                        aIdx,
                        GetTupleElement(states, aIdx),
                        aggSigs(aIdx).state,
                      )
                    })
                  },
                  newKey,
                  seq,
                  "row",
                  aggSigs,
                  bufferSize,
                ),
              ),
              0,
              0,
            ).noSharing(ctx)

          val analyses = LoweringAnalyses(partiallyAggregated, ctx)
          val preShuffleStage = ctx.backend.tableToTableStage(ctx, partiallyAggregated, analyses)
          // annoying but no better alternative right now
          val rt = analyses.requirednessAnalysis.lookup(partiallyAggregated).asInstanceOf[RTable]
          val partiallyAggregatedReader = ctx.backend.lowerDistributedSort(
            ctx,
            preShuffleStage,
            newKeyType.fieldNames.map(k => SortField(k, Ascending)),
            rt,
            nPartitions,
          )

          val takeVirtualSig =
            TakeStateSig(VirtualTypeWithReq(newKeyType, rt.rowType.select(newKeyType.fieldNames)))
          val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)
          val aggStateSigsPlusTake = aggs.states ++ Array(takeVirtualSig)

          val postAggUID = genUID()
          val resultFromTakeUID = genUID()
          val result = ResultOp(aggs.aggs.length, takeAggSig)

          val shuffleRead =
            TableRead(partiallyAggregatedReader.fullType, false, partiallyAggregatedReader)

          val partStream = Ref(genUID(), TStream(shuffleRead.typ.rowType))
          val tmp = TableMapPartitions(
            shuffleRead,
            insGlob.name,
            partStream.name,
            Let(
              FastSeq("global" -> GetField(insGlob, "oldGlobals")),
              mapIR(StreamGroupByKey(
                partStream,
                newKeyType.fieldNames.toIndexedSeq,
                missingEqual = true,
              )) { groupRef =>
                RunAgg(
                  Begin(FastSeq(
                    bindIR(GetField(insGlob, "__initState")) { states =>
                      Begin(aggSigs.indices.map { aIdx =>
                        InitFromSerializedValue(
                          aIdx,
                          GetTupleElement(states, aIdx),
                          aggSigs(aIdx).state,
                        )
                      })
                    },
                    InitOp(
                      aggSigs.length,
                      IndexedSeq(I32(1)),
                      PhysicalAggSig(Take(), takeVirtualSig),
                    ),
                    forIR(groupRef) { elem =>
                      Begin(FastSeq(
                        SeqOp(
                          aggSigs.length,
                          IndexedSeq(SelectFields(elem, newKeyType.fieldNames)),
                          PhysicalAggSig(Take(), takeVirtualSig),
                        ),
                        Begin((0 until aggSigs.length).map { aIdx =>
                          CombOpValue(
                            aIdx,
                            GetTupleElement(GetField(elem, "agg"), aIdx),
                            aggSigs(aIdx),
                          )
                        }),
                      ))
                    },
                  )),
                  Let(
                    FastSeq(
                      resultUID -> ResultOp.makeTuple(aggs.aggs),
                      postAggUID -> postAggIR,
                      resultFromTakeUID -> result,
                    ), {
                      val keyIRs: IndexedSeq[(String, IR)] =
                        newKeyType.fieldNames.map(keyName =>
                          keyName -> GetField(
                            ArrayRef(Ref(resultFromTakeUID, result.typ), 0),
                            keyName,
                          )
                        )

                      MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                        (f, GetField(Ref(postAggUID, postAggIR.typ), f))
                      })
                    },
                  ),
                  aggStateSigsPlusTake,
                )
              },
            ),
            newKeyType.size,
            newKeyType.size - 1,
          )
          Some(TableMapGlobals(tmp, GetField(Ref("global", insGlob.typ), "oldGlobals")))
        case _ => None
      },
    )
  }
}
