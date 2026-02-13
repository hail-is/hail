package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{Requiredness, _}
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.defs._
import is.hail.types._
import is.hail.types.virtual._

object LowerAndExecuteShuffles {

  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR =
    ctx.time {
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

            val aggs = Extract(ctx, expr, Requiredness(t, ctx)).independent
            val aggSigs = aggs.sigs

            var ts = child

            val origGlobalTyp = ts.typ.globalType
            ts = TableKeyBy(child, IndexedSeq())
            ts = TableMapGlobals(
              ts,
              MakeStruct(FastSeq(
                ("oldGlobals", Ref(TableIR.globalName, origGlobalTyp)),
                (
                  "__initState",
                  RunAgg(aggs.init, aggSigs.valuesOp, aggSigs.states),
                ),
              )),
            )

            val partiallyAggregated =
              mapPartitions(ts) { (insGlob, partStream) =>
                Let(
                  FastSeq(TableIR.globalName -> GetField(insGlob, "oldGlobals")),
                  StreamBufferedAggregate(
                    partStream,
                    bindIR(GetField(insGlob, "__initState"))(aggSigs.initFromSerializedValueOp),
                    newKey,
                    aggs.seqPerElt,
                    TableIR.rowName,
                    aggSigs.sigs,
                    bufferSize,
                  ),
                )
              }.noSharing(ctx)

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
            val aggStateSigsPlusTake = aggSigs.states ++ Array(takeVirtualSig)

            val result = ResultOp(aggSigs.nAggs, takeAggSig)

            val shuffleRead =
              TableRead(partiallyAggregatedReader.fullType, false, partiallyAggregatedReader)

            val tmp = mapPartitions(
              shuffleRead,
              newKeyType.size,
              newKeyType.size - 1,
            ) { (insGlob, shuffledPartStream) =>
              Let(
                FastSeq(TableIR.globalName -> GetField(insGlob, "oldGlobals")),
                mapIR(StreamGroupByKey(
                  shuffledPartStream,
                  newKeyType.fieldNames.toIndexedSeq,
                  missingEqual = true,
                )) { groupRef =>
                  RunAgg(
                    Begin(FastSeq(
                      bindIR(GetField(insGlob, "__initState"))(aggSigs.initFromSerializedValueOp),
                      InitOp(
                        aggSigs.nAggs,
                        IndexedSeq(I32(1)),
                        PhysicalAggSig(Take(), takeVirtualSig),
                      ),
                      forIR(groupRef) { elem =>
                        Begin(FastSeq(
                          SeqOp(
                            aggSigs.nAggs,
                            IndexedSeq(SelectFields(elem, newKeyType.fieldNames)),
                            PhysicalAggSig(Take(), takeVirtualSig),
                          ),
                          bindIR(GetField(elem, "agg"))(aggSigs.combOpValues),
                        ))
                      },
                    )),
                    bindIRs(aggs.result, result) { case Seq(postAgg, resultFromTake) =>
                      val keyIRs: IndexedSeq[(String, IR)] =
                        newKeyType.fieldNames.map(keyName =>
                          keyName -> GetField(ArrayRef(resultFromTake, 0), keyName)
                        )

                      MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                        (f, GetField(postAgg, f))
                      })
                    },
                    aggStateSigsPlusTake,
                  )
                },
              )
            }
            Some(TableMapGlobals(
              tmp,
              GetField(Ref(TableIR.globalName, tmp.typ.globalType), "oldGlobals"),
            ))
          case _ => None
        },
      )
    }
}
