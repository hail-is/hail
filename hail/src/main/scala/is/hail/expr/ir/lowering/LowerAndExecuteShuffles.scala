package is.hail.expr.ir.lowering

import cats.syntax.all._
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.{Requiredness, _}
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq

import scala.language.higherKinds


object LowerAndExecuteShuffles {

  def apply[M[_]](ir: BaseIR, passesBelow: LoweringPipeline)(implicit M: MonadLower[M]): M[BaseIR] =
    RewriteBottomUp(ir, {
      case t@TableKeyBy(child, key, _) if !t.definitelyDoesNotShuffle =>
        for {
          r <- Requiredness(child)
          backend <- M.reader(_.backend)
          reader <- backend.lowerDistributedSort(child, key.map(k => SortField(k, Ascending)), r.lookup(child).asInstanceOf[RTable])
        } yield Some(TableRead(t.typ, dropRows = false, reader))

      case t@TableOrderBy(child, sortFields) if !t.definitelyDoesNotShuffle =>
        for {
          r <- Requiredness(child)
          backend <- M.reader(_.backend)
          reader <- backend.lowerDistributedSort(child, sortFields, r.lookup(child).asInstanceOf[RTable])
        } yield Some(TableRead(t.typ, false, reader))

      case t@TableKeyByAndAggregate(child, expr, newKey, _, bufferSize) =>
        for {
          req <- Requiredness(t)
          resultUID = genUID()

          aggs = Extract(expr, resultUID, req)
          postAggIR = aggs.postAggIR
          init = aggs.init
          seq = aggs.seqPerElt
          aggSigs = aggs.aggs

          streamName = genUID()
          streamTyp = TStream(child.typ.rowType)

          ts = TableMapGlobals(
            TableKeyBy(child, IndexedSeq()),
            MakeStruct(FastIndexedSeq(
              "oldGlobals" -> Ref("global", child.typ.globalType),
              "__initState" -> RunAgg(
                init,
                MakeTuple.ordered(aggSigs.indices.map { aIdx => AggStateValue(aIdx, aggSigs(aIdx).state) }),
                aggSigs.map(_.state)
              )
            ))
          )

          insGlobName = genUID()
          insGlob = Ref(insGlobName, ts.typ.globalType)
          partiallyAggregated =
            TableMapPartitions(ts, insGlob.name, streamName,
              Let("global", GetField(insGlob, "oldGlobals"),
                StreamBufferedAggregate(Ref(streamName, streamTyp), bindIR(GetField(insGlob, "__initState")) { states =>
                  Begin(aggSigs.indices.map { aIdx => InitFromSerializedValue(aIdx, GetTupleElement(states, aIdx), aggSigs(aIdx).state) })
                }, newKey, seq, "row", aggSigs, bufferSize)),
              0,
              0
            ).noSharing

          analyses <- LoweringAnalyses(partiallyAggregated)
          backend <- M.reader(_.backend)

          preShuffleStage <- backend.tableToTableStage(partiallyAggregated, analyses)

          rt = analyses.requirednessAnalysis.lookup(partiallyAggregated).asInstanceOf[RTable]
          newKeyType = newKey.typ.asInstanceOf[TStruct]

          partiallyAggregatedReader <- backend.lowerDistributedSort(
            preShuffleStage,
            newKeyType.fieldNames.map(k => SortField(k, Ascending)),
            rt
          )
        } yield {
          val takeVirtualSig = TakeStateSig(VirtualTypeWithReq(newKeyType, rt.rowType.select(newKeyType.fieldNames)))
          val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)
          val aggStateSigsPlusTake = aggs.states ++ Array(takeVirtualSig)

          val postAggUID = genUID()
          val resultFromTakeUID = genUID()
          val result = ResultOp(aggs.aggs.length, takeAggSig)

          val shuffleRead = TableRead(partiallyAggregatedReader.fullType, false, partiallyAggregatedReader)

          val partStream = Ref(genUID(), TStream(shuffleRead.typ.rowType))
          val tmp = TableMapPartitions(shuffleRead, insGlob.name, partStream.name,
            Let("global", GetField(insGlob, "oldGlobals"),
              mapIR(StreamGroupByKey(partStream, newKeyType.fieldNames.toIndexedSeq, missingEqual = true)) { groupRef =>
                RunAgg(Begin(FastIndexedSeq(
                  bindIR(GetField(insGlob, "__initState")) { states =>
                    Begin(aggSigs.indices.map { aIdx => InitFromSerializedValue(aIdx, GetTupleElement(states, aIdx), aggSigs(aIdx).state) })
                  },
                  InitOp(aggSigs.length, IndexedSeq(I32(1)), PhysicalAggSig(Take(), takeVirtualSig)),
                  forIR(groupRef) { elem =>
                    Begin(FastIndexedSeq(
                      SeqOp(aggSigs.length, IndexedSeq(SelectFields(elem, newKeyType.fieldNames)), PhysicalAggSig(Take(), takeVirtualSig)),
                      Begin(aggSigs.indices.map { aIdx =>
                        CombOpValue(aIdx, GetTupleElement(GetField(elem, "agg"), aIdx), aggSigs(aIdx))
                      })))
                  })),
                  Let(
                    resultUID,
                    ResultOp.makeTuple(aggs.aggs),
                    Let(postAggUID, postAggIR,
                      Let(resultFromTakeUID,
                        result, {
                          val keyIRs: IndexedSeq[(String, IR)] = newKeyType.fieldNames.map(keyName => keyName -> GetField(ArrayRef(Ref(resultFromTakeUID, result.typ), 0), keyName))
                          MakeStruct(keyIRs ++ expr.typ.asInstanceOf[TStruct].fieldNames.map { f => (f, GetField(Ref(postAggUID, postAggIR.typ), f))
                          })
                        }
                      )
                    )
                  ),
                  aggStateSigsPlusTake)
              }), newKeyType.size, newKeyType.size - 1)
          Some(TableMapGlobals(tmp, GetField(Ref("global", insGlob.typ), "oldGlobals")))
        }

      case _ => M.pure(None)
    })
}
