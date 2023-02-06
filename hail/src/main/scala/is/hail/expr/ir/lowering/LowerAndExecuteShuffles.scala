package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{LoweringAnalyses, BaseIR, IR, PruneDeadFields, RewriteBottomUp, TableExecuteIntermediate, TableKeyBy, TableKeyByAndAggregate, TableOrderBy, TableReader, TableValue}
import is.hail.types.virtual._
import is.hail.types._
import is.hail.expr.ir._
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.io.TypedCodecSpec
import is.hail.rvd.RVDPartitioner
import is.hail.utils.{FastIndexedSeq, Interval}
import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.JsonAST.JString


object LowerAndExecuteShuffles {

  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR = {

    RewriteBottomUp(ir, {
      case t@TableKeyBy(child, key, isSorted) if !t.definitelyDoesNotShuffle =>
        val r = Requiredness(child, ctx)
        val reader = ctx.backend.lowerDistributedSort(ctx, child, key.map(k => SortField(k, Ascending)), r.lookup(child).asInstanceOf[RTable])
        Some(TableRead(t.typ, false, reader))

      case t@TableOrderBy(child, sortFields) if !t.definitelyDoesNotShuffle =>
        val r = Requiredness(child, ctx)
        val reader = ctx.backend.lowerDistributedSort(ctx, child, sortFields, r.lookup(child).asInstanceOf[RTable])
        Some(TableRead(t.typ, false, reader))

      case t@TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val newKeyType = newKey.typ.asInstanceOf[TStruct]
        val resultUID = genUID()

        val req = Requiredness(child, ctx)

        val aggs = Extract(expr, resultUID, req)
        val postAggIR = aggs.postAggIR
        val init = aggs.init
        val seq = aggs.seqPerElt
        val aggSigs = aggs.aggs

        val globalName = genUID()
        val streamName = genUID()
        val streamTyp = TStream(child.typ.rowType)
        val partiallyAggregated = TableMapPartitions(child, "global", streamName,
          StreamBufferedAggregate(Ref(streamName, streamTyp), init, newKey, seq, "row", aggSigs, bufferSize),
          0, child.typ.key.length)

        // annoying but no better alternative right now
        val req2 = Requiredness(partiallyAggregated, ctx)
        val rt = req2.lookup(partiallyAggregated).asInstanceOf[RTable]
        val partiallyAggregatedReader = ctx.backend.lowerDistributedSort(ctx,
          partiallyAggregated,
          newKeyType.fieldNames.map(k => SortField(k, Ascending)),
          rt)

        val takeVirtualSig = TakeStateSig(VirtualTypeWithReq(newKeyType, rt.rowType.select(newKeyType.fieldNames)))
        val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)
        val aggStateSigsPlusTake = aggs.states ++ Array(takeVirtualSig)

        val postAggUID = genUID()
        val resultFromTakeUID = genUID()
        val result = ResultOp(aggs.aggs.length, takeAggSig)

        val aggSigsPlusTake = aggSigs ++ IndexedSeq(takeAggSig)

        val shuffleRead = TableRead(partiallyAggregatedReader.fullType, false, partiallyAggregatedReader)

        val partStream = Ref(genUID(), TStream(shuffleRead.typ.rowType))
        val tmp = TableMapPartitions(shuffleRead, "global", partStream.name,
          mapIR(StreamGroupByKey(partStream, newKeyType.fieldNames.toIndexedSeq, missingEqual = true)) { groupRef =>
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
                  }))
              },

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
          }, newKeyType.size, newKeyType.size - 1)
        Some(tmp )
      case _ => None
    })
  }
}
