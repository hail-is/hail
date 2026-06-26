package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.defs._
import is.hail.types.{RTable, VirtualTypeWithReq}
import is.hail.types.virtual.TStruct
import is.hail.utils.TimedBlock

object LowerTableKeyByAndAggregate {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter {
      val rewritten =
        RewriteBottomUp(
          ir,
          {
            case t @ TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
              val newKeyType = newKey.typ.asInstanceOf[TStruct]

              val aggs = Extract(ctx, expr, Requiredness(t, ctx)).independent
              val aggSigs = aggs.sigs

              val ts =
                child
                  .keyBy(FastSeq())
                  .mapGlobals { global =>
                    makestruct(
                      "__old_global" -> global,
                      "__init_state" -> RunAgg(aggs.init, aggSigs.valuesOp, aggSigs.states),
                    )
                  }.mapPartitions() { (global, partStream) =>
                    Let(
                      FastSeq(TableIR.globalName -> global.get("__old_global")),
                      StreamBufferedAggregate(
                        partStream,
                        global.get("__init_state").bind(aggSigs.initFromSerializedValueOp),
                        newKey,
                        aggs.seqPerElt,
                        TableIR.rowName,
                        aggSigs.sigs,
                        bufferSize,
                      ),
                    )
                  }

              val rt = Requiredness(ts, ctx).lookup(ts).asInstanceOf[RTable]

              val takeVirtualSig =
                TakeStateSig(VirtualTypeWithReq(
                  newKeyType,
                  rt.rowType.select(newKeyType.fieldNames),
                ))

              val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)

              Some {
                ts
                  .keyBy(t.key, nPartitions = nPartitions)
                  .mapPartitions(newKeyType.size, newKeyType.size - 1) { (global, partStream) =>
                    Let(
                      FastSeq(TableIR.globalName -> global.get("__old_global")),
                      partStream.groupedByKey(t.key, missingEqual = true).streamMap { group =>
                        RunAgg(
                          Begin(FastSeq(
                            global.get("__init_state").bind(aggSigs.initFromSerializedValueOp),
                            InitOp(aggSigs.nAggs, FastSeq(I32(1)), takeAggSig),
                            group.streamFor { elem =>
                              Begin(FastSeq(
                                SeqOp(
                                  aggSigs.nAggs,
                                  FastSeq(elem.select(newKeyType.fieldNames)),
                                  takeAggSig,
                                ),
                                elem.get("agg").bind(aggSigs.combOpValues),
                              ))
                            },
                          )),
                          bindIRs(aggs.result, ResultOp(aggSigs.nAggs, takeAggSig)) {
                            case Seq(postAgg, resultFromTake) =>
                              MakeStruct(
                                t.key.map(k => k -> resultFromTake.at(0).get(k)) ++
                                  expr.typ.asInstanceOf[TStruct].fieldNames.map { f =>
                                    f -> postAgg.get(f)
                                  }
                              )
                          },
                          aggSigs.states ++ FastSeq(takeVirtualSig),
                        )
                      },
                    )
                  }
                  .mapGlobals(_.get("__old_global"))
              }
            case _ => None
          },
        )

      if (rewritten ne ir) NormalizeNames()(ctx, rewritten)
      else ir
    }
}
