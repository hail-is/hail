package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.defs.{
  ApplyIR, ArrayRef, Begin, GetField, I32, InitOp, Let, MakeStruct, Ref, ResultOp, RunAgg,
  RunAggScan, SelectFields, SeqOp, StreamAgg, StreamAggScan, StreamBufferedAggregate, StreamFor,
  StreamGroupByKey,
}
import is.hail.expr.ir.lowering.invariant._
import is.hail.types.{RTable, VirtualTypeWithReq}
import is.hail.types.virtual.TStruct

final class IrMetadata() {
  private[this] var hashCounter: Int = 0
  private[this] var markCounter: Int = 0

  var semhash: Option[SemanticHash.Type] = None

  def nextHash: Option[SemanticHash.Type] = {
    hashCounter += 1
    semhash.map(SemanticHash.extend(_, SemanticHash.Bytes.fromInt(hashCounter)))
  }

  def nextFlag: Int = {
    markCounter += 1
    markCounter
  }
}

abstract class LoweringPass(implicit E: sourcecode.Enclosing) {
  def context: String
  def before: Invariant
  def after: Invariant

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      before.verify(ctx, ir)
      val result = transform(ctx, ir)
      after.verify(ctx, result)
      result
    }

  protected def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR
}

case class OptimizePass(_context: String) extends LoweringPass {
  override val context = s"Optimize: ${_context}"
  override def before: Invariant = AnyIR
  override def after: Invariant = AnyIR

  override def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    if (ctx.flags.isDefined(Optimize.Flags.Optimize)) super.apply(ctx, ir)
    else ir

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = Optimize(ctx, ir)
}

case object LowerMatrixToTablePass extends LoweringPass {
  override val context: String = "LowerMatrixToTable"
  override def before: Invariant = AnyIR
  override def after: Invariant = before and NoMatrixIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case x: IR => LowerMatrixIR(ctx, x)
    case x: TableIR => LowerMatrixIR(ctx, x)
    case x: MatrixIR => LowerMatrixIR(ctx, x)
    case x: BlockMatrixIR => LowerMatrixIR(ctx, x)
  }
}

case object LiftRelationalValuesToRelationalLets extends LoweringPass {
  override val context: String = "LiftRelationalValuesToRelationalLets"
  override def before: Invariant = AnyIR and NoMatrixIR
  override def after: Invariant = before

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = LiftRelationalValues(ir)
}

case object LegacyInterpretNonCompilablePass extends LoweringPass {
  override val context: String = "InterpretNonCompilable"
  override def before: Invariant = AnyIR and NoMatrixIR
  override def after: Invariant = before and NoRelationalLets and CompilableValueIRs

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case object LowerOrInterpretNonCompilablePass extends LoweringPass {
  override val context: String = "LowerOrInterpretNonCompilable"
  override def before: Invariant = AnyIR and NoMatrixIR
  override def after: Invariant = AnyIR and CompilableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerOrInterpretNonCompilable(ctx, ir)
}

case class LowerToDistributedArrayPass(t: DArrayLowering.Type) extends LoweringPass {
  override val context: String = "LowerToDistributedArray"
  override def before: Invariant = AnyIR and NoMatrixIR
  override def after: Invariant = AnyIR and CompilableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerToCDA(ir.asInstanceOf[IR], t, ctx)
}

case object InlineApplyIR extends LoweringPass {
  override val context: String = "InlineApplyIR"
  override def before: Invariant = AnyIR and CompilableIR
  override def after: Invariant = before and NoApplyIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      RewriteBottomUp(
        ir,
        {
          case x: ApplyIR => Some(x.explicitNode)
          case _ => None
        },
      )
    }
}

case object LowerArrayAggsToRunAggsPass extends LoweringPass {
  override val context: String = "LowerArrayAggsToRunAggs"
  override def before: Invariant = CompilableIR and NoApplyIR
  override def after: Invariant = EmittableIR

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      val x = ir.noSharing(ctx)
      val r = Requiredness(x, ctx)
      RewriteBottomUp(
        x,
        {
          case x @ StreamAgg(a, name, query) =>
            val aggs = Extract(ctx, query, r)

            val newNode = Let(
              aggs.initBindings,
              RunAgg(
                Begin(FastSeq(
                  aggs.init,
                  StreamFor(a, name, aggs.seqPerElt),
                )),
                aggs.result,
                aggs.sigs.states,
              ),
            )

            if (newNode.typ != x.typ)
              throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
            Some(newNode.noSharing(ctx))
          case x @ StreamAggScan(a, name, query) =>
            val aggs = Extract(ctx, query, r, isScan = true)

            val newNode = Let(
              aggs.initBindings,
              RunAggScan(
                a,
                name,
                aggs.init,
                aggs.seqPerElt,
                aggs.result,
                aggs.sigs.states,
              ),
            )
            if (newNode.typ != x.typ)
              throw new RuntimeException(s"types differ:\n  new: ${newNode.typ}\n  old: ${x.typ}")
            Some(newNode.noSharing(ctx))
          case _ => None
        },
      )
    }
}

case class EvalRelationalLetsPass(passesBelow: LoweringPipeline) extends LoweringPass {
  override val context: String = "EvalRelationalLets"
  override def before: Invariant = NoMatrixIR
  override def after: Invariant = before and NoRelationalLets

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    EvalRelationalLets(ir, ctx, passesBelow)
}

case object LowerTableKeyByAndAggregatePass extends LoweringPass {
  override val context: String = "LowerTableKeyByAndAggregate"
  override def before: Invariant = NoRelationalLets and NoMatrixIR
  override def after: Invariant = before and NoTableKeyByAndAggregate

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR = ctx.time {
    RewriteBottomUp(
      ir,
      {
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
          val rt = analyses.requirednessAnalysis.lookup(partiallyAggregated).asInstanceOf[RTable]

          val takeVirtualSig =
            TakeStateSig(VirtualTypeWithReq(newKeyType, rt.rowType.select(newKeyType.fieldNames)))
          val takeAggSig = PhysicalAggSig(Take(), takeVirtualSig)
          val aggStateSigsPlusTake = aggSigs.states ++ Array(takeVirtualSig)

          val result = ResultOp(aggSigs.nAggs, takeAggSig)

          val shuffled = TableKeyBy(
            partiallyAggregated,
            newKey.typ.asInstanceOf[TStruct].fieldNames,
            nPartitions = nPartitions,
          )

          val tmp =
            mapPartitions(shuffled, newKeyType.size, newKeyType.size - 1) {
              (insGlob, shuffledPartStream) =>
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

case object LowerAndExecuteShufflesPass extends LoweringPass {
  override val context: String = "LowerAndExecuteShuffles"
  override def before: Invariant = NoRelationalLets and NoMatrixIR and NoTableKeyByAndAggregate
  override def after: Invariant = before and LoweredShuffles

  override def transform(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    LowerAndExecuteShuffles(ir, ctx)
}
