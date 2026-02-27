package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{Requiredness, _}
import is.hail.expr.ir.agg.{Extract, PhysicalAggSig, TakeStateSig}
import is.hail.expr.ir.defs._
import is.hail.types._
import is.hail.types.virtual._

object LowerAndExecuteShuffles {

  def apply(ir: BaseIR, ctx: ExecuteContext): BaseIR =
    ctx.time {
      RewriteBottomUp(
        ir,
        {
          case t @ TableKeyBy(child, key, _, nPartitions) if !t.definitelyDoesNotShuffle =>
            val r = Requiredness(child, ctx)
            val reader = ctx.backend.lowerDistributedSort(
              ctx,
              child,
              key.map(k => SortField(k, Ascending)),
              r.lookup(child).asInstanceOf[RTable],
              nPartitions,
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

          case _ => None
        },
      )
    }
}
