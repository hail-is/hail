package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.types.RTable
import is.hail.types.virtual.TStruct

import scala.collection.compat._

object LowerTableIRHelpers {

  def lowerTableJoin(
    ctx: ExecuteContext,
    analyses: LoweringAnalyses,
    tj: TableJoin,
    loweredLeft: TableStage,
    loweredRight: TableStage,
  ): TableStage = {
    val TableJoin(left, right, joinType, joinKey) = tj
    val lKeyFields = left.typ.key.take(joinKey)
    val lValueFields = left.typ.rowType.fieldNames.filter(f => !lKeyFields.contains(f))
    val rKeyFields = right.typ.key.take(joinKey)
    val rValueFields = right.typ.rowType.fieldNames.filter(f => !rKeyFields.contains(f))
    val lReq = analyses.requirednessAnalysis.lookup(left).asInstanceOf[RTable]
    val rReq = analyses.requirednessAnalysis.lookup(right).asInstanceOf[RTable]
    val rightKeyIsDistinct = analyses.distinctKeyedAnalysis.contains(right)

    val joinedStage = loweredLeft.orderedJoin(
      ctx,
      loweredRight,
      joinKey,
      joinType,
      (l, r) => l.insert(r.typ.asInstanceOf[TStruct].fieldNames.map(f => f -> r.get(f))),
      (lElem, rElem) =>
        MakeStruct(
          lKeyFields.lazyZip(rKeyFields).map { (lKey, rKey) =>
            val die =
              if (joinType == "outer" && lReq.field(lKey).required && rReq.field(rKey).required)
                FastSeq(
                  Die("TableJoin expected non-missing key", left.typ.rowType.fieldType(lKey), -1)
                )
              else
                FastSeq()

            lKey -> Coalesce(FastSeq(lElem.get(lKey), rElem.get(rKey)) ++ die)
          }
            ++ lValueFields.map(f => f -> GetField(lElem, f))
            ++ rValueFields.map(f => f -> GetField(rElem, f))
        ),
      rightKeyIsDistinct,
    )

    assert(joinedStage.rowType == tj.typ.rowType)
    joinedStage
  }
}
