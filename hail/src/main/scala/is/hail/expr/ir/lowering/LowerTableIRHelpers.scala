package is.hail.expr.ir.lowering

import cats.implicits.toFlatMapOps
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.utils.assertA
import is.hail.types.RTable
import is.hail.types.virtual.TStruct
import is.hail.utils.FastSeq

import scala.language.higherKinds

object LowerTableIRHelpers {

  def lowerTableJoin[M[_]: MonadLower](analyses: LoweringAnalyses,
                                       tj: TableJoin,
                                       loweredLeft: TableStage,
                                       loweredRight: TableStage
                                      ): M[TableStage] = {
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

    joinedStage.flatTap(stage => assertA(stage.rowType == tj.typ.rowType))
  }
}
