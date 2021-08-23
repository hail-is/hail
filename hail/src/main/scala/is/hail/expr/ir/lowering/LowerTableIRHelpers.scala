package is.hail.expr.ir.lowering

import is.hail.expr.ir.{Coalesce, Die, ExecuteContext, GetField, IR, InsertFields, Let, MakeStruct, Ref, RequirednessAnalysis, SelectFields, StreamJoinRightDistinct, TableExplode, TableIR, TableJoin, TableLeftJoinRightDistinct, ToStream, flatMapIR, genUID, mapIR}
import is.hail.types.RTable
import is.hail.types.virtual.TStruct
import is.hail.utils.FastSeq

object LowerTableIRHelpers {

  def lowerTableJoin(ctx: ExecuteContext, tj: TableJoin, loweredLeft: TableStage, loweredRight: TableStage, r: RequirednessAnalysis): TableStage = {
    val TableJoin(left, right, joinType, joinKey) = tj

    val lKeyFields = left.typ.key.take(joinKey)
    val lValueFields = left.typ.rowType.fieldNames.filter(f => !lKeyFields.contains(f))
    val rKeyFields = right.typ.key.take(joinKey)
    val rValueFields = right.typ.rowType.fieldNames.filter(f => !rKeyFields.contains(f))
    val lReq = r.lookup(left).asInstanceOf[RTable]
    val rReq = r.lookup(right).asInstanceOf[RTable]

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
      })

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
    val repartitionedRight = loweredRight.repartitionNoShuffle(newRightPartitioner)

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
          mapIR(ToStream(refs.last)) { elt =>
            path.zip(refs.init).foldRight[IR](elt) { case ((p, ref), inserted) =>
              InsertFields(ref, FastSeq(p -> inserted))
            }
          }) { case ((ref, root), accum) => Let(ref.name, root, accum) }
      }
    }
  }
}
