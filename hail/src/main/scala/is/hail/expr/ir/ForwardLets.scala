package is.hail.expr.ir

import is.hail.utils._

import scala.collection.mutable

object ForwardLets {

  def memoizeRefsByNestingLevel(ir: BaseIR, binding: String, nestingLevel: Int, memo: Memo[Int]): Unit = {
      ir match {
        case Ref(`binding`, _) => memo.bind(ir, nestingLevel)
        case ArrayMap(a, name, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case ArrayFilter(a, name, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case ArrayFlatMap(a, name, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case ArrayFor(a, name, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case ArrayAgg(a, name, query) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(query, binding, nestingLevel + 1, memo)
        case ArrayFold(a, zero, accumName, valueName, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          memoizeRefsByNestingLevel(zero, binding, nestingLevel, memo)
          if (accumName != binding && valueName != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case ArrayScan(a, zero, accumName, valueName, body) =>
          memoizeRefsByNestingLevel(a, binding, nestingLevel, memo)
          memoizeRefsByNestingLevel(zero, binding, nestingLevel, memo)
          if (accumName != binding && valueName != binding)
            memoizeRefsByNestingLevel(body, binding, nestingLevel + 1, memo)
        case AggExplode(array, name, aggBody) =>
          memoizeRefsByNestingLevel(array, binding, nestingLevel, memo)
          if (name != binding)
            memoizeRefsByNestingLevel(aggBody, binding, nestingLevel + 1, memo)
        case _: AggFilter | _: AggGroupBy | _: ApplyAggOp | _: ArrayAgg =>
          ir.children.foreach(memoizeRefsByNestingLevel(_, binding, nestingLevel + 1, memo))
        case _ =>
          ir.children.foreach(memoizeRefsByNestingLevel(_, binding, nestingLevel, memo))
      }
  }

  def apply(ir: BaseIR, copy: Boolean = true): BaseIR = {
    RewriteBottomUp(if (copy) ir.deepCopy() else ir, {
      case let@Let(binding, value, letBody) =>
        val memo = Memo.empty[Int]
        memoizeRefsByNestingLevel(letBody, binding, 0, memo)

        if (memo.m.isEmpty)
          Some(letBody)
        else if (memo.m.size == 1 && memo.m.forall(_._2 == 0))
          Some(Subst(letBody, Env(binding -> value)))
        else
          None
      case _ => None
    })
  }
}
