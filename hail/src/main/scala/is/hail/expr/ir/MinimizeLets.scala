package is.hail.expr.ir

import is.hail.utils.HailException

import scala.collection.mutable

object MinimizeLets {

  def memoize(ir: BaseIR, binding: String, memo: Memo[Boolean]): Boolean = {
    val consumesBinding: Boolean = ir match {
      case Ref(`binding`, _) => true
      case Let(binding2, value, body) =>
        val vm = memoize(value, binding, memo)
        val bm = if (binding2 == binding) false else memoize(body, binding, memo)
        vm || bm
      case ArrayMap(a, name, body) =>
        val am = memoize(a, binding, memo)
        val bm = if (name == binding) false else memoize(body, binding, memo)
        am || bm
      case ArrayFilter(a, name, body) =>
        val am = memoize(a, binding, memo)
        val bm = if (name == binding) false else memoize(body, binding, memo)
        am || bm
      case ArrayFlatMap(a, name, body) =>
        val am = memoize(a, binding, memo)
        val bm = if (name == binding) false else memoize(body, binding, memo)
        am || bm
      case ArrayFor(a, name, body) =>
        val am = memoize(a, binding, memo)
        val bm = if (name == binding) false else memoize(body, binding, memo)
        am || bm
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val am = memoize(a, binding, memo)
        val zerom = memoize(zero, binding, memo)
        val bodym = if (accumName == binding || valueName == binding) false else memoize(body, binding, memo)
        am || zerom || bodym
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val am = memoize(a, binding, memo)
        val zerom = memoize(zero, binding, memo)
        val bodym = if (accumName == binding || valueName == binding) false else memoize(body, binding, memo)
        am || zerom || bodym
      case x =>
        val childrenContains = x.children.filter(_.isInstanceOf[IR]).map(memoize(_, binding, memo))
        childrenContains.count(x => x) > 0
    }

    memo.bind(ir, consumesBinding)
    consumesBinding
  }


  def apply(ir: BaseIR, copy: Boolean = true): BaseIR = {
    var i = 0
    val m = mutable.HashSet.empty[(String, String)]
    RewriteTopDown.rewriteTopDown(if (copy) ir.deepCopy() else ir, {
      case let@Let(binding, value, letBody) =>
        val memo = Memo.empty[Boolean]
        val mentions = memoize(letBody, binding, memo)

        def baseCase(x: IR): BaseIR = {
          val nContains = x.children.count(memo.getOrElse(_, false))
          assert(nContains > 0)
          if (nContains == 1)
            Copy(x, x.children.map { child =>
              if (memo.get(child).contains(true))
                Let(binding, value, child.asInstanceOf[IR])
              else
                child
            })
          else
            let
        }

        if (!mentions)
          letBody
        else letBody match {
          case x@Ref(`binding`, _) => value
          case x@Let(binding2, value2, letBody2) =>
            val pair = (binding, binding2)
            if (m.contains(pair))
              let
            else {
              m += pair
              Let(binding2, value2, Let(binding, value, letBody2))
            }
          case ArrayMap(a, name, body) =>
            if (memo.lookup(body))
              let
            else
              ArrayMap(Let(binding, value, a), name, body)
          case ArrayFilter(a, name, body) =>
            if (memo.lookup(body))
              let
            else
              ArrayFilter(Let(binding, value, a), name, body)
          case ArrayFlatMap(a, name, body) =>
            if (memo.lookup(body))
              let
            else
              ArrayFlatMap(Let(binding, value, a), name, body)
          case ArrayFor(a, name, body) =>
            if (memo.lookup(body))
              let
            else
              ArrayFor(Let(binding, value, a), name, body)
          case x@ArrayFold(a, zero, accumName, valueName, body) =>
            if (memo.lookup(body))
              let
            else
              baseCase(x)
          case x@ArrayScan(a, zero, accumName, valueName, body) =>
            if (memo.lookup(body))
              let
            else
              baseCase(x)
          case _: ApplyAggOp => let
          case _: ApplyScanOp => let
          case _: AggFilter => let
          case _: AggExplode => let
          case _: AggGroupBy => let
          case _: TableAggregate => let
          case _: MatrixAggregate => let
          case x => baseCase(x)
        }
      case other => other
    })
  }
}
