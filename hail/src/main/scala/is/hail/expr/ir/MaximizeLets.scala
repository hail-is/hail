package is.hail.expr.ir

import scala.collection.mutable

object MaximizeLets {
  def apply(ir: BaseIR): BaseIR = {
    val m = mutable.HashSet.empty[(String, String)]

    RewriteBottomUp(ir, {
      case ir: IR =>
        val children = ir.children
        val letIndices: Array[Int] = children.zipWithIndex.flatMap {
          case (x@Let(letName, letValue, letBody), idx) => ir match {
            case Let(parentName, _, _) =>
              assert(letName != parentName)
              val pair = letName -> parentName
              if (m.contains(pair) || Mentions(letValue, parentName))
                None
              else {
                m += pair
                Some(idx)
              }
            case ArrayMap(_, name, _) if Mentions(letValue, name) => None
            case ArrayFilter(_, name, body) if Mentions(letValue, name) => None
            case ArrayFlatMap(_, name, body) if Mentions(letValue, name) => None
            case ArrayFor(_, name, body) if Mentions(letValue, name) => None
            case ArrayFold(_, _, name1, name2, body) if Mentions(letValue, name1) || Mentions(letValue, name2) => None
            case ArrayScan(_, _, name1, name2, body) if Mentions(letValue, name1) || Mentions(letValue, name2) => None
            case _: ApplyAggOp => None
            case _: ApplyScanOp => None
            case _: AggFilter => None
            case _: AggExplode => None
            case _: AggGroupBy => None
            case _: TableAggregate => None
            case _: MatrixAggregate => None
            case _ => Some(idx)
          }
          case _ => None
        }.toArray
        if (letIndices.isEmpty)
          None
        else {
          val letIndicesSet = letIndices.toSet
          val replaced = ir.copy(ir.children.zipWithIndex.map {
            case (x@Let(varName, _, body), idx) =>
              if (letIndices.contains(idx))
                body
              else
                x
            case (child, idx) =>
            assert(!letIndices.contains(idx))
            child
          })
          Some(letIndices.map(children).foldLeft(replaced) {
            case (Let(name, value, body), let: Let) if value == let.value =>
            // elide duplicated bindings
              Let(name, value, Subst(body, Env(let.name -> Ref(name, value.typ))))
            case (replacedIR, let: Let) =>
              Let(let.name, let.value, replacedIR)
          })
        }
      case _ => None
    })
  }
}