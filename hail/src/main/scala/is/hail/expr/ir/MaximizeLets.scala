package is.hail.expr.ir

import scala.collection.mutable

object MaximizeLets {
  def apply(ir: BaseIR): BaseIR = {
    val m = mutable.HashSet.empty[(String, String)]

    RewriteBottomUp(ir, {
      case ir: IR =>
        val letChildren: IndexedSeq[Let] = ir.children.flatMap {
          case x@Let(letName, letValue, letBody) => ir match {
            case Let(parentName, _, _) =>
              assert(letName != parentName)
              val pair = letName -> parentName
              if (m.contains(pair))
                None
              else {
                m += pair
                Some(x)
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
            case _ => Some(x)
          }
          case _ => None
        }
        if (letChildren.isEmpty)
          None
        else {
          val replaced = ir.copy(ir.children.map {
            case Let(_, _, body) => body
            case child => child
          })
          Some(letChildren.foldLeft(replaced) {
            // elide duplicated bindings
            case (Let(name, value, body), let) if value == let.value => Let(name, value, Subst(body, Env(let.name -> Ref(name, value.typ))))
            case (replacedIR, let) => Let(let.name, let.value, replacedIR)
          })
        }
      case _ => None
    })
  }
}