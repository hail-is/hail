package is.hail.expr.ir

import is.hail.utils._

import scala.collection.mutable

object ForwardLets {
  type Strategy = Int
  private val ELIMINATE: Strategy = 0
  private val FORWARD: Strategy = 1
  private val PROCEED: Strategy = 2

  def valueAndBody(x: IR): (IR, IR) = (x: @unchecked) match {
    case Let(_, value, body) => (value, body)
    case AggLet(_, value, body) => (value, body)
  }

  private def rewriteIR(ir0: IR): IR = {
    val UsesAndDefs(uses, defs) = ComputeUsesAndDefs(ir0)
    val nestingDepth = NestingDepth(ir0)

    val m = Memo.empty[IR]

    var needsRewrite = true

    def strategy(x: RefEquality[IR], refs: mutable.Set[RefEquality[Ref]]): Strategy = {
      val size = refs.size

      if (size == 0)
        ELIMINATE
      else if (size == 1) {
        if (nestingDepth.lookup(refs.head) == nestingDepth.lookup(x))
          FORWARD
        else
          PROCEED
      } else PROCEED
    }

    def rewrite(ir: IR): IR = {
      ir match {
        case _: Let | _: AggLet =>
          val (value, body) = valueAndBody(ir)
          val refs = uses.lookup(ir)

          (strategy(RefEquality(ir), refs): @unchecked) match {
            case ELIMINATE =>
              def visit(ir: IR): Unit = {
                ir match {
                  case _: Let | _: AggLet =>
                    val refs = uses.lookup(ir)
                    refs.foreach { r =>
                      defs.delete(r)
                    }
                    uses.delete(ir)
                  case r: Ref =>
                    val re = RefEquality(r)

                    m.delete(re) // there could possibly be pending substitutions for `re`

                    defs.get(re).foreach { definition =>
                      val otherUses = uses.lookup(definition)
                      otherUses -= re

                      // with a mutable IR, we could directly mutate the definition
                      // instead we trigger another rewrite pass
                      if (!needsRewrite && strategy(definition, otherUses) != PROCEED)
                        needsRewrite = true
                    }
                  case _ => VisitIRChildren(visit)(ir)
                }
              }

              visit(value)
              rewrite(body)

            case FORWARD =>
              val ref = refs.head
              m.bind(ref, rewrite(value))
              rewrite(body)

            case PROCEED =>
              MapIR(rewrite)(ir)
          }
        case r: Ref =>
          val re = RefEquality(r)
          m.get(re) match {
            case Some(replacement) =>
              m.delete(re)
              replacement
            case None => r
          }
        case _ =>
          MapIR(rewrite)(ir)
      }
    }

    var ir = ir0
    var i = 0
    val maxIter = 25
    while (needsRewrite && i < maxIter) {
      needsRewrite = false
      ir = rewrite(ir0)
      i += 1
    }

    log.info(s"ForwardLets: iterated $i times")

    ir
  }

  def apply(ir0: BaseIR, needsCopy: Boolean = true): BaseIR = MapIRSubtrees(rewriteIR)(if (needsCopy) ir0.deepCopy() else ir0)
}
