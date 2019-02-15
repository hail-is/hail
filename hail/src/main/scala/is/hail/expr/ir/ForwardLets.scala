package is.hail.expr.ir

import is.hail.utils._

import scala.collection.mutable

object ForwardLets {
  type Strategy = Int
  private val ELIMINATE: Strategy = 0
  private val FORWARD: Strategy = 1
  private val PROCEED: Strategy = 2

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
        val depth = nestingDepth.lookup(x)
        if (nestingDepth.lookup(refs.head) == nestingDepth.lookup(x))
          FORWARD
        else
          PROCEED
      } else PROCEED
    }

    def rewrite(ir: IR): IR = {
      ir match {
        case x: LetNode =>
          val refs = uses.lookup(x)

          (strategy(RefEquality(x), refs): @unchecked) match {
            case ELIMINATE =>
              def visit(ir: IR): Unit = {
                ir match {
                  case x: LetNode =>
                    val refs = uses.lookup(x)
                    refs.foreach { r =>
                      defs.delete(r)
                    }
                    uses.delete(x)
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
                  case _ => VisitIR(visit)(ir)
                }
              }

              visit(x.value)
              rewrite(x.body)

            case FORWARD =>
              val ref = refs.head
              m.bind(ref, x.value)

              rewrite(x.body)

            case PROCEED =>
              MapIR(rewrite)(x)
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
