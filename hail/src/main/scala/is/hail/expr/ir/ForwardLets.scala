package is.hail.expr.ir

import is.hail.backend.ExecuteContext

import scala.collection.mutable

object ForwardLets {
  def apply[T <: BaseIR](ctx: ExecuteContext)(ir0: T): T = {
    val ir1 = new NormalizeNames(_ => genUID(), allowFreeVariables = true)(ctx, ir0)
    val UsesAndDefs(uses, defs, _) = ComputeUsesAndDefs(ir1, errorIfFreeVariables = false)
    val nestingDepth = NestingDepth(ir1)

    def rewrite(ir: BaseIR, env: BindingEnv[IR]): BaseIR = {

      def shouldForward(value: IR, refs: mutable.Set[RefEquality[BaseRef]], base: IR): Boolean = {
        value.isInstanceOf[Ref] ||
        value.isInstanceOf[In] ||
          (IsConstant(value) && !value.isInstanceOf[Str]) ||
          refs.isEmpty ||
          (refs.size == 1 &&
            nestingDepth.lookup(refs.head) == nestingDepth.lookup(base) &&
            !ContainsScan(value) &&
            !ContainsAgg(value)) &&
            !ContainsAggIntermediate(value)
      }

      def mapRewrite(): BaseIR = ir.mapChildrenWithIndex { (ir1, i) =>
        rewrite(ir1, ChildEnvWithoutBindings(ir, i, env))
      }

      ir match {
        case l@Let(name, value, body) =>
          val refs = uses.lookup(ir)
          val rewriteValue = rewrite(value, env).asInstanceOf[IR]
          if (shouldForward(rewriteValue, refs, l))
            rewrite(body, env.bindEval(name -> rewriteValue))
          else
            Let(name, rewriteValue, rewrite(body, env).asInstanceOf[IR])
        case l@AggLet(name, value, body, isScan) =>
          val refs = uses.lookup(ir)
          val rewriteValue = rewrite(value, if (isScan) env.promoteScan else env.promoteAgg).asInstanceOf[IR]
          if (shouldForward(rewriteValue, refs, l))
            if (isScan)
              rewrite(body, env.copy(scan = Some(env.scan.get.bind(name -> rewriteValue))))
            else
              rewrite(body, env.copy(agg = Some(env.agg.get.bind(name -> rewriteValue))))
          else
            AggLet(name, rewriteValue, rewrite(body, env).asInstanceOf[IR], isScan)
        case x@Ref(name, _) => env.eval.lookupOption(name)
          .map { forwarded => if (uses.lookup(defs.lookup(x)).size > 1) forwarded.deepCopy() else forwarded }
          .getOrElse(x)
        case _ =>
          mapRewrite()
      }
    }

    rewrite(ir1, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty))).asInstanceOf[T]
  }
}
