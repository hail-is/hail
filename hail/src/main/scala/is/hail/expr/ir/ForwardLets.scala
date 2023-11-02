package is.hail.expr.ir

import is.hail.utils.BoxedArrayBuilder

import scala.collection.mutable

object ForwardLets {
  def apply[T <: BaseIR](ir0: T): T = {
    val ir1 = new NormalizeNames(_ => genUID(), allowFreeVariables = true).apply(ir0)
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

      ir match {
        case l@Let(bindings, body) =>
          val refs = uses.lookup(ir)
          val keep = new BoxedArrayBuilder[(String, IR)]
          val newEnv = bindings.foldLeft(env) { case (env, (name, value)) =>
            val rewriteValue = rewrite(value, env).asInstanceOf[IR]
            if (shouldForward(rewriteValue, refs, l))
              env.bindEval(name -> rewriteValue)
            else {keep += (name -> rewriteValue); env}
          }

          Let(keep.underlying(), rewrite(body, newEnv).asInstanceOf[IR])

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
          ir.mapChildrenWithIndex { (ir1, i) =>
            rewrite(ir1, ChildEnvWithoutBindings(ir, i, env))
          }
      }
    }

    rewrite(ir1, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty))).asInstanceOf[T]
  }
}
