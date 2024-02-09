package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.types.virtual.TVoid
import is.hail.utils.BoxedArrayBuilder

import scala.collection.Set

object ForwardLets {
  def apply[T <: BaseIR](ctx: ExecuteContext)(ir0: T): T = {
    val ir1 = new NormalizeNames(_ => genUID(), allowFreeVariables = true)(ctx, ir0)
    val UsesAndDefs(uses, defs, _) = ComputeUsesAndDefs(ir1, errorIfFreeVariables = false)
    val nestingDepth = NestingDepth(ir1)

    def rewrite(ir: BaseIR, env: BindingEnv[IR]): BaseIR = {

      def shouldForward(value: IR, refs: Set[RefEquality[BaseRef]], base: IR): Boolean = {
        IsPure(value) && (
          value.isInstanceOf[Ref] ||
            value.isInstanceOf[In] ||
            (IsConstant(value) && !value.isInstanceOf[Str]) ||
            refs.isEmpty ||
            (refs.size == 1 &&
              nestingDepth.lookup(refs.head) == nestingDepth.lookup(base) &&
              !ContainsScan(value) &&
              !ContainsAgg(value)) &&
            !ContainsAggIntermediate(value)
        )
      }

      ir match {
        case l @ Let(bindings, body) =>
          val keep = new BoxedArrayBuilder[(String, IR)]
          val refs = uses(ir)
          val newEnv = bindings.foldLeft(env) { case (env, (name, value)) =>
            val rewriteValue = rewrite(value, env).asInstanceOf[IR]
            if (
              rewriteValue.typ != TVoid
              && shouldForward(rewriteValue, refs.filter(_.t.name == name), l)
            ) {
              env.bindEval(name -> rewriteValue)
            } else {
              keep += (name -> rewriteValue)
              env
            }
          }

          val newBody = rewrite(body, newEnv).asInstanceOf[IR]
          if (keep.isEmpty) newBody
          else Let(keep.result(), newBody)

        case l @ AggLet(name, value, body, isScan) =>
          val refs = uses.lookup(ir)
          val rewriteValue =
            rewrite(value, if (isScan) env.promoteScan else env.promoteAgg).asInstanceOf[IR]
          if (shouldForward(rewriteValue, refs, l))
            if (isScan)
              rewrite(body, env.copy(scan = Some(env.scan.get.bind(name -> rewriteValue))))
            else
              rewrite(body, env.copy(agg = Some(env.agg.get.bind(name -> rewriteValue))))
          else
            AggLet(name, rewriteValue, rewrite(body, env).asInstanceOf[IR], isScan)
        case x @ Ref(name, _) =>
          env.eval
            .lookupOption(name)
            .map { forwarded =>
              if (uses.lookup(defs.lookup(x)).count(_.t.name == name) > 1) forwarded.deepCopy()
              else forwarded
            }
            .getOrElse(x)
        case _ =>
          ir.mapChildrenWithIndex((ir1, i) => rewrite(ir1, ChildEnvWithoutBindings(ir, i, env)))
      }
    }

    rewrite(ir1, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty))).asInstanceOf[T]
  }
}
