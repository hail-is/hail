package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.defs.{Atom, BaseRef, Binding, Block, Ref}
import is.hail.utils.Logging

import scala.collection.Set

object ForwardLets extends Logging {
  def apply[T <: BaseIR](ctx: ExecuteContext, ir0: T): T =
    ctx.time {
      val ir1 = NormalizeNames(allowFreeVariables = true)(ctx, ir0)
      val UsesAndDefs(uses, _, _) = ComputeUsesAndDefs(ir1, errorIfFreeVariables = false)
      val nestingDepth = NestingDepth(ctx, ir1)

      def shouldForward(value: IR, refs: Set[RefEquality[BaseRef]], base: Block, scope: Scope)
        : Boolean =
        IsPure(value) && (
          value.isInstanceOf[Atom] ||
            refs.isEmpty ||
            (refs.size == 1 &&
              nestingDepth.lookupRef(refs.head) == nestingDepth.lookupBinding(base, scope) &&
              !ContainsScan(value) &&
              !ContainsAgg(value)) &&
            !ContainsAggIntermediate(value)
        )

      def rewrite(ir: BaseIR, env: BindingEnv[IR]): BaseIR =
        ir match {
          case l: Block if l.bindings.nonEmpty =>
            val keep = ArraySeq.newBuilder[Binding]
            val refs = uses(l)
            val newEnv = l.bindings.foldLeft(env) {
              case (env, Binding(name, value, scope)) =>
                val rewriteValue = rewrite(value, env.promoteScope(scope)).asInstanceOf[IR]
                val refs_ = refs.filter(_.t.name == name)
                if (shouldForward(rewriteValue, refs_, l, scope)) {
                  if (refs_.nonEmpty) env.bindInScope(name, rewriteValue, scope)
                  else {
                    logger.info(
                      f"Eliminating unused binding:\n" +
                        f"$name: ${value.typ} = ($scope) ${Pretty.ssaStyle(value, preserveNames = true).trim}"
                    )
                    env
                  }
                } else {
                  keep += Binding(name, rewriteValue, scope)
                  env
                }
            }

            val newBody = rewrite(l.body, newEnv).asInstanceOf[IR]
            val newBindings = keep.result()
            if (newBindings.isEmpty) newBody
            else Block(newBindings, newBody)

          case x @ Ref(name, _) =>
            env.eval
              .lookupOption(name)
              .map {
                case forwarded: Atom => forwarded.ir
                case big => big
              }
              .getOrElse(x)
          case _ =>
            ir.mapChildrenWithIndex((ir1, i) =>
              rewrite(ir1, env.extend(Bindings.get(ir, i).dropBindings))
            )
        }

      rewrite(ir1, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty))).asInstanceOf[T]
    }
}
