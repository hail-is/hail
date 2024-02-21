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

      def shouldForward(value: IR, refs: Set[RefEquality[BaseRef]], base: Let, scope: Int): Boolean = {
        IsPure(value) && (
          value.isInstanceOf[Ref] ||
            value.isInstanceOf[In] ||
            (IsConstant(value) && !value.isInstanceOf[Str]) ||
            refs.isEmpty ||
            (refs.size == 1 &&
              nestingDepth.lookupRef(refs.head) == nestingDepth.lookupBinding(base, scope) &&
              !ContainsScan(value) &&
              !ContainsAgg(value)) &&
            !ContainsAggIntermediate(value)
        )
      }

      ir match {
        case l: Let =>
          val keep = new BoxedArrayBuilder[Binding]
          val refs = uses(l)
          val newEnv = l.bindings.foldLeft(env) {
            case (env, Binding(name, value, scope)) =>
              val rewriteValue = rewrite(value, env.promoteScope(scope)).asInstanceOf[IR]
              if (
                rewriteValue.typ != TVoid
                && shouldForward(rewriteValue, refs.filter(_.t.name == name), l, scope)
              ) {
                scope match {
                  case Scope.EVAL => env.bindEval(name -> rewriteValue)
                  case Scope.AGG => env.copy(agg = Some(env.agg.get.bind(name -> rewriteValue)))
                  case Scope.SCAN => env.copy(scan = Some(env.scan.get.bind(name -> rewriteValue)))
                }
              } else {
                keep += Binding(name, rewriteValue, scope)
                env
              }
          }

          val newBody = rewrite(l.body, newEnv).asInstanceOf[IR]
          if (keep.isEmpty) newBody
          else Let.withAgg(keep.result(), newBody)

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
