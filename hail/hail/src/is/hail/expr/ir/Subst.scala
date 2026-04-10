package is.hail.expr.ir

import is.hail.expr.ir.defs.{Ref, TrivialIR}

object Subst {

  def apply(e: IR, env: BindingEnv[IR]): IR =
    if (env.allEmpty) e
    else e match {
      case x @ Ref(name, _) =>
        env.eval.lookupOption(name)
          .map {
            case y: TrivialIR => y.clone
            case y => y
          }
          .getOrElse(x)
      case _ =>
        e.mapChildrenWithIndex {
          case (child: IR, i) => Subst(child, env.subtract(Bindings.get(e, i)))
          case (child, _) => child
        }
    }
}
