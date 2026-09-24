package is.hail.expr.ir

import is.hail.expr.ir.defs.{Atom, Ref}

object Subst {

  def apply(e: IR, env: BindingEnv[IR]): IR =
    if (env.allEmpty) e
    else e match {
      case x @ Ref(name, _) =>
        env.eval.lookupOption(name)
          .map {
            // We don't know how many uses of `y` there might be. For Atom, we
            // can safely clone `y` knowing work won't be duplicated. If `y` is big
            // then this is probably a bug and the TreeIR invariant will complain.
            case y: Atom => y.ir
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
