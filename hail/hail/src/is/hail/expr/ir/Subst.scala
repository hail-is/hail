package is.hail.expr.ir

object Subst {
  def apply(e: IR): IR = apply(e, BindingEnv.empty[IR])

  def apply(e: IR, env: BindingEnv[IR]): IR =
    subst(e, env)

  private def subst(e: IR, env: BindingEnv[IR]): IR = {
    if (env.allEmpty)
      return e

    e match {
      case x @ Ref(name, _) =>
        env.eval.lookupOption(name).getOrElse(x)
      case _ =>
        e.mapChildrenWithIndex {
          case (child: IR, i) => subst(child, env.subtract(Bindings.get(e, i)))
          case (child, _) => child
        }
    }
  }
}
