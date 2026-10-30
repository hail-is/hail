package is.hail.expr.ir

import is.hail.expr.ir.defs.{Atom, Ref}

object Subst {

  private[Subst] type SubstEnv[_] <: BindingEnv[IR]

  private def SubstEnv[A](env: BindingEnv[IR]): SubstEnv[A] =
    env.asInstanceOf[SubstEnv[A]]

  // bind deletes the names: a name the child binds shadows its substitute
  implicit private object scopes extends Scopes[SubstEnv] {
    private val S = BindingEnv.bindingEnvScopes

    override def fresh[A](e: SubstEnv[A]): SubstEnv[A] = SubstEnv(S.fresh(e))
    override def create[A](e: SubstEnv[A], s: Scope): SubstEnv[A] = SubstEnv(S.create(e, s))
    override def drop[A](e: SubstEnv[A], s: Scope): SubstEnv[A] = SubstEnv(S.drop(e, s))
    override def promote[A](e: SubstEnv[A], s: Scope): SubstEnv[A] = SubstEnv(S.promote(e, s))

    override def bind[A](e: SubstEnv[A], s: EnvScope, bs: Seq[(Name, A)]): SubstEnv[A] = {
      val names = bs.map(_._1)
      SubstEnv(s match {
        case Scope.EVAL => e.copy(eval = e.eval.delete(names))
        case Scope.AGG => e.copy(agg = Some(e.agg.get.delete(names)))
        case Scope.SCAN => e.copy(scan = Some(e.scan.get.delete(names)))
        case Scope.REL => e.copy(relational = e.relational.delete(names))
      })
    }
  }

  def apply(e: IR, env: BindingEnv[IR]): IR =
    subst(e, SubstEnv[Unit](env))

  private def subst(e: IR, env: SubstEnv[Unit]): IR =
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
        e.mapChildrenWithEnv(env)((_, _) => ()) {
          case (child: IR, env) => subst(child, env)
          case (child, _) => child
        }.asInstanceOf[IR]
    }
}
