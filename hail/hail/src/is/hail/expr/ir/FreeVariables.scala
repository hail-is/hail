package is.hail.expr.ir

import is.hail.expr.ir.defs.Ref

import scala.collection.mutable

case class FreeVariableEnv(boundVars: Set[Name], freeVars: mutable.Set[Name]) {
  def visitRef(name: Name): Unit =
    if (!boundVars.contains(name))
      freeVars += name

  def bindIterable(bindings: Seq[(Name, Any)]): FreeVariableEnv =
    copy(boundVars = boundVars ++ bindings.view.map(_._1))

  def getFreeVars: Env[Unit] = new Env[Unit].bindIterable(freeVars.view.map(n => (n, ())))
}

// The free variables of eval, agg and scan.
case class FreeVariableBindingEnv(
  evalVars: Option[FreeVariableEnv],
  aggVars: Option[FreeVariableEnv],
  scanVars: Option[FreeVariableEnv],
) {
  def visitRef(name: Name): Unit =
    evalVars.foreach(_.visitRef(name))

  def getFreeVars: BindingEnv[Unit] = BindingEnv(
    evalVars.map(_.getFreeVars).getOrElse(Env.empty),
    aggVars.map(_.getFreeVars),
    scanVars.map(_.getFreeVars),
  )
}

object FreeVariableBindingEnv {
  type E[_] = FreeVariableBindingEnv

  // As BindingEnv's, but binding or promoting a missing scope does nothing, and relational
  // names aren't tracked
  implicit object freeVariableScopes extends Scopes[E] {
    override def bind[A](e: E[A], s: EnvScope, bs: Seq[(Name, A)]): E[A] =
      s match {
        case Scope.EVAL => e.copy(evalVars = e.evalVars.map(_.bindIterable(bs)))
        case Scope.AGG => e.copy(aggVars = e.aggVars.map(_.bindIterable(bs)))
        case Scope.SCAN => e.copy(scanVars = e.scanVars.map(_.bindIterable(bs)))
        case Scope.REL => e
      }

    override def create[A](e: E[A], s: Scope): E[A] =
      s match {
        case Scope.AGG => e.copy(aggVars = e.evalVars)
        case Scope.SCAN => e.copy(scanVars = e.evalVars)
        case Scope.EVAL => throw new IllegalArgumentException("can't create eval")
      }

    override def drop[A](e: E[A], s: Scope): E[A] =
      s match {
        case Scope.EVAL => e.copy(evalVars = None)
        case Scope.AGG => e.copy(aggVars = None)
        case Scope.SCAN => e.copy(scanVars = None)
      }

    override def promote[A](e: E[A], s: Scope): E[A] =
      s match {
        case Scope.EVAL => e
        case Scope.AGG => e.copy(evalVars = e.aggVars, aggVars = None)
        case Scope.SCAN => e.copy(evalVars = e.scanVars, scanVars = None)
      }

    override def fresh[A](e: E[A]): E[A] =
      FreeVariableBindingEnv(None, None, None)
  }
}

object FreeVariables {
  def apply(ir: IR, supportsAgg: Boolean, supportsScan: Boolean): BindingEnv[Unit] = {
    val env =
      FreeVariableBindingEnv(
        Some(FreeVariableEnv(Set.empty, mutable.Set.empty)),
        Option.when(supportsAgg)(FreeVariableEnv(Set.empty, mutable.Set.empty)),
        Option.when(supportsScan)(FreeVariableEnv(Set.empty, mutable.Set.empty)),
      )

    VisitIR.withEnv[FreeVariableBindingEnv.E, Unit](ir, env)((_, _) => ()) { (ir, env) =>
      ir match {
        case Ref(name, _) => env.visitRef(name)
        case _ =>
      }
    }
    env.getFreeVars
  }
}
