package is.hail.expr.ir

import is.hail.expr.ir.defs.{
  AggArrayPerElement, AggExplode, AggFilter, AggFold, AggGroupBy, ApplyAggOp, ApplyScanOp, Block,
}

object UsesAggEnv {
  def apply(ir0: BaseIR, i: Int): Boolean = ir0 match {
    case Block(bindings, _) if i < bindings.length =>
      bindings(i).scope == Scope.AGG
    case AggGroupBy(_, _, false) => i == 0
    case AggFilter(_, _, false) => i == 0
    case AggExplode(_, _, _, false) => i == 0
    case AggArrayPerElement(_, _, _, _, _, false) => i == 0
    case ApplyAggOp(initOp, _, _) => i >= initOp.length
    case AggFold(_, _, _, _, _, false) => i == 1
    case _ => false
  }
}

object UsesScanEnv {
  def apply(ir0: BaseIR, i: Int): Boolean = ir0 match {
    case Block(bindings, _) if i < bindings.length =>
      bindings(i).scope == Scope.SCAN
    case AggGroupBy(_, _, true) => i == 0
    case AggFilter(_, _, true) => i == 0
    case AggExplode(_, _, _, true) => i == 0
    case AggArrayPerElement(_, _, _, _, _, true) => i == 0
    case ApplyScanOp(initOp, _, _) => i >= initOp.length
    case AggFold(_, _, _, _, _, true) => i == 1
    case _ => false
  }
}

// a scope an env binds names in: a Scope, or Scope.REL, the relational names. REL isn't a
// Scope, so it can't be a Block binding's scope
sealed trait EnvScope

sealed trait Scope extends EnvScope {
  import Scope.{AGG, EVAL, SCAN}

  def toInt: Int =
    this match {
      case EVAL => 0
      case AGG => 1
      case SCAN => 2
    }
}

object Scope {
  case object EVAL extends Scope
  case object AGG extends Scope
  case object SCAN extends Scope
  case object REL extends EnvScope
}

trait Scopes[E[_]] {
  def fresh[A](e: E[A]): E[A]
  def create[A](e: E[A], s: Scope): E[A]
  def drop[A](e: E[A], s: Scope): E[A]
  def promote[A](e: E[A], s: Scope): E[A]
  def bind[A](e: E[A], s: EnvScope, bs: Seq[(Name, A)]): E[A]
}

object Scopes {
  implicit class ScopesSyntax[E[_], A](private val e: E[A]) extends AnyVal {
    def create(s: Scope)(implicit S: Scopes[E]): E[A] = S.create(e, s)
    def drop(s: Scope)(implicit S: Scopes[E]): E[A] = S.drop(e, s)
    def promote(s: Scope)(implicit S: Scopes[E]): E[A] = S.promote(e, s)
    def fresh(implicit S: Scopes[E]): E[A] = S.fresh(e)
    def bind(s: EnvScope, bs: Seq[(Name, A)])(implicit S: Scopes[E]): E[A] = S.bind(e, s, bs)
  }
}
