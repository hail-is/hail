package is.hail.expr.ir

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.Scope.{AGG, EVAL, SCAN}
import is.hail.expr.ir.defs.{Atom, Binding, Block, Ref}

import scala.annotation.tailrec

object IRBuilder {
  def scoped(f: IRBuilder => IR): IR = {
    val builder = new IRBuilder()
    val result = f(builder)
    Block(builder.getBindings, result)
  }
}

class IRBuilder {
  private val bindings = ArraySeq.newBuilder[Binding]

  def getBindings: IndexedSeq[Binding] = bindings.result()

  def memoize(ir: IR, scope: Scope = Scope.EVAL): Atom =
    ir match {
      case ir: Atom => ir
      case _ => strictMemoize(ir, scope = scope)
    }

  def strictMemoize(ir: IR, name: Name = freshName(), scope: Scope = Scope.EVAL): Atom = {
    bindings += Binding(name, ir, scope)
    Ref(name, ir.typ)
  }
}

sealed abstract class Memoized[S] {
  import Memoized.{Cont, memo}

  def map(f: Atom => IR): Memoized[S] = flatMap(a => memo(f(a)))
  def flatMap(f: Atom => Memoized[S]): Memoized[S] = new Cont(this, f)
  def >>(m: Memoized[S]): Memoized[S] = flatMap(_ => m)
}

object Memoized {
  private[this] val unit_ : Memoized[Nothing] = pure(0) // ir has no unit
  @inline def unit[S]: Memoized[S] = unit_.asInstanceOf[Memoized[S]]
  @inline def pure[S](a: Atom): Memoized[S] = new Pure(a)
  @inline def let[S](binding: (Name, IR)): Memoized[S] = new Let[S](binding)
  @inline def defer[S](f: IRBuilder => IR): Memoized[S] = new Suspend(f)
  @inline def lift[S](m: Memoized[S]): Lifted[S] = new Lifted[S](m)

  @inline def sequence[S](exprs: Memoized[S]*): Memoized[S] = exprs.foldLeft(unit[S])(_ >> _)

  @inline def memo[S](ir: IR): Memoized[S] =
    ir match {
      case a: Atom => pure(a)
      case big => let(freshName() -> big)
    }

  sealed abstract class HasScope[S](val scope: Scope)
  implicit object evalScope extends HasScope[EVAL.type](EVAL)
  implicit object aggScope extends HasScope[AGG.type](AGG)
  implicit object scanScope extends HasScope[SCAN.type](SCAN)

  @inline def eval(m: Memoized[EVAL.type]): IR = m
  @inline def agg(m: Memoized[AGG.type]): IR = m
  @inline def scan(m: Memoized[SCAN.type]): IR = m

  final private class Pure[S](val a: Atom) extends Memoized[S]
  final private class Let[S](val binding: (Name, IR)) extends Memoized[S]
  final private class Cont[S](val m: Memoized[S], val f: Atom => Memoized[S]) extends Memoized[S]
  final private class Suspend[S](val f: IRBuilder => IR) extends Memoized[S]

  final class Lifted[S](val m: Memoized[S]) extends AnyVal

  private def run[S](m0: Memoized[S])(ib: IRBuilder)(implicit ev: HasScope[S]): IR = {
    @tailrec def go(m: Memoized[S], stack: List[Atom => Memoized[S]]): IR =
      m match {
        case m: Let[S @unchecked] =>
          val b = m.binding
          stack match {
            case Nil => b._2
            case k :: rest => go(k(ib.strictMemoize(b._2, b._1, ev.scope)), rest)
          }

        case m: Cont[S @unchecked] =>
          go(m.m, m.f :: stack)

        case m: Pure[S @unchecked] =>
          stack match {
            case Nil => m.a
            case k :: rest => go(k(m.a), rest)
          }

        case m: Suspend[S @unchecked] =>
          go(memo(m.f(ib)), stack)
      }

    go(m0, Nil)
  }

  implicit private def memoizedToIR[S: HasScope](m: Memoized[S]): IR =
    IRBuilder.scoped(run[S](m))

  implicit def liftedToMemoized[S0: HasScope, S](m: Lifted[S0]): Memoized[S] =
    defer[S](run(m.m))
}
