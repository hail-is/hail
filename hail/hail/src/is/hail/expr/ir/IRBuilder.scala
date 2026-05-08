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
  import Memoized.{Cont, pure}

  def map(f: Atom => IR): Memoized[S] = flatMap(a => pure(f(a)))
  def flatMap(f: Atom => Memoized[S]): Memoized[S] = new Cont(this, f)
}

object Memoized {

  def pure[S](ir: IR): Memoized[S] = new Mem[S](ir)
  def let[S](n: Name, ir: IR): Memoized[S] = new Let[S](n, ir)
  def defer[S](f: IRBuilder => IR): Memoized[S] = new Suspend(f)

  sealed abstract class HasScope[S](val scope: Scope)
  implicit object evalScope extends HasScope[EVAL.type](EVAL)
  implicit object aggScope extends HasScope[AGG.type](AGG)
  implicit object scanScope extends HasScope[SCAN.type](SCAN)

  def eval(m: Memoized[EVAL.type]): IR = m
  def agg(m: Memoized[AGG.type]): IR = m
  def scan(m: Memoized[SCAN.type]): IR = m

  final private class Mem[S](val expr: IR) extends Memoized[S]
  final private class Let[S](val name: Name, val expr: IR) extends Memoized[S]
  final private class Cont[S](val m: Memoized[S], val f: Atom => Memoized[S]) extends Memoized[S]
  final private class Suspend[S](val f: IRBuilder => IR) extends Memoized[S]

  private def run[S](m0: Memoized[S])(b: IRBuilder)(implicit ev: HasScope[S]): IR = {
    @tailrec def go(m: Memoized[S], stack: List[Atom => Memoized[S]]): IR =
      m match {
        case m: Mem[S @unchecked] =>
          stack match {
            case Nil => m.expr
            case k :: rest => go(k(b.memoize(m.expr, scope = ev.scope)), rest)
          }

        case m: Let[S @unchecked] =>
          stack match {
            case Nil => m.expr
            case k :: rest => go(k(b.strictMemoize(m.expr, m.name, ev.scope)), rest)
          }

        case m: Cont[S @unchecked] =>
          go(m.m, m.f :: stack)

        case m: Suspend[S @unchecked] =>
          go(pure(m.f(b)), stack)
      }

    go(m0, Nil)
  }

  implicit def memoizedToIR[S: HasScope](m: Memoized[S]): IR =
    IRBuilder.scoped(run[S](m))
}
