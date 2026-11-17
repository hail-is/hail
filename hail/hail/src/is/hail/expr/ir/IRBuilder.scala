package is.hail.expr.ir

import is.hail.expr.ir.Scope.{AGG, EVAL, SCAN}
import is.hail.expr.ir.defs.{Atom, Binding, Block, Ref}

import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq

object IRBuilder {
  def scoped(f: IRBuilder => IR): IR = {
    val builder = new IRBuilder()
    val result = f(builder)
    val bindings = builder.getBindings
    if (bindings.nonEmpty) Block(bindings, result)
    else result
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

sealed abstract class Memoized {
  import Memoized.{Cont, ret}

  final def map(f: Atom => IR): Memoized = flatMap(a => ret(f(a)))
  final def flatMap(f: Atom => Memoized): Memoized = new Cont(this, f)
  final def >>(m: Memoized): Memoized = flatMap(_ => m)
}

object Memoized {
  val unit: Memoized = ret(0) // hack: ir has no unit
  @inline def ret(ir: IR): Memoized = new Return(ir)
  @inline def let(binding: (Name, IR)): Memoized = new Let(binding)
  @inline def defer(f: IRBuilder => IR): Memoized = new Suspend((b, s) => (new Return(f(b)), s))

  @inline def when(cond: Boolean)(m: Memoized): Memoized = if (cond) m else unit
  @inline def sequence(exprs: Memoized*): Memoized = exprs.foldLeft(unit)(_ >> _)
  @inline def interleave(scope: Scope)(m: Memoized): Memoized = new Suspend((_, _) => (m, scope))

  @inline def eval(m: Memoized): IR = IRBuilder.scoped(run(m, EVAL))
  @inline def agg(m: Memoized): IR = IRBuilder.scoped(run(m, AGG))
  @inline def scan(m: Memoized): IR = IRBuilder.scoped(run(m, SCAN))

  final private class Return(val ir: IR) extends Memoized
  final private class Let(val binding: (Name, IR)) extends Memoized
  final private class Cont(val m: Memoized, val f: Atom => Memoized) extends Memoized
  final private class Suspend(val f: (IRBuilder, Scope) => (Memoized, Scope)) extends Memoized

  private def run(m0: Memoized, scope0: Scope)(ib: IRBuilder): IR = {
    @tailrec def go(m: Memoized, scope: Scope, stack: List[(Scope, Atom => Memoized)]): IR =
      m match {
        case m: Return =>
          stack match {
            case Nil => m.ir
            case (s, k) :: rest => go(k(ib.memoize(m.ir, s)), s, rest)
          }

        case m: Cont =>
          go(m.m, scope, (scope, m.f) :: stack)

        case m: Let =>
          val b = m.binding
          val a = ib.strictMemoize(b._2, b._1, scope)
          go(ret(a), scope, stack)

        case m: Suspend =>
          val (next, nextScope) = m.f(ib, scope)
          go(next, nextScope, stack)
      }

    go(m0, scope0, Nil)
  }
}
