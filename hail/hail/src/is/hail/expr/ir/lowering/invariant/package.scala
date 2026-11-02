package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.NormalizeNames.needsRenaming
import is.hail.expr.ir.defs.{ApplyIR, RelationalLet, RelationalRef}
import is.hail.expr.ir.lowering.invariant.Flags.StrictInvariants
import is.hail.utils.StackSafe._
import is.hail.utils.TimedBlock
import is.hail.utils.implicits.toRichPredicate

import scala.collection.mutable

import sourcecode.Enclosing

package invariant {
  object Flags {
    val StrictInvariants = "strict_invariants"
  }

  class UnsatisfiedInvariantError(msg: String) extends AssertionError(msg)

  sealed abstract class Invariant {
    private[invariant] def verify(ctx: ExecuteContext, ir: BaseIR): Unit
  }

  private case class <>(a: Invariant, b: Invariant) extends Invariant {
    override def verify(ctx: ExecuteContext, ir: BaseIR): Unit = {
      a.verify(ctx, ir)
      b.verify(ctx, ir)
    }
  }

  // Special case for invariants that apply a predicate to every element of the ir.
  // These predicates can be fused together so that we traverse the ir at most once.
  private case class Fused(invariant: BaseIR => Boolean)(implicit E: Enclosing) extends Invariant {
    override def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
      TimedBlock.enter {
        IRTraversal.trace(ir).foreach { case trace @ ir :: _ =>
          if (!invariant(ir)) throw new UnsatisfiedInvariantError(
            s"""Invariant ${E.value} forbids
               |${trace.take(5).map(Pretty(ctx, _)).mkString("\nin\n")}
               |""".stripMargin
          )
        }
      }
  }
}

package object invariant {

  implicit class InvariantOps(private val x: Invariant) extends AnyVal {

    def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
      if (ctx.flags.isDefined(StrictInvariants)) x.verify(ctx, ir)

    def <>(y: Invariant): Invariant =
      new <>(x, y)

    def and(y: Invariant)(implicit E: sourcecode.Enclosing): Invariant =
      (x, y) match {
        case (Fused(p), Fused(q)) => p and q
        case (a <> Fused(p), Fused(q)) => a <> (p and q)
        case _ => x <> y
      }
  }

  implicit def Invariant(p: BaseIR => Boolean)(implicit E: sourcecode.Enclosing): Invariant =
    Fused(p)

  def LowerableIR(implicit E: Enclosing): Invariant =
    NoSharedNodes and UniquelyNamed

  def NoSharedNodes: Invariant = {
    var mark: Int = 0

    val setup = new Invariant {
      override def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
        mark = ctx.irMetadata.nextFlag
    }

    setup <> Invariant(ir => ir.mark != mark && { ir.mark = mark; true })
  }

  object UniquelyNamed extends Invariant {

    private type E[_] = Unit

    // calls record with the names each bind adds
    private def recording(record: Seq[Name] => Unit): Scopes[E] =
      new Scopes[E] {
        override def fresh[A](e: E[A]): E[A] = ()
        override def create[A](e: E[A], s: Scope): E[A] = ()
        override def drop[A](e: E[A], s: Scope): E[A] = ()
        override def promote[A](e: E[A], s: Scope): E[A] = ()
        override def bind[A](e: E[A], s: EnvScope, bs: Seq[(Name, A)]): E[A] = record(bs.map(_._1))
      }

    // A name bound by a node that needs renaming is bound by no other node. Nodes that keep
    // their names (e.g. `row`) may share them with each other, but not with a renamed node.
    override private[invariant] def verify(ctx: ExecuteContext, ir0: BaseIR): Unit = {
      val renamed = mutable.HashMap.empty[Name, BaseIR]
      val kept = mutable.HashMap.empty[Name, BaseIR]

      def collision(name: Name, ir: BaseIR, original: BaseIR): Nothing =
        throw new UnsatisfiedInvariantError(
          s"Invariant '${getClass.getName}' forbids redefinition of '$name' in" + Pretty(ctx, ir) +
            "Originally bound in" + Pretty(ctx, original)
        )

      def record(ir: BaseIR, newNames: mutable.Set[Name])(names: Seq[Name]): Unit = {
        // ir may bind the same name in multiple children
        val filtered = names.filter(newNames.add)
        if (needsRenaming(ir)) filtered.foreach { name =>
          kept.get(name).foreach(collision(name, ir, _))
          renamed.put(name, ir).foreach(collision(name, ir, _))
        }
        else filtered.foreach { name =>
          renamed.get(name).foreach(collision(name, ir, _))
          kept.update(name, ir)
        }
      }

      def go(ir: BaseIR): StackFrame[Unit] = {
        implicit val scopes: Scopes[E] = recording(record(ir, mutable.HashSet.empty))
        ir.forEachChildWithEnvM((): E[Unit])((_, _) => ())((child, _, _) => go(child))
      }

      go(ir0).run()
    }
  }

  lazy val NoMatrixIR: Invariant =
    Invariant(!_.isInstanceOf[MatrixIR])

  lazy val NoTableIR: Invariant =
    Invariant(!_.isInstanceOf[TableIR])

  lazy val NoBlockMatrixIR: Invariant =
    Invariant(!_.isInstanceOf[BlockMatrixIR])

  lazy val NoRelationalLets: Invariant =
    Invariant {
      case _: RelationalLet => false
      case _: RelationalLetMatrixTable => false
      case _: RelationalLetTable => false
      case _: RelationalRef => false
      case _ => true
    }

  lazy val NoTableKeyByAndAggregate: Invariant =
    Invariant(!_.isInstanceOf[TableKeyByAndAggregate])

  lazy val NoApplyIR: Invariant =
    Invariant(!_.isInstanceOf[ApplyIR])

  lazy val ValueIROnly: Invariant =
    Invariant(_.isInstanceOf[IR])

  lazy val CompilableValueIRs: Invariant =
    Invariant {
      case x: IR => Compilable(x)
      case _ => true
    }

  lazy val CompilableIR: Invariant =
    ValueIROnly and CompilableValueIRs

  lazy val EmittableValueIRs: Invariant =
    Invariant {
      case x: IR => Emittable(x)
      case _ => true
    }

  lazy val EmittableIR: Invariant =
    ValueIROnly and EmittableValueIRs

  lazy val LoweredShuffles: Invariant =
    Invariant {
      case t: TableKeyBy => t.definitelyDoesNotShuffle
      case _: TableKeyByAndAggregate => false
      case t: TableOrderBy => t.definitelyDoesNotShuffle
      case _ => true
    }
}
