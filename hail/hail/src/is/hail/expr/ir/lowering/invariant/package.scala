package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  BaseIR, Bindings, BlockMatrixIR, Compilable, Emittable, IR, IRTraversal, MatrixIR, Name, Pretty,
  RelationalLetMatrixTable, RelationalLetTable, TableIR, TableKeyBy, TableKeyByAndAggregate,
  TableOrderBy,
}
import is.hail.expr.ir.NormalizeNames.needsRenaming
import is.hail.expr.ir.defs.{ApplyIR, RelationalLet, RelationalRef}
import is.hail.expr.ir.lowering.invariant.Flags.StrictInvariants
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
      ctx.time {
        IRTraversal.trace(ir).foreach { case trace @ ir :: _ =>
          if (!invariant(ir)) throw new UnsatisfiedInvariantError(
            s"""Invariant ${E.value} forbids
               |${trace.take(5).map(Pretty(ctx, _, preserveNames = true)).mkString("\nin\n")}
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

  lazy val AnyIR: Invariant =
    new Invariant {
      override private[invariant] def verify(ctx: ExecuteContext, ir: BaseIR): Unit = ()
    }

  def LowerableIR(implicit E: Enclosing): Invariant =
    TreeIR and NoRedefinedNames

  def TreeIR: Invariant = {
    var mark: Int = 0

    val setup = new Invariant {
      override def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
        mark = ctx.irMetadata.nextFlag
    }

    setup <> Invariant(ir => ir.mark != mark && { ir.mark = mark; true })
  }

  def NoRedefinedNames: Invariant = {
    val names: mutable.Map[Name, BaseIR] =
      is.hail.collection.compat.mutable.AnyRefMap.empty

    (ir: BaseIR) =>
      !needsRenaming(ir) || {
        // ir may bind the same name in multiple children
        val newNames = mutable.HashSet.empty[Name]
        (0 until ir.children.size).forall { i =>
          Bindings.get(ir, i).all.forall { case (name, _) =>
            !newNames.add(name) || names.put(name, ir).forall { orig =>
              throw new UnsatisfiedInvariantError(
                s"""Invariant ${implicitly[Enclosing].value} forbids redefinition of '$name' in
                   |${Pretty.ssaStyle(ir, preserveNames = true)}
                   |Originally bound in
                   |${Pretty.ssaStyle(orig, preserveNames = true)}""".stripMargin
              )
            }
          }
        }
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
