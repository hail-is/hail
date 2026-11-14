package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, Family, IR}
import is.hail.types.virtual.TVoid
import is.hail.utils.TimedBlock

import scala.collection.mutable

object EvalRelationalValues {
  // Replaces each relational value, a node in the Relational family whose type isn't void, with a
  // literal of its value, innermost first. `passesBelow` lowers each value so it can be evaluated.
  // A relational value reads no name bound outside it (each table node's IR children start from a
  // fresh env), so equal values are evaluated once.
  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR =
    TimedBlock.enter {
      val memo = mutable.HashMap.empty[IR, IR]

      def execute(value: IR): IR =
        CompileAndEvaluate.evalToIR(ctx, passesBelow(ctx, value).asInstanceOf[IR])

      def lower(ir: BaseIR): BaseIR =
        ir.mapChildren(lower) match {
          case x: IR if x.tag.family.toInt == Family.Relational && x.typ != TVoid =>
            memo.getOrElseUpdate(x, execute(x)).deepCopy
          case x => x
        }

      lower(ir)
    }
}
