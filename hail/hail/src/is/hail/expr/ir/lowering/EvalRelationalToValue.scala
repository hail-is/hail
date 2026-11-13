package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, Family, IR}
import is.hail.types.virtual.TVoid
import is.hail.utils.TimedBlock

import scala.collection.mutable

object EvalRelationalToValue {
  def apply(ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): BaseIR =
    TimedBlock.enter {
      // Sound because a RelationalToValue node reads no name bound outside it: each table node's
      // IR children start from a fresh env.
      val memo = mutable.HashMap.empty[IR, IR]

      def execute(value: IR): IR =
        CompileAndEvaluate.evalToIR(ctx, passesBelow(ctx, value).asInstanceOf[IR])

      def lower(ir: BaseIR): BaseIR =
        ir.mapChildren(lower) match {
          // Void nodes are effects; LowerOrInterpretNonCompilable runs them in program order.
          case x: IR if x.tag.family.toInt == Family.RelationalToValue && x.typ != TVoid =>
            memo.getOrElseUpdate(x, execute(x)).deepCopy
          case x => x
        }

      lower(ir)
    }
}
