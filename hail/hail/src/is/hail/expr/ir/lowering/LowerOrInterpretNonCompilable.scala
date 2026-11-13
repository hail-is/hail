package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, IR, Interpret}
import is.hail.expr.ir.defs.{Begin, Literal}
import is.hail.types.virtual.TVoid
import is.hail.utils._

object LowerOrInterpretNonCompilable extends Logging {

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {

    def evaluate(value: IR): IR = {
      val preTime = System.nanoTime()
      val result = CanLowerEfficiently(ctx, value) match {
        case Some(failReason) =>
          logger.info(s"LowerOrInterpretNonCompilable: cannot efficiently lower query: $failReason")
          logger.info(s"interpreting non-compilable result: ${value.getClass.getSimpleName}")
          val v = Interpret.alreadyLowered(ctx, value)
          if (value.typ == TVoid) {
            Begin(FastSeq())
          } else Literal.coerce(value.typ, v)
        case None =>
          logger.info(s"LowerOrInterpretNonCompilable: whole stage code generation is a go!")
          logger.info(s"lowering result: ${value.getClass.getSimpleName}")
          val fullyLowered = LowerToDistributedArrayPass(DArrayLowering.All)(ctx, value)
            .asInstanceOf[IR]
          logger.info(s"compiling and evaluating result: ${value.getClass.getSimpleName}")
          CompileAndEvaluate.evalToIR(ctx, fullyLowered)
      }
      logger.info(s"took ${formatTime(System.nanoTime() - preTime)}")
      assert(result.typ == value.typ)
      result
    }

    def rewrite(x: BaseIR): BaseIR =
      x match {
        case x: IR if InterpretableButNotCompilable(x) =>
          evaluate(x.mapChildren(rewrite).asInstanceOf[IR])
        case _ => x.mapChildren(rewrite)
      }

    rewrite(ir)
  }
}
