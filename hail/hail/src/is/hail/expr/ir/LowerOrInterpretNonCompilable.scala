package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{Begin, Literal, RelationalLet, RelationalRef}
import is.hail.expr.ir.lowering.{CanLowerEfficiently, DArrayLowering, LowerToDistributedArrayPass}
import is.hail.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

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
          val fullyLowered = LowerToDistributedArrayPass(DArrayLowering.All).transform(ctx, value)
            .asInstanceOf[IR]
          logger.info(s"compiling and evaluating result: ${value.getClass.getSimpleName}")
          CompileAndEvaluate.evalToIR(ctx, fullyLowered)
      }
      logger.info(s"took ${formatTime(System.nanoTime() - preTime)}")
      assert(result.typ == value.typ)
      result
    }

    def rewriteChildren(x: BaseIR, m: mutable.Map[Name, IR]): BaseIR =
      x.mapChildren(rewrite(_, m))

    def rewrite(x: BaseIR, m: mutable.Map[Name, IR]): BaseIR = {

      x match {
        case RelationalLet(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetTable(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetMatrixTable(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalRef(name, t) =>
          m.get(name) match {
            case Some(res) =>
              assert(res.typ == t)
              res
            case None => throw new RuntimeException(name.str)
          }
        case x: IR if InterpretableButNotCompilable(x) =>
          evaluate(rewriteChildren(x, m).asInstanceOf[IR])
        case _ => rewriteChildren(x, m)
      }
    }

    rewrite(ir.noSharing(ctx), mutable.HashMap.empty)
  }
}
