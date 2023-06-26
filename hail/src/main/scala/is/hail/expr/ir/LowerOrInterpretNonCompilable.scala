package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.{CanLowerEfficiently, DArrayLowering, LowerToDistributedArrayPass}
import is.hail.types.virtual.TVoid
import is.hail.utils._

import scala.collection.mutable

object LowerOrInterpretNonCompilable {

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {

    def evaluate(value: IR): IR = {
      val preTime = System.nanoTime()
      val result = CanLowerEfficiently(ctx, value) match {
        case Some(failReason) =>
          log.info(s"LowerOrInterpretNonCompilable: cannot efficiently lower query: $failReason")
          log.info(s"interpreting non-compilable result: ${ value.getClass.getSimpleName }")
          val v = Interpret.alreadyLowered(ctx, value)
          if (value.typ == TVoid) {
            Begin(FastIndexedSeq())
          } else Literal.coerce(value.typ, v)
        case None =>
          log.info(s"LowerOrInterpretNonCompilable: whole stage code generation is a go!")
          log.info(s"lowering result: ${ value.getClass.getSimpleName }")
          val fullyLowered = LowerToDistributedArrayPass(DArrayLowering.All).transform(ctx, value)
            .asInstanceOf[IR]
          log.info(s"compiling and evaluating result: ${ value.getClass.getSimpleName }")
          CompileAndEvaluate.evalToIR(ctx, fullyLowered, true)
      }
      log.info(s"took ${ formatTime(System.nanoTime() - preTime) }")
      assert(result.typ == value.typ)
      result
    }

    def rewriteChildren(x: BaseIR, m: mutable.Map[String, IR]): BaseIR =
      x.mapChildren(rewrite(_, m))

    def rewrite(x: BaseIR, m: mutable.Map[String, IR]): BaseIR = {

      x match {
        case RelationalLet(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetTable(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetMatrixTable(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalLetBlockMatrix(name, value, body) =>
          rewrite(body, m += (name -> evaluate(rewrite(value, m).asInstanceOf[IR])))
        case RelationalRef(name, t) =>
          m.get(name) match {
            case Some(res) =>
              assert(res.typ == t)
              res
            case None => throw new RuntimeException(name)
          }
        case x: IR if InterpretableButNotCompilable(x) => evaluate(rewriteChildren(x, m).asInstanceOf[IR])
        case _ => rewriteChildren(x, m)
      }
    }

    rewrite(ir.noSharing, mutable.HashMap.empty)
  }
}
