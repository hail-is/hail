package is.hail.expr.ir.lowering

import cats.syntax.all._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, IR, RelationalLet, RelationalLetMatrixTable, RelationalLetTable, RelationalRef}
import is.hail.utils.traverseInstanceGenTraversable

import scala.language.higherKinds

object EvalRelationalLets {
  // need to run the rest of lowerings to eval.
  def apply[M[_]: MonadLower](ir: BaseIR, ctx: ExecuteContext, passesBelow: LoweringPipeline): M[BaseIR] = {
    def execute(value: BaseIR, letsAbove: Map[String, IR]): M[IR] =
      for {
        lowered <- lower(value, letsAbove)
        compilable <- passesBelow.apply(ctx, lowered)
        result <- CompileAndEvaluate.evalToIR(ctx, compilable.asInstanceOf[IR])
    } yield result

    def lower(ir: BaseIR, letsAbove: Map[String, IR]): M[BaseIR] = {
      ir match {
        case RelationalLet(name, value, body) =>
          for {
            valueLit <- execute(value, letsAbove)
            lowered <- lower(body, letsAbove + (name -> valueLit))
          } yield lowered

        case RelationalLetTable(name, value, body) =>
          for {
            valueLit <- execute(value, letsAbove)
            lowered <- lower(body, letsAbove + (name -> valueLit))
          } yield lowered

        case RelationalLetMatrixTable(name, value, body) =>
          for {
            valueLit <- execute(value, letsAbove)
            lowered <- lower(body, letsAbove + (name -> valueLit))
          } yield lowered

        case RelationalRef(name, _) =>
          MonadLower[M].pure(letsAbove(name))

        case x =>
          val children = x.children
          children.traverse(lower(_, letsAbove)).map { newChildren =>
            if (children.zip(newChildren).forall { case (a, b) => a eq b }) x
            else x.copy(newChildren)
          }
      }
    }

    lower(ir, Map())
  }
}
