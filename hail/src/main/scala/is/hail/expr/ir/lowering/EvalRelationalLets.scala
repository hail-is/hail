package is.hail.expr.ir.lowering

import cats.syntax.all._
import is.hail.expr.ir.{BaseIR, CompileAndEvaluate, IR, RelationalLet, RelationalLetMatrixTable, RelationalLetTable, RelationalRef}

import scala.language.higherKinds

object EvalRelationalLets {
  // need to run the rest of lowerings to eval.
  def apply[M[_]: MonadLower](ir: BaseIR, passesBelow: LoweringPipeline): M[BaseIR] = {
    def execute(value: BaseIR, letsAbove: Map[String, IR]): M[IR] =
      for {
        lowered <- lower(value, letsAbove)
        compilable <- passesBelow.apply(lowered)
        result <- CompileAndEvaluate.evalToIR(compilable.asInstanceOf[IR])
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
          x.traverseChildren(lower(_, letsAbove))
      }
    }

    lower(ir, Map())
  }
}
